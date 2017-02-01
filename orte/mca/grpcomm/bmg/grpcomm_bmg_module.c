/*
 *  * Copyright (c) 2007      The Trustees of Indiana University.
 *   *                         All rights reserved.
 *    * Copyright (c) 2011-2015 Cisco Systems, Inc.  All rights reserved.
 *     * Copyright (c) 2011-2016 Los Alamos National Security, LLC. All rights
 *      *                         reserved.
 *       * Copyright (c) 2014-2015 Intel, Inc.  All rights reserved.
 *        * Copyright (c) 2014      Mellanox Technologies, Inc.
 *         *                         All rights reserved.
 *          * $COPYRIGHT$
 *           *
 *            * Additional copyrights may follow
 *             *
 *              * $HEADER$
 *               */

#include "orte_config.h"
#include "orte/constants.h"
#include "orte/types.h"
#include "orte/runtime/orte_wait.h"

#include <math.h>
#include <string.h>

#include "opal/dss/dss.h"
#include "opal/class/opal_list.h"

#include "orte/mca/errmgr/errmgr.h"
#include "orte/mca/rml/rml.h"
#include "orte/mca/state/state.h"
#include "orte/util/name_fns.h"
#include "orte/util/nidmap.h"
#include "orte/util/proc_info.h"
#include "orte/mca/routed/routed.h"
#include "orte/mca/rml/base/rml_contact.h"
#include "orte/mca/errmgr/detector/errmgr_detector.h"
#include "orte/mca/grpcomm/base/base.h"
#include "grpcomm_bmg.h"

/* Static API's */
static int init(void);
static void finalize(void);

static int xcast(orte_vpid_t *vpids,
                 size_t nprocs,
                 opal_buffer_t *buf);

static int allgather(orte_grpcomm_coll_t *coll,
                     opal_buffer_t *buf);

static int rbcast(orte_vpid_t *vpids,
                 size_t nprocs,
                 opal_buffer_t *buf);

static register_cb_type(orte_grpcomm_rbcast_cb_t callback);

static unregister_cb_type(int typr);
/* Module def */
orte_grpcomm_base_module_t orte_grpcomm_bmg_module = {
    init,
    finalize,
    xcast,
    allgather,
    rbcast,
    register_cb_type,
    unregister_cb_type
};

/* Internal functions */
static void xcast_recv(int status, orte_process_name_t* sender,
                       opal_buffer_t* buffer, orte_rml_tag_t tag,
                       void* cbdata);
static void allgather_recv(int status, orte_process_name_t* sender,
                           opal_buffer_t* buffer, orte_rml_tag_t tag,
                           void* cbdata);
static void barrier_release(int status, orte_process_name_t* sender,
                            opal_buffer_t* buffer, orte_rml_tag_t tag,
                            void* cbdata);
static void rbcast_recv(int status, orte_process_name_t* sender,
                       opal_buffer_t* buffer, orte_rml_tag_t tag,
                       void* cbdata);
/* internal variables */
static opal_list_t tracker;

/*
 * registration of callbacks
 */
#define RBCAST_CB_TYPE_MAX 7
static orte_grpcomm_rbcast_cb_t orte_grpcomm_rbcast_cb[RBCAST_CB_TYPE_MAX+1];

int register_cb_type(orte_grpcomm_rbcast_cb_t callback) {
    int i;
    printf("dong grpcomm bmg register cb\n");

    for(i = 0; i < RBCAST_CB_TYPE_MAX; i++) {
        if( NULL == orte_grpcomm_rbcast_cb[i] ) {
            orte_grpcomm_rbcast_cb[i] = callback;
            return i;
        }
    }
    return ORTE_ERR_OUT_OF_RESOURCE;
}

int unregister_cb_type(int type) {
    if( RBCAST_CB_TYPE_MAX < type || 0 > type ) {
        return ORTE_ERR_BAD_PARAM;
    }
    orte_grpcomm_rbcast_cb[type] = NULL;
    return ORTE_SUCCESS;
}

/*
 *  Initialize the module
 */
static int init(void)
{
    OBJ_CONSTRUCT(&tracker, opal_list_t);

   /* post the receives */
    orte_rml.recv_buffer_nb(ORTE_NAME_WILDCARD,
                            ORTE_RML_TAG_BMGXCAST,
                            ORTE_RML_PERSISTENT,
                            xcast_recv, NULL);

    orte_rml.recv_buffer_nb(ORTE_NAME_WILDCARD,
                            ORTE_RML_TAG_ALLGATHER_BMG,
                            ORTE_RML_PERSISTENT,
                            allgather_recv, NULL);

    /* setup recv for barrier release */
    orte_rml.recv_buffer_nb(ORTE_NAME_WILDCARD,
                            ORTE_RML_TAG_BMG_COLL_RELEASE,
                            ORTE_RML_PERSISTENT,
                            barrier_release, NULL);

    orte_rml.recv_buffer_nb(ORTE_NAME_WILDCARD,
                            ORTE_RML_TAG_RBCAST,
                            ORTE_RML_PERSISTENT,
                            rbcast_recv, NULL);

   return OPAL_SUCCESS;
}

/*
 * Finalize the module
 */
static void finalize(void)
{
    /* cancel the xcast recv */
    orte_rml.recv_cancel(ORTE_NAME_WILDCARD, ORTE_RML_TAG_BMGXCAST);
    /* cancel the rbcast recv */
    orte_rml.recv_cancel(ORTE_NAME_WILDCARD, ORTE_RML_TAG_RBCAST);
    OPAL_LIST_DESTRUCT(&tracker);
    return;
}

static int xcast(orte_vpid_t *vpids,
                 size_t nprocs,
                 opal_buffer_t *buf)
{   
    int rc;
    printf("xcast in bmg\n");    
    /* send it to the HNP (could be myself) for relay */
    OBJ_RETAIN(buf);  // we'll let the RML release it
    if (0 > (rc = orte_rml.send_buffer_nb(orte_coll_conduit, ORTE_PROC_MY_HNP, buf, ORTE_RML_TAG_BMGXCAST,
                                          orte_rml_send_callback, NULL))) {
        ORTE_ERROR_LOG(rc);
        OBJ_RELEASE(buf);
       printf("sendbuffer failed \n");
        return rc;
    }
    return ORTE_SUCCESS;
}


static int rbcast(orte_vpid_t *vpids,
                 size_t nprocs,
                 opal_buffer_t *buf)
{
    int rc = false;
    int vpid; 
    int i, d;
    orte_process_name_t daemon; 
    vpid = orte_process_info.my_daemon.vpid;
    printf("xrbcast procs %d buf %s ",nprocs,buf);    
    for(i=1; i <= nprocs/2; i*=2) for(d=1; d >= -1; d-=2) {
        int idx = (nprocs+vpid+d*i)%nprocs;
     redo:
        if( idx == vpid ) continue;

        daemon.jobid = orte_process_info.my_daemon.jobid;
        daemon.vpid = idx;
        OBJ_RETAIN(buf);
        if(0 > (rc = orte_rml.send_buffer_nb(orte_coll_conduit, &daemon, buf, ORTE_RML_TAG_RBCAST, orte_rml_send_callback, NULL))) {
            ORTE_ERROR_LOG(rc);
        }
        if( i == 1 ) {
            /* The ring is cut, find the closest alive neighbor in that direction */
            idx = (nprocs+idx+d)%nprocs;
            /* TODO: find a way to not send twice the message if idx is one of
             * my neighbors for i>1 */
            goto redo;
        }
     }  
    return rc;
}

static int allgather(orte_grpcomm_coll_t *coll,
                     opal_buffer_t *buf)
{
    int rc;
    opal_buffer_t *relay;

    OPAL_OUTPUT_VERBOSE((1, orte_grpcomm_base_framework.framework_output,
                         "%s grpcomm:direct: allgather",
                         ORTE_NAME_PRINT(ORTE_PROC_MY_NAME)));

    /* the base functions pushed us into the event library
 *      * before calling us, so we can safely access global data
 *           * at this point */

    relay = OBJ_NEW(opal_buffer_t);
    /* pack the signature */
    if (OPAL_SUCCESS != (rc = opal_dss.pack(relay, &coll->sig, 1, ORTE_SIGNATURE))) {
        ORTE_ERROR_LOG(rc);
        OBJ_RELEASE(relay);
        return rc;
    }

    /* pass along the payload */
    opal_dss.copy_payload(relay, buf);

    /* send this to ourselves for processing */
    OPAL_OUTPUT_VERBOSE((1, orte_grpcomm_base_framework.framework_output,
                         "%s grpcomm:direct:allgather sending to ourself",
                         ORTE_NAME_PRINT(ORTE_PROC_MY_NAME)));

    /* send the info to ourselves for tracking */
    rc = orte_rml.send_buffer_nb(orte_coll_conduit, ORTE_PROC_MY_NAME, relay,
                                 ORTE_RML_TAG_ALLGATHER_BMG,
                                 orte_rml_send_callback, NULL);
    return rc;
}

static void allgather_recv(int status, orte_process_name_t* sender,
                           opal_buffer_t* buffer, orte_rml_tag_t tag,
                           void* cbdata)
{
    int32_t cnt;
    int rc, ret;
    orte_grpcomm_signature_t *sig;
    opal_buffer_t *reply;
    orte_grpcomm_coll_t *coll;

    OPAL_OUTPUT_VERBOSE((1, orte_grpcomm_base_framework.framework_output,
                         "%s grpcomm:direct allgather recvd from %s",
                         ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),
                         ORTE_NAME_PRINT(sender)));

    /* unpack the signature */
    cnt = 1;
    if (OPAL_SUCCESS != (rc = opal_dss.unpack(buffer, &sig, &cnt, ORTE_SIGNATURE))) {
        ORTE_ERROR_LOG(rc);
        return;
    }

    /* check for the tracker and create it if not found */
    if (NULL == (coll = orte_grpcomm_base_get_tracker(sig, true))) {
        ORTE_ERROR_LOG(ORTE_ERR_NOT_FOUND);
        OBJ_RELEASE(sig);
        return;
    }

    /* increment nprocs reported for collective */
    coll->nreported++;
    /* capture any provided content */
    opal_dss.copy_payload(&coll->bucket, buffer);

    OPAL_OUTPUT_VERBOSE((1, orte_grpcomm_base_framework.framework_output,
                         "%s grpcomm:direct allgather recv nexpected %d nrep %d",
                         ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),
                         (int)coll->nexpected, (int)coll->nreported));

    /* see if everyone has reported */
    if (coll->nreported == coll->nexpected) {
        if (ORTE_PROC_IS_HNP) {
            OPAL_OUTPUT_VERBOSE((1, orte_grpcomm_base_framework.framework_output,
                                 "%s grpcomm:direct allgather HNP reports complete",
                                 ORTE_NAME_PRINT(ORTE_PROC_MY_NAME)));
            /* the allgather is complete - send the xcast */
            reply = OBJ_NEW(opal_buffer_t);
            /* pack the signature */
            if (OPAL_SUCCESS != (rc = opal_dss.pack(reply, &sig, 1, ORTE_SIGNATURE))) {
                ORTE_ERROR_LOG(rc);
                OBJ_RELEASE(reply);
                OBJ_RELEASE(sig);
                return;
            }
            /* pack the status - success since the allgather completed. This
 *              * would be an error if we timeout instead */
            ret = ORTE_SUCCESS;
            if (OPAL_SUCCESS != (rc = opal_dss.pack(reply, &ret, 1, OPAL_INT))) {
                ORTE_ERROR_LOG(rc);
                OBJ_RELEASE(reply);
                OBJ_RELEASE(sig);
                return;
            }
            /* transfer the collected bucket */
            opal_dss.copy_payload(reply, &coll->bucket);
            /* send the release via xcast */
            (void)orte_grpcomm.xcast(sig, ORTE_RML_TAG_BMG_COLL_RELEASE, reply);
            OBJ_RELEASE(reply);
        } else {
            OPAL_OUTPUT_VERBOSE((1, orte_grpcomm_base_framework.framework_output,
                                 "%s grpcomm:direct allgather rollup complete - sending to %s",
                                 ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),
                                 ORTE_NAME_PRINT(ORTE_PROC_MY_PARENT)));
            /* relay the bucket upward */
            reply = OBJ_NEW(opal_buffer_t);
            /* pack the signature */
            if (OPAL_SUCCESS != (rc = opal_dss.pack(reply, &sig, 1, ORTE_SIGNATURE))) {
                ORTE_ERROR_LOG(rc);
                OBJ_RELEASE(reply);
                OBJ_RELEASE(sig);
                return;
            }
            /* transfer the collected bucket */
            opal_dss.copy_payload(reply, &coll->bucket);
            /* send the info to our parent */
            rc = orte_rml.send_buffer_nb(orte_coll_conduit, ORTE_PROC_MY_PARENT, reply,
                                         ORTE_RML_TAG_ALLGATHER_BMG,
                                         orte_rml_send_callback, NULL);
        }
    }
    OBJ_RELEASE(sig);
}



static void xcast_recv(int status, orte_process_name_t* sender,
                       opal_buffer_t* buffer, orte_rml_tag_t tg,
                       void* cbdata)
{
    opal_list_item_t *item;
    orte_namelist_t *nm;
    int ret, cnt;
    opal_buffer_t *relay, *rly;
    orte_daemon_cmd_flag_t command = ORTE_DAEMON_NULL_CMD;
    opal_buffer_t wireup;
    opal_byte_object_t *bo;
    int8_t flag;
    orte_job_t *jdata;
    orte_proc_t *rec;
    opal_list_t coll;
    char *rtmod;
    orte_grpcomm_signature_t *sig;
    orte_rml_tag_t tag;
    printf("xcast recv \n");
    OPAL_OUTPUT_VERBOSE((1, orte_grpcomm_base_framework.framework_output,
                         "%s grpcomm:direct:xcast:recv: with %d bytes",
                         ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),
                         (int)buffer->bytes_used));

    /* we need a passthru buffer to send to our children */
    rly = OBJ_NEW(opal_buffer_t);
    opal_dss.copy_payload(rly, buffer);

    /* get the signature that we do not need */
    cnt=1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &sig, &cnt, ORTE_SIGNATURE))) {
        ORTE_ERROR_LOG(ret);
        ORTE_FORCED_TERMINATE(ret);
        return;
    }
    OBJ_RELEASE(sig);

    /* get the target tag */
    cnt=1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &tag, &cnt, ORTE_RML_TAG))) {
        ORTE_ERROR_LOG(ret);
        ORTE_FORCED_TERMINATE(ret);
        return;
    }

    /* setup a buffer we can pass to ourselves - this just contains
     * the initial message, minus the headers inserted by xcast itself */
    relay = OBJ_NEW(opal_buffer_t);
    opal_dss.copy_payload(relay, buffer);
    /* setup the relay list */
    OBJ_CONSTRUCT(&coll, opal_list_t);

    /* get our conduit's routed module name */
    rtmod = orte_rml.get_routed(orte_coll_conduit);


    /* if this is headed for the daemon command processor,
     * then we first need to check for add_local_procs
     * as that command includes some needed wireup info */
    if (ORTE_RML_TAG_DAEMON == tag) {
        /* peek at the command */
        cnt=1;
        if (ORTE_SUCCESS == (ret = opal_dss.unpack(buffer, &command, &cnt, ORTE_DAEMON_CMD))) {
            /* if it is add_procs, then... */
            if (ORTE_DAEMON_ADD_LOCAL_PROCS == command ||
                ORTE_DAEMON_DVM_NIDMAP_CMD == command) {
                /* extract the byte object holding the daemonmap */
                cnt=1;
                if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &bo, &cnt, OPAL_BYTE_OBJECT))) {
                    ORTE_ERROR_LOG(ret);
                    goto relay;
                }

                /* update our local nidmap, if required - the decode function
                 * knows what to do - it will also free the bytes in the byte object
                 */
                if (ORTE_PROC_IS_HNP) {
                    /* no need - already have the info */
                    if (NULL != bo) {
                        if (NULL != bo->bytes) {
                            free(bo->bytes);
                        }
                        free(bo);
                    }
                } else {
                    OPAL_OUTPUT_VERBOSE((5, orte_grpcomm_base_framework.framework_output,
                                         "%s grpcomm:direct:xcast updating daemon nidmap",
                                         ORTE_NAME_PRINT(ORTE_PROC_MY_NAME)));

                    if (ORTE_SUCCESS != (ret = orte_util_decode_daemon_nodemap(bo))) {
                        ORTE_ERROR_LOG(ret);
                        goto relay;
                    }
                }

                /* update the routing plan */
                orte_routed.update_routing_plan(rtmod);

                /* see if we have wiring info as well */
                cnt=1;
                if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &flag, &cnt, OPAL_INT8))) {
                    ORTE_ERROR_LOG(ret);
                    goto relay;
                }

                if (ORTE_DAEMON_ADD_LOCAL_PROCS == command) {
                    OBJ_RELEASE(relay);
                    relay = OBJ_NEW(opal_buffer_t);
                    /* repack the command */
                    if (OPAL_SUCCESS != (ret = opal_dss.pack(relay, &command, 1, ORTE_DAEMON_CMD))) {
                        ORTE_ERROR_LOG(ret);
                        goto relay;
                    }
                    if (0 == flag) {
                        /* copy the remainder of the payload */
                        opal_dss.copy_payload(relay, buffer);
                        /* no - just return */
                        goto relay;
                    }
                }

                /* unpack the byte object */
                cnt=1;
                if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &bo, &cnt, OPAL_BYTE_OBJECT))) {
                    ORTE_ERROR_LOG(ret);
                    goto relay;
                }
                if (0 < bo->size) {
                    /* load it into a buffer */
                    OBJ_CONSTRUCT(&wireup, opal_buffer_t);
                    opal_dss.load(&wireup, bo->bytes, bo->size);
                    /* pass it for processing */
                    if (ORTE_SUCCESS != (ret = orte_rml_base_update_contact_info(&wireup))) {
                        ORTE_ERROR_LOG(ret);
                        OBJ_DESTRUCT(&wireup);
                        goto relay;
                    }
                    /* done with the wireup buffer - dump it */
                    OBJ_DESTRUCT(&wireup);
                }
                free(bo);
                if (ORTE_DAEMON_ADD_LOCAL_PROCS == command) {
                    /* copy the remainder of the payload */
                    opal_dss.copy_payload(relay, buffer);
                }
            }
        } else {
            ORTE_ERROR_LOG(ret);
            goto CLEANUP;
        }
    }

 relay:

    /* get the list of next recipients from the routed module */
    orte_routed.get_routing_list(rtmod,&coll);

    /* if list is empty, no relay is required */
    if (opal_list_is_empty(&coll)) {
        OPAL_OUTPUT_VERBOSE((5, orte_grpcomm_base_framework.framework_output,
                             "%s grpcomm:direct:send_relay - recipient list is empty!",
                             ORTE_NAME_PRINT(ORTE_PROC_MY_NAME)));
        OBJ_RELEASE(rly);
        goto CLEANUP;
    }
    /* send the message to each recipient on list, deconstructing it as we go */
    while (NULL != (item = opal_list_remove_first(&coll))) {
        nm = (orte_namelist_t*)item;

        OPAL_OUTPUT_VERBOSE((5, orte_grpcomm_base_framework.framework_output,
                             "%s grpcomm:direct:send_relay sending relay msg of %d bytes to %s",
                             ORTE_NAME_PRINT(ORTE_PROC_MY_NAME), (int)rly->bytes_used,
                             ORTE_NAME_PRINT(&nm->name)));
        OBJ_RETAIN(rly);
        /* check the state of the recipient - no point
         * sending to someone not alive    
         */
        jdata = orte_get_job_data_object(nm->name.jobid);
        if (NULL == (rec = (orte_proc_t*)opal_pointer_array_get_item(jdata->procs, nm->name.vpid))) {
            opal_output(0, "%s grpcomm:direct:send_relay proc %s not found - cannot relay",
                        ORTE_NAME_PRINT(ORTE_PROC_MY_NAME), ORTE_NAME_PRINT(&nm->name));
            OBJ_RELEASE(rly);
            OBJ_RELEASE(item);
            continue;
        }
        if (ORTE_PROC_STATE_RUNNING < rec->state) {
            opal_output(0, "%s grpcomm:direct:send_relay proc %s not running - cannot relay",
                        ORTE_NAME_PRINT(ORTE_PROC_MY_NAME), ORTE_NAME_PRINT(&nm->name));
            OBJ_RELEASE(rly);
            OBJ_RELEASE(item);
            continue;
        }
        if (ORTE_SUCCESS != (ret = orte_rml.send_buffer_nb(orte_coll_conduit, &nm->name, rly, ORTE_RML_TAG_BMGXCAST,
                                                           orte_rml_send_callback, NULL))) {
            ORTE_ERROR_LOG(ret);
            OBJ_RELEASE(rly);
            OBJ_RELEASE(item);
            continue;
        }
        OBJ_RELEASE(item);
    }
    OBJ_RELEASE(rly);  // retain accounting

 CLEANUP:
    /* cleanup */
    OBJ_DESTRUCT(&coll);
   printf("bcast recv cleanup\n");
    /* now send the relay buffer to myself for processing */
    if (ORTE_DAEMON_DVM_NIDMAP_CMD != command) {
        if (ORTE_SUCCESS != (ret = orte_rml.send_buffer_nb(orte_coll_conduit, ORTE_PROC_MY_NAME, relay, tag,
                                                           orte_rml_send_callback, NULL))) {
            ORTE_ERROR_LOG(ret);
            OBJ_RELEASE(relay);
            printf("send buffer my name failed]\n");
        }
    }
}


static void rbcast_recv(int status, orte_process_name_t* sender,
                       opal_buffer_t* buffer, orte_rml_tag_t tg,
                       void* cbdata)
{
    int ret, cnt;
    opal_buffer_t *relay, *rly;
    orte_grpcomm_signature_t *sig;
    orte_rml_tag_t tag;
    int cbtype;
    orte_vpid_t *dmns;
    size_t ndmns;

    OPAL_OUTPUT_VERBOSE((1, orte_grpcomm_base_framework.framework_output,
                         "%s grpcomm:direct:rbcast:recv: with %d bytes",
                         ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),
                         (int)buffer->bytes_used));

    /* we need a passthru buffer to forward and to the callback */
    rly = OBJ_NEW(opal_buffer_t);
    opal_dss.copy_payload(rly, buffer);

    relay = OBJ_NEW(opal_buffer_t);
    opal_dss.copy_payload(relay, buffer);

    /* get the signature that we need to create the dmns*/
    cnt=1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &sig, &cnt, ORTE_SIGNATURE))) {
        ORTE_ERROR_LOG(ret);
        ORTE_FORCED_TERMINATE(ret);
        goto CLEANUP;  
    }
    OBJ_RELEASE(sig);

    /* get the target tag */
    cnt=1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &tag, &cnt, ORTE_RML_TAG))) {
        ORTE_ERROR_LOG(ret);
        ORTE_FORCED_TERMINATE(ret);
        goto CLEANUP;
    }
    printf("HELLO tag %d",tag);  
    /* get the cbtype */
    cnt=1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &cbtype, &cnt,OPAL_INT ))) {
        ORTE_ERROR_LOG(ret);
        ORTE_FORCED_TERMINATE(ret);
        goto CLEANUP;
    }
    if( orte_grpcomm_rbcast_cb[cbtype](relay) ) {
        /* forward the rbcast */
        /* create the array of participating daemons */
          if (ORTE_SUCCESS != (ret = create_dmns(sig, &dmns, &ndmns))) {
              ORTE_ERROR_LOG(ret);
              goto CLEANUP;
          }
          if (ORTE_SUCCESS == (ret = rbcast(dmns, ndmns, rly))) {
              if (NULL != dmns) {
                 free(dmns);
              }
          }
      }

CLEANUP: 
    OBJ_RELEASE(rly);
    OBJ_RELEASE(relay);
}

static void barrier_release(int status, orte_process_name_t* sender,
                            opal_buffer_t* buffer, orte_rml_tag_t tag,
                            void* cbdata)
{
    int32_t cnt;
    int rc, ret;
    orte_grpcomm_signature_t *sig;
    orte_grpcomm_coll_t *coll;

    OPAL_OUTPUT_VERBOSE((5, orte_grpcomm_base_framework.framework_output,
                         "%s grpcomm:direct: barrier release called with %d bytes",
                         ORTE_NAME_PRINT(ORTE_PROC_MY_NAME), (int)buffer->bytes_used));

    /* unpack the signature */
    cnt = 1;
    if (OPAL_SUCCESS != (rc = opal_dss.unpack(buffer, &sig, &cnt, ORTE_SIGNATURE))) {
        ORTE_ERROR_LOG(rc);
        return;
    }

    /* unpack the return status */
    cnt = 1;
    if (OPAL_SUCCESS != (rc = opal_dss.unpack(buffer, &ret, &cnt, OPAL_INT))) {
        ORTE_ERROR_LOG(rc);
        return;
    }

    /* check for the tracker - it is not an error if not
 *      * found as that just means we wre not involved
 *           * in the collective */
    if (NULL == (coll = orte_grpcomm_base_get_tracker(sig, false))) {
        OBJ_RELEASE(sig);
        return;
    }

    /* execute the callback */
    if (NULL != coll->cbfunc) {
        coll->cbfunc(ret, buffer, coll->cbdata);
    }
    opal_list_remove_item(&orte_grpcomm_base.ongoing, &coll->super);
    OBJ_RELEASE(coll);
}

