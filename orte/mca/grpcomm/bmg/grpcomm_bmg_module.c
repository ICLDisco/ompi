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
#include "orte/mca/rml/rml.h"i
#include "orte/mca/state/state.h"
#include "orte/util/name_fns.h"
#include "orte/util/proc_info.h"

#include "orte/mca/errmgr/detector/errmgr_detector.h"
#include "orte/mca/grpcomm/base/base.h"
#include "grpcomm_bmg.h"

#include "orte/mca/errmgr/detector/errmgr_detector.h"
/* Static API's */
static int init(void);
static void finalize(void);

static int xcast(orte_vpid_t *vpids,
                 size_t nprocs,
                 opal_buffer_t *buf);

/* Module def */
orte_grpcomm_base_module_t orte_grpcomm_bmg_module = {
    init,
    finalize,
    xcast,
    NULL
};

/* Internal functions */
static void xcast_recv(int status, orte_process_name_t* sender,
                       opal_buffer_t* buffer, orte_rml_tag_t tag,
                       void* cbdata);

/*
 * registration of callbacks
 */
#define XCAST_CB_TYPE_MAX 7
static orte_errmgr_xcast_cb_t orte_errmgr_xcast_cb[XCAST_CB_TYPE_MAX+1];

int orte_errmgr_xcast_register_cb_type(orte_errmgr_xcast_cb_t callback) {
    int i;
    for(i = 0; i < XCAST_CB_TYPE_MAX; i++) {
        if( NULL == orte_errmgr_xcast_cb[i] ) {
            orte_errmgr_xcast_cb[i] = callback;
            return i;
        }
    }
    return ORTE_ERR_OUT_OF_RESOURCE;
}

int orte_errmgr_xcast_unregister_cb_type(int type) {
    if( XCAST_CB_TYPE_MAX < type || 0 > type ) {
        return ORTE_ERR_BAD_PARAM;
    }
    orte_errmgr_xcast_cb[type] = NULL;
    return ORTE_SUCCESS;
}

/*
 *  Initialize the module
 */
static int init(void)
{
   /* post the receives */
    orte_rml.recv_buffer_nb(ORTE_NAME_WILDCARD,
                            ORTE_RML_TAG_XCAST,
                            ORTE_RML_PERSISTENT,
                            xcast_recv, NULL);
 
   return OPAL_SUCCESS;
}

/*
 * Finalize the module
 */
static void finalize(void)
{
    /* cancel the recv */
    orte_rml.recv_cancel(ORTE_NAME_WILDCARD, ORTE_RML_TAG_XCAST);
    return;
}

static int xcast(orte_vpid_t *vpids,
                 size_t nprocs,
                 opal_buffer_t *buf)
{
    int rc = false;
    int vpid; 
    
    vpid = orte_process_info.my_daemon.vpid;
    
    for(i=1; i <= nprocs/2; i*=2) for(d=1; d >= -1; d-=2) {
        int idx = (nprocs+vpid+d*i)%nprocs;
     redo:
        if( idx == vpid ) continue;

        daemon.jobid = orte_process_info.my_daemon.jobid;
        daemon.vpid = idx;

        if (0 > (rc = orte_rml.send_buffer_nb(&daemon, buf, ORTE_RML_TAG_XCAST,orte_rml_send_callback , NULL))) {
            ORTE_ERROR_LOG(rc);
        }
        OBJ_RETAIN(buf); 
        if( i == 1 ) {
            /* The ring is cut, find the closest alive neighbor in that direction */
            idx = (nprocs+idx+d)%nprocs;
            /* TODO: find a way to not send twice the message if idx is one of
             * my neighbors for i>1 */
            goto redo;
        }
     }  
    OBJ_RELEASE(buf);     
    return rc;
}

static void xcast_recv(int status, orte_process_name_t* sender,
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
                         "%s grpcomm:direct:xcast:recv: with %d bytes",
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
        
    /* get the cbtype */
    cnt=1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &cbtype, &cnt,OPAL_INT ))) {
        ORTE_ERROR_LOG(ret);
        ORTE_FORCED_TERMINATE(ret);
        goto CLEANUP;
    }
    if( orte_errmgr_xcast_cb[cbtype](relay) ) {
        /* forward the rbcast */
        /* create the array of participating daemons */
          if (ORTE_SUCCESS != (ret = create_dmns(sig, &dmns, &ndmns))) {
              ORTE_ERROR_LOG(ret);
              goto CLEANUP;
          }
          if (ORTE_SUCCESS == (ret = active->module->xcast(dmns, ndmns, rly))) {
              if (NULL != dmns) {
                 free(dmns);
              }
          }
      }

CLEARUP: 
    OBJ_RELEASE(rly);
    OBJ_RELEASE(relay);
}

