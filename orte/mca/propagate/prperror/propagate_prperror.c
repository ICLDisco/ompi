/*
 * Copyright (c) 2017      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "orte_config.h"

#include <sys/types.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif  /* HAVE_UNISTD_H */
#include <string.h>
#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif

#include "opal/util/output.h"
#include "opal/dss/dss.h"
#include "opal/mca/pmix/pmix.h"

#include "orte/mca/rml/rml.h"
#include "orte/mca/odls/odls.h"
#include "orte/mca/odls/base/base.h"
#include "orte/mca/odls/base/odls_private.h"
#include "orte/mca/plm/base/plm_private.h"
#include "orte/mca/plm/plm.h"
#include "orte/mca/rmaps/rmaps_types.h"
#include "orte/mca/routed/routed.h"
#include "orte/mca/grpcomm/grpcomm.h"
#include "orte/mca/grpcomm/bmg/grpcomm_bmg.h"
#include "orte/mca/ess/ess.h"
#include "orte/mca/state/state.h"

#include "orte/util/error_strings.h"
#include "orte/util/name_fns.h"
#include "orte/util/proc_info.h"
#include "orte/util/show_help.h"
#include "orte/util/nidmap.h"

#include "orte/runtime/orte_globals.h"
#include "orte/runtime/orte_locks.h"
#include "orte/runtime/orte_quit.h"
#include "orte/runtime/data_type_support/orte_dt_support.h"

#include "orte/mca/propagate/propagate.h"
#include "orte/mca/propagate/base/base.h"

#include "orte/mca/errmgr/errmgr.h"
#include "orte/mca/errmgr/base/base.h"
#include "orte/mca/errmgr/base/errmgr_private.h"
#include "orte/mca/errmgr/detector/errmgr_detector.h"

#include "propagate_prperror.h"

opal_list_t orte_error_procs = {{0}};

static int orte_propagate_error_cb_type = -1;

static int init(void);
static int finalize(void);
static int register_prp_callback(void);
static int orte_propagate_prperror(orte_jobid_t *job, orte_process_name_t *source,
        orte_process_name_t *sickproc, orte_proc_state_t state);

int orte_propagate_prperror_recv(opal_buffer_t* buffer);

/* flag use to register callback for grpcomm rbcast forward */
int cbflag = 1;

orte_propagate_base_module_t orte_propagate_prperror_module ={
    init,
    finalize,
    orte_propagate_prperror,
    register_prp_callback
};

/*
 *Local functions
 */
static int init(void)
{
    OBJ_CONSTRUCT(&orte_error_procs, opal_list_t);
    return ORTE_SUCCESS;
}

static int register_prp_callback(void)
{
    int ret;
    if(cbflag)
    {
        if(orte_grpcomm.register_cb!=NULL)
            ret= orte_grpcomm.register_cb((orte_grpcomm_rbcast_cb_t)orte_propagate_prperror_recv);
        if ( 0 <= ret ){
            orte_propagate_error_cb_type = ret;
        }
        cbflag = 0;
    }
    return ORTE_SUCCESS;
}

static int finalize(void)
{
    int ret=0;
    if ( -1 == orte_propagate_error_cb_type){
        return ORTE_SUCCESS;
    }
    /* do we need unregister ? cause all module is unloading, those memory maybe collecred may have illegal access */
    //ret = orte_grpcomm.unregister_cb(orte_propagate_error_cb_type);
    orte_propagate_error_cb_type = -1;
    OBJ_DESTRUCT(&orte_error_procs);
    return ret;
}

/*
 * uplevel call from the error handler to initiate a failure_propagator
 */
static int orte_propagate_prperror(orte_jobid_t *job, orte_process_name_t *source,
        orte_process_name_t *sickproc, orte_proc_state_t state) {

    int rc = ORTE_SUCCESS;
    // don't need to check jobid because this can be different: daemon and process has different jobids

    /* namelist for tracking error procs */
    orte_namelist_t *nmcheck, *nm;
    nmcheck = OBJ_NEW(orte_namelist_t);

    OPAL_LIST_FOREACH(nmcheck, &orte_error_procs, orte_namelist_t){
        if ((nmcheck->name.jobid == sickproc->jobid) && (nmcheck->name.vpid == sickproc->vpid))
        {
            OPAL_OUTPUT_VERBOSE((5, orte_propagate_base_framework.framework_output,
                        "propagate:prperror already propagated this error msg process id %d:%d",sickproc->jobid, sickproc->vpid));
            return rc;
        }
    }
    nm = OBJ_NEW(orte_namelist_t);
    nm->name.jobid = sickproc->jobid;
    nm->name.vpid = sickproc->vpid;
    opal_list_append(&orte_error_procs, &(nm->super));

    orte_grpcomm_signature_t *sig;
    int cnt=0;
    /* ---------------------------------------------------------
     * | cb_type | status | sickproc | nprocs afftected | procs|
     * --------------------------------------------------------*/
    opal_buffer_t prperror_buffer;
    opal_list_t affected_list;
    opal_list_t *info;
    opal_value_t *val;

    /*set the status for pmix to use*/
    orte_proc_state_t status;
    status = state;
    /*use a tempproc to get the orte_proc obj*/
    orte_process_name_t temp_daemon;

    // register callback for rbcast for forwarding
    int ret;
    if(cbflag)
    {
        if(orte_grpcomm.register_cb!=NULL)
        ret= orte_grpcomm.register_cb((orte_grpcomm_rbcast_cb_t)orte_propagate_prperror_recv);
        if ( 0 <= ret ){
            orte_propagate_error_cb_type = ret;
        }
        cbflag = 0;
    }

    temp_daemon.jobid = ORTE_PROC_MY_NAME->jobid;
    temp_daemon.vpid = source->vpid;

    /* change the error daemon state*/
    OPAL_OUTPUT_VERBOSE((5, orte_propagate_base_framework.framework_output,
                        "propagate:daemon %d prperror error msg from daemon %d:%d with error proc %d:%d",
                        ORTE_PROC_MY_NAME->vpid,
                        temp_daemon.jobid,
                        temp_daemon.vpid,
                        sickproc->jobid,sickproc->vpid));

    OBJ_CONSTRUCT(&prperror_buffer, opal_buffer_t);
    info = OBJ_NEW(opal_list_t);
    /* pack the callback type */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&prperror_buffer, &orte_propagate_error_cb_type, 1, OPAL_INT))) {
        ORTE_ERROR_LOG(rc);
        OBJ_DESTRUCT(&prperror_buffer);
        return rc;
    }

    /* pack the status */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&prperror_buffer, &status, 1, OPAL_INT))) {
        ORTE_ERROR_LOG(rc);
        OBJ_DESTRUCT(&prperror_buffer);
        return rc;
    }
    /* pack dead proc first */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&prperror_buffer, sickproc, 1, ORTE_NAME))) {
        ORTE_ERROR_LOG(rc);
        OBJ_DESTRUCT(&prperror_buffer);
        return rc;
    }

    // add the dead proc in info list for local prp
    val = OBJ_NEW(opal_value_t);
    val->key = strdup(OPAL_PMIX_EVENT_AFFECTED_PROC);
    val->type = OPAL_NAME;
    val->data.name.jobid = sickproc->jobid;
    val->data.name.vpid = sickproc->vpid;
    opal_list_append(info, &(val->super));

    /* check this is a daemon or not, if vpid is same cannot ensure this is daemon also need check jobid*/
    if (sickproc->vpid == orte_get_proc_daemon_vpid(sickproc) && (sickproc->jobid == ORTE_PROC_MY_NAME->jobid) ){
        /* Given a node name, return an array of processes within the specified jobid
         * on that node. If the specified node does not currently host any processes,
         * then the returned list will be empty.
         */
        OPAL_OUTPUT_VERBOSE((5, orte_propagate_base_framework.framework_output,"propagate:daemon prperror this is a daemon error %s", orte_get_proc_hostname(sickproc)));
        OBJ_CONSTRUCT(&affected_list, opal_list_t);
        // daemon and procs have different jobid, so cannot pass sickproc->jobid, need to pass wildcard
        opal_pmix.resolve_peers(orte_get_proc_hostname(sickproc), ORTE_PROC_MY_NAME->jobid, &affected_list);//sickproc->jobid
        OPAL_OUTPUT_VERBOSE((5, orte_propagate_base_framework.framework_output,"propagate:daemon prperror this is a daemon error"));
        /* add those procs in the buffer*/
        if (!opal_list_is_empty(&affected_list)){
            cnt =affected_list.opal_list_length;
            if (ORTE_SUCCESS != (rc = opal_dss.pack(&prperror_buffer, &cnt, 1, OPAL_INT))) {
                ORTE_ERROR_LOG(rc);
                OBJ_DESTRUCT(&prperror_buffer);
                return rc;
            }
            if (0 < cnt) {
                OPAL_LIST_FOREACH(val, &affected_list, opal_value_t) {
                    if (OPAL_SUCCESS != (rc = opal_dss.pack(&prperror_buffer, &val, 1, OPAL_VALUE))) {
                        ORTE_ERROR_LOG(rc);
                        OBJ_DESTRUCT(&prperror_buffer);
                        return rc;
                     }
                    opal_list_append(info, &val->super);
                }
            }
        }
    }

    //notify this error locally
    if (OPAL_SUCCESS != opal_pmix.notify_event(OPAL_ERR_PROC_ABORTED, (opal_process_name_t*)ORTE_PROC_MY_NAME,
                OPAL_PMIX_RANGE_LOCAL,info,
                NULL,NULL ))
    {
        ORTE_ERROR_LOG(rc);
        OBJ_RELEASE(info);
    }

    /* goes to all daemons */
    sig = OBJ_NEW(orte_grpcomm_signature_t);
    sig->signature = (orte_process_name_t*)malloc(sizeof(orte_process_name_t));
    sig->signature[0].jobid = ORTE_PROC_MY_NAME->jobid;
    /* all daemons hosting this jobid are participating */
    sig->signature[0].vpid = ORTE_VPID_WILDCARD;
    if (ORTE_SUCCESS != (rc = orte_grpcomm.rbcast(sig, ORTE_RML_TAG_PROPAGATE, &prperror_buffer))) {
        ORTE_ERROR_LOG(rc);
    }

    OBJ_DESTRUCT(&prperror_buffer);
    OBJ_RELEASE(sig);
    /* we're done! */
    return ORTE_SUCCESS;
}

int orte_propagate_prperror_recv(opal_buffer_t* buffer)
{
    int ret, cnt, state;
    orte_process_name_t sickproc;
    int cbtype;

    /* get the cbtype */
    cnt=1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &cbtype, &cnt,OPAL_INT ))) {
        ORTE_ERROR_LOG(ret);
        return false;
    }
    cnt = 1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &state, &cnt, OPAL_INT))) {
        ORTE_ERROR_LOG(ret);
        return false;
    }
    /* for propagate, only one major sickproc is affected per call */
    cnt = 1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &sickproc, &cnt, OPAL_NAME))) {
        ORTE_ERROR_LOG(ret);
        return false;
    }

    OPAL_OUTPUT_VERBOSE((5, orte_propagate_base_framework.framework_output,
                "propagete:prperror daemon %d  received %d:%d gone begin forwarding with status %d",
                orte_process_info.my_name.vpid, sickproc.jobid, sickproc.vpid, state));

    orte_propagate_prperror(&orte_process_info.my_name.jobid, ORTE_PROC_MY_NAME, &sickproc, state);
    return false;
}
