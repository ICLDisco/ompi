/*
 * Copyright (c) 2010-2012 Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2011-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
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

#include "orte/mca/errmgr/errmgr.h"
#include "orte/mca/errmgr/base/base.h"
#include "orte/mca/errmgr/base/errmgr_private.h"

#include "errmgr_detector.h"

static int orte_errmgr_failure_propagator_cb_type = -1;

int orte_errmgr_failure_propagate(orte_jobid_t *job, orte_process_name_t *daemon, orte_proc_state_t state);
int orte_errmgr_failure_propagate_recv(opal_buffer_t* buffer);

/*
 *Local functions
 */

int orte_errmgr_init_failure_propagate(void)
{
    int ret;
    if(orte_grpcomm.register_cb!=NULL)
    ret= orte_grpcomm.register_cb((orte_grpcomm_rbcast_cb_t)orte_errmgr_failure_propagate_recv);
    if ( 0 <= ret ){
        orte_errmgr_failure_propagator_cb_type = ret;
        return ORTE_SUCCESS;
    }
    return ret;
}

int orte_errmgr_finalize_failure_propagate(void)
{
    int ret;
    if ( -1 == orte_errmgr_failure_propagator_cb_type){
        return ORTE_SUCCESS;
    }
    ret = orte_grpcomm.unregister_cb(orte_errmgr_failure_propagator_cb_type);
    orte_errmgr_failure_propagator_cb_type = -1;
    return ret;
}

/*
 * uplevel call from the error handler to initiate a failure_propagator
 */
int orte_errmgr_failure_propagate(orte_jobid_t *job, orte_process_name_t *daemon, orte_proc_state_t state) {
    int rc = ORTE_SUCCESS;

    orte_grpcomm_signature_t *sig;
    opal_buffer_t propagete_buff;
    opal_list_t *info;
    opal_value_t *kv, *kvptr;
    /*packing variables*/
    int status;
    status = state;
    orte_process_name_t temp_daemon;
    temp_daemon.jobid =  daemon->jobid; temp_daemon.vpid = daemon->vpid;

    /* change the error daemon state*/
    orte_proc_t *temp_orte_proc;
    temp_orte_proc = (orte_proc_t*)orte_get_proc_object(daemon);
    OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                        "errmgr:daemon %d prp error daemon %d with state",
                        ORTE_PROC_MY_NAME->vpid,
                        daemon->vpid,
                        errmgr_get_daemon_status(temp_daemon)));

    if(!errmgr_get_daemon_status(temp_daemon))
        return rc;
    OBJ_CONSTRUCT(&propagete_buff, opal_buffer_t);
    info = OBJ_NEW(opal_list_t);
    /* pack the callback type */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&propagete_buff, &orte_errmgr_failure_propagator_cb_type, 1, OPAL_INT))) {
        ORTE_ERROR_LOG(rc);
        OBJ_DESTRUCT(&propagete_buff);
        return rc;
    }

    /* pack the status */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&propagete_buff, &status, 1, OPAL_INT))) {
        ORTE_ERROR_LOG(rc);
        OBJ_DESTRUCT(&propagete_buff);
        return rc;
    }

    /* pack dead daemom */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&propagete_buff, daemon, 1, ORTE_NAME))) {
        ORTE_ERROR_LOG(rc);
        OBJ_DESTRUCT(&propagete_buff);
        return rc;
    }

    /* pass thru opal_values */
    rc = 1;
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&propagete_buff, &rc, 1, OPAL_INT))) {
        ORTE_ERROR_LOG(rc);
        OBJ_DESTRUCT(&propagete_buff);
        return rc;
    }

    /* pass along the affected proc(s) */
    kv = OBJ_NEW(opal_value_t);
    /* #define OPAL_PMIX_EVENT_TERMINATE_NODE "pmix.evterm.node"      (bool) RM intends to terminate all procs on this node */
    // kv->key = strdup(OPAL_PMIX_EVENT_AFFECTED_PROC);
    kv->key = strdup(OPAL_PMIX_EVENT_TERMINATE_NODE);

    kv->type = OPAL_NAME;
    kv->data.name.jobid = daemon->jobid;
    kv->data.name.vpid = daemon->vpid;

    if (ORTE_SUCCESS != (rc = opal_dss.pack(&propagete_buff, &kv, 1, OPAL_VALUE))) {
        ORTE_ERROR_LOG(rc);
        //OBJ_DESTRUCT(&kv);
        OBJ_DESTRUCT(&propagete_buff);
        return rc;
    }
    opal_list_append(info, &kv->super);
    //OBJ_DESTRUCT(&kv);

    // #define PMIX_ERR_NODE_DOWN  (PMIX_ERR_SYS_BASE -  1) source: ORTE_PROC_MY_NAME
    /* notify this error locally */
   if (OPAL_SUCCESS != (rc = opal_pmix.server_notify_event(status, daemon, info, NULL, NULL))) {
        ORTE_ERROR_LOG(rc);
        if (NULL != info) {
            OPAL_LIST_RELEASE(info);
        }
    }

    /* goes to all daemons */
    sig = OBJ_NEW(orte_grpcomm_signature_t);
    sig->signature = (orte_process_name_t*)malloc(sizeof(orte_process_name_t));
    sig->signature[0].jobid = daemon->jobid;
    /* all daemons hosting this jobid are participating */
    sig->signature[0].vpid = ORTE_VPID_WILDCARD;
    if (ORTE_SUCCESS != (rc = orte_grpcomm.rbcast(sig, ORTE_RML_TAG_PROPAGATE, &propagete_buff))) {
        ORTE_ERROR_LOG(rc);
    }

    OBJ_DESTRUCT(&propagete_buff);
    OBJ_RELEASE(sig);
    /* we're done! */
    return ORTE_SUCCESS;
}

int orte_errmgr_failure_propagate_recv(opal_buffer_t* buffer)
{
    int ret, cnt, state;
    orte_process_name_t temp_proc_name;
    orte_proc_t *temp_orte_proc, *checkoproc;
    orte_grpcomm_signature_t *sig;
    orte_rml_tag_t tag;
    int cbtype, nprocs;

    opal_pointer_array_t cmd;
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
    /* for propagate, only one daemon is affected per call */
    cnt = 1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &temp_proc_name, &cnt, OPAL_NAME))) {
        ORTE_ERROR_LOG(ret);
        return false;
    }
    temp_orte_proc = orte_get_proc_object(&temp_proc_name);
    OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                "errmgr:daemon %d received %d",
                orte_process_info.my_name.vpid,
                temp_proc_name.vpid));

    if(errmgr_get_daemon_status(temp_proc_name)){ 
    OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                "errmgr:daemon %d begin forwarding state is %d",
                orte_process_info.my_name.vpid,
                 errmgr_get_daemon_status(temp_proc_name)));

        orte_errmgr_failure_propagate(&orte_process_info.my_name.jobid, &temp_proc_name, state);
        errmgr_set_daemon_status(temp_proc_name , false) ;
    }
    return false;
}
