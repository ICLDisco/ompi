/*
 * Copyright (c) 2016      The University of Tennessee and The University
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

#include "opal/mca/base/mca_base_var.h"
#include "opal/mca/timer/base/base.h"
#include "opal/threads/threads.h"
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

#include "orte/mca/propagate/propagate.h"
#include "errmgr_detector.h"
#include <math.h>

static int init(void);
static int finalize(void);

int orte_errmgr_enable_detector(bool flag);
/******************
 * detector module
 ******************/
orte_errmgr_base_module_t orte_errmgr_detector_module = {
    init,
    finalize,
    orte_errmgr_base_log,
    orte_errmgr_base_abort,
    orte_errmgr_base_abort_peers,
    orte_errmgr_enable_detector
};

orte_errmgr_base_module_t orte_errmgr = {
    init,
    finalize,
    orte_errmgr_base_log,
    NULL,
    NULL,
    orte_errmgr_enable_detector
};

/*
 * Local functions
 */
static int fd_heartbeat_request(orte_errmgr_detector_t* detector);
static int fd_heartbeat_send(orte_errmgr_detector_t* detector);

static int fd_heartbeat_request_cb(int status, orte_process_name_t* sender,
                                  opal_buffer_t *buffer,
                                  orte_rml_tag_t tg, void *cbdata);
static int fd_heartbeat_recv_cb(int status, orte_process_name_t* sender,
                                  opal_buffer_t *buffer,
                                  orte_rml_tag_t tg, void *cbdata);

static double Wtime();
static double orte_errmgr_heartbeat_period = 2e-1;
static double orte_errmgr_heartbeat_timeout = 5e-1;
static opal_event_base_t* fd_event_base = NULL;

static void fd_event_cb(int fd, short flags, void* pdetector);


static int pack_state_for_proc(opal_buffer_t *alert, orte_proc_t *child)
{
    int rc;

    /* pack the child's vpid */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(alert, &(child->name.vpid), 1, ORTE_VPID))) {
        ORTE_ERROR_LOG(rc);
        return rc;
    }
    /* pack the pid */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(alert, &child->pid, 1, OPAL_PID))) {
        ORTE_ERROR_LOG(rc);
        return rc;
    }
    /* pack its state */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(alert, &child->state, 1, ORTE_PROC_STATE))) {
        ORTE_ERROR_LOG(rc);
        return rc;
    }
    /* pack its exit code */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(alert, &child->exit_code, 1, ORTE_EXIT_CODE))) {
        ORTE_ERROR_LOG(rc);
        return rc;
    }

    return ORTE_SUCCESS;
}

static void register_cbfunc(int status, size_t errhndler, void *cbdata)
{
    OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                "errmgr:detector:event register cbfunc with status %d ", status));
}

static void error_notify_cbfunc(int status,
        const opal_process_name_t *source,
        opal_list_t *info, opal_list_t *results,
        opal_pmix_notification_complete_fn_t cbfunc, void *cbdata)
{
    orte_process_name_t proc;
    opal_value_t *kv;
    proc.jobid = ORTE_JOBID_INVALID;
    proc.vpid = ORTE_VPID_INVALID;

    int rc;
    orte_proc_t *temp_orte_proc;
    opal_buffer_t *alert;
    orte_job_t *jdata;
    orte_plm_cmd_flag_t cmd;

    OPAL_LIST_FOREACH(kv, info, opal_value_t) {
        if (0 == strcmp(kv->key, OPAL_PMIX_EVENT_AFFECTED_PROC)) {

            proc.jobid = kv->data.name.jobid;
            proc.vpid = kv->data.name.vpid;

            OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                        "errmgr:detector %s error proc %d:%d with key %s notified from %d:%d",
                        ORTE_NAME_PRINT(ORTE_PROC_MY_NAME), proc.jobid, proc.vpid,
                        kv->key, source->jobid, source->vpid));

            if (NULL == (jdata = orte_get_job_data_object(proc.jobid))) {
                /* must already be complete */
                OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                            "%s errmgr:detector:error_notify_callback NULL jdata - ignoring error",
                            ORTE_NAME_PRINT(ORTE_PROC_MY_NAME)));
            }

            if (!orte_get_attribute(&jdata->attributes, ORTE_JOB_FAIL_NOTIFIED, NULL, OPAL_BOOL)) {
                alert = OBJ_NEW(opal_buffer_t);
                /* pack update state command */
                cmd = ORTE_PLM_UPDATE_PROC_STATE;
                if (ORTE_SUCCESS != (rc = opal_dss.pack(alert, &cmd, 1, ORTE_PLM_CMD))) {
                    ORTE_ERROR_LOG(rc);
                    return;
                }

                /* pack jobid */
                if (ORTE_SUCCESS != (rc = opal_dss.pack(alert, &proc.jobid, 1, ORTE_JOBID))) {
                    ORTE_ERROR_LOG(rc);
                    return;
                }

                temp_orte_proc= (orte_proc_t*)opal_pointer_array_get_item(jdata->procs, proc.vpid);

                if (!ORTE_FLAG_TEST(temp_orte_proc, ORTE_PROC_FLAG_WAITPID)) {
                    ORTE_FLAG_SET(temp_orte_proc, ORTE_PROC_FLAG_WAITPID);
                }

                OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                            "errmgr:detector:packed proc is %d:%d  state %d, code %d num_terminated:%d",
                            temp_orte_proc->name.jobid, temp_orte_proc->name.vpid,
                            temp_orte_proc->state, temp_orte_proc->exit_code, jdata->num_terminated));

                /* proc state now is ORTE_PROC_STATE_ABORTED_BY_SIG, cause odls set state to this; code is 128+9 */
                temp_orte_proc->state = ORTE_PROC_STATE_ABORTED_BY_SIG;

                /* now pack the child's info */
                if (ORTE_SUCCESS != (rc = pack_state_for_proc(alert, temp_orte_proc))) {
                    ORTE_ERROR_LOG(rc);
                    return;
                }

                /* send this process's info to hnp */
                if (0 > (rc = orte_rml.send_buffer_nb(orte_mgmt_conduit,
                                ORTE_PROC_MY_HNP, alert,
                                ORTE_RML_TAG_PLM,
                                orte_rml_send_callback, NULL))) {
                    OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                                "errmgr:detector: send to hnp failed"));
                    ORTE_ERROR_LOG(rc);
                    OBJ_RELEASE(alert);
                }

                orte_set_attribute(&jdata->attributes, ORTE_JOB_FAIL_NOTIFIED, ORTE_ATTR_LOCAL, NULL, OPAL_BOOL);

                /* if the proc has terminated, notify the state machine */
                if (ORTE_FLAG_TEST(temp_orte_proc, ORTE_PROC_FLAG_IOF_COMPLETE) &&
                        ORTE_FLAG_TEST(temp_orte_proc, ORTE_PROC_FLAG_WAITPID) &&
                        !ORTE_FLAG_TEST(temp_orte_proc, ORTE_PROC_FLAG_RECORDED)) {
                    ORTE_ACTIVATE_PROC_STATE(&proc, ORTE_PROC_STATE_TERMINATED);
                }
            }
            orte_propagate.prp(&source->jobid, source, &proc, OPAL_ERR_PROC_ABORTED);
            break;
       }
    }

    if (NULL != cbfunc) {
        cbfunc(ORTE_SUCCESS, NULL, NULL, NULL, cbdata);
    }
}

static int init(void) {
    int ret;
    fd_event_base = opal_sync_event_base;
    volatile int active;
    active = -1;
    if ( ORTE_PROC_IS_DAEMON )
    {
    orte_rml.recv_buffer_nb(ORTE_NAME_WILDCARD, ORTE_RML_TAG_HEARTBEAT_REQUEST,
                           ORTE_RML_PERSISTENT,fd_heartbeat_request_cb,NULL);
    orte_rml.recv_buffer_nb(ORTE_NAME_WILDCARD, ORTE_RML_TAG_HEARTBEAT,
                           ORTE_RML_PERSISTENT,fd_heartbeat_recv_cb,NULL);
    }
    return ORTE_SUCCESS;
}

int finalize(void) {
    if ( ORTE_PROC_IS_DAEMON )
    {
        orte_errmgr_detector_t* detector = &orte_errmgr_world_detector;

        if(detector->hb_observer != ORTE_VPID_INVALID)
        {
            detector->hb_observer = orte_process_info.my_name.vpid;
            OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,"errmgr:detector: send last heartbeat message"));
            fd_heartbeat_send(detector);
            detector->hb_period = INFINITY;
        }
        opal_event_del(&orte_errmgr_world_detector.fd_event);
        orte_rml.recv_cancel(ORTE_NAME_WILDCARD, ORTE_RML_TAG_HEARTBEAT_REQUEST);
        orte_rml.recv_cancel(ORTE_NAME_WILDCARD, ORTE_RML_TAG_HEARTBEAT);
        if( opal_sync_event_base != fd_event_base ) opal_event_base_free(fd_event_base);

        /* set heartbeat peroid to infinity and observer to invalid */
        orte_errmgr_world_detector.hb_period = INFINITY;
        orte_errmgr_world_detector.hb_observer = ORTE_VPID_INVALID;
    }
    return ORTE_SUCCESS;
}

int errmgr_get_daemon_status(orte_process_name_t daemon)
{
    orte_errmgr_detector_t* detector = &orte_errmgr_world_detector;
    return detector->daemons_state[daemon.vpid];
}

void errmgr_set_daemon_status(orte_process_name_t daemon, bool state)
{
    orte_errmgr_detector_t* detector = &orte_errmgr_world_detector;
    detector->daemons_state[daemon.vpid] = state;
}

static double Wtime(void)
{
    double wtime;

#if OPAL_TIMER_CYCLE_NATIVE
    wtime = ((double) opal_timer_base_get_cycles()) / opal_timer_base_get_freq();
#elif OPAL_TIMER_USEC_NATIVE
    wtime = ((double) opal_timer_base_get_usec()) / 1000000.0;
#else
    /* Fall back to gettimeofday() if we have nothing else */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    wtime = tv.tv_sec;
    wtime += (double)tv.tv_usec / 1000000.0;
#endif
    OPAL_CR_NOOP_PROGRESS();
    return wtime;
}

int orte_errmgr_enable_detector(bool enable_flag)
{
    OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,"%s errmgr:detector report detector_enable_status %d",
                ORTE_NAME_PRINT(ORTE_PROC_MY_NAME), enable_flag));

    if ( ORTE_PROC_IS_DAEMON && enable_flag )
    {
        orte_errmgr_detector_t* detector = &orte_errmgr_world_detector;
        int  ndmns, i;
        uint32_t vpid;

        opal_list_t *codes;
        opal_value_t *ekv;

        codes = OBJ_NEW(opal_list_t);
        ekv = OBJ_NEW(opal_value_t);
        ekv->key = strdup("OPAL_PMIX_EVENT_AFFECTED_PROC");
        ekv->type = OPAL_INT;
        ekv->data.integer =OPAL_ERR_PROC_ABORTED;
        opal_list_append(codes, &ekv->super);
        OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                    "errmgr:detector: register evhandler in errmgr.."));

        opal_pmix.register_evhandler(codes, NULL, error_notify_cbfunc, register_cbfunc, NULL);

        orte_propagate.register_cb();

        ndmns = orte_process_info.num_procs-1;                /* num of daemon in this jobid */
        vpid = orte_process_info.my_name.vpid;
        if( 0 != (vpid - 1) )
            detector->hb_observing = vpid - 1;               /*  we observing somebody {n,1,2,...n-1}, the ring */
        else detector->hb_observing = ndmns;

        detector->hb_observer = (ndmns+vpid) % ndmns + 1 ;   /* someone is observing us: range [1~n], the observing ring */
        detector->hb_period = orte_errmgr_heartbeat_period;
        detector->hb_timeout = orte_errmgr_heartbeat_timeout;
        detector->hb_sstamp = 0.;
        detector->hb_rstamp = Wtime()+(double)ndmns;        /* give some slack for MPIInit */

        for(i=0; i<orte_process_info.num_procs; i++)
        {
            detector->daemons_state[i] = true;
        }
        OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                    "errmgr:detector daemon %d observering %d observer %d",
                    vpid,
                    detector->hb_observing,
                    detector->hb_observer));

        opal_event_set(fd_event_base, &detector->fd_event, -1, OPAL_EV_TIMEOUT | OPAL_EV_PERSIST, fd_event_cb, detector);
        struct timeval tv;
        tv.tv_sec = (int)(detector->hb_period / 10.);
        tv.tv_usec = (-tv.tv_sec + (detector->hb_period / 10.)) * 1e6;
        opal_event_add(&orte_errmgr_world_detector.fd_event, &tv);
    }
    return ORTE_SUCCESS;
}

static int fd_heartbeat_request(orte_errmgr_detector_t* detector) {

    int ret,  ndmns;
    uint32_t vpid;

    orte_process_name_t temp_proc_name;
    orte_proc_t * proc;
    temp_proc_name.jobid = orte_process_info.my_name.jobid;
    temp_proc_name.vpid = detector->hb_observing;

    if( errmgr_get_daemon_status(temp_proc_name) )
    {
       	/* already observing a live process, so nothing to do. */
        return ORTE_SUCCESS;
    }

    ndmns = orte_process_info.num_procs-1;

    opal_buffer_t *buffer = NULL;
    orte_process_name_t daemon;
    for( vpid = (ndmns+detector->hb_observing) % ndmns;
         vpid != orte_process_info.my_name.vpid;
         vpid = (ndmns+vpid-1) % ndmns ) {
            daemon.jobid = orte_process_info.my_name.jobid;
            if(0 != vpid){
                daemon.vpid = vpid;
             }
            else daemon.vpid = ndmns;

            // this daemon is not alive
            if(!errmgr_get_daemon_status(daemon)) continue;

            /* everyone is gone, i dont need to monitor myself */
            if(daemon.vpid == orte_process_info.my_name.vpid)
            {
                detector->hb_observer = detector->hb_observing = ORTE_VPID_INVALID;
                detector->hb_rstamp = INFINITY;
                detector->hb_period = INFINITY;
                return ORTE_SUCCESS;
            }

            OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                                    "errmgr:detector hb request updating ring"));
            detector->hb_observing = daemon.vpid;
    	    buffer = OBJ_NEW(opal_buffer_t);
            if (OPAL_SUCCESS != (ret = opal_dss.pack(buffer, &orte_process_info.my_name.jobid, 1,OPAL_JOBID))) {
                ORTE_ERROR_LOG(ret); }
            if (OPAL_SUCCESS != (ret = opal_dss.pack(buffer, &orte_process_info.my_name.vpid, 1,OPAL_VPID))) {
                ORTE_ERROR_LOG(ret); }
	        if (0 > (ret = orte_rml.send_buffer_nb(orte_mgmt_conduit, &daemon, buffer, ORTE_RML_TAG_HEARTBEAT_REQUEST, orte_rml_send_callback, NULL))) {
	            ORTE_ERROR_LOG(ret);}
            break;
    }
    OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                "errmgr:detector updated ring daemon %d observering %d observer %d",
                ORTE_PROC_MY_NAME->vpid,
                detector->hb_observing,
                detector->hb_observer));

    detector->hb_rstamp = Wtime()+detector->hb_timeout; /* we add one timeout slack to account for the send time */
    return ORTE_SUCCESS;
}

static int fd_heartbeat_request_cb(int status, orte_process_name_t* sender,
                                  opal_buffer_t *buffer,
                                  orte_rml_tag_t tg, void *cbdata) {
    orte_errmgr_detector_t* detector = &orte_errmgr_world_detector;
    int ndmns, rr, ro;
    opal_jobid_t vpid, jobid;
    int temp;
    temp =1;
    int rc;
    if (OPAL_SUCCESS != (rc = opal_dss.unpack(buffer, &jobid,&temp,OPAL_JOBID)))
                ORTE_ERROR_LOG(rc);
    if (OPAL_SUCCESS != (rc = opal_dss.unpack(buffer, &vpid, &temp,OPAL_VPID)))
		ORTE_ERROR_LOG(rc);
    OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                        "errmgr:detector %d receive %d",
                        orte_process_info.my_name.vpid,
                        detector->hb_observer));
    ndmns = orte_process_info.num_nodes;
    rr = (ndmns-orte_process_info.my_name.vpid+vpid) % ndmns; /* translate msg->from in circular space so that myrank==0 */
    ro = (ndmns-orte_process_info.my_name.vpid+detector->hb_observer) % ndmns; /* same for the observer rank */
    if( rr < ro ) {
        return false; /* never forward on the rbcast */
    }

    detector->hb_observer = vpid;
    detector->hb_sstamp = 0.;

    fd_heartbeat_send(detector);
    return false;
}

/*
 * event loop and thread
 */

static void fd_event_cb(int fd, short flags, void* pdetector) {
    // need to find a new time func
    double stamp = Wtime();
    orte_errmgr_detector_t* detector = pdetector;

    // temp proc name for get the orte object
    orte_process_name_t temp_proc_name;

    if( (stamp - detector->hb_sstamp) >= detector->hb_period ) {
        fd_heartbeat_send(detector);
    }
    if( INFINITY == detector->hb_rstamp ) return;

    // rte_process_info.my_name.vpid == observing ? or observe mean quit msg

    if( (stamp - detector->hb_rstamp) > detector->hb_timeout ) {
        /* this process is now suspected dead. */
        temp_proc_name.jobid = orte_process_info.my_name.jobid;
        temp_proc_name.vpid = detector->hb_observing;
        /* if first time detected */
        if (errmgr_get_daemon_status(temp_proc_name)){
            OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                        "errmgr:detector %d observing %d",
                        orte_process_info.my_name.vpid, detector->hb_observing));
            orte_propagate.prp(&temp_proc_name.jobid, &temp_proc_name, &temp_proc_name,OPAL_ERR_PROC_ABORTED );
            errmgr_set_daemon_status(temp_proc_name, false);
            fd_heartbeat_request(detector);
        }
    }
}

/*
 * send eager based heartbeats
 */
static int fd_heartbeat_send(orte_errmgr_detector_t* detector) {

    double now = Wtime();
    if( 0. != detector->hb_sstamp
            && (now - detector->hb_sstamp) >= 2.*detector->hb_period ) {
         /* missed my send deadline find a verbose to use */
         OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                             "errmgr:detector: daemon %s MISSED my deadline by %.1e, this could trigger a false suspicion for me",
                             ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),
                             now-detector->hb_sstamp));
    }
    detector->hb_sstamp = now;

    opal_buffer_t *buffer = NULL;
    int ret;

    buffer = OBJ_NEW(opal_buffer_t);
    orte_process_name_t daemon;
    daemon.jobid = orte_process_info.my_name.jobid;
    daemon.vpid = detector->hb_observer;
    if (OPAL_SUCCESS != (ret = opal_dss.pack(buffer, &orte_process_info.my_name.jobid, 1, OPAL_JOBID))) {
            ORTE_ERROR_LOG(ret);
    }
    if (OPAL_SUCCESS != (ret = opal_dss.pack(buffer, &orte_process_info.my_name.vpid, 1, OPAL_VPID))) {
            ORTE_ERROR_LOG(ret);
    }
    /* send the heartbeat with eager send */
    if (0 > (ret  = orte_rml.send_buffer_nb(orte_mgmt_conduit, &daemon, buffer, ORTE_RML_TAG_HEARTBEAT, orte_rml_send_callback, NULL))) {
        OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,"errmgr:detector:failed to send heartbeat to %d:%d", daemon.jobid, daemon.vpid));
        ORTE_ERROR_LOG(ret);
    }
    return ORTE_SUCCESS;
}

static int fd_heartbeat_recv_cb(int status, orte_process_name_t* sender,
                                  opal_buffer_t *buffer,
                                  orte_rml_tag_t tg, void *cbdata) {
    orte_errmgr_detector_t* detector = &orte_errmgr_world_detector;
    int rc;
    int32_t cnt;
    uint32_t vpid, jobid;

    if ( sender->vpid == orte_process_info.my_name.vpid)
    {
        /* this is a quit msg from observed process, stop detector */
        OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                    "errmgr:detector:%s %s Received heartbeat from %d, which is myself, quit msg to close detector",
                    ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),__func__, sender->vpid));
         detector->hb_observing = detector->hb_observer = ORTE_VPID_INVALID;
         detector->hb_rstamp = INFINITY;
         detector->hb_period = INFINITY;
         return false;
    }

    cnt = 1;
    if (OPAL_SUCCESS != (rc = opal_dss.unpack(buffer, &jobid, &cnt, OPAL_JOBID))){
           ORTE_ERROR_LOG(rc);
    }
    cnt = 1;
    if (OPAL_SUCCESS != (rc = opal_dss.unpack(buffer, &vpid, &cnt, OPAL_VPID))){
          ORTE_ERROR_LOG(rc);
    }

    if(vpid != detector->hb_observing ) {
        OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                             "errmgr:detector: daemon %s receive heartbeat from vpid %d, but I am monitoring vpid %d ",
                             ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),
                             vpid,
                             detector->hb_observing ));
    }
    else {
        double stamp = Wtime();
        double grace = detector->hb_timeout - (stamp - detector->hb_rstamp);
        OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                             "errmgr:detector: daemon %s receive heartbeat from vpid %d tag %d at timestamp %g (remained %.1e of %.1e before suspecting)",
                             ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),
                             vpid,
                             tg,
                             stamp,
                             grace,
                             detector->hb_timeout));
        detector->hb_rstamp = stamp;
        if( grace < 0.0 ) {
            OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                             "errmgr:detector: daemon %s  MISSED (%.1e)",
                             ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),
                             grace));
	    }
    }
    return false;
}


