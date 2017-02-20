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

#include "orte/mca/rml/rml.h"
#include "orte/mca/odls/odls.h"
#include "orte/mca/odls/base/base.h"
#include "orte/mca/odls/base/odls_private.h"
#include "orte/mca/plm/base/plm_private.h"
#include "orte/mca/plm/plm.h"
#include "orte/mca/rmaps/rmaps_types.h"
#include "orte/mca/routed/routed.h"
#include "orte/mca/grpcomm/grpcomm.h"
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

#include "opal/mca/base/mca_base_var.h"
#include "opal/mca/timer/base/base.h"
#include "opal/threads/threads.h"

#include "orte/mca/grpcomm/bmg/grpcomm_bmg.h"
#include "errmgr_detector.h""

static int init(void);
static int finalize(void);

static int predicted_fault(opal_list_t *proc_list,
                           opal_list_t *node_list,
                           opal_list_t *suggested_map);

static int uggest_map_targetsuggest_map_targets(orte_proc_t *proc,
                               orte_node_t *oldnode,
                               opal_list_t *node_list);

/******************
 * detector module
 ******************/
orte_errmgr_base_module_t orte_errmgr_detector_module = {
    init,
    finalize,
    orte_errmgr_base_log,
    orte_errmgr_base_abort,
    orte_errmgr_base_abort_peers,
    predicted_fault,
    suggest_map_targets,
    NULL,
    orte_errmgr_base_register_migration_warning,
    NULL,
    orte_errmgr_base_execute_error_callbacks
};

typedef struct {
    opal_event_t fd_event; /* to trigger timeouts with opal_events */
    int hb_observing;      /* the deamon vpid of the process we observe */
    int hb_observer;       /* the daemon vpid of the process that observes us */
    double hb_rstamp;      /* the date of the last hb reception */
    double hb_timeout;     /* the timeout before we start suspecting observed process as dead (delta) */
    double hb_period;      /* the time spacing between heartbeat emission (eta) */
    double hb_sstamp;      /* the date at which the last hb emission was done */
} orte_errmgr_detector_t;
static orte_errmgr_detector_t orte_errmgr_world_detector;

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

static int orte_errmgr_start_detector();

static double orte_errmgr_heartbeat_period = 2e-1;
static double orte_errmgr_heartbeat_timeout = 5e-1;
static opal_event_base_t* fd_event_base = NULL;
static void fd_event_cb(int fd, short flags, void* pdetector);

#if OPAL_ENABLE_MULTI_THREADS
static bool orte_errmgr_detector_use_thread = true;
static volatile int32_t fd_thread_active = 0;
static opal_thread_t fd_thread;
static void* fd_progress(opal_object_t* obj);
#endif /* OPAL_ENABLE_MULTI_THREADS */

int init(void) {
    int ret;
    fd_event_base = opal_sync_event_base;

    /* setup state machine to trap job errors */
    orte_state.add_job_state(ORTE_JOB_STATE_ERROR, job_errors, ORTE_ERROR_PRI);

    /* set the lost connection state to run at MSG priority so
     * we can process any last messages from the proc
     */
    orte_state.add_proc_state(ORTE_PROC_STATE_COMM_FAILED, proc_errors, ORTE_MSG_PRI);

    /* setup state machine to trap proc errors */
    orte_state.add_proc_state(ORTE_PROC_STATE_ERROR, proc_errors, ORTE_ERROR_PRI);

    /* registering the cb types */
    orte_rml.recv_buffer_nb(ORTE_NAME_WILDCARD, ORTE_RML_TAG_HEARTBEAT_REQUEST, 
                            ORTE_RML_PERSISTENT,fd_heartbeat_request_cb,NULL);
    orte_rml.recv_buffer_nb(ORTE_NAME_WILDCARD, ORTE_RML_TAG_HEARTBEAT, 
                            ORTE_RML_PERSISTENT,fd_heartbeat_recv_cb,NULL);
    /* init propagator*/ 
    orte_errmgr_init_failure_propagate();
#if OPAL_ENABLE_MULTI_THREADS
    if( orte_errmgr_detector_use_thread ) {
        fd_event_base = opal_event_base_create();
        if( NULL == fd_event_base ) {
            fd_event_base = opal_sync_event_base;
            ret = ORTE_ERR_OUT_OF_RESOURCE;
            goto cleanup;
        }
        opal_event_use_threads();
        opal_set_using_threads(true);
        OBJ_CONSTRUCT(&fd_thread, opal_thread_t);
        fd_thread.t_run = fd_progress;
        fd_thread.t_arg = NULL;
        ret = opal_thread_start(&fd_thread);
        if( OPAL_SUCCESS != ret ) goto cleanup;
        while( 0 == fd_thread_active ); /* wait for the fd thread initialization */
        return ORTE_SUCCESS;
    }
#endif /* OPAL_ENABLE_MULTI_THREADS */
    opal_progress_event_users_increment();

    /* setting up the default detector */
    return orte_errmgr_start_detector();

cleanup:
    orte_errmgr_finalize_failure_detector();
    return ORTE_SUCCESS;
}

int finalize(void) {

#if OPAL_ENABLE_MULTI_THREADS
    if( 1 == fd_thread_active ) {
        void* tret;
        OPAL_THREAD_ADD32(&fd_thread_active, -1);
        opal_event_base_loopbreak(fd_event_base);
        opal_thread_join(&fd_thread, &tret);
    }
#endif /* OPAL_ENABLE_MULTI_THREADS */

    opal_event_del(&orte_errmgr_world_detector.fd_event);
    
    /* unregister callbacks */
    if( opal_sync_event_base != fd_event_base ) opal_event_base_free(fd_event_base);
    orte_rml.recv_cancel(ORTE_NAME_WILDCARD, ORTE_RML_TAG_HEARTBEAT_REQUEST);   
    orte_rml.recv_cancel(ORTE_NAME_WILDCARD, ORTE_RML_TAG_HEARTBEAT);
    orte_errmgr_finalize_failure_propagate();
    return ORTE_SUCCESS;
}

static int orte_errmgr_start_detector() {

    orte_errmgr_detector_t* detector = &orte_errmgr_world_detector;
    int  ndmns;
    uint32_t vpid;
    
    ndmns = orte_process_info.num_nodes;                /* num of daemon in this jobid */
    vpid = orte_process_info.my_daemon.vpid;            /* my deamon's vpid */
    if( 0 != (vpid - 1) )
        detector->hb_observing = vpid - 1;              /*  we observing somebody {n,1,2,...n-1}, the ring */
    else detector->hb_observing = ndmns;

    detector->hb_observer = (ndmns+vpid) % ndmns + 1 ;  /* someone is observing us: range [1~n], the observing ring */ 
    detector->hb_period = orte_errmgr_heartbeat_period;
    detector->hb_timeout = orte_errmgr_heartbeat_timeout;
    detector->hb_sstamp = 0.;
    detector->hb_rstamp = PMPI_Wtime()+(double)ndmns;      /* give some slack for MPIInit */
   
    fd_heartbeat_request(detector);    

    opal_event_set(fd_event_base, &detector->fd_event, -1, OPAL_EV_TIMEOUT | OPAL_EV_PERSIST, fd_event_cb, detector);
    struct timeval tv;
    tv.tv_sec = (int)(detector->hb_period / 10.);
    tv.tv_usec = (-tv.tv_sec + (detector->hb_period / 10.)) * 1e6;
    opal_event_add(&detector->fd_event, &tv);
    return ORTE_SUCCESS;
}

static int fd_heartbeat_request(orte_errmgr_detector_t* detector) {
    
    int ret,  ndmns;
    uint32_t vpid;
  
    //using proc name to get orte_proc object
    orte_process_name_t temp_proc_name;
    orte_proc_t *temp_orte_proc;

    temp_proc_name.jobid = orte_process_info.my_daemon.jobid;
    temp_proc_name.vpid = detector->hb_observing;  
    temp_orte_proc = orte_get_proc_object(&temp_proc_name);   
    //or ORTE_PROC_STATE_RUNNING == temp_orte_proc->state
    if( ORTE_PROC_STATE_HEARTBEAT_FAILED != temp_orte_proc->state )
    {
       	/* already observing a live process, so nothing to do. */
        return ORTE_SUCCESS;
    }

    ndmns = orte_process_info.num_nodes;

    opal_buffer_t *buffer = NULL;
    orte_process_name_t daemon;
	
    for( vpid = (ndmns+detector->hb_observing) % ndmns;
         vpid != orte_process_info.my_daemon.vpid;
         vpid = (ndmns+vpid-1) % ndmns ) {
        
        daemon.jobid = orte_process_info.my_daemon.jobid;
        if(0 != vpid){
            daemon.vpid = vpid;
        }
        else daemon.vpid = ndmns;
        temp_orte_proc = orte_get_proc_object(&daemon);
        if( ORTE_PROC_STATE_HEARTBEAT_FAILED == temp_orte_proc->state ) continue;

        detector->hb_observing = daemon.vpid;		
	buffer = OBJ_NEW(opal_buffer_t);
        if (OPAL_SUCCESS != (ret = opal_dss.pack(buffer, &orte_process_info.my_daemon.jobid, 1,OPAL_JOBID))) {
            ORTE_ERROR_LOG(ret); }
        if (OPAL_SUCCESS != (ret = opal_dss.pack(buffer, &orte_process_info.my_daemon.vpid, 1,OPAL_VPID))) {
            ORTE_ERROR_LOG(ret); }
	if (0 > (ret = orte_rml.send_buffer_nb(&daemon, buffer, ORTE_RML_TAG_HEARTBEAT_REQUEST, orte_rml_send_callback, NULL))) {
	    ORTE_ERROR_LOG(ret);}
        break;
    }
    /* if everybody else is dead, then it's a success */
    detector->hb_rstamp = PMPI_Wtime()+detector->hb_timeout; /* we add one timeout slack to account for the send time */
    return ORTE_SUCCESS;
}

// int or void cause it dont have to return a value
static int  fd_heartbeat_request_cb(int status, orte_process_name_t* sender,
                                  opal_buffer_t *buffer,
                                  orte_rml_tag_t tg, void *cbdata) {
  
    orte_errmgr_detector_t* detector = &orte_errmgr_world_detector;

    int ndmns, rr, ro;
    opal_jobid_t vpid, jobid;
    int temp;
    int rc;
	
    if (OPAL_SUCCESS != (rc = opal_dss.unpack(buffer, &jobid,&temp,OPAL_JOBID)))
                ORTE_ERROR_LOG(rc);
    if (OPAL_SUCCESS != (rc = opal_dss.unpack(buffer, &vpid, &temp,OPAL_VPID)))
		ORTE_ERROR_LOG(rc);
   
    ndmns = orte_process_info.num_nodes;
    rr = (ndmns-orte_process_info.my_daemon.vpid+vpid) % ndmns; /* translate msg->from in circular space so that myrank==0 */
    ro = (ndmns-orte_process_info.my_daemon.vpid+detector->hb_observer) % ndmns; /* same for the observer rank */
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
    double stamp = PMPI_Wtime();
    orte_errmgr_detector_t* detector = pdetector;

    // temp proc name for get the orte object 
    orte_process_name_t temp_proc_name;
    orte_proc_t *temp_orte_proc;

    if( (stamp - detector->hb_sstamp) >= detector->hb_period ) {
        fd_heartbeat_send(detector);
    }
 
    if( (stamp - detector->hb_rstamp) > detector->hb_timeout ) {
        /* this process is now suspected dead. */
        temp_proc_name.jobid = orte_process_info.my_daemon.jobid;
        temp_proc_name.vpid = detector->hb_observing;
 
        temp_orte_proc =  orte_get_proc_object(&temp_proc_name); 
        /* if first time detected */
        if( ORTE_PROC_STATE_HEARTBEAT_FAILED != temp_orte_proc->state ){
            orte_errmgr_failure_propagate(temp_proc_name.jobid, &temp_proc_name, ORTE_PROC_STATE_HEARTBEAT_FAILED); 
            fd_heartbeat_request(detector);
        } 
    }
}

#if OPAL_ENABLE_MULTI_THREADS
void* fd_progress(opal_object_t* obj) {
    if( ORTE_SUCCESS != orte_errmgr_start_detector()) {
        OPAL_THREAD_ADD32(&fd_thread_active, -1);
        return OPAL_THREAD_CANCELLED;
    }
    OPAL_THREAD_ADD32(&fd_thread_active, 1);
    while( fd_thread_active ) { 
        opal_event_loop(fd_event_base, OPAL_EVLOOP_ONCE);
    }
    return OPAL_THREAD_CANCELLED;
}
#endif /* OPAL_ENABLE_MULTI_THREADS */

/*
 * send eager based heartbeats
 */
static int fd_heartbeat_send(orte_errmgr_detector_t* detector) {
   
    double now = PMPI_Wtime();

    if( 0. != detector->hb_sstamp
     && (now - detector->hb_sstamp) >= 2.*detector->hb_period ) {
         // missed my send deadline  find a verbose to use 
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
    daemon.jobid = orte_process_info.my_daemon.jobid;
    daemon.vpid = detector->hb_observer;

    if (OPAL_SUCCESS != (ret = opal_dss.pack(buffer, &orte_process_info.my_daemon.jobid, 1,OPAL_JOBID))) {
            ORTE_ERROR_LOG(ret);        
    }
    if (OPAL_SUCCESS != (ret = opal_dss.pack(buffer, &orte_process_info.my_daemon.vpid, 1,OPAL_VPID))) {
            ORTE_ERROR_LOG(ret);
    }					 
    /* send the heartbeat with eager send */
    if (0 > (ret  = orte_rml.send_buffer_nb(&daemon, buffer,ORTE_RML_TAG_HEARTBEAT, orte_rml_send_callback, NULL))) {
           ORTE_ERROR_LOG(ret);
    }
    return ORTE_SUCCESS;
}

static int fd_heartbeat_recv_cb(int status, orte_process_name_t* sender,
                                  opal_buffer_t *buffer,
                                  orte_rml_tag_t tg, void *cbdata) {
    orte_errmgr_detector_t* detector = &orte_errmgr_world_detector;
    int rc;
    uint32_t  vpid, jobid;
    if (OPAL_SUCCESS != (rc = opal_dss.unpack(buffer, &jobid, 1, OPAL_JOBID))){
           ORTE_ERROR_LOG(rc);
    }
    if (OPAL_SUCCESS != (rc = opal_dss.unpack(buffer, &vpid, 1, OPAL_VPID))){
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
        double stamp = PMPI_Wtime();
        double grace = detector->hb_timeout - (stamp - detector->hb_rstamp);   
        OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                             "errmgr:detector: daemon %s receive heartbeat from vpid %d at timestamp %g (remained %.1e of %.1e before suspecting)",
                             ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),
                             vpid,
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

static int predicted_fault(opal_list_t *proc_list,
                           opal_list_t *node_list,
                           opal_list_t *suggested_map)
{
    return ORTE_ERR_NOT_IMPLEMENTED;
}

static int suggest_map_targets(orte_proc_t *proc,
                               orte_node_t *oldnode,
                               opal_list_t *node_list)
{
    return ORTE_ERR_NOT_IMPLEMENTED;
}

