/*
 * Copyright (c) 2010      Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2004-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2016      Intel, Inc. All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/**
 * @file
 *
 */

#ifndef MCA_ERRMGR_detector_EXPORT_H
#define MCA_ERRMGR_detector_EXPORT_H

#include "orte_config.h"

#include "orte/mca/errmgr/errmgr.h"
#include "opal/mca/pmix/pmix_types.h"

BEGIN_C_DECLS

typedef struct {
    opal_event_t fd_event; /* to trigger timeouts with opal_events */
    int hb_observing;      /* the deamon vpid of the process we observe */
    int hb_observer;       /* the daemon vpid of the process that observes us */
    double hb_rstamp;      /* the date of the last hb reception */
    double hb_timeout;     /* the timeout before we start suspecting observed process as dead (delta) */
    double hb_period;      /* the time spacing between heartbeat emission (eta) */
    double hb_sstamp;      /* the date at which the last hb emission was done */
    bool daemons_state[256];
} orte_errmgr_detector_t;
static orte_errmgr_detector_t orte_errmgr_world_detector;

/*
 * Local Component structures
 */

ORTE_MODULE_DECLSPEC extern orte_errmgr_base_component_t mca_errmgr_detector_component;

ORTE_DECLSPEC extern orte_errmgr_base_module_t orte_errmgr_detector_module;

/*
 * Propagator functions
 */
int orte_errmgr_failure_propagate(orte_jobid_t *job, orte_process_name_t *daemon, orte_proc_state_t state);
int orte_errmgr_failure_propagate_recv(opal_buffer_t* buffer);
int orte_errmgr_init_failure_propagate(void);
int orte_errmgr_finalize_failure_propagate(void);
int errmgr_get_daemon_status(orte_process_name_t daemon);
void errmgr_set_daemon_status(orte_process_name_t daemon, bool state);
extern int orte_errmgr_enable_detector(bool flag);
END_C_DECLS

#endif /* MCA_ERRMGR_detector_EXPORT_H */
