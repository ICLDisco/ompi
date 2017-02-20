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

BEGIN_C_DECLS

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

END_C_DECLS

#endif /* MCA_ERRMGR_detector_EXPORT_H */
