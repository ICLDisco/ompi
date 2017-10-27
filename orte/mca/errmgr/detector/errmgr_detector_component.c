/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2010      Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2016      Intel, Inc. All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "orte_config.h"
#include "opal/util/output.h"

#include "orte/mca/errmgr/errmgr.h"
#include "orte/mca/errmgr/base/base.h"
#include "orte/mca/errmgr/base/errmgr_private.h"
#include "errmgr_detector.h"

/*
 * Public string for version number
 */
const char *orte_errmgr_detector_component_version_string =
    "ORTE ERRMGR detector MCA component version " ORTE_VERSION;

/*
 * Local functionality
 */
static int errmgr_detector_register(void);
static int errmgr_detector_open(void);
static int errmgr_detector_close(void);
static int errmgr_detector_component_query(mca_base_module_t **module, int *priority);

/*
 * Instantiate the public struct with all of our public information
 * and pointer to our public functions in it
 */
orte_errmgr_base_component_t mca_errmgr_detector_component = {
    /* Handle the general mca_component_t struct containing
     *  meta information about the component detector
     */
    .base_version = {
        ORTE_ERRMGR_BASE_VERSION_3_0_0,
        /* Component name and version */
        .mca_component_name = "detector",
        MCA_BASE_MAKE_VERSION(component, ORTE_MAJOR_VERSION, ORTE_MINOR_VERSION,
                              ORTE_RELEASE_VERSION),

        /* Component open and close functions */
        .mca_open_component = errmgr_detector_open,
        .mca_close_component = errmgr_detector_close,
        .mca_query_component = errmgr_detector_component_query,
        .mca_register_component_params = errmgr_detector_register,
    },
    .base_data = {
        /* The component is checkpoint ready */
        MCA_BASE_METADATA_PARAM_CHECKPOINT
    },
};

static int my_priority;

static int errmgr_detector_register(void)
{
    mca_base_component_t *c = &mca_errmgr_detector_component.base_version;

    my_priority = 900;
    (void) mca_base_component_var_register(c, "priority",
                                           "Priority of the detector errmgr component",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &my_priority);

    return ORTE_SUCCESS;
}

static int errmgr_detector_open(void)
{
    return ORTE_SUCCESS;
}

static int errmgr_detector_close(void)
{
    return ORTE_SUCCESS;
}

static int errmgr_detector_component_query(mca_base_module_t **module, int *priority)
{
    /* used by DVM masters */
   // if (ORTE_PROC_IS_DAEMON || ORTE_PROC_IS_HNP ) {
        *priority = my_priority;
        *module = (mca_base_module_t *)&orte_errmgr_detector_module;
        return ORTE_SUCCESS;
    //}

    *module = NULL;
    *priority = -1;
    return ORTE_ERROR;
}
