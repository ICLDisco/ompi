
/*
 * Copyright (c) 2014      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 */
/**
 * @file
 *
 * Warning: this is not for the faint of heart -- don't even bother
 * reading this source code if you don't have a strong understanding
 * of nested data structures and pointer math (remember that
 * associativity and order of C operations is *critical* in terms of
 * pointer math!).
 */

#include "ompi_config.h"

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_SCHED_H
#include <sched.h>
#endif
#include <sys/types.h>
#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif  /* HAVE_SYS_MMAN_H */
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif  /* HAVE_UNISTD_H */

#include "mpi.h"
#include "opal_stdint.h"
#include "opal/mca/hwloc/base/base.h"
#include "opal/util/os_path.h"

#include "ompi/communicator/communicator.h"
#include "ompi/group/group.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/rte/rte.h"
#include "ompi/proc/proc.h"
#include "coll_adapt.h"

#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_adapt_algorithms.h"

#define OPAL_HAVE_HWLOC 1
/*
 * Global variables
 */
uint32_t mca_coll_adapt_one = 1;


/*
 * Local functions
 */
static int adapt_module_enable(mca_coll_base_module_t *module,
                          struct ompi_communicator_t *comm);
static int bootstrap_comm(ompi_communicator_t *comm,
                          mca_coll_adapt_module_t *module);

/*
 * Module constructor
 */
static void mca_coll_adapt_module_construct(mca_coll_adapt_module_t *module)
{
    module->enabled = false;
    module->adapt_comm_data = NULL;
    module->previous_reduce = NULL;
    module->previous_reduce_module = NULL;
}

/*
 * Module destructor
 */
static void mca_coll_adapt_module_destruct(mca_coll_adapt_module_t *module)
{
    mca_coll_adapt_comm_t *c = module->adapt_comm_data;

    if (NULL != c) {
        /* Munmap the per-communicator shmem data segment */
        if (NULL != c->adapt_bootstrap_meta) {
            /* Ignore any errors -- what are we going to do about
               them? */
        }
        free(c);
    }

    /* It should always be non-NULL, but just in case */
    if (NULL != module->previous_reduce_module) {
        OBJ_RELEASE(module->previous_reduce_module);
    }

    module->enabled = false;
}


OBJ_CLASS_INSTANCE(mca_coll_adapt_module_t,
                   mca_coll_base_module_t,
                   mca_coll_adapt_module_construct,
                   mca_coll_adapt_module_destruct);

/*
 * Initial query function that is invoked during MPI_INIT, allowing
 * this component to disqualify itself if it doesn't support the
 * required level of thread support.  This function is invoked exactly
 * once.
 */
int mca_coll_adapt_init_query(bool enable_progress_threads,
                           bool enable_mpi_threads)
{
    /* if no session directory was created, then we cannot be used */
    if (NULL == ompi_process_info.job_session_dir) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    /* Don't do much here because we don't really want to allocate any
       shared memory until this component is selected to be used. */
    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:adapt:init_query: pick me! pick me!");
    return OMPI_SUCCESS;
}


/*
 * Invoked when there's a new communicator that has been created.
 * Look at the communicator and decide which set of functions and
 * priority we want to return.
 */
mca_coll_base_module_t *
mca_coll_adapt_comm_query(struct ompi_communicator_t *comm, int *priority)
{
    mca_coll_adapt_module_t *adapt_module;

    /* If we're intercomm, or if there's only one process in the
       communicator, or if not all the processes in the communicator
       are not on this node, then we don't want to run */
    if (OMPI_COMM_IS_INTER(comm) || 1 == ompi_comm_size(comm)) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:adapt:comm_query (%d/%s): intercomm, comm is too small, disqualifying myself", comm->c_contextid, comm->c_name);
	return NULL;
    }

    /* Get the priority level attached to this module. If priority is less
     * than or equal to 0, then the module is unavailable. */
    *priority = mca_coll_adapt_component.adapt_priority;
    if (mca_coll_adapt_component.adapt_priority <= 0) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:adapt:comm_query (%d/%s): priority too low; disqualifying myself", comm->c_contextid, comm->c_name);
	return NULL;
    }

    adapt_module = OBJ_NEW(mca_coll_adapt_module_t);
    if (NULL == adapt_module) {
        return NULL;
    }
    mca_coll_base_comm_t * data = OBJ_NEW(mca_coll_base_comm_t);
    if (NULL == data) {
        return NULL;
    }

    /* All is good -- return a module */
    adapt_module->super.coll_module_enable = adapt_module_enable;
    adapt_module->super.ft_event        = NULL;
    adapt_module->super.coll_allgather  = NULL;
    adapt_module->super.coll_allgatherv = NULL;
    adapt_module->super.coll_allreduce  = mca_coll_adapt_allreduce;
    adapt_module->super.coll_alltoall   = NULL;
    //adapt_module->super.coll_alltoallv  = mca_coll_adapt_alltoallv;
    adapt_module->super.coll_alltoallw  = NULL;
    adapt_module->super.coll_barrier    = NULL;
    adapt_module->super.coll_bcast      = mca_coll_adapt_bcast;
    adapt_module->super.coll_exscan     = NULL;
    adapt_module->super.coll_gather     = NULL;
    adapt_module->super.coll_gatherv    = NULL;
    adapt_module->super.coll_reduce     = mca_coll_adapt_reduce;
    adapt_module->super.coll_reduce_scatter = NULL;
    adapt_module->super.coll_scan       = NULL;
    adapt_module->super.coll_scatter    = NULL;
    adapt_module->super.coll_scatterv   = NULL;
    adapt_module->super.coll_ibcast     = mca_coll_adapt_ibcast;
    adapt_module->super.coll_ireduce    = mca_coll_adapt_ireduce;
    //adapt_module->super.coll_ialltoallv = mca_coll_adapt_ialltoallv;
    adapt_module->super.coll_iallreduce = NULL; //mca_coll_adapt_iallreduce;

    /* general n fan out tree */
    data->cached_ntree = NULL;
    /* binary tree */
    data->cached_bintree = NULL;
    /* binomial tree */
    data->cached_bmtree = NULL;
    /* binomial tree */
    data->cached_in_order_bmtree = NULL;
    /* chains (fanout followed by pipelines) */
    data->cached_chain = NULL;
    /* standard pipeline */
    data->cached_pipeline = NULL;
    /* in-order binary tree */
    data->cached_in_order_bintree = NULL;
    /* linear */
    data->cached_linear = NULL;
    /* topo linear */
    data->cached_topolinear = NULL;
    /* topo chain */
    data->cached_topochain = NULL;
    /* two tree binary */
    data->cached_two_trees_binary = NULL;
    /* two tree binomial */
    data->cached_two_trees_binomial = NULL;
    /* All done */
    adapt_module->super.base_data = data;
    
    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:adapt:comm_query (%d/%s): pick me! pick me!",
                        comm->c_contextid, comm->c_name);
    return &(adapt_module->super);
}


/*
 * Init module on the communicator
 */
static int adapt_module_enable(mca_coll_base_module_t *module,
                            struct ompi_communicator_t *comm)
{
    /* We do everything lazily in ompi_coll_adapt_enable() */
    return OMPI_SUCCESS;
}

int ompi_coll_adapt_lazy_enable(mca_coll_base_module_t *module,
                             struct ompi_communicator_t *comm)
{
    int ret;
    int rank = ompi_comm_rank(comm);
    int size = ompi_comm_size(comm);
    mca_coll_adapt_module_t *adapt_module = (mca_coll_adapt_module_t*) module;
    size_t control_size, frag_size;
    mca_coll_adapt_component_t *c = &mca_coll_adapt_component;
#if OPAL_HAVE_HWLOC
    opal_hwloc_base_memory_segment_t *maffinity;
#endif

    /* Just make sure we haven't been here already */
    if (adapt_module->enabled) {
        return OMPI_SUCCESS;
    }
    adapt_module->enabled = true;

#if OPAL_HAVE_HWLOC
    /* Get some space to setup memory affinity (just easier to try to
       alloc here to handle the error case) */
    maffinity = (opal_hwloc_base_memory_segment_t*)
        malloc(sizeof(opal_hwloc_base_memory_segment_t) * 
               c->adapt_comm_num_segments * 3);
    if (NULL == maffinity) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:adapt:lazy_enable (%d/%s): malloc failed (1)",
                            comm->c_contextid, comm->c_name);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
#endif

    /* Attach to this communicator's shmem data segment */
    if (OMPI_SUCCESS != (ret = bootstrap_comm(comm, adapt_module))) {
#if OPAL_HAVE_HWLOC
        free(maffinity);
#endif
        adapt_module->adapt_comm_data = NULL;
        return ret;
    }

    control_size = size * c->adapt_control_size;
    frag_size = size * c->adapt_fragment_size;

#if OPAL_HAVE_HWLOC
    /* Setup memory affinity so that the pages that belong to this
       process are local to this process */
    int j=0;
    opal_hwloc_base_memory_set(maffinity, j);
    free(maffinity);
#endif

    /* Save previous component's reduce information */
    adapt_module->previous_reduce = comm->c_coll->coll_reduce;
    adapt_module->previous_reduce_module = comm->c_coll->coll_reduce_module;
    OBJ_RETAIN(adapt_module->previous_reduce_module);

    /* Wait for everyone in this communicator to attach and setup */
    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:adapt:lazy_enable (%d/%s): waiting for peers to attach",
                        comm->c_contextid, comm->c_name);

    /* Once we're all here, remove the mmap file; it's not needed anymore */
    if (0 == rank) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:adapt:lazy_enable (%d/%s)",
                            comm->c_contextid, comm->c_name);
    }

    /* All done */

    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:adapt:lazy_enable (%d/%s): success!",
                        comm->c_contextid, comm->c_name);
    return OMPI_SUCCESS;
}



static int bootstrap_comm(ompi_communicator_t *comm,
                          mca_coll_adapt_module_t *module)
{
    int i;
    char *shortpath, *fullpath;
    mca_coll_adapt_component_t *c = &mca_coll_adapt_component;
    int comm_size = ompi_comm_size(comm);
    int num_segments = c->adapt_comm_num_segments;
    int num_in_use = c->adapt_comm_num_in_use_flags;
    int frag_size = c->adapt_fragment_size;
    int control_size = c->adapt_control_size;
    ompi_process_name_t *lowest_name = NULL;
    size_t size;
    ompi_proc_t *proc;

    /* Make the rendezvous filename for this communicators shmem data
       segment.  The CID is not guaranteed to be unique among all
       procs on this node, so also pair it with the PID of the proc
       with the lowest ORTE name to form a unique filename. */
    proc = ompi_group_peer_lookup(comm->c_local_group, 0);
    lowest_name = OMPI_CAST_RTE_NAME(&proc->super.proc_name);
    for (i = 1; i < comm_size; ++i) {
        proc = ompi_group_peer_lookup(comm->c_local_group, i);
        if (ompi_rte_compare_name_fields(OMPI_RTE_CMP_ALL, 
                                          OMPI_CAST_RTE_NAME(&proc->super.proc_name),
                                          lowest_name) < 0) {
            lowest_name = OMPI_CAST_RTE_NAME(&proc->super.proc_name);
        }
    }
    asprintf(&shortpath, "coll-adapt-cid-%d-name-%s.mmap", comm->c_contextid,
             OMPI_NAME_PRINT(lowest_name));
    if (NULL == shortpath) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:adapt:enable:bootstrap comm (%d/%s): asprintf failed",
                            comm->c_contextid, comm->c_name);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    fullpath = opal_os_path(false, ompi_process_info.job_session_dir,
                            shortpath, NULL);
    free(shortpath);
    if (NULL == fullpath) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:adapt:enable:bootstrap comm (%d/%s): opal_os_path failed",
                            comm->c_contextid, comm->c_name);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* Calculate how much space we need in the per-communicator shmem
       data segment.  There are several values to add:

       - size of the barrier data (2 of these):
           - fan-in data (num_procs * control_size)
           - fan-out data (num_procs * control_size)
       - size of the "in use" buffers:
           - num_in_use_buffers * control_size
       - size of the message fragment area (one for each segment):
           - control (num_procs * control_size)
           - fragment data (num_procs * (frag_size))

       So it's:

           barrier: 2 * control_size + 2 * control_size
           in use:  num_in_use * control_size
           control: num_segments * (num_procs * control_size * 2 +
                                    num_procs * control_size)
           message: num_segments * (num_procs * frag_size)
     */

    size = 4 * control_size +
        (num_in_use * control_size) +
        (num_segments * (comm_size * control_size * 2)) +
        (num_segments * (comm_size * frag_size));
    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:adapt:enable:bootstrap comm (%d/%s): attaching to %" PRIsize_t " byte mmap: %s",
                        comm->c_contextid, comm->c_name, size, fullpath);
    /* All done */
    return OMPI_SUCCESS;
}


int mca_coll_adapt_ft_event(int state) {
    if(OPAL_CRS_CHECKPOINT == state) {
        ;
    }
    else if(OPAL_CRS_CONTINUE == state) {
        ;
    }
    else if(OPAL_CRS_RESTART == state) {
        ;
    }
    else if(OPAL_CRS_TERM == state ) {
        ;
    }
    else {
        ;
    }

    return OMPI_SUCCESS;
}
