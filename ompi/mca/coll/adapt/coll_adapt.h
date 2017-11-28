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
/** @file */

#ifndef MCA_COLL_ADAPT_EXPORT_H
#define MCA_COLL_ADAPT_EXPORT_H

#include "ompi_config.h"

#include "mpi.h"
#include "opal/mca/mca.h"
#include "opal/datatype/opal_convertor.h"
#include "ompi/mca/coll/coll.h"
#if OPAL_CUDA_SUPPORT
#include "opal/mca/mpool/mpool.h"
#endif

#if OPAL_CUDA_SUPPORT  
#define CPU_BUFFER  0
#define GPU_BUFFER  1
#else
#define CPU_BUFFER  1
#define GPU_BUFFER  0
#endif

BEGIN_C_DECLS

typedef struct mca_coll_adapt_module_t mca_coll_adapt_module_t;

/** 
 * Structure to hold the adapt coll component.  First it holds the
 * base coll component, and then holds a bunch of
 * adapt-coll-component-specific stuff (e.g., current MCA param
 * values). 
 */
typedef struct mca_coll_adapt_component_t {
    /** Base coll component */
    mca_coll_base_component_2_0_0_t super;

    /** MCA parameter: Priority of this component */
    int adapt_priority;

    /** MCA parameter: Length of a cache line or page (in bytes) */
    int adapt_control_size;

    /** MCA parameter: Number of "in use" flags in each
        communicator's area in the data mpool */
    int adapt_comm_num_in_use_flags;

    /** MCA parameter: Number of segments for each communicator in
        the data mpool */
    int adapt_comm_num_segments;

    /** MCA parameter: Fragment size for data */
    int adapt_fragment_size;

    /** MCA parameter: Degree of tree for tree-based collectives */
    int adapt_tree_degree;

    /** MCA parameter: Number of processes to use in the
        calculation of the "info" MCA parameter */
    int adapt_info_comm_size;
    
    /** MCA parameter: Output verbose level */
    int adapt_output;
    
    /** MCA parameter: Maximum number of segment in context free list */
    int adapt_context_free_list_max;
        
    /** MCA parameter: Minimum number of segment in context free list */
    int adapt_context_free_list_min;
    
    /** MCA parameter: Increasment number of segment in context free list */
    int adapt_context_free_list_inc;

    /******* end of MCA params ********/

    /** How many fragment segments are protected by a single
        in-use flags.  This is solely so that we can only perform
        the division once and then just use the value without
        having to re-calculate. */
    int adapt_segs_per_inuse_flag;
    
    /** cuda support */
    int coll_adapt_cuda_enabled;

#if OPAL_CUDA_SUPPORT    
    /** pinned cpu memory for GPU use */
    mca_mpool_base_module_t *pined_cpu_mpool;
    
    /** pinned gpu memory for GPU use */
    mca_mpool_base_module_t *pined_gpu_mpool;
#endif
} mca_coll_adapt_component_t;

/**
 * Structure for the sm coll module to hang off the communicator.
 * Contains communicator-specific information, including pointers
 * into the per-communicator shmem data data segment for this
 * comm's sm collective operations area.
 */
typedef struct mca_coll_adapt_comm_t {
    /* Meta data that we get back from the common mmap allocation
       function */
    mca_coll_adapt_module_t *adapt_bootstrap_meta;
    
    /** Pointer to my barrier control pages (odd index pages are
        "in", even index pages are "out") */
    uint32_t *mcb_barrier_control_me;
    
    /** Pointer to my parent's barrier control pages (will be NULL
        for communicator rank 0; odd index pages are "in", even
        index pages are "out") */
    uint32_t *mcb_barrier_control_parent;
    
    /** Pointers to my childrens' barrier control pages (they're
        contiguous in memory, so we only point to the base -- the
        number of children is in my entry in the mcb_tree); will
        be NULL if this process has no children (odd index pages
        are "in", even index pages are "out") */
    uint32_t *mcb_barrier_control_children;
    
    /** Number of barriers that we have executed (i.e., which set
        of barrier buffers to use). */
    int mcb_barrier_count;
    
    /** Operation number (i.e., which segment number to use) */
    uint32_t mcb_operation_count;
} mca_coll_adapt_comm_t;

/** Coll sm module */
struct mca_coll_adapt_module_t {
    /** Base module */
	mca_coll_base_module_t super;
    
    /* Whether this module has been lazily initialized or not yet */
    bool enabled;
    
    /* Data that hangs off the communicator */
	mca_coll_adapt_comm_t *adapt_comm_data;
    
        /* Underlying reduce function and module */
	mca_coll_base_module_reduce_fn_t previous_reduce;
	mca_coll_base_module_t *previous_reduce_module;
};
OBJ_CLASS_DECLARATION(mca_coll_adapt_module_t);

/**
 * Global component instance
 */
OMPI_MODULE_DECLSPEC extern mca_coll_adapt_component_t mca_coll_adapt_component;



/*
 * coll module functions
 */
int mca_coll_adapt_init_query(bool enable_progress_threads,
                              bool enable_mpi_threads);

mca_coll_base_module_t *
mca_coll_adapt_comm_query(struct ompi_communicator_t *comm, int *priority);

/* Lazily enable a module (since it involves expensive/slow mmap
   allocation, etc.) */
int ompi_coll_adapt_lazy_enable(mca_coll_base_module_t *module,
                                struct ompi_communicator_t *comm);


int mca_coll_adapt_ft_event(int state);

#endif /* MCA_COLL_ADAPT_EXPORT_H */
