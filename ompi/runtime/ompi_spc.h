/*
 * Copyright (c) 2018-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2018      Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OMPI_SPC
#define OMPI_SPC

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <fcntl.h>    /* For O_* constants */
#include <limits.h>

#include "ompi/communicator/communicator.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/runtime/params.h"
#include "opal/mca/timer/timer.h"
#include "opal/mca/base/mca_base_pvar.h"
#include "opal/util/argv.h"
#include "opal/util/show_help.h"
#include "opal/util/output.h"
#include "opal/mca/shmem/base/base.h"
#include "opal/util/sys_limits.h"

#define SPC_CACHE_LINE opal_cache_line_size /* The number of bytes in a cache line. */
#define SPC_MAX_FILENAME PATH_MAX /* The maximum length allowed for the spc file strings */
#define SPC_SHM_DIR "/dev/shm" /* The default directory for shared memory files */

#include MCA_timer_IMPLEMENTATION_HEADER

/* MCA Parameter Variables */
/**
 * A comma delimited list of SPC counters to turn on or 'attach'.  To turn
 * all counters on, the string can be simply "all".  An empty string will
 * keep all counters turned off.
 */
OMPI_DECLSPEC extern char * ompi_mpi_spc_attach_string;

/**
 * A string to append to the SPC XML files for using the mmap interface.
 * This is to make the filename easier to identify.
 */
OMPI_DECLSPEC extern char * ompi_mpi_spc_xml_string;

/**
 * A boolean value that determines whether or not to dump the SPC counter
 * values in MPI_Finalize.  A value of true dumps the counters and false does not.
 */
OMPI_DECLSPEC extern bool ompi_mpi_spc_dump_enabled;

/**
 * A boolean value that determines whether or not to dump the SPC counter
 * values in an mmap'd file during execution.  A value of true dumps the
 * counters and false does not.
 */
OMPI_DECLSPEC extern bool ompi_mpi_spc_mmap_enabled;

/**
 * An integer value that denotes the time period between snapshots with the
 * SPC mmap interface.
 */
OMPI_DECLSPEC extern int ompi_mpi_spc_snapshot_period;

/**
 * An integer value that denotes the boundary at which a message is qualified
 * as a small/large message for the point to point message counter.
 */
OMPI_DECLSPEC extern int ompi_mpi_spc_p2p_message_boundary;

/**
 * An integer value that denotes the boundary at which a message is qualified
 * as a small/large message for collective bin counters.
 */
OMPI_DECLSPEC extern int ompi_mpi_spc_collective_message_boundary;

/**
 * An integer value that denotes the boundary at which a communicator is qualified
 * as a small/large communicator for collective bin counters.
 */
OMPI_DECLSPEC extern int ompi_mpi_spc_collective_comm_boundary;

/* INSTRUCTIONS FOR ADDING COUNTERS
 * 1.) Add a new counter name in the ompi_spc_counters_t enum before
 *     OMPI_SPC_NUM_COUNTERS below.
 * 2.) Add corresponding counter description(s) to the
 *     ompi_spc_events_names definition in
 *     ompi_spc.c  NOTE: The names and descriptions
 *     MUST be in the same array location as where you added the
 *     counter name in step 1.
 *     Search For: 'STEP 2'
 * 3.) If this counter is a specialized counter like a timer,
 *     bin, or collective bin add its enum name to the logic for
 *     specialized counters in the ompi_spc_init function in ompi_spc.c
 *     Search For: 'STEP 3'
 *     NOTE: If this is a bin counter, and not a collective bin counter,
 *           you will need to initialize it. Search for: 'STEP 3a'
 * 4.) If this counter is a watermark counter, additional logic is required
 *     for when this counter is read.  The standard behavior of watermark
 *     counters is to keep track of updates to another counter and increase
 *     when that tracked counter exceeds the current value of the high
 *     watermark.  These counters are reset to the current value of the
 *     tracked counter whenever they are read through MPI_T in the
 *     ompi_spc_get_count function in ompi_spc.c
 *     Search For: 'STEP 4'
 * 5.) Instrument the Open MPI code where it makes sense for
 *     your counter to be modified using the appropriate SPC  macro.
 *     This will typically be SPC_RECORD, but could be SPC_BIN_RECORD,
 *     SPC_COLL_BIN_RECORD, SPC_UPDATE_WATERMARK, or SPC_TIMER_START/STOP
 *     depending on the counter.
 *     Note: If your counter is timer-based you should use the
 *     SPC_TIMER_START and SPC_TIMER_STOP macros to record
 *     the time in cycles to then be converted to microseconds later
 *     in the ompi_spc_get_count function when requested by MPI_T.
 */

/* This enumeration serves as event ids for the various events */
typedef enum ompi_spc_counters {
    OMPI_SPC_SEND,
    OMPI_SPC_BSEND,
    OMPI_SPC_RSEND,
    OMPI_SPC_SSEND,
    OMPI_SPC_RECV,
    OMPI_SPC_MRECV,
    OMPI_SPC_ISEND,
    OMPI_SPC_IBSEND,
    OMPI_SPC_IRSEND,
    OMPI_SPC_ISSEND,
    OMPI_SPC_IRECV,
    OMPI_SPC_SENDRECV,
    OMPI_SPC_SENDRECV_REPLACE,
    OMPI_SPC_PUT,
    OMPI_SPC_RPUT,
    OMPI_SPC_GET,
    OMPI_SPC_RGET,
    OMPI_SPC_PROBE,
    OMPI_SPC_IPROBE,
    OMPI_SPC_BCAST,
    OMPI_SPC_IBCAST,
    OMPI_SPC_BCAST_INIT,
    OMPI_SPC_REDUCE,
    OMPI_SPC_REDUCE_SCATTER,
    OMPI_SPC_REDUCE_SCATTER_BLOCK,
    OMPI_SPC_IREDUCE,
    OMPI_SPC_IREDUCE_SCATTER,
    OMPI_SPC_IREDUCE_SCATTER_BLOCK,
    OMPI_SPC_REDUCE_INIT,
    OMPI_SPC_REDUCE_SCATTER_INIT,
    OMPI_SPC_REDUCE_SCATTER_BLOCK_INIT,
    OMPI_SPC_ALLREDUCE,
    OMPI_SPC_IALLREDUCE,
    OMPI_SPC_ALLREDUCE_INIT,
    OMPI_SPC_SCAN,
    OMPI_SPC_EXSCAN,
    OMPI_SPC_ISCAN,
    OMPI_SPC_IEXSCAN,
    OMPI_SPC_SCAN_INIT,
    OMPI_SPC_EXSCAN_INIT,
    OMPI_SPC_SCATTER,
    OMPI_SPC_SCATTERV,
    OMPI_SPC_ISCATTER,
    OMPI_SPC_ISCATTERV,
    OMPI_SPC_SCATTER_INIT,
    OMPI_SPC_SCATTERV_INIT,
    OMPI_SPC_GATHER,
    OMPI_SPC_GATHERV,
    OMPI_SPC_IGATHER,
    OMPI_SPC_IGATHERV,
    OMPI_SPC_GATHER_INIT,
    OMPI_SPC_GATHERV_INIT,
    OMPI_SPC_ALLTOALL,
    OMPI_SPC_ALLTOALLV,
    OMPI_SPC_ALLTOALLW,
    OMPI_SPC_IALLTOALL,
    OMPI_SPC_IALLTOALLV,
    OMPI_SPC_IALLTOALLW,
    OMPI_SPC_ALLTOALL_INIT,
    OMPI_SPC_ALLTOALLV_INIT,
    OMPI_SPC_ALLTOALLW_INIT,
    OMPI_SPC_NEIGHBOR_ALLTOALL,
    OMPI_SPC_NEIGHBOR_ALLTOALLV,
    OMPI_SPC_NEIGHBOR_ALLTOALLW,
    OMPI_SPC_INEIGHBOR_ALLTOALL,
    OMPI_SPC_INEIGHBOR_ALLTOALLV,
    OMPI_SPC_INEIGHBOR_ALLTOALLW,
    OMPI_SPC_NEIGHBOR_ALLTOALL_INIT,
    OMPI_SPC_NEIGHBOR_ALLTOALLV_INIT,
    OMPI_SPC_NEIGHBOR_ALLTOALLW_INIT,
    OMPI_SPC_ALLGATHER,
    OMPI_SPC_ALLGATHERV,
    OMPI_SPC_IALLGATHER,
    OMPI_SPC_IALLGATHERV,
    OMPI_SPC_ALLGATHER_INIT,
    OMPI_SPC_ALLGATHERV_INIT,
    OMPI_SPC_NEIGHBOR_ALLGATHER,
    OMPI_SPC_NEIGHBOR_ALLGATHERV,
    OMPI_SPC_INEIGHBOR_ALLGATHER,
    OMPI_SPC_INEIGHBOR_ALLGATHERV,
    OMPI_SPC_NEIGHBOR_ALLGATHER_INIT,
    OMPI_SPC_NEIGHBOR_ALLGATHERV_INIT,
    OMPI_SPC_TEST,
    OMPI_SPC_TESTALL,
    OMPI_SPC_TESTANY,
    OMPI_SPC_TESTSOME,
    OMPI_SPC_WAIT,
    OMPI_SPC_WAITALL,
    OMPI_SPC_WAITANY,
    OMPI_SPC_WAITSOME,
    OMPI_SPC_BARRIER,
    OMPI_SPC_IBARRIER,
    OMPI_SPC_BARRIER_INIT,
    OMPI_SPC_WTIME,
    OMPI_SPC_CANCEL,
    OMPI_SPC_BYTES_RECEIVED_USER,
    OMPI_SPC_BYTES_RECEIVED_MPI,
    OMPI_SPC_BYTES_SENT_USER,
    OMPI_SPC_BYTES_SENT_MPI,
    OMPI_SPC_BYTES_PUT,
    OMPI_SPC_BYTES_GET,
    OMPI_SPC_UNEXPECTED,
    OMPI_SPC_OUT_OF_SEQUENCE,
    OMPI_SPC_OOS_QUEUE_HOPS,
    OMPI_SPC_MATCH_TIME,
    OMPI_SPC_MATCH_QUEUE_TIME,
    OMPI_SPC_OOS_MATCH_TIME,
    OMPI_SPC_OOS_MATCH_QUEUE_TIME,
    OMPI_SPC_UNEXPECTED_IN_QUEUE,
    OMPI_SPC_OOS_IN_QUEUE,
    OMPI_SPC_MAX_UNEXPECTED_IN_QUEUE,
    OMPI_SPC_MAX_OOS_IN_QUEUE,
    OMPI_SPC_BASE_BCAST_LINEAR,
    OMPI_SPC_BASE_BCAST_CHAIN,
    OMPI_SPC_BASE_BCAST_PIPELINE,
    OMPI_SPC_BASE_BCAST_SPLIT_BINTREE,
    OMPI_SPC_BASE_BCAST_BINTREE,
    OMPI_SPC_BASE_BCAST_BINOMIAL,
    OMPI_SPC_BASE_REDUCE_CHAIN,
    OMPI_SPC_BASE_REDUCE_PIPELINE,
    OMPI_SPC_BASE_REDUCE_BINARY,
    OMPI_SPC_BASE_REDUCE_BINOMIAL,
    OMPI_SPC_BASE_REDUCE_IN_ORDER_BINTREE,
    OMPI_SPC_BASE_REDUCE_LINEAR,
    OMPI_SPC_BASE_REDUCE_SCATTER_NONOVERLAPPING,
    OMPI_SPC_BASE_REDUCE_SCATTER_RECURSIVE_HALVING,
    OMPI_SPC_BASE_REDUCE_SCATTER_RING,
    OMPI_SPC_BASE_ALLREDUCE_NONOVERLAPPING,
    OMPI_SPC_BASE_ALLREDUCE_RECURSIVE_DOUBLING,
    OMPI_SPC_BASE_ALLREDUCE_RING,
    OMPI_SPC_BASE_ALLREDUCE_RING_SEGMENTED,
    OMPI_SPC_BASE_ALLREDUCE_LINEAR,
    OMPI_SPC_BASE_SCATTER_BINOMIAL,
    OMPI_SPC_BASE_SCATTER_LINEAR,
    OMPI_SPC_BASE_GATHER_BINOMIAL,
    OMPI_SPC_BASE_GATHER_LINEAR_SYNC,
    OMPI_SPC_BASE_GATHER_LINEAR,
    OMPI_SPC_BASE_ALLTOALL_INPLACE,
    OMPI_SPC_BASE_ALLTOALL_PAIRWISE,
    OMPI_SPC_BASE_ALLTOALL_BRUCK,
    OMPI_SPC_BASE_ALLTOALL_LINEAR_SYNC,
    OMPI_SPC_BASE_ALLTOALL_TWO_PROCS,
    OMPI_SPC_BASE_ALLTOALL_LINEAR,
    OMPI_SPC_BASE_ALLGATHER_BRUCK,
    OMPI_SPC_BASE_ALLGATHER_RECURSIVE_DOUBLING,
    OMPI_SPC_BASE_ALLGATHER_RING,
    OMPI_SPC_BASE_ALLGATHER_NEIGHBOR_EXCHANGE,
    OMPI_SPC_BASE_ALLGATHER_TWO_PROCS,
    OMPI_SPC_BASE_ALLGATHER_LINEAR,
    OMPI_SPC_BASE_BARRIER_DOUBLE_RING,
    OMPI_SPC_BASE_BARRIER_RECURSIVE_DOUBLING,
    OMPI_SPC_BASE_BARRIER_BRUCK,
    OMPI_SPC_BASE_BARRIER_TWO_PROCS,
    OMPI_SPC_BASE_BARRIER_LINEAR,
    OMPI_SPC_BASE_BARRIER_TREE,
    OMPI_SPC_P2P_MESSAGE_SIZE,
    OMPI_SPC_EAGER_MESSAGES,
    OMPI_SPC_NOT_EAGER_MESSAGES,
    OMPI_SPC_QUEUE_ALLOCATION,
    OMPI_SPC_MAX_QUEUE_ALLOCATION,
    OMPI_SPC_UNEXPECTED_QUEUE_DATA,
    OMPI_SPC_MAX_UNEXPECTED_QUEUE_DATA,
    OMPI_SPC_OOS_QUEUE_DATA,
    OMPI_SPC_MAX_OOS_QUEUE_DATA,
    OMPI_SPC_NUM_COUNTERS /* This serves as the number of counters.  It must be last. */
} ompi_spc_counters_t;

extern uint32_t ompi_spc_attached_event[];

/* There is currently no support for atomics on long long values so we will default to
 * size_t for now until support for such atomics is implemented.
 */
typedef opal_atomic_size_t ompi_spc_value_t;

/* A structure for storing the event data */
typedef struct ompi_spc_s {
    char *name;
    ompi_spc_value_t value;
    int *bin_rules; /* The first element is the number of bins, the rest represent when each bin starts */
    ompi_spc_value_t *bins;
} ompi_spc_t;

/* A structure for indexing into the event data */
typedef struct ompi_spc_offset_s {
    int num_bins;
    int rules_offset;
    int bins_offset;
} ompi_spc_offset_t;

/* MCA Parameters Initialization Function */
int ompi_spc_register_params(void);

/* Events data structure initialization function */
void ompi_spc_events_init(void);

/* OMPI SPC utility functions */
void ompi_spc_init(void);
void ompi_spc_fini(void);
void ompi_spc_record(unsigned int event_id, ompi_spc_value_t value);
void ompi_spc_bin_record(unsigned int event_id, ompi_spc_value_t value);
void ompi_spc_collective_bin_record(unsigned int event_id, ompi_spc_value_t bytes, ompi_spc_value_t procs);
void ompi_spc_timer_start(unsigned int event_id, opal_timer_t *cycles);
void ompi_spc_timer_stop(unsigned int event_id, opal_timer_t *cycles);
void ompi_spc_user_or_mpi(int tag, ompi_spc_value_t value, unsigned int user_enum, unsigned int mpi_enum);
void ompi_spc_cycles_to_usecs(ompi_spc_value_t *cycles);
void ompi_spc_update_watermark(unsigned int watermark_enum, unsigned int value_enum, ompi_spc_value_t value);
ompi_spc_value_t ompi_spc_get_value(unsigned int event_id);
bool IS_SPC_BIT_SET(uint32_t* array, int32_t pos);

/* Macros for using the SPC utility functions throughout the codebase.
 * If SPC_ENABLE is not 1, the macros become no-ops.
 */
#if SPC_ENABLE == 1

#define SPC_INIT()  \
    ompi_spc_init()

#define SPC_FINI()  \
    ompi_spc_fini()

#define SPC_RECORD(event_id, value)  \
    if( OPAL_UNLIKELY(IS_SPC_BIT_SET(ompi_spc_attached_event, event_id)) ) \
        ompi_spc_record(event_id, value)

#define SPC_BIN_RECORD(event_id, value)  \
    if( OPAL_UNLIKELY(IS_SPC_BIT_SET(ompi_spc_attached_event, event_id)) ) \
        ompi_spc_bin_record(event_id, value)

#define SPC_COLL_BIN_RECORD(event_id, bytes, procs)   \
    if( OPAL_UNLIKELY(IS_SPC_BIT_SET(ompi_spc_attached_event, event_id)) ) \
        ompi_spc_collective_bin_record(event_id, bytes, procs)

#define SPC_TIMER_START(event_id, usec)  \
    if( OPAL_UNLIKELY(IS_SPC_BIT_SET(ompi_spc_attached_event, event_id)) ) \
        ompi_spc_timer_start(event_id, usec)

#define SPC_TIMER_STOP(event_id, usec)  \
    if( OPAL_UNLIKELY(IS_SPC_BIT_SET(ompi_spc_attached_event, event_id)) ) \
        ompi_spc_timer_stop(event_id, usec)

#define SPC_USER_OR_MPI(tag, value, enum_if_user, enum_if_mpi) \
    if( OPAL_UNLIKELY(IS_SPC_BIT_SET(ompi_spc_attached_event, enum_if_user) || IS_SPC_BIT_SET(ompi_spc_attached_event, enum_if_mpi)) ) \
        ompi_spc_user_or_mpi(tag, value, enum_if_user, enum_if_mpi)

#define SPC_CYCLES_TO_USECS(cycles) \
    ompi_spc_cycles_to_usecs(cycles)

/* WARNING: This macro assumes that it was called while a lock has already been taken.
 *          This function is NOT thread safe otherwise!
 */
#define SPC_UPDATE_WATERMARK(watermark_enum, value_enum, value) \
    if( OPAL_UNLIKELY(IS_SPC_BIT_SET(ompi_spc_attached_event, value_enum)) ) \
        ompi_spc_update_watermark(watermark_enum, value_enum, value)

#define SPC_GET(event_id)  \
    ompi_spc_get_value(event_id)

#else /* SPCs are not enabled */

#define SPC_INIT()  \
    ((void)0)

#define SPC_FINI()  \
    ((void)0)

#define SPC_RECORD(event_id, value)  \
    ((void)0)

#define SPC_BIN_RECORD(event_id, value)  \
    ((void)0)

#define SPC_COLL_BIN_RECORD(event_id, bytes, procs)        \
    ((void)0)

#define SPC_TIMER_START(event_id, usec)  \
    ((void)0)

#define SPC_TIMER_STOP(event_id, usec)  \
    ((void)0)

#define SPC_USER_OR_MPI(tag, value, enum_if_user, enum_if_mpi) \
    ((void)0)

#define SPC_CYCLES_TO_USECS(cycles) \
    ((void)0)

#define SPC_UPDATE_WATERMARK(watermark_enum, value_enum, value) \
    ((void)0)

#define SPC_GET(event_id)  \
    ((void)0)

#endif

#endif
