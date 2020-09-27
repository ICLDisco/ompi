/*
 * Copyright (c) 2018-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * Copyright (c) 2018      Cisco Systems, Inc.  All rights reserved
 * Copyright (c) 2018      Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2019      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_spc.h"

static opal_timer_t sys_clock_freq_mhz = 0;
static int mpi_t_offset = -1;
static bool mpi_t_enabled = false;
static bool spc_enabled = true;
static bool need_free = false;
static bool mmap_failed = false;
static ompi_communicator_t *ompi_spc_comm = NULL;

/* MCA Parameter Variables */
char *ompi_mpi_spc_attach_string = NULL;
char *ompi_mpi_spc_xml_string = NULL;
bool ompi_mpi_spc_dump_enabled = false;
bool ompi_mpi_spc_mmap_enabled = false;
int ompi_mpi_spc_snapshot_period = 0;
int ompi_mpi_spc_p2p_message_boundary = 12288;
int ompi_mpi_spc_collective_message_boundary = 12288;
int ompi_mpi_spc_collective_comm_boundary = 64;

typedef struct ompi_spc_event_t {
    const char* counter_name;
    const char* counter_description;
} ompi_spc_event_t;

static void ompi_spc_dump(void);

#define SET_COUNTER_ARRAY(NAME, DESC)   [NAME] = { .counter_name = #NAME, .counter_description = DESC }

/* STEP 2: Add the counter descriptions for new counters here along with their
 *         enumeration to be converted to a name value.  NOTE: The names and
 *         descriptions MUST be in the same array location as where you added
 *         the counter name in the ompi_spc_counters_t enumeration!
 */
static ompi_spc_event_t ompi_spc_events_names[OMPI_SPC_NUM_COUNTERS] = {
    SET_COUNTER_ARRAY(OMPI_SPC_SEND, "The number of times MPI_Send was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_BSEND, "The number of times MPI_Bsend was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_RSEND, "The number of times MPI_Rsend was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_SSEND, "The number of times MPI_Ssend was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_RECV, "The number of times MPI_Recv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_MRECV, "The number of times MPI_Mrecv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ISEND, "The number of times MPI_Isend was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IBSEND, "The number of times MPI_Ibsend was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IRSEND, "The number of times MPI_Irsend was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ISSEND, "The number of times MPI_Issend was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IRECV, "The number of times MPI_Irecv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_SENDRECV, "The number of times MPI_Sendrecv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_SENDRECV_REPLACE, "The number of times MPI_Sendrecv_replace was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_PUT, "The number of times MPI_Put was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_RPUT, "The number of times MPI_Rput was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_GET, "The number of times MPI_Get was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_RGET, "The number of times MPI_Rget was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_PROBE, "The number of times MPI_Probe was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IPROBE, "The number of times MPI_Iprobe was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_BCAST, "The number of times MPI_Bcast was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IBCAST, "The number of times MPI_Ibcast was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_BCAST_INIT, "The number of times MPIX_Bcast_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_REDUCE, "The number of times MPI_Reduce was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_REDUCE_SCATTER, "The number of times MPI_Reduce_scatter was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_REDUCE_SCATTER_BLOCK, "The number of times MPI_Reduce_scatter_block was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IREDUCE, "The number of times MPI_Ireduce was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IREDUCE_SCATTER, "The number of times MPI_Ireduce_scatter was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IREDUCE_SCATTER_BLOCK, "The number of times MPI_Ireduce_scatter_block was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_REDUCE_INIT, "The number of times MPIX_Reduce_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_REDUCE_SCATTER_INIT, "The number of times MPIX_Reduce_scatter_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_REDUCE_SCATTER_BLOCK_INIT, "The number of times MPIX_Reduce_scatter_block_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ALLREDUCE, "The number of times MPI_Allreduce was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IALLREDUCE, "The number of times MPI_Iallreduce was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ALLREDUCE_INIT, "The number of times MPIX_Allreduce_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_SCAN, "The number of times MPI_Scan was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_EXSCAN, "The number of times MPI_Exscan was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ISCAN, "The number of times MPI_Iscan was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IEXSCAN, "The number of times MPI_Iexscan was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_SCAN_INIT, "The number of times MPIX_Scan_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_EXSCAN_INIT, "The number of times MPIX_Exscan_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_SCATTER, "The number of times MPI_Scatter was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_SCATTERV, "The number of times MPI_Scatterv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ISCATTER, "The number of times MPI_Iscatter was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ISCATTERV, "The number of times MPI_Iscatterv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_SCATTER_INIT, "The number of times MPIX_Scatter_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_SCATTERV_INIT, "The number of times MPIX_Scatterv_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_GATHER, "The number of times MPI_Gather was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_GATHERV, "The number of times MPI_Gatherv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IGATHER, "The number of times MPI_Igather was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IGATHERV, "The number of times MPI_Igatherv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_GATHER_INIT, "The number of times MPIX_Gather_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_GATHERV_INIT, "The number of times MPIX_Gatherv_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ALLTOALL, "The number of times MPI_Alltoall was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ALLTOALLV, "The number of times MPI_Alltoallv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ALLTOALLW, "The number of times MPI_Alltoallw was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IALLTOALL, "The number of times MPI_Ialltoall was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IALLTOALLV, "The number of times MPI_Ialltoallv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IALLTOALLW, "The number of times MPI_Ialltoallw was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ALLTOALL_INIT, "The number of times MPIX_Alltoall_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ALLTOALLV_INIT, "The number of times MPIX_Alltoallv_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ALLTOALLW_INIT, "The number of times MPIX_Alltoallw_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_NEIGHBOR_ALLTOALL, "The number of times MPI_Neighbor_alltoall was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_NEIGHBOR_ALLTOALLV, "The number of times MPI_Neighbor_alltoallv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_NEIGHBOR_ALLTOALLW, "The number of times MPI_Neighbor_alltoallw was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_INEIGHBOR_ALLTOALL, "The number of times MPI_Ineighbor_alltoall was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_INEIGHBOR_ALLTOALLV, "The number of times MPI_Ineighbor_alltoallv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_INEIGHBOR_ALLTOALLW, "The number of times MPI_Ineighbor_alltoallw was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_NEIGHBOR_ALLTOALL_INIT, "The number of times MPIX_Neighbor_alltoall_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_NEIGHBOR_ALLTOALLV_INIT, "The number of times MPIX_Neighbor_alltoallv_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_NEIGHBOR_ALLTOALLW_INIT, "The number of times MPIX_Neighbor_alltoallw_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ALLGATHER, "The number of times MPI_Allgather was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ALLGATHERV, "The number of times MPI_Allgatherv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IALLGATHER, "The number of times MPI_Iallgather was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IALLGATHERV, "The number of times MPI_Iallgatherv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ALLGATHER_INIT, "The number of times MPIX_Allgather_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_ALLGATHERV_INIT, "The number of times MPIX_Allgatherv_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_NEIGHBOR_ALLGATHER, "The number of times MPI_Neighbor_allgather was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_NEIGHBOR_ALLGATHERV, "The number of times MPI_Neighbor_allgatherv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_INEIGHBOR_ALLGATHER, "The number of times MPI_Ineighbor_allgather was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_INEIGHBOR_ALLGATHERV, "The number of times MPI_Ineighbor_allgatherv was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_NEIGHBOR_ALLGATHER_INIT, "The number of times MPIX_Neighbor_allgather_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_NEIGHBOR_ALLGATHERV_INIT, "The number of times MPIX_Neighbor_allgatherv_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_TEST, "The number of times MPI_Test was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_TESTALL, "The number of times MPI_Testall was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_TESTANY, "The number of times MPI_Testany was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_TESTSOME, "The number of times MPI_Testsome was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_WAIT, "The number of times MPI_Wait was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_WAITALL, "The number of times MPI_Waitall was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_WAITANY, "The number of times MPI_Waitany was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_WAITSOME, "The number of times MPI_Waitsome was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_BARRIER, "The number of times MPI_Barrier was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_IBARRIER, "The number of times MPI_Ibarrier was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_BARRIER_INIT, "The number of times MPIX_Barrier_init was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_WTIME, "The number of times MPI_Wtime was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_CANCEL, "The number of times MPI_Cancel was called."),
    SET_COUNTER_ARRAY(OMPI_SPC_BYTES_RECEIVED_USER, "The number of bytes received by the user through point-to-point communications. Note: Includes bytes transferred using internal RMA operations."),
    SET_COUNTER_ARRAY(OMPI_SPC_BYTES_RECEIVED_MPI, "The number of bytes received by MPI through collective, control, or other internal communications."),
    SET_COUNTER_ARRAY(OMPI_SPC_BYTES_SENT_USER, "The number of bytes sent by the user through point-to-point communications.  Note: Includes bytes transferred using internal RMA operations."),
    SET_COUNTER_ARRAY(OMPI_SPC_BYTES_SENT_MPI, "The number of bytes sent by MPI through collective, control, or other internal communications."),
    SET_COUNTER_ARRAY(OMPI_SPC_BYTES_PUT, "The number of bytes sent/received using RMA Put operations both through user-level Put functions and internal Put functions."),
    SET_COUNTER_ARRAY(OMPI_SPC_BYTES_GET, "The number of bytes sent/received using RMA Get operations both through user-level Get functions and internal Get functions."),
    SET_COUNTER_ARRAY(OMPI_SPC_UNEXPECTED, "The number of messages that arrived as unexpected messages."),
    SET_COUNTER_ARRAY(OMPI_SPC_OUT_OF_SEQUENCE, "The number of messages that arrived out of the proper sequence."),
    SET_COUNTER_ARRAY(OMPI_SPC_OOS_QUEUE_HOPS, "The number of times we jumped to the next element in the out of sequence message queue's ordered list."),
    SET_COUNTER_ARRAY(OMPI_SPC_MATCH_TIME, "The amount of time (MPI_T reports microseconds) spent matching unexpected messages.  Note: The timer used on the back end is in cycles, which could potentially be problematic on a system where the clock frequency can change.  On such a system, this counter could be inaccurate since we assume a fixed clock rate."),
    SET_COUNTER_ARRAY(OMPI_SPC_MATCH_QUEUE_TIME, "The amount of time (MPI_T reports microseconds, stored in cycles) spent inserting unexpected messages into the unexpected message queue.  Note: The timer used on the back end is in cycles, which could potentially be problematic on a system where the clock frequency can change.  On such a system, this counter could be inaccurate since we assume a fixed clock rate."),
    SET_COUNTER_ARRAY(OMPI_SPC_OOS_MATCH_TIME, "The amount of time (MPI_T reports microseconds, stored in cycles) spent matching out-of-sequence messages.  Note: The timer used on the back end is in cycles, which could potentially be problematic on a system where the clock frequency can change.  On such a system, this counter could be inaccurate since we assume a fixed clock rate."),
    SET_COUNTER_ARRAY(OMPI_SPC_OOS_MATCH_QUEUE_TIME, "The amount of time (MPI_T reports microseconds, stored in cycles) spent inserting out-of-sequence messages into the unexpected message queue.  Note: The timer used on the back end is in cycles, which could potentially be problematic on a system where the clock frequency can change.  On such a system, this counter could be inaccurate since we assume a fixed clock rate."),
    SET_COUNTER_ARRAY(OMPI_SPC_UNEXPECTED_IN_QUEUE, "The number of messages that are currently in the unexpected message queue(s) of an MPI process."),
    SET_COUNTER_ARRAY(OMPI_SPC_OOS_IN_QUEUE, "The number of messages that are currently in the out of sequence message queue(s) of an MPI process."),
    SET_COUNTER_ARRAY(OMPI_SPC_MAX_UNEXPECTED_IN_QUEUE, "The maximum number of messages that the unexpected message queue(s) within an MPI process contained at once since the last reset of this counter. Note: This counter is reset each time it is read.  Note: The OMPI_SPC_UNEXPECTED_IN_QUEUE counter must also be activated."),
    SET_COUNTER_ARRAY(OMPI_SPC_MAX_OOS_IN_QUEUE, "The maximum number of messages that the out of sequence message queue(s) within an MPI process contained at once since the last reset of this counter. Note: This counter is reset each time it is read.  Note: The OMPI_SPC_OOS_IN_QUEUE counter must also be activated."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_BCAST_LINEAR, "The number of times the base broadcast used the linear algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_BCAST_CHAIN, "The number of times the base broadcast used the chain algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_BCAST_PIPELINE, "The number of times the base broadcast used the pipeline algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_BCAST_SPLIT_BINTREE, "The number of times the base broadcast used the split binary tree algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_BCAST_BINTREE, "The number of times the base broadcast used the binary tree algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_BCAST_BINOMIAL, "The number of times the base broadcast used the binomial algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_REDUCE_CHAIN, "The number of times the base reduce used the chain algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_REDUCE_PIPELINE, "The number of times the base reduce used the pipeline algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_REDUCE_BINARY, "The number of times the base reduce used the binary tree algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_REDUCE_BINOMIAL, "The number of times the base reduce used the binomial tree algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_REDUCE_IN_ORDER_BINTREE, "The number of times the base reduce used the in order binary tree algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_REDUCE_LINEAR, "The number of times the base reduce used the basic linear algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_REDUCE_SCATTER_NONOVERLAPPING, "The number of times the base reduce scatter used the nonoverlapping algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_REDUCE_SCATTER_RECURSIVE_HALVING, "The number of times the base reduce scatter used the recursive halving algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_REDUCE_SCATTER_RING, "The number of times the base reduce scatter used the ring algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLREDUCE_NONOVERLAPPING, "The number of times the base allreduce used the nonoverlapping algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLREDUCE_RECURSIVE_DOUBLING, "The number of times the base allreduce used the recursive doubling algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLREDUCE_RING, "The number of times the base allreduce used the ring algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLREDUCE_RING_SEGMENTED, "The number of times the base allreduce used the segmented ring algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLREDUCE_LINEAR, "The number of times the base allreduce used the linear algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_SCATTER_BINOMIAL, "The number of times the base scatter used the binomial tree algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_SCATTER_LINEAR, "The number of times the base scatter used the linear algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_GATHER_BINOMIAL, "The number of times the base gather used the binomial tree algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_GATHER_LINEAR_SYNC, "The number of times the base gather used the synchronous linear algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_GATHER_LINEAR, "The number of times the base gather used the linear algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLTOALL_INPLACE, "The number of times the base alltoall used the in-place algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLTOALL_PAIRWISE, "The number of times the base alltoall used the pairwise algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLTOALL_BRUCK, "The number of times the base alltoall used the bruck algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLTOALL_LINEAR_SYNC, "The number of times the base alltoall used the synchronous linear algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLTOALL_TWO_PROCS, "The number of times the base alltoall used the two process algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLTOALL_LINEAR, "The number of times the base alltoall used the linear algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLGATHER_BRUCK, "The number of times the base allgather used the bruck algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLGATHER_RECURSIVE_DOUBLING, "The number of times the base allgather used the recursive doubling algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLGATHER_RING, "The number of times the base allgather used the ring algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLGATHER_NEIGHBOR_EXCHANGE, "The number of times the base allgather used the neighbor exchange algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLGATHER_TWO_PROCS, "The number of times the base allgather used the two process algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_ALLGATHER_LINEAR, "The number of times the base allgather used the linear algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_BARRIER_DOUBLE_RING, "The number of times the base barrier used the double ring algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_BARRIER_RECURSIVE_DOUBLING, "The number of times the base barrier used the recursive doubling algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_BARRIER_BRUCK, "The number of times the base barrier used the bruck algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_BARRIER_TWO_PROCS, "The number of times the base barrier used the two process algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_BARRIER_LINEAR, "The number of times the base barrier used the linear algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_BASE_BARRIER_TREE, "The number of times the base barrier used the tree algorithm."),
    SET_COUNTER_ARRAY(OMPI_SPC_P2P_MESSAGE_SIZE, "This is a bin counter with two subcounters.  The first is messages that are less than or equal to mpi_spc_p2p_message_boundary bytes and the second is those that are larger than mpi_spc_p2p_message_boundary bytes."),
    SET_COUNTER_ARRAY(OMPI_SPC_EAGER_MESSAGES, "The number of messages that fall within the eager size."),
    SET_COUNTER_ARRAY(OMPI_SPC_NOT_EAGER_MESSAGES, "The number of messages that do not fall within the eager size."),
    SET_COUNTER_ARRAY(OMPI_SPC_QUEUE_ALLOCATION, "The amount of memory allocated after runtime currently in use for temporary message queues like the unexpected message queue and the out of sequence message queue."),
    SET_COUNTER_ARRAY(OMPI_SPC_MAX_QUEUE_ALLOCATION, "The maximum amount of memory allocated after runtime at one point for temporary message queues like the unexpected message queue and the out of sequence message queue.  Note: The OMPI_SPC_QUEUE_ALLOCATION counter must also be activated."),
    SET_COUNTER_ARRAY(OMPI_SPC_UNEXPECTED_QUEUE_DATA, "The amount of memory currently in use for the unexpected message queue."),
    SET_COUNTER_ARRAY(OMPI_SPC_MAX_UNEXPECTED_QUEUE_DATA, "The maximum amount of memory in use for the unexpected message queue.  Note: The OMPI_SPC_UNEXPECTED_QUEUE_DATA counter must also be activated."),
    SET_COUNTER_ARRAY(OMPI_SPC_OOS_QUEUE_DATA, "The amount of memory currently in use for the out-of-sequence message queue."),
    SET_COUNTER_ARRAY(OMPI_SPC_MAX_OOS_QUEUE_DATA, "The maximum amount of memory in use for the out-of-sequence message queue.  Note: The OMPI_SPC_OOS_QUEUE_DATA counter must also be activated.")
};

/* A bitmap to denote whether an event is activated (1) or not (0) 
 * This is not static beacuse it is needed in the recording macros
 * for instrumentation.
 */
OMPI_DECLSPEC uint32_t ompi_spc_attached_event[OMPI_SPC_NUM_COUNTERS / sizeof(uint32_t)] = { 0 };
/* A bitmap to denote whether an event is timer-based (1) or not (0) */
static uint32_t ompi_spc_timer_event[OMPI_SPC_NUM_COUNTERS / sizeof(uint32_t)] = { 0 };
/* A bitmap to denote whether an event is bin-based (1) or not (0) */
static uint32_t ompi_spc_bin_event[OMPI_SPC_NUM_COUNTERS / sizeof(uint32_t)] = { 0 };
/* A bitmap to denote whether an event is collective bin-based (1) or not (0) */
static uint32_t ompi_spc_collective_bin_event[OMPI_SPC_NUM_COUNTERS / sizeof(uint32_t)] = { 0 };

/* A contiguous data structure for storing the counter values, rules, and bins */
void *ompi_spc_events = NULL;
/* An array of offset structures for indexing into the ompi_spc_events data */
static ompi_spc_offset_t ompi_spc_offsets[OMPI_SPC_NUM_COUNTERS] = {-1};
/* A pointer into the ompi_spc_events data for the SPC values */
static ompi_spc_value_t *ompi_spc_values = NULL;

/* ##############################################################
 * ################## SPC Bitmap Functions ######################
 * ##############################################################
 */
static inline void SET_SPC_BIT(uint32_t* array, int32_t pos)
{
    assert(pos < OMPI_SPC_NUM_COUNTERS);
    array[pos / (8 * sizeof(uint32_t))] |= (1U << (pos % (8 * sizeof(uint32_t))));
}

inline bool IS_SPC_BIT_SET(uint32_t* array, int32_t pos)
{
    assert(pos < OMPI_SPC_NUM_COUNTERS);
    return !!(array[pos / (8 * sizeof(uint32_t))] & (1U << (pos % (8 * sizeof(uint32_t)))));
}

static inline void CLEAR_SPC_BIT(uint32_t* array, int32_t pos)
{
    assert(pos < OMPI_SPC_NUM_COUNTERS);
    array[pos / (8 * sizeof(uint32_t))] &= ~(1U << (pos % (8 * sizeof(uint32_t))));
}

/* Registers all of the SPC MCA parameter variables.  If SPCs are enabled, this is called from
 * ompi_mpi_params.c in its registration function.
 */
int ompi_spc_register_params()
{   
    ompi_mpi_spc_attach_string = NULL;
    (void) mca_base_var_register("ompi", "mpi", NULL, "spc_attach",
                                 "A comma delimeted string listing the software-based performance counters (SPCs) to enable.",
                                 MCA_BASE_VAR_TYPE_STRING, NULL, 0, 0,
                                 OPAL_INFO_LVL_4,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &ompi_mpi_spc_attach_string);

    ompi_mpi_spc_xml_string = NULL;
    (void) mca_base_var_register("ompi", "mpi", NULL, "spc_xml_string",
                                 "A string to add to SPC XML files for easier identification.  The format will be: spc_data.[nodename].[jobid or spc_xml_string].[world_rank].xml",
                                 MCA_BASE_VAR_TYPE_STRING, NULL, 0, 0,
                                 OPAL_INFO_LVL_4,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &ompi_mpi_spc_xml_string);

    ompi_mpi_spc_dump_enabled = false;
    (void) mca_base_var_register("ompi", "mpi", NULL, "spc_dump_enabled",
                                 "A boolean value for whether (true) or not (false) to enable dumping SPC counters in MPI_Finalize.",
                                 MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                 OPAL_INFO_LVL_4,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &ompi_mpi_spc_dump_enabled);

    ompi_mpi_spc_mmap_enabled = false;
    (void) mca_base_var_register("ompi", "mpi", NULL, "spc_mmap_enabled",
                                 "A boolean value for whether (true) or not (false) to enable dumping SPC counters to an mmap'd file.",
                                 MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                 OPAL_INFO_LVL_4,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &ompi_mpi_spc_mmap_enabled);

    ompi_mpi_spc_p2p_message_boundary = 12288;
    (void) mca_base_var_register("ompi", "mpi", NULL, "spc_p2p_message_boundary",
                                 "An integer value for determining the boundary for whether a message is small/large for point to point message size bin counter (<= this value is small).",
                                 MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                 OPAL_INFO_LVL_4,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &ompi_mpi_spc_p2p_message_boundary);

    ompi_mpi_spc_collective_message_boundary = 12288;
    (void) mca_base_var_register("ompi", "mpi", NULL, "spc_collective_message_boundary",
                                 "An integer value for determining the boundary for whether a message is small/large for collective bin counters (<= this value is small).",
                                 MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                 OPAL_INFO_LVL_4,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &ompi_mpi_spc_collective_message_boundary);

    ompi_mpi_spc_collective_comm_boundary = 64;
    (void) mca_base_var_register("ompi", "mpi", NULL, "spc_collective_comm_boundary",
                                 "An integer value for determining the boundary for whether a communicator is small/large for collective bin counters (<= this value is small).",
                                 MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                 OPAL_INFO_LVL_4,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &ompi_mpi_spc_collective_comm_boundary);

    return OMPI_SUCCESS;
}

/* ##############################################################
 * ################# Begin MPI_T Functions ######################
 * ##############################################################
 */
static int ompi_spc_notify(mca_base_pvar_t *pvar, mca_base_pvar_event_t event, void *obj_handle, int *count)
    __opal_attribute_unused__;

static int ompi_spc_notify(mca_base_pvar_t *pvar, mca_base_pvar_event_t event, void *obj_handle, int *count)
{
    int index;

    if(OPAL_LIKELY(!mpi_t_enabled)) {
        return MPI_SUCCESS;
    }

    index = (int)(uintptr_t)pvar->ctx;  /* Convert from MPI_T pvar index to SPC index */

    /* For this event, we need to set count to the number of long long type
     * values for this counter.  Most SPC counters are one long long so the
     * default is 1, however bin counters and the xml string can be longer.
     */
    do {
        if(MCA_BASE_PVAR_HANDLE_BIND == event) {
            /* Convert from MPI_T pvar index to SPC index */
            index = pvar->pvar_index - mpi_t_offset;
            if(index < 0) {
                char *shm_dir;
                if(0 == access(SPC_SHM_DIR, W_OK)) {
                    shm_dir = SPC_SHM_DIR;
                } else {
                    opal_show_help("help-mpi-runtime.txt", "spc: default shared memory directory failed", true, opal_process_info.job_session_dir);
                    shm_dir = opal_process_info.job_session_dir;
                }

                int rank = ompi_comm_rank(ompi_spc_comm), rc;
                char filename[SPC_MAX_FILENAME];

                if(ompi_mpi_spc_xml_string == NULL) {
                    rc = snprintf(filename, SPC_MAX_FILENAME, "%s" OPAL_PATH_SEP "spc_data.%s.%d.%d.xml", shm_dir,
                                 opal_process_info.nodename, OPAL_PROC_MY_NAME.jobid, rank);
                } else {
                    rc = snprintf(filename, SPC_MAX_FILENAME, "%s" OPAL_PATH_SEP "spc_data.%s.%s.%d.xml", shm_dir,
                                 opal_process_info.nodename, ompi_mpi_spc_xml_string, rank);
                }

                *count = strlen(filename);
                break;
            }
            if( IS_SPC_BIT_SET(ompi_spc_bin_event, index) || IS_SPC_BIT_SET(ompi_spc_collective_bin_event, index) ) {
                *count = *(int*)(ompi_spc_events+ompi_spc_offsets[index].rules_offset);
            } else {
                *count = 1;
            }
        }
        /* For this event, we need to turn on the counter */
        else if(MCA_BASE_PVAR_HANDLE_START == event) {
            /* Convert from MPI_T pvar index to SPC index */
            index = pvar->pvar_index - mpi_t_offset;
            if(index > 0) {
                SET_SPC_BIT(ompi_spc_attached_event, index);
            }
        }
        /* For this event, we need to turn off the counter */
        else if(MCA_BASE_PVAR_HANDLE_STOP == event) {
            /* Convert from MPI_T pvar index to SPC index */
            index = pvar->pvar_index - mpi_t_offset;
            if(index > 0) {
                CLEAR_SPC_BIT(ompi_spc_attached_event, index);
            }
        }
    } while(0);

    return MPI_SUCCESS;
}

static int ompi_spc_get_xml_filename(const struct mca_base_pvar_t *pvar, void *value, void *obj_handle)
    __opal_attribute_unused__;

static int ompi_spc_get_xml_filename(const struct mca_base_pvar_t *pvar, void *value, void *obj_handle)
{
    int rc;
    char **filename, *shm_dir;

    if(OPAL_LIKELY(!mpi_t_enabled)) {
        filename = (char**)value;
        rc = sprintf(*filename, "");

        return MPI_SUCCESS;
    }

    if(0 == access(SPC_SHM_DIR, W_OK)) {
        shm_dir = SPC_SHM_DIR;
    } else {
        opal_show_help("help-mpi-runtime.txt", "spc: default shared memory directory failed", true, opal_process_info.job_session_dir);
        shm_dir = opal_process_info.job_session_dir;
    }

    int rank = ompi_comm_rank(ompi_spc_comm);

    filename = (char**)value;
    if(ompi_mpi_spc_xml_string == NULL) {
        rc = sprintf(*filename, "%s" OPAL_PATH_SEP "spc_data.%s.%d.%d.xml", shm_dir,
                     opal_process_info.nodename, OPAL_PROC_MY_NAME.jobid, rank);
    } else {
        rc = sprintf(*filename, "%s" OPAL_PATH_SEP "spc_data.%s.%s.%d.xml", shm_dir,
                     opal_process_info.nodename, ompi_mpi_spc_xml_string, rank);
    }

    return MPI_SUCCESS;
}

/* This function returns the current count of an SPC counter that has been retistered
 * as an MPI_T pvar.  The MPI_T index is not necessarily the same as the SPC index,
 * so we need to convert from MPI_T index to SPC index and then set the 'value' argument
 * to the correct value for this pvar.  Watermark counters are also reset here.
 */
static int ompi_spc_get_count(const struct mca_base_pvar_t *pvar, void *value, void *obj_handle)
    __opal_attribute_unused__;

static int ompi_spc_get_count(const struct mca_base_pvar_t *pvar, void *value, void *obj_handle)
{
    if(OPAL_LIKELY(!mpi_t_enabled)) {
        long long *counter_value = (long long*)value;
        *counter_value = 0;
        return MPI_SUCCESS;
    }

    /* Convert from MPI_T pvar index to SPC index */
    int index = pvar->pvar_index - mpi_t_offset;

    /* If this is a bin-based counter, set 'value' to the array of bin values */
    if( IS_SPC_BIT_SET(ompi_spc_bin_event, index) || IS_SPC_BIT_SET(ompi_spc_collective_bin_event, index) ) {
        long long *bin_value = (long long*)value;
        int count = ((int*)(ompi_spc_events+ompi_spc_offsets[index].rules_offset))[0];
        for(int i = 0; i < count; i++) {
            bin_value[i] = ((long long*)(ompi_spc_events+ompi_spc_offsets[index].bins_offset))[i];
        }
        return MPI_SUCCESS;
    }

    long long *counter_value = (long long*)value;
    /* Set the counter value to the current SPC value */
    *counter_value = ompi_spc_values[index];

    /* If this is a timer-based counter, convert from cycles to microseconds */
    if( IS_SPC_BIT_SET(ompi_spc_timer_event, index) ) {
        *counter_value /= sys_clock_freq_mhz;
    }
    /* STEP 4: If this is a high watermark counter, reset it after it has been read.
     *         Be sure to reset the counter to the current value of the counter it
     *         is tracking.
     */
    if(index == OMPI_SPC_MAX_UNEXPECTED_IN_QUEUE) {
        ompi_spc_values[index] = ompi_spc_values[OMPI_SPC_UNEXPECTED_IN_QUEUE];
    }
    if(index == OMPI_SPC_MAX_OOS_IN_QUEUE) {
        ompi_spc_values[index] = ompi_spc_values[OMPI_SPC_OOS_IN_QUEUE];
    }
    if(index == OMPI_SPC_MAX_QUEUE_ALLOCATION) {
        ompi_spc_values[index] = ompi_spc_values[OMPI_SPC_QUEUE_ALLOCATION];
    }
    if(index == OMPI_SPC_MAX_UNEXPECTED_QUEUE_DATA) {
        ompi_spc_values[index] = ompi_spc_values[OMPI_SPC_UNEXPECTED_QUEUE_DATA];
    }
    if(index == OMPI_SPC_MAX_OOS_QUEUE_DATA) {
        ompi_spc_values[index] = ompi_spc_values[OMPI_SPC_OOS_QUEUE_DATA];
    }

    return MPI_SUCCESS;
}

/* ##############################################################
 * ################# Begin SPC Functions ########################
 * ##############################################################
 */

/* Initializes the events data structure and allocates memory for it if needed. */
void ompi_spc_events_init(void)
{
    ompi_comm_dup(&ompi_mpi_comm_world.comm, &ompi_spc_comm);

    int i, value_offset = 0, bin_offset = OMPI_SPC_NUM_COUNTERS*sizeof(ompi_spc_value_t), rank = ompi_comm_rank(ompi_spc_comm), shm_fd, rc, ret, mod;
    char filename[SPC_MAX_FILENAME], *shm_dir;
    void *ptr;

    /* Make sure the bin offset is cache aligned to avoid false sharing */
    mod = bin_offset % SPC_CACHE_LINE;
    if(mod != 0) {
        bin_offset += SPC_CACHE_LINE - mod;
    }

    FILE *fptr, *shm_fptr = NULL;
    char sm_file[SPC_MAX_FILENAME], *my_segment;
    opal_shmem_ds_t shm_ds;

    if(ompi_mpi_spc_mmap_enabled) {
        /* Determine the location for saving the shared memory file */
        if(0 == access(SPC_SHM_DIR, W_OK)) {
            shm_dir = SPC_SHM_DIR;
        } else {
            opal_show_help("help-mpi-runtime.txt", "spc: default shared memory directory failed", true, opal_process_info.job_session_dir);
            shm_dir = opal_process_info.job_session_dir;
        }

        /* Create a shared memory file */

        rc = snprintf(sm_file, SPC_MAX_FILENAME, "%s" OPAL_PATH_SEP "spc_data.%s.%d.%d", shm_dir,
                     opal_process_info.nodename, OPAL_PROC_MY_NAME.jobid, rank);

        if (0 > rc) {
            opal_show_help("help-mpi-runtime.txt", "spc: filename creation failure", true);
        }

        if(ompi_mpi_spc_xml_string == NULL) {
            rc = snprintf(filename, SPC_MAX_FILENAME, "%s" OPAL_PATH_SEP "spc_data.%s.%d.%d.xml", shm_dir,
                         opal_process_info.nodename, OPAL_PROC_MY_NAME.jobid, rank);
        } else {
            rc = snprintf(filename, SPC_MAX_FILENAME, "%s" OPAL_PATH_SEP "spc_data.%s.%s.%d.xml", shm_dir,
                         opal_process_info.nodename, ompi_mpi_spc_xml_string, rank);
        }
        if (0 > rc) {
            opal_show_help("help-mpi-runtime.txt", "spc: filename creation failure", true);
        }
        fptr = fopen(filename, "w+");

        /* Registers the name/path of the XML file as an MPI_T pvar */
        ret = mca_base_pvar_register("ompi", "runtime", "spc", "OMPI_SPC_XML_FILE", "The filename for the SPC XML file for using the mmap interface.",
                                     OPAL_INFO_LVL_4, MCA_BASE_PVAR_CLASS_GENERIC,
                                     MCA_BASE_VAR_TYPE_STRING, NULL, MPI_T_BIND_NO_OBJECT,
                                     MCA_BASE_PVAR_FLAG_READONLY | MCA_BASE_PVAR_FLAG_CONTINUOUS,
                                     ompi_spc_get_xml_filename, NULL, ompi_spc_notify, NULL);
        if(ret < 0) {
            opal_output(0, "There was an error registering an MPI_T pvar -> %s\n", opal_strerror(ret));
        }


        fprintf(fptr, "<?xml version=\"1.0\"?>\n");
        fprintf(fptr, "<SPC>\n");
    }

    /* STEP 3: Add specialized counter enumerations to the appropriate bitmap here.
     *         
     */
    /* ########################################################################
     * ################## Add Timer Based Counter Enums Here ##################
     * ########################################################################
     */

    SET_SPC_BIT(ompi_spc_timer_event, OMPI_SPC_MATCH_TIME);
    SET_SPC_BIT(ompi_spc_timer_event, OMPI_SPC_MATCH_QUEUE_TIME);
    SET_SPC_BIT(ompi_spc_timer_event, OMPI_SPC_OOS_MATCH_TIME);
    SET_SPC_BIT(ompi_spc_timer_event, OMPI_SPC_OOS_MATCH_QUEUE_TIME);

    /* ###############################################################################
     * ###################### Put Bin Counter Sizes Here #############################
     * ###############################################################################
     */
    int data_size = OMPI_SPC_NUM_COUNTERS * sizeof(ompi_spc_value_t);

    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_P2P_MESSAGE_SIZE);
    ompi_spc_offsets[OMPI_SPC_P2P_MESSAGE_SIZE].num_bins = 2;
    data_size += 2 * (sizeof(int) + sizeof(ompi_spc_value_t));

    /* ########################################################################
     * ############## Add Collective Bin-Based Counter Enums Here #############
     * ########################################################################
     */
    /* For each collective bin counter we must set the bitmap bit, allocate memory for the arrays and populate the bin_rules array */

    /* Allgather Algorithms */
    /* Collective bin counter for the Bruck Allgather algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLGATHER_BRUCK);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLGATHER_BRUCK);
    /* Collective bin counter for the Recursive Doubling Allgather algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLGATHER_RECURSIVE_DOUBLING);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLGATHER_RECURSIVE_DOUBLING);
    /* Collective bin counter for the Ring Allgather algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLGATHER_RING);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLGATHER_RING);
    /* Collective bin counter for the Neighbor Exchange Allgather algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLGATHER_NEIGHBOR_EXCHANGE);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLGATHER_NEIGHBOR_EXCHANGE);
    /* Collective bin counter for the Two Process Allgather algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLGATHER_TWO_PROCS);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLGATHER_TWO_PROCS);
    /* Collective bin counter for the Linear Allgather algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLGATHER_LINEAR);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLGATHER_LINEAR);

    /* Allreduce Algorithms */
    /* Collective bin counter for the Nonoverlapping Allreduce algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLREDUCE_NONOVERLAPPING);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLREDUCE_NONOVERLAPPING);
    /* Collective bin counter for the Recursive Doubling Allreduce algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLREDUCE_RECURSIVE_DOUBLING);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLREDUCE_RECURSIVE_DOUBLING);
    /* Collective bin counter for the Ring Allreduce algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLREDUCE_RING);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLREDUCE_RING);
    /* Collective bin counter for the Segmented Ring Allreduce algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLREDUCE_RING_SEGMENTED);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLREDUCE_RING_SEGMENTED);
    /* Collective bin counter for the Linear Allreduce algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLREDUCE_LINEAR);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLREDUCE_LINEAR);

    /* All-to-All Algorithms */
    /* Collective bin counter for the Inplace Alltoall algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLTOALL_INPLACE);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLTOALL_INPLACE);
    /* Collective bin counter for the Pairwise Alltoall algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLTOALL_PAIRWISE);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLTOALL_PAIRWISE);
    /* Collective bin counter for the Bruck Alltoall algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLTOALL_BRUCK);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLTOALL_BRUCK);
    /* Collective bin counter for the Linear Sync Alltoall algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLTOALL_LINEAR_SYNC);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLTOALL_LINEAR_SYNC);
    /* Collective bin counter for the Two Process Alltoall algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLTOALL_TWO_PROCS);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLTOALL_TWO_PROCS);
    /* Collective bin counter for the Linear Alltoall algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_ALLTOALL_LINEAR);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_ALLTOALL_LINEAR);

    /* Broadcast Algorithms */
    /* Collective bin counter for the Chain Broadcast algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_BCAST_CHAIN);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_BCAST_CHAIN);
    /* Collective bin counter for the Binomial Broadcast algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_BCAST_BINOMIAL);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_BCAST_BINOMIAL);
    /* Collective bin counter for the Pipeline Broadcast algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_BCAST_PIPELINE);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_BCAST_PIPELINE);
    /* Collective bin counter for the Split Binary Tree Broadcast algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_BCAST_SPLIT_BINTREE);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_BCAST_SPLIT_BINTREE);
    /* Collective bin counter for the Binary Tree Broadcast algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_BCAST_BINTREE);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_BCAST_BINTREE);

    /* Gather Algorithms */
    /* Collective bin counter for the Binomial Gather algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_GATHER_BINOMIAL);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_GATHER_BINOMIAL);
    /* Collective bin counter for the Linear Sync Gather algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_GATHER_LINEAR_SYNC);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_GATHER_LINEAR_SYNC);
    /* Collective bin counter for the Linear Gather algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_GATHER_LINEAR);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_GATHER_LINEAR);

    /* Reduce Algorithms */
    /* Collective bin counter for the Chain Reduce algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_REDUCE_CHAIN);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_REDUCE_CHAIN);
    /* Collective bin counter for the Pipeline Reduce algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_REDUCE_PIPELINE);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_REDUCE_PIPELINE);
    /* Collective bin counter for the Binary Reduce algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_REDUCE_BINARY);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_REDUCE_BINARY);
    /* Collective bin counter for the Binomial Reduce algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_REDUCE_BINOMIAL);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_REDUCE_BINOMIAL);
    /* Collective bin counter for the In Order Binary Tree Reduce algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_REDUCE_IN_ORDER_BINTREE);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_REDUCE_IN_ORDER_BINTREE);
    /* Collective bin counter for the Linear Reduce algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_REDUCE_LINEAR);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_REDUCE_LINEAR);

    /* Reduce Scatter Algorithms */
    /* Collective bin counter for the Nonoverlapping Reduce Scatter algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_REDUCE_SCATTER_NONOVERLAPPING);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_REDUCE_SCATTER_NONOVERLAPPING);
    /* Collective bin counter for the Recursive Halving Reduce Scatter algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_REDUCE_SCATTER_RECURSIVE_HALVING);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_REDUCE_SCATTER_RECURSIVE_HALVING);
    /* Collective bin counter for the Ring Reduce Scatter algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_REDUCE_SCATTER_RING);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_REDUCE_SCATTER_RING);

    /* Scatter Algorithms */
    /* Collective bin counter for the Binomial Scatter algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_SCATTER_BINOMIAL);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_SCATTER_BINOMIAL);
    /* Collective bin counter for the Linear Scatter algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_SCATTER_LINEAR);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_SCATTER_LINEAR);

/* Form for adding new collective bin counters */
#if 0
    /* X Algorithms */
    /* Collective bin counter for the X algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_);
    /* Collective bin counter for the X algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_);
    /* Collective bin counter for the X algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_);
    /* Collective bin counter for the X algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_);
    /* Collective bin counter for the X algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_);
    /* Collective bin counter for the X algorithm */
    SET_SPC_BIT(ompi_spc_collective_bin_event, OMPI_SPC_BASE_);
    SET_SPC_BIT(ompi_spc_bin_event, OMPI_SPC_BASE_);
#endif

    for(i = 0; i < OMPI_SPC_NUM_COUNTERS; i++) {
        if(IS_SPC_BIT_SET(ompi_spc_collective_bin_event,i)) {
            data_size += 4 * (sizeof(int) + sizeof(ompi_spc_value_t));
            ompi_spc_offsets[i].num_bins = 4;
        }
    }

    /* ###############################################################################
     * ############################# MMAP Initialization #############################
     * ###############################################################################
     */
    uint page_size = opal_getpagesize();
    int bytes_needed = page_size * ((data_size + page_size - 1) % page_size);

    if(ompi_mpi_spc_mmap_enabled) {
        rc = opal_shmem_segment_create(&shm_ds, sm_file, bytes_needed);
        if (OPAL_SUCCESS != rc) {
            opal_show_help("help-mpi-runtime.txt", "spc: shm segment creation failure", true);
            goto map_failed;
        }
        int default_permissions = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
        shm_fd = open(sm_file, O_RDWR | O_CREAT | O_NONBLOCK, default_permissions);
        if(0 > shm_fd) {
            opal_show_help("help-mpi-runtime.txt", "spc: shm file open failure", true, strerror(errno));
            goto map_failed;
        }

        my_segment = opal_shmem_segment_attach(&shm_ds);
        if(NULL == my_segment) {
            opal_show_help("help-mpi-runtime.txt", "spc: shm attach failure", true);
            goto map_failed;
        }
    }

    /* If the mmap fails, we can fall back to malloc to allocate the data.  If malloc fails, then we can't
     * continue and the counters will have to be disabled.
     */
    if(!ompi_mpi_spc_mmap_enabled) {
        goto map_failed;
    }
    if(MAP_FAILED == (ompi_spc_events = mmap(0, bytes_needed, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0))) {
        opal_show_help("help-mpi-runtime.txt", "spc: mmap failure", true, strerror(errno));
    map_failed:
        mmap_failed = true;
        ompi_spc_events = NULL;
        /* mmap failed, so try malloc */
        if(NULL == (ompi_spc_events = malloc(bytes_needed))) {
            opal_show_help("help-mpi-runtime.txt", "lib-call-fail", true,
                           "malloc", __FILE__, __LINE__);
            spc_enabled = false;
            return;
        } else {
            need_free = true; /* Since we malloc'd this data we will need to free it */
        }
    }

    ompi_spc_values = (ompi_spc_value_t*)ompi_spc_events;

    /* Write the XML file header with the basic spc information.
     * NOTE: If we failed to mmap the data file, there is no need to write the XML file.
     */
    if(ompi_mpi_spc_mmap_enabled && !mmap_failed) {
        fprintf(fptr, "\t<filename>%s</filename>\n", sm_file);
        fprintf(fptr, "\t<file_size>%d</file_size>\n", OMPI_SPC_NUM_COUNTERS * sizeof(ompi_spc_t));
        fprintf(fptr, "\t<num_counters>%d</num_counters>\n", OMPI_SPC_NUM_COUNTERS);
        fprintf(fptr, "\t<freq_mhz>%d</freq_mhz>\n", sys_clock_freq_mhz);
    }

    /* The data structure has been allocated, so we simply initialize all of the counters
     * with their names and an initial count of 0.
     */
    for(i = 0; i < OMPI_SPC_NUM_COUNTERS; i++) {
        ompi_spc_values[i] = 0;

        /* Add this counter to the XML document */
        if(ompi_mpi_spc_mmap_enabled && !mmap_failed) {
            fprintf(fptr, "\t<counter>\n");
            fprintf(fptr, "\t\t<name>%s</name>\n", ompi_spc_events_names[i].counter_name);
            fprintf(fptr, "\t\t<value_offset>%d</value_offset>\n", value_offset);
        }
        value_offset += sizeof(ompi_spc_value_t);

        if(ompi_spc_offsets[i].num_bins > 0) {
            ompi_spc_offsets[i].rules_offset = bin_offset;
            bin_offset += ompi_spc_offsets[i].num_bins*sizeof(int);
            ompi_spc_offsets[i].bins_offset = bin_offset;
            bin_offset += ompi_spc_offsets[i].num_bins*sizeof(ompi_spc_value_t);

            /* Make sure the bin offset is cache aligned to avoid false sharing */
            mod = bin_offset % SPC_CACHE_LINE;
            if(mod != 0) {
                bin_offset += SPC_CACHE_LINE - mod;
            }
        } else {
            ompi_spc_offsets[i].rules_offset = -1;
            ompi_spc_offsets[i].bins_offset = -1;
        }
        if(ompi_mpi_spc_mmap_enabled && !mmap_failed) {
            fprintf(fptr, "\t\t<rules_offset>%d</rules_offset>\n", ompi_spc_offsets[i].rules_offset);
            fprintf(fptr, "\t\t<bin_offset>%d</bin_offset>\n", ompi_spc_offsets[i].bins_offset);
            fprintf(fptr, "\t</counter>\n");
        }
    }

    if(ompi_mpi_spc_mmap_enabled && !mmap_failed) {
        fprintf(fptr, "</SPC>\n");
        fclose(fptr);
    }
}

/* Initializes the SPC data structures and registers all counters as MPI_T pvars.
 * Turns on only the counters that were specified in the mpi_spc_attach MCA parameter.  
 */
void ompi_spc_init(void)
{
    int i, j, ret, found = 0, all_on = 0, matched = 0;

    /* Initialize the clock frequency variable as the CPU's frequency in MHz */
    sys_clock_freq_mhz = opal_timer_base_get_freq() / 1000000;

    ompi_spc_events_init();
    if(!spc_enabled) {
        return;
    }

    /* Get the MCA params string of counters to turn on */
    char **arg_strings = opal_argv_split(ompi_mpi_spc_attach_string, ',');
    int num_args       = opal_argv_count(arg_strings);

    /* If there is only one argument and it is 'all', then all counters
     * should be turned on.  If the size is 0, then no counters will be enabled.
     */
    if(1 == num_args) {
        if(strcmp(arg_strings[0], "all") == 0) {
            all_on = 1;
        }
    } else if(0 == num_args) {
        goto no_counters;
    }

    for(i = 0; i < OMPI_SPC_NUM_COUNTERS; i++) {
        matched = all_on;

        if( !matched ) {
            /* Turn on only the counters that were specified in the MCA parameter */
            for(j = 0; j < num_args; j++) {
                if( 0 == strcmp(ompi_spc_events_names[i].counter_name, arg_strings[j]) ) {
                    matched = 1;
                    break;
                }
            }
        }
        if( matched ) {
            SET_SPC_BIT(ompi_spc_attached_event, i);
            mpi_t_enabled = true;
            found++;
        }

        /* Registers the current counter as an MPI_T pvar regardless of whether it's been turned on or not */
        ret = mca_base_pvar_register("ompi", "runtime", "spc", ompi_spc_events_names[i].counter_name, ompi_spc_events_names[i].counter_description,
                                     OPAL_INFO_LVL_4, MPI_T_PVAR_CLASS_COUNTER,
                                     MCA_BASE_VAR_TYPE_UNSIGNED_LONG_LONG, NULL, MPI_T_BIND_NO_OBJECT,
                                     MCA_BASE_PVAR_FLAG_READONLY | MCA_BASE_PVAR_FLAG_CONTINUOUS,
                                     ompi_spc_get_count, NULL, ompi_spc_notify, NULL);

        /* Check to make sure that ret is a valid index and not an error code.
         */
        if( ret >= 0 ) {
            if( mpi_t_offset == -1 ) {
                mpi_t_offset = ret;
            }
        }
        if( (ret < 0) || (all_on && (ret != (mpi_t_offset + found - 1))) ) {
            mpi_t_enabled = false;
            opal_show_help("help-mpi-runtime.txt", "spc: MPI_T disabled", true);
            break;
        }
    }

    /* ########################################################################
     * ###################### Initialize Bin Counters Here ####################
     * ########################################################################
     */
    /* NOTE: These should be initialized even if they aren't currently turned on.
     *       This is just in case they are turned on mid-run through MPI_T.
     */
    /* STEP 3a: Regular bin counters initialized here. */
    int *rules = NULL;
    ompi_spc_value_t *bins = NULL;

    rules = (int*)(ompi_spc_events+ompi_spc_offsets[OMPI_SPC_P2P_MESSAGE_SIZE].rules_offset);
    bins = (ompi_spc_value_t*)(ompi_spc_events+ompi_spc_offsets[OMPI_SPC_P2P_MESSAGE_SIZE].bins_offset);

    bins[0] = bins[1] = 0;

    rules[0] = 2; /* The number of bins */
    rules[1] = ompi_mpi_spc_p2p_message_boundary; /* The number after which counters go in the second bin */

    /* Initialize Collective Bin Counters Here */
    int num_bins = 4; /* This can be expanded to be more flexible */

    for(i = 0; i < OMPI_SPC_NUM_COUNTERS; i++) {
        if(IS_SPC_BIT_SET(ompi_spc_collective_bin_event,i)) {
            rules = (int*)(ompi_spc_events+ompi_spc_offsets[i].rules_offset);
            bins = (ompi_spc_value_t*)(ompi_spc_events+ompi_spc_offsets[i].bins_offset);

            bins[0] = bins[1] = bins[2] = bins[3] = 0;

            rules[0] = num_bins; /* The number of bins */
            rules[1] = ompi_mpi_spc_collective_message_boundary; /* The 'small message' break point */
            rules[2] = ompi_mpi_spc_collective_comm_boundary; /* The 'small communicator' break point */
            rules[3] = 0; /* Placeholder for now */

            ompi_spc_offsets[i].num_bins = 4;
        }
    }

no_counters:
    opal_argv_free(arg_strings);
}

/* Gathers all of the SPC data onto rank 0 of MPI_COMM_WORLD and prints out all
 * of the counter values to stdout.
 */
static void ompi_spc_dump(void)
{
    int i, j, k, world_size, offset, bin_offset;
    long long *recv_buffer = NULL, *send_buffer;
    int *rules;
    ompi_spc_value_t *bins;

    int rank = ompi_comm_rank(ompi_spc_comm);
    world_size = ompi_comm_size(ompi_spc_comm);

    /* Convert from cycles to usecs before sending */
    for(i = 0; i < OMPI_SPC_NUM_COUNTERS; i++) {
        if( IS_SPC_BIT_SET(ompi_spc_timer_event, i) ) {
            SPC_CYCLES_TO_USECS(&ompi_spc_values[i]);
        }
    }

    size_t buffer_size = OMPI_SPC_NUM_COUNTERS * sizeof(long long);
    int buffer_len = OMPI_SPC_NUM_COUNTERS;
    for(i = 0; i < OMPI_SPC_NUM_COUNTERS; i++){
        if(IS_SPC_BIT_SET(ompi_spc_bin_event, i)) {
            /* Increment the buffer size enough to store the bin_rules and bins values */
            rules = (int*)(ompi_spc_events+ompi_spc_offsets[i].rules_offset);
            buffer_size += rules[0] * 2 * sizeof(long long);
            buffer_len += rules[0] * 2;
        }
    }

    /* Aggregate all of the information on rank 0 using MPI_Gather on MPI_COMM_WORLD */
    send_buffer = (long long*)malloc(buffer_size);
    if (NULL == send_buffer) {
        opal_show_help("help-mpi-runtime.txt", "lib-call-fail", true,
                       "malloc", __FILE__, __LINE__);
        return;
    }
    bin_offset = OMPI_SPC_NUM_COUNTERS;
    for(i = 0; i < OMPI_SPC_NUM_COUNTERS; i++) {
        send_buffer[i] = (long long)ompi_spc_values[i];
        /* If this is a bin counter we need to append its arrays to the end of the send buffer */
        if(IS_SPC_BIT_SET(ompi_spc_bin_event, i)) {
            rules = (int*)(ompi_spc_events+ompi_spc_offsets[i].rules_offset);
            for(j = 0; j < rules[0]; j++) {
                send_buffer[bin_offset] = (long long)rules[j];
                bin_offset++;
            }
            /* A flag to check if all bins are 0 */
            int is_empty = 1;
            bins = (ompi_spc_value_t*)(ompi_spc_events+ompi_spc_offsets[i].bins_offset);
            for(j = 0; j < rules[0]; j++) {
                send_buffer[bin_offset] = (long long)bins[j];
                bin_offset++;

                if(bins[j] > 0) {
                    is_empty = 0;
                }
            }
            /* Even if all bins are 0 we still send it for ease, even though it won't be printed */
            if(!is_empty && !IS_SPC_BIT_SET(ompi_spc_collective_bin_event, i)) {
                send_buffer[i] = 1;
            }
        }
    }
    if( 0 == rank ) {
        recv_buffer = (long long*)malloc(world_size * buffer_size);
        if (NULL == recv_buffer) {
            opal_show_help("help-mpi-runtime.txt", "lib-call-fail", true,
                           "malloc", __FILE__, __LINE__);
            return;
        }
    }
    (void)ompi_spc_comm->c_coll->coll_gather(send_buffer, buffer_len, MPI_LONG_LONG,
                                             recv_buffer, buffer_len, MPI_LONG_LONG,
                                             0, ompi_spc_comm,
                                             ompi_spc_comm->c_coll->coll_gather_module);

    /* Once rank 0 has all of the information, print the aggregated counter values for each rank in order */
    if(rank == 0) {
        opal_output(0, "Open MPI Software-based Performance Counters:\n");
        offset = 0; /* Offset into the recv_buffer for each rank */
        bin_offset = OMPI_SPC_NUM_COUNTERS;
        for(j = 0; j < world_size; j++) {
            opal_output(0, "MPI_COMM_WORLD Rank %d:\n", j);
            for(i = 0; i < OMPI_SPC_NUM_COUNTERS; i++) {
                /* Don't print counters with zero values */
                if( 0 == recv_buffer[offset+i] ) {
                    if(IS_SPC_BIT_SET(ompi_spc_bin_event, i)) {
                        bin_offset += recv_buffer[bin_offset]*2;
                    }
                    continue;
                }
                /* This is a non-zero bin counter */
                if(IS_SPC_BIT_SET(ompi_spc_bin_event, i)) {
                    /* This is a non-zero collective bin counter */
                    if(IS_SPC_BIT_SET(ompi_spc_collective_bin_event, i)) {
                        opal_output(0, "%s -> %lld\n", ompi_spc_events_names[i].counter_name, recv_buffer[offset+i]);

                        int num_bins = recv_buffer[bin_offset];
                        int message_boundary = recv_buffer[bin_offset+1];
                        int process_boundary = recv_buffer[bin_offset+2];

                        opal_output(0, "\tSmall Messages (<= %lld bytes)\n", message_boundary);
                        opal_output(0, "\t\tSmall Comm (<= %lld processes) -> %lld\n", process_boundary, recv_buffer[bin_offset+num_bins]);
                        opal_output(0, "\t\tLarge Comm (>  %lld processes) -> %lld\n", process_boundary, recv_buffer[bin_offset+num_bins+1]);
                        opal_output(0, "\tLarge Messages (> %lld bytes)\n", message_boundary);
                        opal_output(0, "\t\tSmall Comm (<= %lld processes) -> %lld\n", process_boundary, recv_buffer[bin_offset+num_bins+2]);
                        opal_output(0, "\t\tLarge Comm (>  %lld processes) -> %lld\n", process_boundary, recv_buffer[bin_offset+num_bins+3]);

                        bin_offset += num_bins*2;
                        continue;
                    }
                    opal_output(0, "%s\n", ompi_spc_events_names[i].counter_name);
                    int num_bins = recv_buffer[bin_offset];
                    for(k = 0; k < num_bins; k++){
                        if(k == 0) {
                            opal_output(0, "\t-inf to %lld -> %lld\n", recv_buffer[bin_offset+1], recv_buffer[bin_offset+num_bins]);
                        } else if(k < num_bins-1) {
                            opal_output(0, "\t%lld to %lld -> %lld\n", recv_buffer[bin_offset+k]+1, recv_buffer[bin_offset+k+1], recv_buffer[bin_offset+num_bins+k]);
                        } else {
                            opal_output(0, "\t%lld to inf -> %lld\n", recv_buffer[bin_offset+k]+1, recv_buffer[bin_offset+num_bins+k]);
                        }
                    }
                    bin_offset += num_bins*2;
                    continue;
                }
                /* This is a non-zero normal counter */
                opal_output(0, "%s -> %lld\n", ompi_spc_events_names[i].counter_name, recv_buffer[offset+i]);
            }
            opal_output(0, "\n");
            offset += buffer_len;
            bin_offset += OMPI_SPC_NUM_COUNTERS;
        }
        opal_output(0, "###########################################################################\n");
        opal_output(0, "NOTE: Any counters not shown here were either disabled or had a value of 0.\n");
        opal_output(0, "###########################################################################\n");

        free(recv_buffer); recv_buffer = NULL;
    }
    free(send_buffer); send_buffer = NULL;

    ompi_spc_comm->c_coll->coll_barrier(ompi_spc_comm, ompi_spc_comm->c_coll->coll_barrier_module);
}

/* Frees any dynamically alocated OMPI SPC data structures */
void ompi_spc_fini(void)
{
    int fd, rc;
    char sm_file[SPC_MAX_FILENAME];
    char *shm_dir = SPC_SHM_DIR;

    int rank = ompi_comm_rank(ompi_spc_comm);

    if (SPC_ENABLE == 1 && ompi_mpi_spc_dump_enabled) {
        ompi_spc_dump();
    }

    int i;
    for(i = 0; i < OMPI_SPC_NUM_COUNTERS; i++) {
        CLEAR_SPC_BIT(ompi_spc_attached_event, i);
        CLEAR_SPC_BIT(ompi_spc_timer_event, i);
        CLEAR_SPC_BIT(ompi_spc_bin_event, i);
        CLEAR_SPC_BIT(ompi_spc_collective_bin_event, i);
    }
    if(need_free) {
        free(ompi_spc_events); ompi_spc_events = NULL;
    }
    ompi_comm_free(&ompi_spc_comm); ompi_spc_comm = NULL;
}

/* Records an update to a counter using an atomic add operation. */
void ompi_spc_record(unsigned int event_id, ompi_spc_value_t value)
{
    OPAL_THREAD_ADD_FETCH_SIZE_T(&(ompi_spc_values[event_id]), value);
}

/* Records an update to a bin counter using an atomic add operation. */
void ompi_spc_bin_record(unsigned int event_id, ompi_spc_value_t value)
{
    int *rules;
    ompi_spc_value_t *bins;

    /* Update the total number of times this counter has been triggered */
    OPAL_THREAD_ADD_FETCH_SIZE_T(&(ompi_spc_values[event_id]), 1);
    rules = (int*)(ompi_spc_events+ompi_spc_offsets[event_id].rules_offset);
    bins = (ompi_spc_value_t*)(ompi_spc_events+ompi_spc_offsets[event_id].bins_offset);

    int i, num_bins = rules[0];
    /* Update the appropriate bin */
    for(i = 1; i < num_bins; i++) {
        if(value <= rules[i]) {
            OPAL_THREAD_ADD_FETCH_SIZE_T(&(bins[i-1]), 1);
            return;
        }
    }
    /* This didn't fall within any of the other bins, so it must belong to the last bin */
    OPAL_THREAD_ADD_FETCH_SIZE_T(&(bins[num_bins-1]), 1);
}

/* Records an update to a collective bin counter using an atomic add operation. */
void ompi_spc_collective_bin_record(unsigned int event_id, ompi_spc_value_t bytes, ompi_spc_value_t procs)
{
    int *rules;
    ompi_spc_value_t *bins;

    rules = (int*)(ompi_spc_events+ompi_spc_offsets[event_id].rules_offset);
    bins = (ompi_spc_value_t*)(ompi_spc_events+ompi_spc_offsets[event_id].bins_offset);

    uint small_message = (bytes <= rules[1]);
    uint small_comm    = (procs <= rules[2]);
    /* Always update the total number of times this collective algorithm was called */
    OPAL_THREAD_ADD_FETCH_SIZE_T(&(ompi_spc_values[event_id]), 1);

    /* Update the appropriate bin for the message size and number of processes */
    if(small_message && small_comm) {
        OPAL_THREAD_ADD_FETCH_SIZE_T(&(bins[0]), 1);
    } else if(small_message && !small_comm) {
        OPAL_THREAD_ADD_FETCH_SIZE_T(&(bins[1]), 1);
    } else if(!small_message && small_comm) {
        OPAL_THREAD_ADD_FETCH_SIZE_T(&(bins[2]), 1);
    } else {
        OPAL_THREAD_ADD_FETCH_SIZE_T(&(bins[3]), 1);
    }
}

/* Starts cycle-precision timer and stores the start value in the 'cycles' argument.
 * Note: This assumes that the 'cycles' argument is initialized to 0 if the timer
 *       hasn't been started yet.
 */
void ompi_spc_timer_start(unsigned int event_id, opal_timer_t *cycles)
{
    /* Check whether cycles == 0.0 to make sure the timer hasn't started yet. */
    if( *cycles == 0 ) {
        *cycles = opal_timer_base_get_cycles();
    }
}

/* Stops a cycle-precision timer and calculates the total elapsed time
 * based on the starting time in 'cycles' and stores the result in the
 * 'cycles' argument.
 */
void ompi_spc_timer_stop(unsigned int event_id, opal_timer_t *cycles)
{
    *cycles = opal_timer_base_get_cycles() - *cycles;
    OPAL_THREAD_ADD_FETCH_SIZE_T(&ompi_spc_values[event_id], (ompi_spc_value_t) *cycles);
}

/* Checks a tag, and records the user version of the counter if it's greater
 * than or equal to 0 and records the mpi version of the counter otherwise.
 */
void ompi_spc_user_or_mpi(int tag, ompi_spc_value_t value, unsigned int user_enum, unsigned int mpi_enum)
{
    SPC_RECORD( (tag >= 0 ? user_enum : mpi_enum), value);
}

/* Checks whether the counter denoted by value_enum exceeds the current value of the
 * counter denoted by watermark_enum, and if so sets the watermark_enum counter to the
 * value of the value_enum counter.
 *
 * WARNING: This assumes that this function was called while a lock has already been taken.
 *          This function is NOT thread safe otherwise!
 */
void ompi_spc_update_watermark(unsigned int watermark_enum, unsigned int value_enum, ompi_spc_value_t value)
{
    OPAL_THREAD_ADD_FETCH_SIZE_T(&(ompi_spc_values[value_enum]), value);
    if(IS_SPC_BIT_SET(ompi_spc_attached_event, watermark_enum)) {
        if(ompi_spc_values[value_enum] > ompi_spc_values[watermark_enum]) {
            ompi_spc_values[watermark_enum] = ompi_spc_values[value_enum];
        }
    }
}

/* Gets the value of an SPC counter.
 *
 * WARNING: This function is not performed atomically and may not return the most up to date
 *          value in a threaded run.
 */
ompi_spc_value_t ompi_spc_get_value(unsigned int event_id)
{
    if( OPAL_UNLIKELY(IS_SPC_BIT_SET(ompi_spc_attached_event, event_id)) ) {
        return ompi_spc_values[event_id];
    }
    return 0;
}

/* Converts a counter value that is in cycles to microseconds.
 */
void ompi_spc_cycles_to_usecs(ompi_spc_value_t *cycles)
{
    *cycles = *cycles / sys_clock_freq_mhz;
}
