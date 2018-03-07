/*
 * Copyright (c) 2018      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
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
#include <dlfcn.h>

#include "ompi/communicator/communicator.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/runtime/params.h"
#include "opal/mca/timer/timer.h"
#include "opal/mca/base/mca_base_pvar.h"
#include "opal/util/argv.h"

#include MCA_timer_IMPLEMENTATION_HEADER

/* INSTRUCTIONS FOR ADDING COUNTERS
 * 1.) Add a new counter name in the OMPI_COUNTERS enum before
 *     OMPI_NUM_COUNTERS below.
 * 2.) Add corresponding counter name and descriptions to the
 *     counter_names and counter_descriptions arrays in
 *     ompi_spc.c  NOTE: The names and descriptions
 *     MUST be in the same array location as where you added the
 *     counter name in step 1.
 * 3.) If this counter is based on a timer, add its enum name to
 *     the logic for timer-based counters in the ompi_spc_init
 *     function in ompi_spc.c
 * 4.) Instrument the Open MPI code base where it makes sense for
 *     your counter to be modified using the SPC_RECORD macro.
 *     Note: If your counter is timer-based you should use the
 *     SPC_TIMER_START and SPC_TIMER_STOP macros to record
 *     the time in cycles to then be converted to microseconds later
 *     in the ompi_spc_get_count function when requested by MPI_T
 */

/* This enumeration serves as event ids for the various events */
enum OMPI_COUNTERS{
    OMPI_SEND,
    OMPI_BSEND,
    OMPI_RSEND,
    OMPI_SSEND,
    OMPI_RECV,
    OMPI_MRECV,
    OMPI_ISEND,
    OMPI_IBSEND,
    OMPI_IRSEND,
    OMPI_ISSEND,
    OMPI_IRECV,
    OMPI_SENDRECV,
    OMPI_SENDRECV_REPLACE,
    OMPI_PUT,
    OMPI_RPUT,
    OMPI_GET,
    OMPI_RGET,
    OMPI_PROBE,
    OMPI_IPROBE,
    OMPI_BCAST,
    OMPI_IBCAST,
    OMPI_REDUCE,
    OMPI_IREDUCE,
    OMPI_IREDUCE_SCATTER,
    OMPI_IREDUCE_SCATTER_BLOCK,
    OMPI_ALLREDUCE,
    OMPI_IALLREDUCE,
    OMPI_SCAN,
    OMPI_ISCAN,
    OMPI_SCATTER,
    OMPI_SCATTERV,
    OMPI_ISCATTER,
    OMPI_ISCATTERV,
    OMPI_GATHER,
    OMPI_GATHERV,
    OMPI_IGATHER,
    OMPI_ALLTOALL,
    OMPI_IALLTOALL,
    OMPI_IALLTOALLV,
    OMPI_IALLTOALLW,
    OMPI_NEIGHBOR_ALLTOALL,
    OMPI_NEIGHBOR_ALLTOALLV,
    OMPI_NEIGHBOR_ALLTOALLW,
    OMPI_ALLGATHER,
    OMPI_ALLGATHERV,
    OMPI_IALLGATHER,
    OMPI_IALLGATHERV,
    OMPI_NEIGHBOR_ALLGATHER,
    OMPI_NEIGHBOR_ALLGATHERV,
    OMPI_TEST,
    OMPI_TESTALL,
    OMPI_TESTANY,
    OMPI_TESTSOME,
    OMPI_WAIT,
    OMPI_WAITALL,
    OMPI_WAITANY,
    OMPI_WAITSOME,
    OMPI_BARRIER,
    OMPI_IBARRIER,
    OMPI_WTICK,
    OMPI_WTIME,
    OMPI_CANCEL,
    OMPI_BYTES_RECEIVED_USER,
    OMPI_BYTES_RECEIVED_MPI,
    OMPI_BYTES_SENT_USER,
    OMPI_BYTES_SENT_MPI,
    OMPI_BYTES_PUT,
    OMPI_BYTES_GET,
    OMPI_UNEXPECTED,
    OMPI_OUT_OF_SEQUENCE,
    OMPI_MATCH_TIME,
    OMPI_UNEXPECTED_IN_QUEUE,
    OMPI_OOS_IN_QUEUE,
    OMPI_MAX_UNEXPECTED_IN_QUEUE,
    OMPI_MAX_OOS_IN_QUEUE,
    OMPI_NUM_COUNTERS /* This serves as the number of counters.  It must be last. */
};

/* A structure for storing the event data */
typedef struct ompi_event_s{
    char *name;
    long long value;
} ompi_event_t;

OMPI_DECLSPEC extern unsigned int attached_event[OMPI_NUM_COUNTERS];
OMPI_DECLSPEC extern unsigned int timer_event[OMPI_NUM_COUNTERS];
OMPI_DECLSPEC extern ompi_event_t *events;

/* Events data structure initialization function */
void events_init(void);

/* OMPI SPC utility functions */
void ompi_spc_init(void);
void ompi_spc_fini(void);
void ompi_spc_record(unsigned int event_id, long long value);
void ompi_spc_timer_start(unsigned int event_id, opal_timer_t *cycles);
void ompi_spc_timer_stop(unsigned int event_id, opal_timer_t *cycles);
void ompi_spc_user_or_mpi(int tag, long long value, unsigned int user_enum, unsigned int mpi_enum);
void ompi_spc_cycles_to_usecs(long long *cycles);
void ompi_spc_update_watermark(unsigned int watermark_enum, unsigned int value_enum);

/* MPI_T utility functions */
static int ompi_spc_notify(mca_base_pvar_t *pvar, mca_base_pvar_event_t event, void *obj_handle, int *count);
long long ompi_spc_get_counter(int counter_id);

/* Macros for using the SPC utility functions throughout the codebase.
 * If SPC_ENABLE is not 1, the macros become no-ops.
 */
#if SPC_ENABLE == 1

#define SPC_INIT()  \
    ompi_spc_init()

#define SPC_FINI()  \
    ompi_spc_fini()

#define SPC_RECORD(event_id, value)  \
    ompi_spc_record(event_id, value)

#define SPC_TIMER_START(event_id, usec)  \
    ompi_spc_timer_start(event_id, usec)

#define SPC_TIMER_STOP(event_id, usec)  \
    ompi_spc_timer_stop(event_id, usec)

#define SPC_USER_OR_MPI(tag, value, enum_if_user, enum_if_mpi) \
    ompi_spc_user_or_mpi(tag, value, enum_if_user, enum_if_mpi)

#define SPC_CYCLES_TO_USECS(cycles) \
    ompi_spc_cycles_to_usecs(cycles)

#define SPC_UPDATE_WATERMARK(watermark_enum, value_enum) \
    ompi_spc_update_watermark(watermark_enum, value_enum)

#else /* SPCs are not enabled */

#define SPC_INIT()  \
    do {} while (0)

#define SPC_FINI()  \
    do {} while (0)

#define SPC_RECORD(event_id, value)  \
    do {} while (0)

#define SPC_TIMER_START(event_id, usec)  \
    do {} while (0)

#define SPC_TIMER_STOP(event_id, usec)  \
    do {} while (0)

#define SPC_USER_OR_MPI(tag, value, enum_if_user, enum_if_mpi) \
    do {} while (0)

#define SPC_CYCLES_TO_USECS(cycles) \
    do {} while (0)

#define SPC_UPDATE_WATERMARK(watermark_enum, value_enum) \
    do {} while (0)

#endif

#endif
