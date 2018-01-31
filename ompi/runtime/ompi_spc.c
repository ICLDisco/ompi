/*
 * Copyright (c) 2004-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_spc.h"

opal_timer_t sys_clock_freq_mhz = 0;

/* Array for converting from SPC indices to MPI_T indices */
OMPI_DECLSPEC int mpi_t_indices[OMPI_NUM_COUNTERS] = {0};
OMPI_DECLSPEC int mpi_t_offset = -1;
OMPI_DECLSPEC int mpi_t_disabled = 0;

/* Array of names for each counter.  Used for MPI_T and PAPI sde initialization */
OMPI_DECLSPEC const char *counter_names[OMPI_NUM_COUNTERS] = {
    "OMPI_SEND",
    "OMPI_RECV",
    "OMPI_ISEND",
    "OMPI_IRECV",
    "OMPI_BCAST",
    "OMPI_REDUCE",
    "OMPI_ALLREDUCE",
    "OMPI_SCATTER",
    "OMPI_GATHER",
    "OMPI_ALLTOALL",
    "OMPI_ALLGATHER",
    "OMPI_BYTES_RECEIVED_USER",
    "OMPI_BYTES_RECEIVED_MPI",
    "OMPI_BYTES_SENT_USER",
    "OMPI_BYTES_SENT_MPI",
    "OMPI_BYTES_PUT",
    "OMPI_BYTES_GET",
    "OMPI_UNEXPECTED",
    "OMPI_OUT_OF_SEQUENCE",
    "OMPI_MATCH_TIME",
    "OMPI_OOS_MATCH_TIME",
    "OMPI_UNEXPECTED_IN_QUEUE",
    "OMPI_OOS_IN_QUEUE",
    "OMPI_MAX_UNEXPECTED_IN_QUEUE",
    "OMPI_MAX_OOS_IN_QUEUE"
};

/* Array of descriptions for each counter.  Used for MPI_T and PAPI sde initialization */
OMPI_DECLSPEC const char *counter_descriptions[OMPI_NUM_COUNTERS] = {
    "The number of times MPI_Send was called.",
    "The number of times MPI_Recv was called.",
    "The number of times MPI_Isend was called.",
    "The number of times MPI_Irecv was called.",
    "The number of times MPI_Bcast was called.",
    "The number of times MPI_Reduce was called.",
    "The number of times MPI_Allreduce was called.",
    "The number of times MPI_Scatter was called.",
    "The number of times MPI_Gather was called.",
    "The number of times MPI_Alltoall was called.",
    "The number of times MPI_Allgather was called.",
    "The number of bytes received by the user through point-to-point communications. Note: Includes bytes transferred using internal RMA operations.",
    "The number of bytes received by MPI through collective, control, or other internal communications.",
    "The number of bytes sent by the user through point-to-point communications.  Note: Includes bytes transferred using internal RMA operations.",
    "The number of bytes sent by MPI through collective, control, or other internal communications.",
    "The number of bytes sent/received using RMA Put operations both through user-level Put functions and internal Put functions.",
    "The number of bytes sent/received using RMA Get operations both through user-level Get functions and internal Get functions.",
    "The number of messages that arrived as unexpected messages.",
    "The number of messages that arrived out of the proper sequence.",
    "The number of microseconds spent matching unexpected messages.",
    "The number of microseconds spent matching out of sequence messages.",
    "The number of messages that are currently in the unexpected message queue(s) of an MPI process.",
    "The number of messages that are currently in the out of sequence message queue(s) of an MPI process.",
    "The maximum number of messages that the unexpected message queue(s) within an MPI process contained at once since the last reset of this counter.  \
Note: This counter is reset each time it is read.",
    "The maximum number of messages that the out of sequence message queue(s) within an MPI process contained at once since the last reset of this counter.  \
Note: This counter is reset each time it is read."
};

/* An array of integer values to denote whether an event is activated (1) or not (0) */
OMPI_DECLSPEC unsigned int attached_event[OMPI_NUM_COUNTERS] = { 0 };
/* An array of integer values to denote whether an event is timer-based (1) or not (0) */
OMPI_DECLSPEC unsigned int timer_event[OMPI_NUM_COUNTERS] = { 0 };
/* An array of event structures to store the event data (name and value) */
OMPI_DECLSPEC ompi_event_t *events = NULL;

/* ##############################################################
 * ################# Begin MPI_T Functions ######################
 * ##############################################################
 */
static int ompi_spc_notify(mca_base_pvar_t *pvar, mca_base_pvar_event_t event, void *obj_handle, int *count)
{
    (void)obj_handle;

    int index;

    if(OPAL_UNLIKELY(mpi_t_disabled == 1))
        return MPI_SUCCESS;

    /* For this event, we need to set count to the number of long long type
     * values for this counter.  All SPC counters are one long long, so we
     * always set count to 1.
     */
    if(MCA_BASE_PVAR_HANDLE_BIND == event) {
        *count = 1;
    }
    /* For this event, we need to turn on the counter */
    else if(MCA_BASE_PVAR_HANDLE_START == event) {
        /* Convert from MPI_T pvar index to SPC index */
        index = pvar->pvar_index - mpi_t_offset;
        attached_event[index] = 1;
    }
    /* For this event, we need to turn off the counter */
    else if(MCA_BASE_PVAR_HANDLE_STOP == event) {
        /* Convert from MPI_T pvar index to SPC index */
        index = pvar->pvar_index - mpi_t_offset;
        attached_event[index] = 0;
    }

    return MPI_SUCCESS;
}

/* ##############################################################
 * ################# Begin SPC Functions ########################
 * ##############################################################
 */

/* This function returns the current count of an SPC counter that has been retistered
 * as an MPI_T pvar.  The MPI_T index is not necessarily the same as the SPC index,
 * so we need to convert from MPI_T index to SPC index and then set the 'value' argument
 * to the correct value for this pvar.
 */
static int ompi_spc_get_count(const struct mca_base_pvar_t *pvar, void *value, void *obj_handle)
{   
    (void) obj_handle;

    long long *counter_value = (long long*)value;

    if(OPAL_UNLIKELY(mpi_t_disabled == 1)) {
        *counter_value = 0;
        return MPI_SUCCESS;
    }

    /* Convert from MPI_T pvar index to SPC index */
    int index = pvar->pvar_index - mpi_t_offset;
    /* Set the counter value to the current SPC value */
    *counter_value = events[index].value;
    /* If this is a timer-based counter, convert from cycles to microseconds */
    if(timer_event[index])
        *counter_value /= sys_clock_freq_mhz;
    /* If this is a high watermark counter, reset it after it has been read */
    if(index == OMPI_MAX_UNEXPECTED_IN_QUEUE || index == OMPI_MAX_OOS_IN_QUEUE)
        events[index].value = 0;

    return MPI_SUCCESS;
}

/* Initializes the events data structure and allocates memory for it if needed. */
void events_init()
{
    int i;

    /* If the events data structure hasn't been allocated yet, allocate memory for it */
    if(events == NULL) {
        events = (ompi_event_t*)malloc(OMPI_NUM_COUNTERS * sizeof(ompi_event_t));
    }
    /* The data structure has been allocated, so we simply initialize all of the counters
     * with their names and an initial count of 0.
     */
    for(i = 0; i < OMPI_NUM_COUNTERS; i++) {
        events[i].name = counter_names[i];
        events[i].value = 0;
    }
}

/* Initializes the SPC data structures and registers all counters as MPI_T pvars.
 * Turns on only the counters that were specified in the mpi_spc_attach MCA parameter.  
 */
void ompi_spc_init()
{
    int i, j, ret, found = 0, all_on = 0;

    /* Initialize the clock frequency variable as the CPU's frequency in MHz */
    sys_clock_freq_mhz = opal_timer_base_get_freq() / 1000000;

    events_init();

    /* Get the MCA params string of counters to turn on */
    char **arg_strings = opal_argv_split(ompi_mpi_spc_attach_string, ',');
    int num_args      = opal_argv_count(arg_strings);

    /* If there is only one argument and it is 'all', then all counters
     * should be turned on.  If the size is 0, then no counters will be enabled.
     */
    if(num_args == 1) {
        if(strcmp(arg_strings[0], "all") == 0)
            all_on = 1;
    }

    int prev = -1;
    /* Turn on only the counters that were specified in the MCA parameter */
    for(i = 0; i < OMPI_NUM_COUNTERS; i++) {
        if(all_on)
            attached_event[i] = 1;
        else {
            /* Note: If no arguments were given, this will be skipped */
            for(j = 0; j < num_args && found < num_args; j++) {
                if(strcmp(counter_names[i], arg_strings[j]) == 0) {
                    attached_event[i] = 1;
                    found++;
                }
            }
        }

        /* ########################################################################
         * ################## Add Timer-Based Counter Enums Here ##################
         * ########################################################################
         */
        /* If this is a timer event, sent the corresponding timer_event entry to 1 */
        if(i == OMPI_MATCH_TIME || i == OMPI_OOS_MATCH_TIME)
            timer_event[i] = 1;
        else
            timer_event[i] = 0;

        /* Registers the current counter as an MPI_T pvar regardless of whether it's been turned on or not */
        ret = mca_base_pvar_register("ompi", "runtime", "spc", counter_names[i], counter_descriptions[i],
                                     OPAL_INFO_LVL_4, MPI_T_PVAR_CLASS_SIZE,
                                     MCA_BASE_VAR_TYPE_UNSIGNED_LONG_LONG, NULL, MPI_T_BIND_NO_OBJECT,
                                     MCA_BASE_PVAR_FLAG_READONLY | MCA_BASE_PVAR_FLAG_CONTINUOUS,
                                     ompi_spc_get_count, NULL, ompi_spc_notify, NULL);

        /* Initialize the mpi_t_indices array with the MPI_T indices.
         * The array index indicates the SPC index, while the value indicates
         * the MPI_T index.
         */
        if(ret != OPAL_ERROR) {
            mpi_t_indices[i] = ret;

            if(mpi_t_offset == -1) {
                mpi_t_offset = ret;
                prev = ret;
            } else if(ret != prev + 1) {
                mpi_t_disabled = 1;
                fprintf(stderr, "There was an error registering SPCs as MPI_T pvars.  SPCs will be disabled for MPI_T.\n");
                break;
            } else {
                prev = ret;
            }
        } else {
            mpi_t_indices[i] = -1;
            mpi_t_disabled = 1;
            fprintf(stderr, "There was an error registering SPCs as MPI_T pvars.  SPCs will be disabled for MPI_T.\n");
            break;
        }
    }
}

/* Frees any dynamically alocated OMPI SPC data structures */
void ompi_spc_fini()
{
    free(events);
}

/* Records an update to a counter using an atomic add operation. */
void ompi_spc_record(unsigned int event_id, long long value)
{
    /* Denoted unlikely because counters will often be turned off. */
    if(OPAL_UNLIKELY(attached_event[event_id] == 1)) {
        OPAL_THREAD_ADD_FETCH_SIZE_T(&(events[event_id].value), value);
    }
}

/* Starts cycle-precision timer and stores the start value in the 'cycles' argument.
 * Note: This assumes that the 'cycles' argument is initialized to 0 if the timer
 *       hasn't been started yet.
 */
void ompi_spc_timer_start(unsigned int event_id, opal_timer_t *cycles)
{
    /* Check whether cycles == 0.0 to make sure the timer hasn't started yet.
     * This is denoted unlikely because the counters will often be turned off.
     */
    if(OPAL_UNLIKELY(attached_event[event_id] == 1 && *cycles == 0)) {
        *cycles = opal_timer_base_get_cycles();
    }
}

/* Stops a cycle-precision timer and calculates the total elapsed time
 * based on the starting time in 'cycles' and stores the result in the
 * 'cycles' argument.
 */
void ompi_spc_timer_stop(unsigned int event_id, opal_timer_t *cycles)
{
    /* This is denoted unlikely because the counters will often be turned off. */
    if(OPAL_UNLIKELY(attached_event[event_id] == 1)) {
        *cycles = opal_timer_base_get_cycles() - *cycles;
        OPAL_THREAD_ADD_FETCH_SIZE_T(&events[event_id].value, (long long)*cycles);
    }
}

/* Checks a tag, and records the user version of the counter if it's greater
 * than or equal to 0 and records the mpi version of the counter otherwise.
 */
void ompi_spc_user_or_mpi(int tag, long long value, unsigned int user_enum, unsigned int mpi_enum)
{
    if(tag >= 0 || tag == MPI_ANY_TAG) {
        SPC_RECORD(user_enum, value);
    } else {
        SPC_RECORD(mpi_enum, value);
    }
}

/* Checks whether the counter denoted by value_enum exceeds the current value of the
 * counter denoted by watermark_enum, and if so sets the watermark_enum counter to the
 * value of the value_enum counter.
 */
void ompi_spc_update_watermark(unsigned int watermark_enum, unsigned int value_enum)
{
    /* Denoted unlikely because counters will often be turned off. */
    if(OPAL_UNLIKELY(attached_event[watermark_enum] == 1 && attached_event[value_enum] == 1)) {
        /* WARNING: This assumes that this function was called while a lock has already been taken.
         *          This function is NOT thread safe otherwise!
         */
        if(events[value_enum].value > events[watermark_enum].value)
            events[watermark_enum].value = events[value_enum].value;
    }
}

/* Converts a counter value that is in cycles to microseconds.
 */
void ompi_spc_cycles_to_usecs(long long *cycles)
{
    *cycles = *cycles / sys_clock_freq_mhz;
}
