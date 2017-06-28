#include "ompi_software_events.h"

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
    "OMPI_PROGRESS_SWITCH"
};

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
    "The number of bytes received by the user through point-to-point communications. Note: Excludes RDMA operations.",
    "The number of bytes received by MPI through collective, control, or other internal communications.",
    "The number of bytes sent by the user through point-to-point communications.  Note: Excludes RDMA operations.",
    "The number of bytes sent by MPI through collective, control, or other internal communications.",
    "The number of bytes sent/received using RDMA Put operations.",
    "The number of bytes sent/received using RDMA Get operations.",
    "The number of messages that arrived as unexpected messages.",
    "The number of messages that arrived out of the proper sequence.",
    "The number of microseconds spent matching unexpected messages.",
    "The number of microseconds spent matching out of sequence messages.",
    "The number of times the progress thread changed."
};

/* An array of integer values to denote whether an event is activated (1) or not (0) */
OMPI_DECLSPEC unsigned int attached_event[OMPI_NUM_COUNTERS] = { 0 };
/* An array of event structures to store the event data (name and value) */
OMPI_DECLSPEC ompi_event_t *events = NULL;

/* ##############################################################
 * ################# Begin MPI_T Functions ######################
 * ##############################################################
 */

static int ompi_sw_event_notify(mca_base_pvar_t *pvar, mca_base_pvar_event_t event, void *obj_handle, int *count)
{
    (void)obj_handle;
    if(MCA_BASE_PVAR_HANDLE_BIND == event)
        *count = 1;
 
    return OPAL_SUCCESS;
}

inline long long ompi_sw_event_get_counter(int counter_id)
{
    if(events != NULL)
        return events[counter_id].value;
    else
        return 0; /* -1 would be preferred to indicate lack of initialization, but the type needs to be unsigned */
}

static int ompi_sw_event_get_send(const struct mca_base_pvar_t *pvar, void *value, void *obj_handle)
{   
    (void) obj_handle;
    long long *counter_value = (long long*)value;
    *counter_value = ompi_sw_event_get_counter(OMPI_SEND);

    return OPAL_SUCCESS;
}

/* ##############################################################
 * ############ Begin PAPI software_events Code #################
 * ##############################################################
 */

/* Allocates and initializes the events data structure */
int iter_start()
{
    int i;

    if(events == NULL){
        events = (ompi_event_t*)malloc(OMPI_NUM_COUNTERS * sizeof(ompi_event_t));
    } else {
        fprintf(stderr, "The events data structure has already been allocated.\n");
    }

    for(i = 0; i < OMPI_NUM_COUNTERS; i++){
        events[i].name = counter_names[i];
        events[i].value = 0;
    }
    return 0;
}

/* Returns the name of the next event in the data structure */
char* iter_next()
{
    static int i = 0;

    if(i < OMPI_NUM_COUNTERS){
        i++;
        return events[i-1].name;
    }
    else{
        /* Finished iterating through the list.  Return NULL and reset i */
        i = 0;
        return NULL;
    }
}

/* Frees the events data structure */
int iter_release()
{
    free(events);
    return 0;
}

/* If an event named 'event_name' exists, attach the corresponding event's value
 * to the supplied long long pointer.
 */
int attach_event(char *event_name, long long **value)
{
    int i;

    if(events == NULL){
        fprintf(stderr, "Error: The iterator hasn't been started.  The event cannot be attached.\n");
        return -1;
    }

    if(event_name == NULL){
        fprintf(stderr, "Error: No event name specified for attach_event.\n");
        return -1;
    }

    for(i = 0; i < OMPI_NUM_COUNTERS; i++){
        if(strcmp(event_name, events[i].name) == 0){
            break;
        }
    }

    if(i < OMPI_NUM_COUNTERS){
        *value = &events[i].value;
        attached_event[i] = 1;

        return 0;
    }
    else{
        fprintf(stderr, "Error: Could not find an event by that name.  The event cannot be attached.\n");
        return -1;
    }
}

/* If an event with the name 'event_name' exists, reset its value to 0
 * and set the corresponding value in attached_event to 0.
 */
int detach_event(char *event_name)
{
    int i;

    if(events == NULL){
        fprintf(stderr, "Error: The iterator hasn't been started.  The event cannot be detached.\n");
        return -1;
    }

    if(event_name == NULL){
        fprintf(stderr, "Error: No event name specified for detach_event.\n");
        return -1;
    }

    for(i = 0; i < OMPI_NUM_COUNTERS; i++){
        if(strcmp(event_name, events[i].name) == 0){
            break;
        }
    }

    if(i < OMPI_NUM_COUNTERS){
        attached_event[i] = 0;
        events[i].value = 0;

        return 0;
    }
    else{
        fprintf(stderr, "Error: Could not find an event by that name.  The event cannot be detached.\n");
        return -1;
    }
}

/* A structure to expose to the PAPI software_events component to use these events */
struct PAPI_SOFTWARE_EVENT_S papi_software_events = {"ompi", {0, 0, 0}, iter_start, iter_next, iter_release, attach_event, detach_event};

/* ##############################################################
 * ############ End of PAPI software_events Code ################
 * ##############################################################
 */

/* ##############################################################
 * ############### Begin PAPI sde Code ##########################
 * ##############################################################
 */

/* An initialization function for the PAPI sde component.
 * This creates an sde handle with the name OMPI and registers all events and
 * event descriptions with the sde component.
 */
void ompi_sde_init() {
    int i, event_count = OMPI_NUM_COUNTERS;
    void *sde_handle = (void *)papi_sde_init("OMPI", &event_count);

    /* Required registration of counters and optional counter descriptions */
    for(i = 0; i < OMPI_NUM_COUNTERS; i++){
        printf("Registering: %s (%d of %d)\n", counter_names[i], i, OMPI_NUM_COUNTERS);
        papi_sde_register_counter(sde_handle, counter_names[i], &(events[i].value) );
        papi_sde_describe_counter(sde_handle, counter_names[i], counter_descriptions[i]);
    }
}

/* Define PAPI_DYNAMIC_SDE since we are assuming PAPI is linked dynamically.
 * Note: In the future we should support both dynamic and static linking of PAPI.
 */
#define PAPI_DYNAMIC_SDE
/* This function will be called from papi_native_avail to list all of the OMPI
 * events with their names and descriptions.  In order for the dynamic version
 * to work, the environment variable PAPI_SHARED_LIB must contain the full path
 * to the PAPI shared library like the following:
 * /path/to/papi/install/lib/libpapi.so
 * 
 * This function will use dlsym to get the appropriate functions for initializing
 * the PAPI sde component's environment and register all of the events.
 */
void* papi_sde_hook_list_events(void)
{
    int  i, event_count = OMPI_NUM_COUNTERS;
    char *error;
    void   *papi_handle;
    void*  (*sym_init)(char *name_of_library, int *event_count);
    void   (*sym_reg)( void *handle, char *event_name, long long *counter);
    void   (*sym_desc)(void *handle, char *event_name, char *event_description);
    void   *sde_handle = NULL;

    printf("papi_sde_hook_list_events\n");

#ifdef PAPI_DYNAMIC_SDE
    printf("PAPI_DYNAMIC_SDE defined\n");
    fflush(stdout);

    char *path_to_papi = getenv("PAPI_SHARED_LIB");
    if(path_to_papi == NULL)
        return NULL;

    printf("path_to_papi = %s\n", path_to_papi);

    papi_handle = dlopen(path_to_papi, RTLD_LOCAL | RTLD_LAZY);
    if(!papi_handle){
        fputs(dlerror(), stderr);
        exit(1);
    }
    printf("papi_handle opened\n");
    fflush(stdout);

    dlerror();
    sym_init = (void* (*)(char*, int*)) dlsym(papi_handle, "papi_sde_init");
    if((error = dlerror()) != NULL) {
        fputs(error, stderr);
        exit(1);
    }

    sym_reg = (void (*)(void*, char*, long long int*)) dlsym(papi_handle, "papi_sde_register_counter");
    if((error = dlerror()) != NULL){
        fputs(error, stderr);
        exit(1);
    }

    sym_desc = (void (*)(void*, char*, char*)) dlsym(papi_handle, "papi_sde_describe_counter");
    if((error = dlerror()) != NULL){
        fputs(error, stderr);
        exit(1);
    }

    printf("symbols found\n");
    fflush(stdout);

    sde_handle = (void *) (*sym_init)("OMPI", &event_count);
    printf("sde_handle opened\n");
    fflush(stdout);
    if((error = dlerror()) != NULL){
        fputs(error, stderr);
        exit(1);
    }

    printf("sde_handle preparing to register\n");
    fflush(stdout);

    /* We need to register the counters so they can be printed in papi_native_avail
     * Note: sde::: will be prepended to the names
     */
    iter_start();
    for(i = 0; i < OMPI_NUM_COUNTERS; i++){
        printf("Registering: %s (%d of %d)\n", counter_names[i], i+1, OMPI_NUM_COUNTERS);
        (*sym_reg)(sde_handle, counter_names[i], &(events[i].value));
        (*sym_desc)(sde_handle, counter_names[i], counter_descriptions[i]);
        events[i].value = 0;
    }
#endif

    printf("done papi_sde_hook_list_events %s %d\n", __FILE__, __LINE__);
    return sde_handle;
}

/* ##############################################################
 * ############### End of PAPI sde Code #########################
 * ##############################################################
 */

/* ##############################################################
 * ############### Begin Utility Functions ######################
 * ##############################################################
 */

/* Initializes the OMPI software events.  The default functionality is to
 * turn all of the counters on.
 * Note: in the future, turning events on and off should be done through
 *       an MCA parameter.
 */
void ompi_sw_event_init()
{
    int i;

    iter_start();

    /* Turn all counters on */
    for(i = 0; i < OMPI_NUM_COUNTERS; i++){
        attached_event[i] = 1;
    }

    (void)mca_base_pvar_register("ompi", "opal", "software_events", counter_names[OMPI_SEND], counter_descriptions[OMPI_SEND],
                                 OPAL_INFO_LVL_4, MPI_T_PVAR_CLASS_SIZE,
                                 MCA_BASE_VAR_TYPE_UNSIGNED_LONG_LONG, NULL, MPI_T_BIND_NO_OBJECT,
                                 MCA_BASE_PVAR_FLAG_READONLY | MCA_BASE_PVAR_FLAG_CONTINUOUS,
                                 ompi_sw_event_get_send, NULL, ompi_sw_event_notify, NULL);

    /* For initializing the PAPI sde component environment */
    ompi_sde_init();
}

/* Calls iter_release to free all of the OMPI software events data structures */
void ompi_sw_event_fini()
{
    iter_release();
}

/* Records an update to a counter using an atomic add operation. */
void ompi_sw_event_record(unsigned int event_id, long long value)
{
    if(OPAL_UNLIKELY(attached_event[event_id] == 1)){
        OPAL_THREAD_ADD64(&events[event_id].value, value);
    }
}

/* Starts microsecond-precision timer and stores the start value in usec */
void ompi_sw_event_timer_start(unsigned int event_id, opal_timer_t *usec)
{
    /* Check whether usec == 0.0 to make sure the timer hasn't started yet */
    if(OPAL_UNLIKELY(attached_event[event_id] == 1 && *usec == 0)){
        *usec = opal_timer_base_get_usec();
    }
}

/* Stops a microsecond-precision timer and calculates the total elapsed time
 * based on the starting time in usec and putting the result in usec.
 */
void ompi_sw_event_timer_stop(unsigned int event_id, opal_timer_t *usec)
{
    if(OPAL_UNLIKELY(attached_event[event_id] == 1)){
        *usec = opal_timer_base_get_usec() - *usec;
        OPAL_THREAD_ADD64(&events[event_id].value, (long long)*usec);
    }
}

/* A function to output the value of all of the counters.  This is currently
 * implemented in MPI_Finalize, but we need to find a better way for this to
 * happen.
 */
void ompi_sw_event_print_all()
{
    /*int i, j, rank, world_size, offset;
    long long *recv_buffer, *send_buffer;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(rank == 0){
        send_buffer = (long long*)malloc(OMPI_NUM_COUNTERS * sizeof(long long));
        recv_buffer = (long long*)malloc(world_size * OMPI_NUM_COUNTERS * sizeof(long long));
        for(i = 0; i < OMPI_NUM_COUNTERS; i++){
            send_buffer[i] = events[i].value;
        }
        MPI_Gather(send_buffer, OMPI_NUM_COUNTERS, MPI_LONG_LONG, recv_buffer, OMPI_NUM_COUNTERS, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    }
    else{
        send_buffer = (long long*)malloc(OMPI_NUM_COUNTERS * sizeof(long long));
        for(i = 0; i < OMPI_NUM_COUNTERS; i++){
            send_buffer[i] = events[i].value;
        }
        MPI_Gather(send_buffer, OMPI_NUM_COUNTERS, MPI_LONG_LONG, recv_buffer, OMPI_NUM_COUNTERS, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    }

    if(rank == 0){
        fprintf(stdout, "OMPI Software Counters:\n");
        offset = 0;
        for(j = 0; j < world_size; j++){
            fprintf(stdout, "World Rank %d:\n", j);
            for(i = 0; i < OMPI_NUM_COUNTERS; i++){
                fprintf(stdout, "%s\t%lld\n", counter_names[offset+i], events[offset+i].value);
            }
            offset += OMPI_NUM_COUNTERS;
        }
    }*/
}

