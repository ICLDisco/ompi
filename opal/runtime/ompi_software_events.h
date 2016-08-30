#ifndef OMPI_SOFTWARE_EVENT
#define OMPI_SOFTWARE_EVENT

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <mpi.h>
#include "ompi/include/ompi_config.h"
#include "opal/mca/timer/timer.h"
#include "opal/mca/base/mca_base_pvar.h"

#include MCA_timer_IMPLEMENTATION_HEADER

/* This enumeration serves as event ids for the various events */
enum OMPI_COUNTERS{
    OMPI_SEND,
    OMPI_RECV,
    OMPI_ISEND,
    OMPI_IRECV,
    OMPI_BCAST,
    OMPI_REDUCE,
    OMPI_ALLREDUCE,
    OMPI_SCATTER,
    OMPI_GATHER,
    OMPI_ALLTOALL,
    OMPI_ALLGATHER,
    OMPI_BYTES_RECEIVED_USER,
    OMPI_BYTES_RECEIVED_MPI,
    OMPI_BYTES_SENT_USER,
    OMPI_BYTES_SENT_MPI,
    OMPI_BYTES_PUT,
    OMPI_BYTES_GET,
    OMPI_UNEXPECTED,
    OMPI_OUT_OF_SEQUENCE,
    OMPI_MATCH_TIME,
    OMPI_OOS_MATCH_TIME,
    OMPI_PROGRESS_SWITCH,
    OMPI_NUM_COUNTERS
};

/* A structure for storing the event data */
typedef struct ompi_event_s{
    char *name;
    long long value;
} ompi_event_t;

/* Structure and helper functions for PAPI software_events component
 * Note: This component is being superceded by the sde component.
 */
struct PAPI_SOFTWARE_EVENT_S{
    char  name[32];
    int   version[3];
    int   (*iter_start)(void);
    char* (*iter_next)(void);
    int   (*iter_release)(void);
    int   (*attach_event)(char*, long long**);
    int   (*detach_event)(char*);
};

int iter_start(void);
char* iter_next(void);
int iter_release(void);
int attach_event(char *name, long long **value);
int detach_event(char *name);

/* End of PAPI software_events component stuff */

OMPI_DECLSPEC extern unsigned int attached_event[OMPI_NUM_COUNTERS];
OMPI_DECLSPEC extern ompi_event_t *events;

/* OMPI software event utility functions */
void ompi_sw_event_init(void);
void ompi_sw_event_fini(void);
void ompi_sw_event_record(unsigned int event_id, long long value);
void ompi_sw_event_timer_start(unsigned int event_id, opal_timer_t *usec);
void ompi_sw_event_timer_stop(unsigned int event_id, opal_timer_t *usec);
void ompi_sw_event_print_all(void);

/* MPI_T utility functions */
static int ompi_sw_event_notify(mca_base_pvar_t *pvar, mca_base_pvar_event_t event, void *obj_handle, int *count);
long long ompi_sw_event_get_counter(int counter_id);
static int ompi_sw_event_get_send(const struct mca_base_pvar_t *pvar, void *value, void *obj_handle);

/* Functions for the PAPI sde component */
void ompi_sde_init(void);
/* PAPI sde component interface functions */
typedef void* papi_handle_t;

/* This should be defined at build time through an MCA parameter */
#define SOFTWARE_EVENTS_ENABLE

/* Macros for using the utility functions throughout the codebase.
 * If SOFTWARE_EVENTS_ENABLE is not defined, the macros become no-ops.
 */
#ifdef SOFTWARE_EVENTS_ENABLE

#define SW_EVENT_INIT()  \
    ompi_sw_event_init()

#define SW_EVENT_FINI()  \
    ompi_sw_event_fini()

#define SW_EVENT_RECORD(event_id, value)  \
    ompi_sw_event_record(event_id, value)

#define SW_EVENT_TIMER_START(event_id, usec)  \
    ompi_sw_event_timer_start(event_id, usec)

#define SW_EVENT_TIMER_STOP(event_id, usec)  \
    ompi_sw_event_timer_stop(event_id, usec)

#define SW_EVENT_PRINT_ALL() \
    ompi_sw_event_print_all()

#else /* Software events are not enabled */

#define SW_EVENT_INIT()  \
    do {} while (0)

#define SW_EVENT_FINI()  \
    do {} while (0)

#define SW_EVENT_RECORD(event_id, value)  \
    do {} while (0)

#define SW_EVENT_TIMER_START(event_id, usec)  \
    do {} while (0)

#define SW_EVENT_TIMER_STOP(event_id, usec)  \
    do {} while (0)

#define SW_EVENT_PRINT_ALL() \
    do {} while (0)

#endif

#endif
