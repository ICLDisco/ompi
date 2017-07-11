/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "ompi/mpi/c/bindings.h"
#include "ompi/runtime/params.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/runtime/ompi_software_events.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Finalize = PMPI_Finalize
#endif
#define MPI_Finalize PMPI_Finalize
#endif

static const char FUNC_NAME[] = "MPI_Finalize";


int MPI_Finalize(void)
{
#ifdef SOFTWARE_EVENTS_ENABLE
    int i, j, rank, world_size, offset;
    long long *recv_buffer, *send_buffer;
    char *filename;
    FILE *fptr;

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
        asprintf(&filename, "sw_events_output_XXXXXX");
        filename = mktemp(filename);
        fptr = fopen(filename, "w+");

        fprintf(fptr, "%d %d\n", world_size, OMPI_NUM_COUNTERS);

        fprintf(stdout, "OMPI Software Counters:\n");
        offset = 0;
        for(j = 0; j < world_size; j++){
            fprintf(stdout, "World Rank %d:\n", j);
            fprintf(fptr, "%d\n", j);
            for(i = 0; i < OMPI_NUM_COUNTERS; i++){
                fprintf(stdout, "%s -> %lld\n", events[i].name, recv_buffer[offset+i]);
                fprintf(fptr, "%s %lld\n", events[i].name, recv_buffer[offset+i]);
            }
            fprintf(stdout, "\n");
            offset += OMPI_NUM_COUNTERS;
        }
        free(recv_buffer);
        free(send_buffer);
        fclose(fptr);
    }
    else{
        free(send_buffer);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /*SW_EVENT_PRINT_ALL();*/

    /*SW_EVENT_FINI();*/
#endif

    OPAL_CR_FINALIZE_LIBRARY();

    if (MPI_PARAM_CHECK) {
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
    }

    /* Pretty simple */

    return ompi_mpi_finalize();
}
