/*
 * Copyright (c) 2018-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * Simple example usage of SPCs through MPI_T.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "mpi.h"

/* Sends 'num_messages' messages of 'message_size' bytes from rank 0 to rank 1.
 * All messages are send synchronously and with the same tag in MPI_COMM_WORLD.
 */
void message_exchange(int num_messages, int message_size)
{
    int i, rank;
    /* Use calloc to initialize data to 0's */
    char *data = (char*)calloc(message_size, sizeof(char));
    MPI_Status status;
    MPI_Request req;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* This is designed to have at least num_messages unexpected messages in order to
     * hit the unexpected message queue counters.  The broadcasts are here to showcase
     * the collective bin counters and the P2P and Eager message counters.
     */
    if(rank == 0) {
        for(i = 0; i < num_messages; i++) {
            MPI_Isend(data, message_size, MPI_BYTE, 1, 123, MPI_COMM_WORLD, &req);
        }
        MPI_Send(data, message_size, MPI_BYTE, 1, 321, MPI_COMM_WORLD);
        for(i = 0; i < num_messages; i++) {
            MPI_Bcast(data, message_size, MPI_BYTE, 0, MPI_COMM_WORLD);
        }
    } else if(rank == 1) {
        MPI_Recv(data, message_size, MPI_BYTE, 0, 321, MPI_COMM_WORLD, &status);
        for(i = 0; i < num_messages; i++) {
            MPI_Recv(data, message_size, MPI_BYTE, 0, 123, MPI_COMM_WORLD, &status);
        }
        for(i = 0; i < num_messages; i++) {
            MPI_Bcast(data, message_size, MPI_BYTE, 0, MPI_COMM_WORLD);
        }
    }
    /* This should use the binomial algorithm so it has at least one counter value */
    MPI_Bcast(data, 1, MPI_BYTE, 0, MPI_COMM_WORLD);

    free(data);
}

int main(int argc, char **argv)
{
    int num_messages, message_size, rc;

    if(argc < 3) {
        printf("Usage: mpirun -np 2 --mca mpi_spc_attach all --mca mpi_spc_dump_enabled true ./spc_example [num_messages] [message_size]\n");
        return -1;
    } else {
        num_messages = atoi(argv[1]);
        message_size = atoi(argv[2]);
        if(message_size <= 0) {
            printf("Message size must be positive.\n");
            return -1;
        }
    }

    int i, j, rank, size, provided, num, name_len, desc_len, verbosity, bind, var_class, readonly, continuous, atomic, count, index, xml_index;
    MPI_Datatype datatype;
    MPI_T_enum enumtype;
    MPI_Comm comm;
    char name[256], description[256];

    /* Counter names to be read by ranks 0 and 1 */
    char *counter_names[] = {"runtime_spc_OMPI_SPC_BASE_BCAST_BINOMIAL",
                             "runtime_spc_OMPI_SPC_BYTES_RECEIVED_USER" };
    char *xml_counter = "runtime_spc_OMPI_SPC_XML_FILE";

    MPI_Init(NULL, NULL);
    MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size != 2) {
        fprintf(stderr, "ERROR: This test should be run with two MPI processes.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if(rank == 0) {
        printf("##################################################################\n");
        printf("This test is designed to highlight several different SPC counters.\n");
        printf("The MPI workload of this test will use 1 MPI_Send and %d MPI_Isend\n", num_messages);
        printf("operation(s) on the sender side (rank 0) and %d MPI_Recv operation(s)\n", num_messages+1);
        printf("on the receiver side (rank 1) in such a way that at least %d message(s)\n", num_messages);
        printf("are unexpected.  This highlights the unexpected message queue SPCs.\n");
        printf("There will also be %d MPI_Bcast operation(s) with one of them being of\n", num_messages+1);
        printf("size 1 byte, and %d being of size %d byte(s).  The 1 byte MPI_Bcast is\n", num_messages, message_size);
        printf("meant to ensure that there is at least one MPI_Bcast that uses the\n");
        printf("binomial algorithm so the MPI_T pvar isn't all 0's.  The addition of\n");
        printf("the broadcasts also has the effect of showcasing the P2P message size,\n");
        printf("eager vs not eager message, and bytes sent by the user vs MPI SPCs.\n");
        printf("Be sure to set the mpi_spc_dump_enabled MCA parameter to true in order\n");
        printf("to see all of the tracked SPCs.\n");
        printf("##################################################################\n\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Determine the MPI_T pvar indices for the requested SPCs */
    index = xml_index = -1;
    MPI_T_pvar_get_num(&num);
    for(i = 0; i < num; i++) {
        name_len = desc_len = 256;
        rc = PMPI_T_pvar_get_info(i, name, &name_len, &verbosity,
                                  &var_class, &datatype, &enumtype, description, &desc_len, &bind,
                                  &readonly, &continuous, &atomic);
        if( MPI_SUCCESS != rc )
            continue;

        if(strcmp(name, xml_counter) == 0) {
            xml_index = i;
            printf("[%d] %s -> %s\n", rank, name, description);
        }
        if(strcmp(name, counter_names[rank]) == 0) {
            index = i;
            printf("[%d] %s -> %s\n", rank, name, description);
        }
    }

    /* Make sure we found the counters */
    if(index == -1 || xml_index == -1) {
        fprintf(stderr, "ERROR: Couldn't find the appropriate SPC counter in the MPI_T pvars.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int ret, xml_count;
    long long *values = NULL;
    char *xml_filename = (char*)malloc(128 * sizeof(char));

    MPI_T_pvar_session session;
    MPI_T_pvar_handle handle;
    /* Create the MPI_T sessions/handles for the counters and start the counters */
    ret = MPI_T_pvar_session_create(&session);
    ret = MPI_T_pvar_handle_alloc(session, index, NULL, &handle, &count);
    ret = MPI_T_pvar_start(session, handle);

    values = (long long*)malloc(count * sizeof(long long));

    MPI_T_pvar_session xml_session;
    MPI_T_pvar_handle xml_handle;
    if(xml_index >= 0) {
        ret = MPI_T_pvar_session_create(&xml_session);
        ret = MPI_T_pvar_handle_alloc(xml_session, xml_index, NULL, &xml_handle, &xml_count);
        ret = MPI_T_pvar_start(xml_session, xml_handle);
    }

    double timer = MPI_Wtime();
    message_exchange(num_messages, message_size);
    timer = MPI_Wtime() - timer;

    printf("[%d] Elapsed time: %lf seconds\n", rank, timer);

    ret = MPI_T_pvar_read(session, handle, values);
    if(xml_index >= 0) {
        ret = MPI_T_pvar_read(xml_session, xml_handle, &xml_filename);
    }

    /* Print the counter values in order by rank */
    for(i = 0; i < 2; i++) {
        printf("\n");
        if(i == rank) {
            if(xml_index >= 0) {
                printf("[%d] XML Counter Value Read: %s\n", rank, xml_filename);
            }
            for(j = 0; j < count; j++) {
                printf("[%d] %s Counter Value Read: %lld\n", rank, counter_names[rank], values[j]);
            }
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    /* Stop the MPI_T session, free the handle, and then free the session */
    ret = MPI_T_pvar_stop(session, handle);
    ret = MPI_T_pvar_handle_free(session, &handle);
    ret = MPI_T_pvar_session_free(&session);

    MPI_T_finalize();
    MPI_Finalize();

    return 0;
}
