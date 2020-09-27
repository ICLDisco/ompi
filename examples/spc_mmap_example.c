/*
 * Copyright (c) 2020 The University of Tennessee and The University
 *                    of Tennessee Research Foundation.  All rights
 *                    reserved.
 *
 * Simple example usage of SPCs through an mmap'd file.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#include <mpi.h>

/* This structure will help us store all of the offsets for each
 * counter that we want to print out.
 */
typedef struct spc_s {
    char name[128];
    int offset;
    int rules_offset;
    int bins_offset;
} spc_t;

int main(int argc, char **argv)
{
    if(argc < 4) {
        printf("Usage: ./spc_mmap_test [num_messages] [message_size] [XML string]\n");
        return -1;
    }

    MPI_Init(NULL, NULL);

    int i, num_messages = atoi(argv[1]), message_size = atoi(argv[2]), rank, shm_fd;
    char *buf = (char*)malloc(message_size * sizeof(char));

    MPI_Request *requests = (MPI_Request*)malloc(num_messages * sizeof(MPI_Request));
    MPI_Status  *statuses = (MPI_Status*)malloc(num_messages * sizeof(MPI_Status));
    MPI_Status  status;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int retval, shm_file_size, num_counters, freq_mhz;
    long long value;
    char filename[128], shm_filename[128], line[128], *token;

    char hostname[128];
    gethostname(hostname, 128);

    char *nodename;
    nodename = strtok(hostname, ".");

    char *xml_string = argv[3];
    snprintf(filename, 128, "/dev/shm/spc_data.%s.%s.%d.xml", nodename, xml_string, rank);

    FILE *fptr = NULL;
    void *data_ptr;
    spc_t *spc_data;

    if(NULL == (fptr = fopen(filename, "r"))) {
        printf("Couldn't open xml file.\n");
        MPI_Finalize();
        return -1;
    } else {
        printf("[%d] Successfully opened the XML file!\n", rank);
    }

    /* The following is to read the formatted XML file to get the basic
     * information we need to read the shared memory file and properly
     * format some counters.
     */
    char tmp_filename[128];
    fgets(line, 128, fptr);
    fgets(line, 128, fptr);

    fgets(line, 128, fptr);
    token = strtok(line, ">");
    token = strtok(NULL, "<");
    sscanf(token, "%s", shm_filename);

    if(rank == 0) {
        printf("shm_filename: %s\n", shm_filename);
    }

    fgets(line, 128, fptr);
    token = strtok(line, ">");
    token = strtok(NULL, "<");
    sscanf(token, "%d", &shm_file_size);
    if(rank == 0) {
        printf("shm_file_size: %d\n", shm_file_size);
    }

    fgets(line, 128, fptr);
    token = strtok(line, ">");
    token = strtok(NULL, "<");
    sscanf(token, "%d", &num_counters);
    if(rank == 0) {
        printf("num_counters: %d\n", num_counters);
    }

    fgets(line, 128, fptr);
    token = strtok(line, ">");
    token = strtok(NULL, "<");
    sscanf(token, "%d", &freq_mhz);
    if(rank == 0) {
        printf("freq_mhz: %d\n", freq_mhz);
    }

    if(-1 == (shm_fd = open(shm_filename, O_RDONLY))){
        printf("\nCould not open file '%s'... Error String: %s\n", shm_filename, strerror(errno));
        return -1;
    } else {
        if(MAP_FAILED == (data_ptr = mmap(0, 8192, PROT_READ, MAP_SHARED, shm_fd, 0))) {
            printf("Map failed :(\n");
            return -1;
        }
        printf("Successfully mmap'd file!\n");
    }

    spc_data = (spc_t*)malloc(num_counters * sizeof(spc_t));

    for(i = 0; i < num_counters; i++) {
        fgets(line, 128, fptr); /* Counter begin header */
        /* This should never happen... */
        if(strcmp(line,"</SPC>\n") == 0) {
            printf("Parsing ended prematurely.  There weren't enough counters.\n");
            break;
        }

        fgets(line, 128, fptr); /* Counter name header */
        token = strtok(line, ">");
        token = strtok(NULL, "<");
        sscanf(token, "%s", spc_data[i].name); /* Counter name */

        fgets(line, 128, fptr); /* Counter value offset header */
        token = strtok(line, ">");
        token = strtok(NULL, "<");
        sscanf(token, "%d", &spc_data[i].offset); /* Counter offset */

        fgets(line, 128, fptr); /* Counter rules offset header */
        token = strtok(line, ">");
        token = strtok(NULL, "<");
        sscanf(token, "%d", &spc_data[i].rules_offset); /* Counter rules offset */

        fgets(line, 128, fptr); /* Counter bins offset header */
        token = strtok(line, ">");
        token = strtok(NULL, "<");
        sscanf(token, "%d", &spc_data[i].bins_offset); /* Counter bins offset */

        fgets(line, 128, fptr); /* Counter end header */
    }

    fclose(fptr);

    /* The following communication pattern is intended to cause a certain
     * number of unexpected messages.
     */
    if(rank==0) {
        for(i=num_messages; i > 0; i--) {
            MPI_Isend(buf, message_size, MPI_BYTE, 1, i, MPI_COMM_WORLD, &requests[i-1]);
        }
        MPI_Send(buf, message_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        MPI_Waitall(num_messages, requests, statuses);

        MPI_Barrier(MPI_COMM_WORLD);

        for(i = 0; i < num_counters; i++) {
            if((0 == strcmp(spc_data[i].name, "OMPI_SPC_MATCH_TIME")) || (0 == strcmp(spc_data[i].name, "OMPI_SPC_MATCH_QUEUE_TIME"))) {
                value = (*((long long*)(data_ptr+spc_data[i].offset))) / freq_mhz;
            } else {
                value = *((long long*)(data_ptr+spc_data[i].offset));
            }
            if(value > 0)
                printf("[%d] %s\t%lld\n", rank, spc_data[i].name, value );
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } else {
        MPI_Recv(buf, message_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
        for(i=0; i < num_messages; i++) {
            MPI_Recv(buf, message_size, MPI_BYTE, 0, i+1, MPI_COMM_WORLD, &statuses[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        for(i = 0; i < num_counters; i++) {
            /* These counters are stored in cycles, so we convert them to microseconds.
             */
            if((0 == strcmp(spc_data[i].name, "OMPI_SPC_MATCH_TIME")) || (0 == strcmp(spc_data[i].name, "OMPI_SPC_MATCH_QUEUE_TIME"))) {
                value = (*((long long*)(data_ptr+spc_data[i].offset))) / freq_mhz;
            } else {
                value = *((long long*)(data_ptr+spc_data[i].offset));
            }
            if(value > 0) {
                printf("[%d] %s\t%lld\n", rank, spc_data[i].name, value );
                if(spc_data[i].rules_offset > 0) {
                    int j, *rules = (int*)(data_ptr+spc_data[i].rules_offset);
                    long long *bins = (long long*)(data_ptr+spc_data[i].bins_offset);

                    for(j = 0; j < rules[0]; j++) {
                        if(j == rules[0]-1) {
                            printf("\t>  %d\t", rules[j]);
                        }
                        else {
                            printf("\t<= %d\t", rules[j+1]);
                        }
                        printf("%lld\n", bins[j]);
                    }
                }
            }
        }
    }

    MPI_Finalize();

    return 0;
}
