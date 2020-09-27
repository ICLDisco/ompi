## What Are Software Performance Counters?
Software Performance Counters (SPCs) are modelled after [PAPI's](http://icl.utk.edu/papi/) hardware performance counters, but are designed to keep track of information that originates from software rather than hardware.  The basic idea is to add instrumentation to a software package to keep track of performance metrics of interest and provide access to them in a portable way.  The implementation of SPCs within Open MPI exposes information about the internal operation of Open MPI that would otherwise not be available to users and tool developers.  These counters have been integrated with the MPI Tool Information Interface (MPI_T) as performance variables, or pvars, to make this information more readily available to tool developers and users.  These counters can also be accessed through an mmap-based interface at runtime.

## Building OMPI With SPC Support
By default, Open MPI does not build with SPCs enabled, so all of the instrumentation code becomes no-ops.  Building Open MPI with SPCs is as simple as adding `--enable-spc` to your configure line.  Once SPCs have been built in, there are several MCA parameters that are used to manage SPCs at runtime:

- mpi_spc_attach: A string used to turn specific counters on.  There are two reserved strings, 'all' and 'none', for turning on all and none of the counters respectively.  Otherwise, this should be a comma-separated list of counters to turn on using the counters names.  The default value is 'none'.
- mpi_spc_dump_enabled: A boolean parameter denoted by a 'true' or 'false' string that determines whether or not to print the counter values to stdout during MPI_Finalize.  The default value is 'false'
- mpi_spc_mmap_enabled: A boolean parameter denoted by a 'true' or 'false' string that determines whether or not to use the mmap interface for storing the SPC data.  The default value is 'false'
- mpi_spc_xml_string: A string that is appended to the XML file created by the mmap interface for easy identification.  For example, if this variable is set to the value of 'test', the resultant XML filename would be spc_data.[nodename].test.[world rank].xml.  By default this string is empty, which results in the 'test' value being replaced by the Open MPI jobid.
- mpi_spc_p2p_message boundary: An integer value denoting the point after which messages are determined to be 'large' messages within OMPI_SPC_P2P_MESSAGE_SIZE bin counter.  The default value is 12288.
- mpi_spc_collective_message_boundary: An integer value denoting the point after which messages are determined to be 'large' messages for collective bin counters.  The default value is 12288.
- mpi_spc_collective_comm_boundary: An integer value denoting the point after which a communicator is determined to be 'large' for collective bin counters.  The default value is 64.

Setting these MCA parameters can be done via the command line like so:

`mpirun -np X --mca mpi_spc_attach OMPI_SPC_SEND,OMPI_SPC_RECV --mca mpi_spc_dump_enabled true ./your_app`

These MCA parameters can also be set inside an mca-params.conf file like so: 

```bash
mpi_spc_attach = all
mpi_spc_dump_enabled = true
mpi_spc_mmap_enabled = true
mpi_spc_xml_string = myXMLstring
```

## Using SPCs Through MPI_T
All of the SPC counters are registered with MPI_T as pvars, which means that they can be easily accessed by tools.  It is worth noting that if any of the SPCs fail to register with MPI_T, all counters are turned off with respect to MPI_T.  This is a design decision to create a fast translation between MPI_T indices and SPC indices.

The following is a simple example C program that will show how these counters could be used through MPI_T in practice.  Essentially, this example sends some number of messages of some size both specified by the user from process rank 0 to rank 1.  Rank 0 uses an MPI_T pvar to report the number of times the binomial algorithm was used for a broadcast under different conditions Small/Large communicator/message size, and rank 1 registers an MPI_T pvar to determine the number of bytes received through point to point communications.  This example can also be found in the examples directory of the Open MPI repository.

```c
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
```

### Using SPCs Through the mmap Interface

There is another interface other than MPI_T for accessing SPCs.  In order to use this interface, you need to set the MCA parameter `ompi_mpi_spc_mmap_enabled` to true.  This method uses mmap to create a memory region in which you can directly access SPCs after using mmap on a certain file.  All the information necessary to attach to the correct file is dumped in an XML file for each rank in a shared system location (the default is /dev/shm) with the name `spc_data.[nodename].[SPC XML String or Open MPI jobid].[world rank].xml`.  This XML file will have the format shown in the example below with the filename and file size for performing the mmap, and the number of counters and the clock frequency in MHz for ease of parsing (the clock frequency is needed for converting timer counters to microseconds from cycles).  Each counter tag will have a name, and three offsets.  The three offsets denote how many bytes that information is offset into the counter data.  All counters have a value field, but only bin counters have an offset for the rules and bins (If the counter doesn't have bins, the values in these fields will be -1).

#### XML File Example
```xml
<?xml version="1.0"?>
<SPC>
        <filename>/dev/shm/spc_data.c00.-860356607.0</filename>
        <file_size>5248</file_size>
        <num_counters>164</num_counters>
        <freq_mhz>2127</freq_mhz>
        <counter>
                <name>OMPI_SPC_SEND</name>
                <value_offset>0</value_offset>
                <rules_offset>-1</rules_offset>
                <bin_offset>-1</bin_offset>
        </counter>
        <counter>
                <name>OMPI_SPC_BSEND</name>
                <value_offset>8</value_offset>
                <rules_offset>-1</rules_offset>
                <bin_offset>-1</bin_offset>
        </counter>
        <counter>
                ...
        </counter>
        ...
</SPC>
```

#### mmap Interface Usage in C
```c
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
```

#### Snapshot Feature in the mmap Interface
NOTE: This feature is currently in the process of being moved to an external tool, and will be unavailable until then.
The mmap interface also allows for collecting snapshots of the SPC counter values periodically throughout an execution through a snapshot feature.  These snapshots use an MCA parameter to determine the length of time after which to create a copy of the SPC data file from the mmap interface.  The snapshot data file copies simply append a timestamp to the end of the mmap data file to keep track of when that snapshot was taken.  These snapshot files can be used to show how counter values change over time.

The following is an example python script that takes the values from these snapshot files and creates heatmaps of the change in the counter values over time.  This example script takes three command line arguments: a directory where all of the snapshot, XML, and original data files are stored; the XML string or Open MPI jobid to identify these data and XML files; a comma-separated list of SPCs to be used in creating the heatmaps.

```python
import sys
import glob
import operator
import struct

import numpy as np
import matplotlib
matplotlib.use('Agg') # For use with headless systems
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker

def combine(filename, data):
    f = open(filename, 'rb')
    for i in range(0,num_counters):
        temp = struct.unpack('l', f.read(8))[0]
        if 'TIME' in names[i]:
            temp /= freq_mhz
        data[i].append(temp)

def fmt(x, pos):
    return '{:,.0f}'.format(x)

# Make sure the proper number of arguments have been supplied
if len(sys.argv) < 4:
    print("Usage: ./spc_snapshot_parse.py [/path/to/data/files] [datafile_label] [list_of_spcs]")
    exit()

path = sys.argv[1]
label = sys.argv[2]

xml_filename = ''
# Lists for storing the snapshot data files from each rank
copies = []
ends = []
# Populate the lists with the appropriate data files
for filename in glob.glob(path + "/spc_data*"):
    if label in filename:
        if xml_filename == '' and '.xml' in filename:
            xml_filename = filename
        if '.xml' not in filename:
            temp = filename.split('/')[-1].split('.')
            if len(temp) < 5:
                temp[-1] = int(temp[-1])
                ends.append(temp)
            else:
                temp[-1] = int(temp[-1])
                temp[-2] = int(temp[-2])
                copies.append(temp)

# Sort the lists
ends = sorted(ends, key = operator.itemgetter(-1))
for i in range(0,len(ends)):
    ends[i][-1] = str(ends[i][-1])
copies = sorted(copies, key = operator.itemgetter(-2,-1))
for i in range(0,len(copies)):
    copies[i][-1] = str(copies[i][-1])
    copies[i][-2] = str(copies[i][-2])

sep = '.'

xml_file = open(xml_filename, 'r')
num_counters = 0
freq_mhz = 0
names = []
base = []
# Parse the XML file (same for all data files)
for line in xml_file:
    if 'num_counters' in line:
        num_counters = int(line.split('>')[1].split('<')[0])
    if 'freq_mhz' in line:
        freq_mhz = int(line.split('>')[1].split('<')[0])
    if '<name>' in line:
        names.append(line.split('>')[1].split('<')[0])
        value = [names[-1]]
        base.append(value)

prev = copies[0]
i = 0
ranks = []
values = []
times = []
time = []

# Populate the data lists
for n in range(0,len(base)):
    values.append([0, names[n]])
for c in copies:
    if c[-2] != prev[-2]:
        filename = path + "/" + sep.join(ends[i])
        combine(filename, values)

        ranks.append(values)
        times.append(time)
        for j in range(0, len(names)):
            temp = [ranks[0][j][0]]

        values = []
        time = []
        for n in range(0,len(base)):
            values.append([i+1, names[n]])
        i += 1

    filename = path + "/" + sep.join(c)
    time.append(int(filename.split('.')[-1]))
    combine(filename, values)
    prev = c

filename = path + "/" + sep.join(ends[i])
combine(filename, values)
ranks.append(values)
times.append(time)

spc_list = sys.argv[3].split(",")

for i in range(0, len(names)):
    fig = plt.figure(num=None, figsize=(7, 9), dpi=200, facecolor='w', edgecolor='k')

    plot = False
    # Only plot the SPCs of interest
    if names[i] in spc_list:
        plot = True

    map_data = []
    avg_x = []

    for j in range(0, len(ranks)):
        if avg_x == None:
            avg_x = np.zeros(len(times[j])-1)
        empty = True
        for k in range(2,len(ranks[j][i])):
            if ranks[j][i][k] != 0:
                empty = False
                break
        if not empty:
            if plot:
                xvals = []
                yvals = []
                for l in range(1, len(times[j])):
                    if ranks[j][i][l+2] - ranks[j][i][l+1] < 0:
                        break
                    xvals.append(times[j][l] - times[j][0])
                    yvals.append(ranks[j][i][l+2] - ranks[j][i][l+1])

                map_data.append(yvals)
                for v in range(0,len(avg_x)):
                    avg_x[v] += xvals[v]
    if plot:
        for v in range(0,len(avg_x)):
            avg_x[v] /= float(len(ranks))

        ax = plt.gca()
        im = ax.imshow(map_data, cmap='Reds', interpolation='nearest')

        cbar = ax.figure.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt))
        cbar.ax.set_ylabel("Counter Value", rotation=-90, va="bottom")

        plt.title(names[i] + ' Snapshot Difference')

        plt.xlabel('Time')
        plt.ylabel('MPI Rank')

        ax.set_xticks(np.arange(len(avg_x)))
        ax.set_yticks(np.arange(len(map_data)))
        ax.set_xticklabels(avg_x)

        plt.show()
        fig.savefig(names[i] + '.png')
```

### List of Counters

|Name |Description|
| --- | --------- |
|OMPI_SPC_SEND|The number of times MPI_Send was called.|
|OMPI_SPC_BSEND|The number of times MPI_Bsend was called.|
|OMPI_SPC_RSEND|The number of times MPI_Rsend was called.|
|OMPI_SPC_SSEND|The number of times MPI_Ssend was called.|
|OMPI_SPC_RECV|The number of times MPI_Recv was called.|
|OMPI_SPC_MRECV|The number of times MPI_Mrecv was called.|
|OMPI_SPC_ISEND|The number of times MPI_Isend was called.|
|OMPI_SPC_IBSEND|The number of times MPI_Ibsend was called.|
|OMPI_SPC_IRSEND|The number of times MPI_Irsend was called.|
|OMPI_SPC_ISSEND|The number of times MPI_Issend was called.|
|OMPI_SPC_IRECV|The number of times MPI_Irecv was called.|
|OMPI_SPC_SENDRECV|The number of times MPI_Sendrecv was called.|
|OMPI_SPC_SENDRECV_REPLACE|The number of times MPI_Sendrecv_replace was called.|
|OMPI_SPC_PUT|The number of times MPI_Put was called.|
|OMPI_SPC_RPUT|The number of times MPI_Rput was called.|
|OMPI_SPC_GET|The number of times MPI_Get was called.|
|OMPI_SPC_RGET|The number of times MPI_Rget was called.|
|OMPI_SPC_PROBE|The number of times MPI_Probe was called.|
|OMPI_SPC_IPROBE|The number of times MPI_Iprobe was called.|
|OMPI_SPC_BCAST|The number of times MPI_Bcast was called.|
|OMPI_SPC_IBCAST|The number of times MPI_Ibcast was called.|
|OMPI_SPC_BCAST_INIT|The number of times MPIX_Bcast_init was called.|
|OMPI_SPC_REDUCE|The number of times MPI_Reduce was called.|
|OMPI_SPC_REDUCE_SCATTER|The number of times MPI_Reduce_scatter was called.|
|OMPI_SPC_REDUCE_SCATTER_BLOCK|The number of times MPI_Reduce_scatter_block was called.|
|OMPI_SPC_IREDUCE|The number of times MPI_Ireduce was called.|
|OMPI_SPC_IREDUCE_SCATTER|The number of times MPI_Ireduce_scatter was called.|
|OMPI_SPC_IREDUCE_SCATTER_BLOCK|The number of times MPI_Ireduce_scatter_block was called.|
|OMPI_SPC_REDUCE_INIT|The number of times MPIX_Reduce_init was called.|
|OMPI_SPC_REDUCE_SCATTER_INIT|The number of times MPIX_Reduce_scatter_init was called.|
|OMPI_SPC_REDUCE_SCATTER_BLOCK_INIT|The number of times MPIX_Reduce_scatter_block_init was called.|
|OMPI_SPC_ALLREDUCE|The number of times MPI_Allreduce was called.|
|OMPI_SPC_IALLREDUCE|The number of times MPI_Iallreduce was called.|
|OMPI_SPC_ALLREDUCE_INIT|The number of times MPIX_Allreduce_init was called.|
|OMPI_SPC_SCAN|The number of times MPI_Scan was called.|
|OMPI_SPC_EXSCAN|The number of times MPI_Exscan was called.|
|OMPI_SPC_ISCAN|The number of times MPI_Iscan was called.|
|OMPI_SPC_IEXSCAN|The number of times MPI_Iexscan was called.|
|OMPI_SPC_SCAN_INIT|The number of times MPIX_Scan_init was called.|
|OMPI_SPC_EXSCAN_INIT|The number of times MPIX_Exscan_init was called.|
|OMPI_SPC_SCATTER|The number of times MPI_Scatter was called.|
|OMPI_SPC_SCATTERV|The number of times MPI_Scatterv was called.|
|OMPI_SPC_ISCATTER|The number of times MPI_Iscatter was called.|
|OMPI_SPC_ISCATTERV|The number of times MPI_Iscatterv was called.|
|OMPI_SPC_SCATTER_INIT|The number of times MPIX_Scatter_init was called.|
|OMPI_SPC_SCATTERV_INIT|The number of times MPIX_Scatterv_init was called.|
|OMPI_SPC_GATHER|The number of times MPI_Gather was called.|
|OMPI_SPC_GATHERV|The number of times MPI_Gatherv was called.|
|OMPI_SPC_IGATHER|The number of times MPI_Igather was called.|
|OMPI_SPC_IGATHERV|The number of times MPI_Igatherv was called.|
|OMPI_SPC_GATHER_INIT|The number of times MPIX_Gather_init was called.|
|OMPI_SPC_GATHERV_INIT|The number of times MPIX_Gatherv_init was called.|
|OMPI_SPC_ALLTOALL|The number of times MPI_Alltoall was called.|
|OMPI_SPC_ALLTOALLV|The number of times MPI_Alltoallv was called.|
|OMPI_SPC_ALLTOALLW|The number of times MPI_Alltoallw was called.|
|OMPI_SPC_IALLTOALL|The number of times MPI_Ialltoall was called.|
|OMPI_SPC_IALLTOALLV|The number of times MPI_Ialltoallv was called.|
|OMPI_SPC_IALLTOALLW|The number of times MPI_Ialltoallw was called.|
|OMPI_SPC_ALLTOALL_INIT|The number of times MPIX_Alltoall_init was called.|
|OMPI_SPC_ALLTOALLV_INIT|The number of times MPIX_Alltoallv_init was called.|
|OMPI_SPC_ALLTOALLW_INIT|The number of times MPIX_Alltoallw_init was called.|
|OMPI_SPC_NEIGHBOR_ALLTOALL|The number of times MPI_Neighbor_alltoall was called.|
|OMPI_SPC_NEIGHBOR_ALLTOALLV|The number of times MPI_Neighbor_alltoallv was called.|
|OMPI_SPC_NEIGHBOR_ALLTOALLW|The number of times MPI_Neighbor_alltoallw was called.|
|OMPI_SPC_INEIGHBOR_ALLTOALL|The number of times MPI_Ineighbor_alltoall was called.|
|OMPI_SPC_INEIGHBOR_ALLTOALLV|The number of times MPI_Ineighbor_alltoallv was called.|
|OMPI_SPC_INEIGHBOR_ALLTOALLW|The number of times MPI_Ineighbor_alltoallw was called.|
|OMPI_SPC_NEIGHBOR_ALLTOALL_INIT|The number of times MPIX_Neighbor_alltoall_init was called.|
|OMPI_SPC_NEIGHBOR_ALLTOALLV_INIT|The number of times MPIX_Neighbor_alltoallv_init was called.|
|OMPI_SPC_NEIGHBOR_ALLTOALLW_INIT|The number of times MPIX_Neighbor_alltoallw_init was called.|
|OMPI_SPC_ALLGATHER|The number of times MPI_Allgather was called.|
|OMPI_SPC_ALLGATHERV|The number of times MPI_Allgatherv was called.|
|OMPI_SPC_IALLGATHER|The number of times MPI_Iallgather was called.|
|OMPI_SPC_IALLGATHERV|The number of times MPI_Iallgatherv was called.|
|OMPI_SPC_ALLGATHER_INIT|The number of times MPIX_Allgather_init was called.|
|OMPI_SPC_ALLGATHERV_INIT|The number of times MPIX_Allgatherv_init was called.|
|OMPI_SPC_NEIGHBOR_ALLGATHER|The number of times MPI_Neighbor_allgather was called.|
|OMPI_SPC_NEIGHBOR_ALLGATHERV|The number of times MPI_Neighbor_allgatherv was called.|
|OMPI_SPC_INEIGHBOR_ALLGATHER|The number of times MPI_Ineighbor_allgather was called.|
|OMPI_SPC_INEIGHBOR_ALLGATHERV|The number of times MPI_Ineighbor_allgatherv was called.|
|OMPI_SPC_NEIGHBOR_ALLGATHER_INIT|The number of times MPIX_Neighbor_allgather_init was called.|
|OMPI_SPC_NEIGHBOR_ALLGATHERV_INIT|The number of times MPIX_Neighbor_allgatherv_init was called.|
|OMPI_SPC_TEST|The number of times MPI_Test was called.|
|OMPI_SPC_TESTALL|The number of times MPI_Testall was called.|
|OMPI_SPC_TESTANY|The number of times MPI_Testany was called.|
|OMPI_SPC_TESTSOME|The number of times MPI_Testsome was called.|
|OMPI_SPC_WAIT|The number of times MPI_Wait was called.|
|OMPI_SPC_WAITALL|The number of times MPI_Waitall was called.|
|OMPI_SPC_WAITANY|The number of times MPI_Waitany was called.|
|OMPI_SPC_WAITSOME|The number of times MPI_Waitsome was called.|
|OMPI_SPC_BARRIER|The number of times MPI_Barrier was called.|
|OMPI_SPC_IBARRIER|The number of times MPI_Ibarrier was called.|
|OMPI_SPC_BARRIER_INIT|The number of times MPIX_Barrier_init was called.|
|OMPI_SPC_WTIME|The number of times MPI_Wtime was called.|
|OMPI_SPC_CANCEL|The number of times MPI_Cancel was called.|
|OMPI_SPC_BYTES_RECEIVED_USER|The number of bytes received by the user through point-to-point communications. Note: Includes bytes transferred using internal RMA operations.|
|OMPI_SPC_BYTES_RECEIVED_MPI|The number of bytes received by MPI through collective|
|OMPI_SPC_BYTES_SENT_USER|The number of bytes sent by the user through point-to-point communications.  Note: Includes bytes transferred using internal RMA operations.|
|OMPI_SPC_BYTES_SENT_MPI|The number of bytes sent by MPI through collective|
|OMPI_SPC_BYTES_PUT|The number of bytes sent/received using RMA Put operations both through user-level Put functions and internal Put functions.|
|OMPI_SPC_BYTES_GET|The number of bytes sent/received using RMA Get operations both through user-level Get functions and internal Get functions.|
|OMPI_SPC_UNEXPECTED|The number of messages that arrived as unexpected messages.|
|OMPI_SPC_OUT_OF_SEQUENCE|The number of messages that arrived out of the proper sequence.|
|OMPI_SPC_OOS_QUEUE_HOPS|The number of times we jumped to the next element in the out of sequence message queue's ordered list.|
|OMPI_SPC_MATCH_TIME|The amount of time |
|OMPI_SPC_MATCH_QUEUE_TIME|The amount of time |
|OMPI_SPC_OOS_MATCH_TIME|The amount of time |
|OMPI_SPC_OOS_MATCH_QUEUE_TIME|The amount of time |
|OMPI_SPC_UNEXPECTED_IN_QUEUE|The number of messages that are currently in the unexpected message queue|
|OMPI_SPC_OOS_IN_QUEUE|The number of messages that are currently in the out of sequence message queue|
|OMPI_SPC_MAX_UNEXPECTED_IN_QUEUE|The maximum number of messages that the unexpected message queue|
|OMPI_SPC_MAX_OOS_IN_QUEUE|The maximum number of messages that the out of sequence message queue|
|OMPI_SPC_BASE_BCAST_LINEAR|The number of times the base broadcast used the linear algorithm.|
|OMPI_SPC_BASE_BCAST_CHAIN|The number of times the base broadcast used the chain algorithm.|
|OMPI_SPC_BASE_BCAST_PIPELINE|The number of times the base broadcast used the pipeline algorithm.|
|OMPI_SPC_BASE_BCAST_SPLIT_BINTREE|The number of times the base broadcast used the split binary tree algorithm.|
|OMPI_SPC_BASE_BCAST_BINTREE|The number of times the base broadcast used the binary tree algorithm.|
|OMPI_SPC_BASE_BCAST_BINOMIAL|The number of times the base broadcast used the binomial algorithm.|
|OMPI_SPC_BASE_REDUCE_CHAIN|The number of times the base reduce used the chain algorithm.|
|OMPI_SPC_BASE_REDUCE_PIPELINE|The number of times the base reduce used the pipeline algorithm.|
|OMPI_SPC_BASE_REDUCE_BINARY|The number of times the base reduce used the binary tree algorithm.|
|OMPI_SPC_BASE_REDUCE_BINOMIAL|The number of times the base reduce used the binomial tree algorithm.|
|OMPI_SPC_BASE_REDUCE_IN_ORDER_BINTREE|The number of times the base reduce used the in order binary tree algorithm.|
|OMPI_SPC_BASE_REDUCE_LINEAR|The number of times the base reduce used the basic linear algorithm.|
|OMPI_SPC_BASE_REDUCE_SCATTER_NONOVERLAPPING|The number of times the base reduce scatter used the nonoverlapping algorithm.|
|OMPI_SPC_BASE_REDUCE_SCATTER_RECURSIVE_HALVING|The number of times the base reduce scatter used the recursive halving algorithm.|
|OMPI_SPC_BASE_REDUCE_SCATTER_RING|The number of times the base reduce scatter used the ring algorithm.|
|OMPI_SPC_BASE_ALLREDUCE_NONOVERLAPPING|The number of times the base allreduce used the nonoverlapping algorithm.|
|OMPI_SPC_BASE_ALLREDUCE_RECURSIVE_DOUBLING|The number of times the base allreduce used the recursive doubling algorithm.|
|OMPI_SPC_BASE_ALLREDUCE_RING|The number of times the base allreduce used the ring algorithm.|
|OMPI_SPC_BASE_ALLREDUCE_RING_SEGMENTED|The number of times the base allreduce used the segmented ring algorithm.|
|OMPI_SPC_BASE_ALLREDUCE_LINEAR|The number of times the base allreduce used the linear algorithm.|
|OMPI_SPC_BASE_SCATTER_BINOMIAL|The number of times the base scatter used the binomial tree algorithm.|
|OMPI_SPC_BASE_SCATTER_LINEAR|The number of times the base scatter used the linear algorithm.|
|OMPI_SPC_BASE_GATHER_BINOMIAL|The number of times the base gather used the binomial tree algorithm.|
|OMPI_SPC_BASE_GATHER_LINEAR_SYNC|The number of times the base gather used the synchronous linear algorithm.|
|OMPI_SPC_BASE_GATHER_LINEAR|The number of times the base gather used the linear algorithm.|
|OMPI_SPC_BASE_ALLTOALL_INPLACE|The number of times the base alltoall used the in-place algorithm.|
|OMPI_SPC_BASE_ALLTOALL_PAIRWISE|The number of times the base alltoall used the pairwise algorithm.|
|OMPI_SPC_BASE_ALLTOALL_BRUCK|The number of times the base alltoall used the bruck algorithm.|
|OMPI_SPC_BASE_ALLTOALL_LINEAR_SYNC|The number of times the base alltoall used the synchronous linear algorithm.|
|OMPI_SPC_BASE_ALLTOALL_TWO_PROCS|The number of times the base alltoall used the two process algorithm.|
|OMPI_SPC_BASE_ALLTOALL_LINEAR|The number of times the base alltoall used the linear algorithm.|
|OMPI_SPC_BASE_ALLGATHER_BRUCK|The number of times the base allgather used the bruck algorithm.|
|OMPI_SPC_BASE_ALLGATHER_RECURSIVE_DOUBLING|The number of times the base allgather used the recursive doubling algorithm.|
|OMPI_SPC_BASE_ALLGATHER_RING|The number of times the base allgather used the ring algorithm.|
|OMPI_SPC_BASE_ALLGATHER_NEIGHBOR_EXCHANGE|The number of times the base allgather used the neighbor exchange algorithm.|
|OMPI_SPC_BASE_ALLGATHER_TWO_PROCS|The number of times the base allgather used the two process algorithm.|
|OMPI_SPC_BASE_ALLGATHER_LINEAR|The number of times the base allgather used the linear algorithm.|
|OMPI_SPC_BASE_BARRIER_DOUBLE_RING|The number of times the base barrier used the double ring algorithm.|
|OMPI_SPC_BASE_BARRIER_RECURSIVE_DOUBLING|The number of times the base barrier used the recursive doubling algorithm.|
|OMPI_SPC_BASE_BARRIER_BRUCK|The number of times the base barrier used the bruck algorithm.|
|OMPI_SPC_BASE_BARRIER_TWO_PROCS|The number of times the base barrier used the two process algorithm.|
|OMPI_SPC_BASE_BARRIER_LINEAR|The number of times the base barrier used the linear algorithm.|
|OMPI_SPC_BASE_BARRIER_TREE|The number of times the base barrier used the tree algorithm.|
|OMPI_SPC_P2P_MESSAGE_SIZE|This is a bin counter with two subcounters.  The first is messages that are less than or equal to mpi_spc_p2p_message_boundary bytes and the second is those that are larger than mpi_spc_p2p_message_boundary bytes.|
|OMPI_SPC_EAGER_MESSAGES|The number of messages that fall within the eager size.|
|OMPI_SPC_NOT_EAGER_MESSAGES|The number of messages that do not fall within the eager size.|
|OMPI_SPC_QUEUE_ALLOCATION|The amount of memory allocated after runtime currently in use for temporary message queues like the unexpected message queue and the out of sequence message queue.|
|OMPI_SPC_MAX_QUEUE_ALLOCATION|The maximum amount of memory allocated after runtime at one point for temporary message queues like the unexpected message queue and the out of sequence message queue.  Note: The OMPI_SPC_QUEUE_ALLOCATION counter must also be activated.|
|OMPI_SPC_UNEXPECTED_QUEUE_DATA|The amount of memory currently in use for the unexpected message queue.|
|OMPI_SPC_MAX_UNEXPECTED_QUEUE_DATA|The maximum amount of memory in use for the unexpected message queue.  Note: The OMPI_SPC_UNEXPECTED_QUEUE_DATA counter must also be activated.|
|OMPI_SPC_OOS_QUEUE_DATA|The amount of memory currently in use for the out-of-sequence message queue.|
|OMPI_SPC_MAX_OOS_QUEUE_DATA|The maximum amount of memory in use for the out-of-sequence message queue.  Note: The OMPI_SPC_OOS_QUEUE_DATA counter must also be activated.|
