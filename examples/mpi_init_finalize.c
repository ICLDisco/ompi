/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      Cisco Systems, Inc.  All rights reserved.
 *
 * Simple ring test program in C.
 */

#include <stdio.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    int rank, size, next, prev, message, tag = 201;
     MPI_Init(&argc, &argv);
    /* Start up MPI */
    printf("MPI_initating\n");
    {
                  char name[255];
                  gethostname(name,255);
                  printf("ssh -t zhongdong@%s gdb -p %d\n", name, getpid());
                  int c=1;
                  while (c){}
    }
    //MPI_Init(&argc, &argv);
    printf("Init done\n");
    sleep(100);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Ranks %d\n", rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Finalize();
    return 0;
}
