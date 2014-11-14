/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2009 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006      Sun Microsystems Inc. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "ddt_lib.h"
#include "opal/runtime/opal.h"
#include "opal/datatype/opal_convertor.h"
#include <time.h>
#include <stdlib.h>
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif
#include <stdio.h>
#include <string.h>

#define DDT_TEST_CUDA

#if defined (DDT_TEST_CUDA)
#include <cuda_runtime_api.h>
#include "opal/mca/common/cuda/common_cuda.h"
#include "opal/runtime/opal_params.h"
#endif

/* Compile with:
mpicc -DHAVE_CONFIG_H -I. -I../../include -I../../../ompi-trunk/include  -I../.. -I../../include -I../../../ompi-trunk/opal -I../../../ompi-trunk/orte -I../../../ompi-trunk/ompi -g ddt_test.c -o ddt_test
*/

#define TIMER_DATA_TYPE struct timeval
#define GET_TIME(TV)   gettimeofday( &(TV), NULL )
#define ELAPSED_TIME(TSTART, TEND)  (((TEND).tv_sec - (TSTART).tv_sec) * 1000000 + ((TEND).tv_usec - (TSTART).tv_usec))

#define DUMP_DATA_AFTER_COMMIT 0x00000001
#define CHECK_PACK_UNPACK      0x00000002

uint32_t remote_arch = 0xffffffff;

static int test_upper( unsigned int length )
{
    double *mat1, *mat2, *inbuf;
    ompi_datatype_t *pdt;
    opal_convertor_t * pConv;
    char *ptr;
    int rc;
    unsigned int i, j, iov_count, split_chunk, total_length;
    size_t max_data;
    struct iovec a;
    TIMER_DATA_TYPE start, end;
    long total_time;

    printf( "test upper matrix\n" );
    pdt = upper_matrix( length );
    /*dt_dump( pdt );*/

    mat1 = malloc( length * length * sizeof(double) );
    init_random_upper_matrix( length, mat1 );
    mat2 = calloc( length * length, sizeof(double) );

    total_length = length * (length + 1) * ( sizeof(double) / 2);
    inbuf = (double*)malloc( total_length );
    ptr = (char*)inbuf;
    /* copy upper matrix in the array simulating the input buffer */
    for( i = 0; i < length; i++ ) {
        uint32_t pos = i * length + i;
        for( j = i; j < length; j++, pos++ ) {
            *inbuf = mat1[pos];
            inbuf++;
        }
    }
    inbuf = (double*)ptr;
    pConv = opal_convertor_create( remote_arch, 0 );
    if( OPAL_SUCCESS != opal_convertor_prepare_for_recv( pConv, &(pdt->super), 1, mat2 ) ) {
        printf( "Cannot attach the datatype to a convertor\n" );
        return OMPI_ERROR;
    }

    GET_TIME( start );
    split_chunk = (length + 1) * sizeof(double);
    /*    split_chunk = (total_length + 1) * sizeof(double); */
    for( i = total_length; i > 0; ) {
        if( i <= split_chunk ) {  /* equal test just to be able to set a breakpoint */
            split_chunk = i;
        }
        a.iov_base = ptr;
        a.iov_len = split_chunk;
        iov_count = 1;
        max_data = split_chunk;
        opal_convertor_unpack( pConv, &a, &iov_count, &max_data );
        ptr += max_data;
        i -= max_data;
        if( mat2[0] != inbuf[0] ) assert(0);
    }
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    printf( "complete unpacking in %ld microsec\n", total_time );
    free( inbuf );
    rc = check_diag_matrix( length, mat1, mat2 );
    free( mat1 );
    free( mat2 );

    /* test the automatic destruction pf the data */
    ompi_datatype_destroy( &pdt ); assert( pdt == NULL );

    OBJ_RELEASE( pConv );
    return rc;
}

/**
 * Computing the correct buffer length for moving a multiple of a datatype
 * is not an easy task. Define a function to centralize the complexity in a
 * single location.
 */
static size_t compute_buffer_length(ompi_datatype_t* pdt, int count)
{
    MPI_Aint extent, lb, true_extent, true_lb;
    size_t length;

    ompi_datatype_get_extent(pdt, &lb, &extent);
    ompi_datatype_get_true_extent(pdt, &true_lb, &true_extent); (void)true_lb;
    length = true_lb + true_extent + (count - 1) * extent;

    return  length;
}

/**
 *  Conversion function. They deal with data-types in 3 ways, always making local copies.
 * In order to allow performance testings, there are 3 functions:
 *  - one copying directly from one memory location to another one using the
 *    data-type copy function.
 *  - one which use a 2 convertors created with the same data-type
 *  - and one using 2 convertors created from different data-types.
 *
 */
static int local_copy_ddt_count( ompi_datatype_t* pdt, int count )
{
    void *pdst, *psrc;
    TIMER_DATA_TYPE start, end;
    long total_time;
    size_t length;

    length = compute_buffer_length(pdt, count);

    pdst = malloc(length);
    psrc = malloc(length);

    for( size_t i = 0; i < length; i++ )
	((char*)psrc)[i] = i % 128 + 32;
    memset(pdst, 0, length);

    cache_trash();  /* make sure the cache is useless */

    GET_TIME( start );
    if( OMPI_SUCCESS != ompi_datatype_copy_content_same_ddt( pdt, count, pdst, psrc ) ) {
        printf( "Unable to copy the datatype in the function local_copy_ddt_count."
                " Is the datatype committed ?\n" );
    }
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    printf( "direct local copy in %ld microsec\n", total_time );
    free(pdst);
    free(psrc);

    return OMPI_SUCCESS;
}

static void fill_vectors(double* vp, int itera, int contig, int gap)
{
    int i, j;
    for (i = 0; i < itera-1; i++ ){
        for (j = i*gap; j < (i+1)*gap; j++) {
            if (j >= i*gap && j < i*gap+contig) {
                vp[j] = 1.0;
            } else {
                vp[j] = 0.0;
            }
        }
    }
    for (i = (itera-1)*gap; i < (itera-1)*gap+contig; i++) {
        vp[i] = 1.0;
    }
    
    // printf("vector generated:\n");
    // for (i = 0; i < (itera-1)*gap+contig; i++) {
    //     printf("%1.f ", vp[i]);
    // }
    // printf("\n");
}

static void verify_vectors(double *vp, int itera, int contig, int gap)
{
    int i, j;
    int error = 0;
    for (i = 0; i < itera-1; i++) {
        for (j = i*gap; j < (i+1)*gap; j++) {
            if (j >= i*gap && j < i*gap+contig) {
                if (vp[j] != 1.0) {
                    error ++;
                }
            } 
        }
    }
    for (i = (itera-1)*gap; i < (itera-1)*gap+contig; i++) {
        if (vp[i] != 1.0) {
            error ++;
        }
    }
    // printf("vector received:\n");
    // for (i = 0; i < (itera-1)*gap+contig; i++) {
    //     printf("%1.f ", vp[i]);
    // }
    if (error != 0) {
        printf("%d error is found\n", error);
    } else {
        printf("no error is found\n");
    }
}

static int
local_copy_with_convertor_2datatypes( ompi_datatype_t* send_type, int send_count,
                                      ompi_datatype_t* recv_type, int recv_count,
                                      int chunk, int itera, int contig, int gap )
{
    void *pdst = NULL, *psrc = NULL, *ptemp = NULL, *phost = NULL;
    opal_convertor_t *send_convertor = NULL, *recv_convertor = NULL;
    struct iovec iov;
    uint32_t iov_count;
    size_t max_data;
    int32_t length = 0, done1 = 0, done2 = 0;
    TIMER_DATA_TYPE start, end, unpack_start, unpack_end;
    long total_time, unpack_time = 0;
    size_t slength, rlength;

    rlength = compute_buffer_length(recv_type, recv_count);
    slength = compute_buffer_length(send_type, send_count);

#if defined (DDT_TEST_CUDA)
    cudaError_t error = cudaMalloc((void **)&psrc, slength);
    if ( error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    cudaMemset(psrc, 0, slength);
    printf("cudamalloc psrc %p\n", psrc);
    
    error = cudaMalloc((void **)&pdst, rlength);
    if ( error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    cudaMemset(pdst, 0, rlength); 
    printf("cudamalloc pdst %p\n", pdst);
    
    error = cudaMallocHost((void **)&ptemp, chunk);
    if ( error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    memset(ptemp, 0, chunk);
    printf("cudamallochost ptemp %p\n", ptemp);
    
    error = cudaMallocHost((void **)&phost, slength);
    if ( error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    memset(phost, 0, slength);
    printf("cudamallochost phost %p\n", phost);
#else
    pdst  = malloc( rlength );
    psrc  = malloc( slength );
    ptemp = malloc( chunk );

    /* initialize the buffers to prevent valgrind from complaining */
    for( size_t i = 0; i < slength; i++ )
            ((char*)psrc)[i] = i % 128 + 32;
    memset(pdst, 0, rlength);
#endif
    
#if defined (DDT_TEST_CUDA)
    if (itera > 0) {
        fill_vectors((double *)phost, itera, contig, gap);
    }
    cudaMemcpy(psrc, phost, slength, cudaMemcpyHostToDevice);
#else 
    if (itera > 0) {
        fill_vectors(psrc, itera, contig, gap);
    }
#endif

    send_convertor = opal_convertor_create( remote_arch, 0 );
    if( OPAL_SUCCESS != opal_convertor_prepare_for_send( send_convertor, &(send_type->super), send_count, psrc ) ) {
        printf( "Unable to create the send convertor. Is the datatype committed ?\n" );
        goto clean_and_return;
    }
    recv_convertor = opal_convertor_create( remote_arch, 0 );
    if( OPAL_SUCCESS != opal_convertor_prepare_for_recv( recv_convertor, &(recv_type->super), recv_count, pdst ) ) {
        printf( "Unable to create the recv convertor. Is the datatype committed ?\n" );
        goto clean_and_return;
    }

    cache_trash();  /* make sure the cache is useless */

    GET_TIME( start );
    while( (done1 & done2) != 1 ) {
        /* They are supposed to finish in exactly the same time. */
        if( done1 | done2 ) {
            printf( "WRONG !!! the send is %s but the receive is %s in local_copy_with_convertor_2datatypes\n",
                    (done1 ? "finish" : "not finish"),
                    (done2 ? "finish" : "not finish") );
        }

        max_data = chunk;
        iov_count = 1;
        iov.iov_base = ptemp;
        iov.iov_len = chunk;

        if( done1 == 0 ) {
            done1 = opal_convertor_pack( send_convertor, &iov, &iov_count, &max_data );
        }

        if( done2 == 0 ) {
            GET_TIME( unpack_start );
            done2 = opal_convertor_unpack( recv_convertor, &iov, &iov_count, &max_data );
            GET_TIME( unpack_end );
            unpack_time += ELAPSED_TIME( unpack_start, unpack_end );
        }

        length += max_data;
    }
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    printf( "copying different data-types using convertors in %ld microsec\n", total_time );
    printf( "\t unpack in %ld microsec [pack in %ld microsec]\n", unpack_time,
            total_time - unpack_time );
            
#if defined (DDT_TEST_CUDA)
    memset(phost, 0, slength);
    cudaMemcpy(phost, pdst, rlength, cudaMemcpyDeviceToHost);
    if (itera > 0) {
        verify_vectors((double *)phost, itera, contig, gap);
    }
#else
    if (itera > 0) {
        verify_vectors((double *)pdst, itera, contig, gap);
    }
#endif
 clean_and_return:
    if( send_convertor != NULL ) {
        OBJ_RELEASE( send_convertor ); assert( send_convertor == NULL );
    }
    if( recv_convertor != NULL ) {
        OBJ_RELEASE( recv_convertor ); assert( recv_convertor == NULL );
    }
#if defined (DDT_TEST_CUDA)
    if( NULL != pdst ) cudaFree( pdst );
    if( NULL != psrc ) cudaFree( psrc );
    if( NULL != ptemp ) cudaFreeHost( ptemp );
    if( NULL != phost ) cudaFreeHost( phost );
#else
    if( NULL != pdst ) free( pdst );
    if( NULL != psrc ) free( psrc );
    if( NULL != ptemp ) free( ptemp );
#endif
    return OMPI_SUCCESS;
}

static int
local_copy_with_convertor_2datatypes_struct( ompi_datatype_t* send_type, int send_count,
                                      ompi_datatype_t* recv_type, int recv_count,
                                      int chunk, int count)
{
    void *pdst = NULL, *psrc = NULL, *ptemp = NULL, *phost = NULL;
    opal_convertor_t *send_convertor = NULL, *recv_convertor = NULL;
    struct iovec iov;
    uint32_t iov_count;
    size_t max_data;
    int32_t length = 0, done1 = 0, done2 = 0;
    TIMER_DATA_TYPE start, end, unpack_start, unpack_end;
    long total_time, unpack_time = 0;
    size_t slength, rlength;

    rlength = compute_buffer_length(recv_type, recv_count);
    slength = compute_buffer_length(send_type, send_count);

#if defined (DDT_TEST_CUDA)
    cudaError_t error = cudaMalloc((void **)&psrc, slength);
    if ( error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    cudaMemset(psrc, 0, slength);
    printf("cudamalloc psrc %p\n", psrc);
    
    error = cudaMalloc((void **)&pdst, rlength);
    if ( error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    cudaMemset(pdst, 0, rlength); 
    printf("cudamalloc pdst %p\n", pdst);
    
    error = cudaMallocHost((void **)&ptemp, chunk);
    if ( error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    memset(ptemp, 0, chunk);
    printf("cudamallochost ptemp %p\n", ptemp);
    
    error = cudaMallocHost((void **)&phost, slength);
    if ( error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    memset(phost, 0, slength);
    printf("cudamallochost phost %p\n", phost);
#else
    pdst  = malloc( rlength );
    psrc  = malloc( slength );
    ptemp = malloc( chunk );

    /* initialize the buffers to prevent valgrind from complaining */
    for( size_t i = 0; i < slength; i++ )
            ((char*)psrc)[i] = i % 128 + 32;
    memset(pdst, 0, rlength);
#endif
    
#if defined (DDT_TEST_CUDA)

    cudaMemcpy(psrc, phost, slength, cudaMemcpyHostToDevice);
#else 

#endif

    send_convertor = opal_convertor_create( remote_arch, 0 );
    if( OPAL_SUCCESS != opal_convertor_prepare_for_send( send_convertor, &(send_type->super), send_count, psrc ) ) {
        printf( "Unable to create the send convertor. Is the datatype committed ?\n" );
        goto clean_and_return;
    }
    recv_convertor = opal_convertor_create( remote_arch, 0 );
    if( OPAL_SUCCESS != opal_convertor_prepare_for_recv( recv_convertor, &(recv_type->super), recv_count, pdst ) ) {
        printf( "Unable to create the recv convertor. Is the datatype committed ?\n" );
        goto clean_and_return;
    }

    cache_trash();  /* make sure the cache is useless */

    GET_TIME( start );
    while( (done1 & done2) != 1 ) {
        /* They are supposed to finish in exactly the same time. */
        if( done1 | done2 ) {
            printf( "WRONG !!! the send is %s but the receive is %s in local_copy_with_convertor_2datatypes\n",
                    (done1 ? "finish" : "not finish"),
                    (done2 ? "finish" : "not finish") );
        }

        max_data = chunk;
        iov_count = 1;
        iov.iov_base = ptemp;
        iov.iov_len = chunk;

        if( done1 == 0 ) {
            done1 = opal_convertor_pack( send_convertor, &iov, &iov_count, &max_data );
        }

        if( done2 == 0 ) {
            GET_TIME( unpack_start );
            done2 = opal_convertor_unpack( recv_convertor, &iov, &iov_count, &max_data );
            GET_TIME( unpack_end );
            unpack_time += ELAPSED_TIME( unpack_start, unpack_end );
        }

        length += max_data;
    }
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    printf( "copying different data-types using convertors in %ld microsec\n", total_time );
    printf( "\t unpack in %ld microsec [pack in %ld microsec]\n", unpack_time,
            total_time - unpack_time );
            
#if defined (DDT_TEST_CUDA)
    memset(phost, 0, slength);
    cudaMemcpy(phost, pdst, rlength, cudaMemcpyDeviceToHost);

#else

#endif
 clean_and_return:
    if( send_convertor != NULL ) {
        OBJ_RELEASE( send_convertor ); assert( send_convertor == NULL );
    }
    if( recv_convertor != NULL ) {
        OBJ_RELEASE( recv_convertor ); assert( recv_convertor == NULL );
    }
#if defined (DDT_TEST_CUDA)
    if( NULL != pdst ) cudaFree( pdst );
    if( NULL != psrc ) cudaFree( psrc );
    if( NULL != ptemp ) cudaFreeHost( ptemp );
    if( NULL != phost ) cudaFreeHost( phost );
#else
    if( NULL != pdst ) free( pdst );
    if( NULL != psrc ) free( psrc );
    if( NULL != ptemp ) free( ptemp );
#endif
    return OMPI_SUCCESS;
}


static void fill_upper_matrix(void *matt, int msize)
{
    int i, j, start, end;
    int *blklens, *displs;
#if defined (TEST_DOUBLE)
    double *mat = (double *)matt;
#elif defined (TEST_FLOAT)
    float *mat = (float *)matt;
#elif defined (TEST_CHAR)
    char *mat = (char *)matt;
#else
    void *mat = matt;
#endif
    
    blklens = (int *)malloc(sizeof(int)*msize);
    displs = (int *)malloc(sizeof(int)*msize);
    for (i = 0; i < msize; i++) {
        blklens[i] = msize - i;
        displs[i] = i*msize + i;
    }
    for (i = 0; i < msize; i++) {
        start = displs[i];
        end = start + blklens[i];
        for (j = start; j < end; j++) {
#if defined (TEST_CHAR)
            mat[j] = 'a';
#else
            mat[j] = 0.0 + i;
#endif
        }
    }
    free(blklens);
    free(displs);

   // printf("matrix generate\n");
   // for (i = 0; i < msize; i++) {
   //     for (j = 0; j < msize; j++) {
   //         printf(" %1.f ", mat[i*msize+j]);
   //     }
   //     printf("\n");
   // }
}

static void verify_mat_result(void *matt, int msize)
{
    int *blklens, *displs;
    int i, j, error = 0;
    int start, end;
#if defined (TEST_DOUBLE)
    double *mat = (double *)matt;
#elif defined (TEST_FLOAT)
    float *mat = (float *)matt;
#elif defined (TEST_CHAR)
    char *mat = (char *)matt;
#else
    void *mat = matt;
#endif
    
    blklens = (int *)malloc(sizeof(int)*msize);
    displs = (int *)malloc(sizeof(int)*msize);
    for (i = 0; i < msize; i++) {
        blklens[i] = msize - i;
        displs[i] = i*msize + i;
    }
    for (i = 0; i < msize; i++) {
        start = displs[i];
        end = start + blklens[i];
        for (j = start; j < end; j++) {
#if defined (TEST_CHAR) 
            if (mat[j] != 'a') {
#else
            if (mat[j] != (0.0+i)) {
#endif
                error ++;
            }
        }
    }
    free(blklens);
    free(displs);
    
    // printf("matrix received\n");
    // for (i = 0; i < msize; i++) {
    //     for (j = 0; j < msize; j++) {
    //         printf(" %1.f ", mat[i*msize+j]);
    //     }
    //     printf("\n");
    // }
    
    if (error != 0) {
        printf("error is found %d\n", error);
    } else {
        printf("no error is found\n");
    }
}

static int local_copy_with_convertor( ompi_datatype_t* pdt, int count, int chunk, int msize )
{
    void *pdst = NULL, *psrc = NULL, *ptemp = NULL, *phost = NULL;
    opal_convertor_t *send_convertor = NULL, *recv_convertor = NULL;
    struct iovec iov;
    uint32_t iov_count;
    size_t max_data, dt_length;
    int32_t length = 0, done1 = 0, done2 = 0;
    TIMER_DATA_TYPE start, end, unpack_start, unpack_end;
    long total_time, unpack_time = 0;

    dt_length = compute_buffer_length(pdt, count);
    printf("length %lu\n", dt_length);

#if defined (DDT_TEST_CUDA)
    cudaError_t error = cudaMalloc((void **)&psrc, dt_length);
    if ( error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    cudaMemset(psrc, 0, dt_length);
    printf("cudamalloc psrc %p\n", psrc);
    
    error = cudaMalloc((void **)&pdst, dt_length);
    if ( error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    cudaMemset(pdst, 0, dt_length); 
    printf("cudamalloc pdst %p\n", pdst);
    
    error = cudaMallocHost((void **)&ptemp, chunk);
    if ( error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    memset(ptemp, 0, chunk);
    printf("cudamallochost ptemp %p\n", ptemp);
    
    error = cudaMallocHost((void **)&phost, dt_length);
    if ( error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    memset(phost, 0, dt_length);
    printf("cudamallochost phost %p\n", phost);
#else
    pdst  = malloc(dt_length);
    psrc  = malloc(dt_length);
    ptemp = malloc(chunk);
    
    for( int i = 0; i < length; ((char*)psrc)[i] = i % 128 + 32, i++ );
    memset( pdst, 0, length );
#endif

#if defined (DDT_TEST_CUDA)
    if (msize > 0) {
        fill_upper_matrix(phost, msize);
    }
    cudaMemcpy(psrc, phost, dt_length, cudaMemcpyHostToDevice);
#else 
    if (msize > 0) {
        fill_upper_matrix(psrc, msize);
    }
#endif

    send_convertor = opal_convertor_create( remote_arch, 0 );
    if( OPAL_SUCCESS != opal_convertor_prepare_for_send( send_convertor, &(pdt->super), count, psrc ) ) {
        printf( "Unable to create the send convertor. Is the datatype committed ?\n" );
        goto clean_and_return;
    }

    recv_convertor = opal_convertor_create( remote_arch, 0 );
    if( OPAL_SUCCESS != opal_convertor_prepare_for_recv( recv_convertor, &(pdt->super), count, pdst ) ) {
        printf( "Unable to create the recv convertor. Is the datatype committed ?\n" );
        goto clean_and_return;
    }

    cache_trash();  /* make sure the cache is useless */

    GET_TIME( start );
    while( (done1 & done2) != 1 ) {
        /* They are supposed to finish in exactly the same time. */
        if( done1 | done2 ) {
            printf( "WRONG !!! the send is %s but the receive is %s in local_copy_with_convertor\n",
                    (done1 ? "finish" : "not finish"),
                    (done2 ? "finish" : "not finish") );
        }

        max_data = chunk;
        iov_count = 1;
        iov.iov_base = ptemp;
        iov.iov_len = chunk;

        if( done1 == 0 ) {
            done1 = opal_convertor_pack( send_convertor, &iov, &iov_count, &max_data );
        }

        if( done2 == 0 ) {
            GET_TIME( unpack_start );
            done2 = opal_convertor_unpack( recv_convertor, &iov, &iov_count, &max_data );
            GET_TIME( unpack_end );
            unpack_time += ELAPSED_TIME( unpack_start, unpack_end );
        }

        length += max_data;
    }
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    printf( "copying same data-type using convertors in %ld microsec\n", total_time );
    printf( "\t unpack in %ld microsec [pack in %ld microsec]\n", unpack_time,
            total_time - unpack_time );
            
#if defined (DDT_TEST_CUDA)
    memset(phost, 0, dt_length);
    cudaMemcpy(phost, pdst, dt_length, cudaMemcpyDeviceToHost);
    if (msize > 0) {
        verify_mat_result(phost, msize);
    }
#else
    if (msize > 0) {
        verify_mat_result(pdst, msize);
    }
#endif
clean_and_return:
    if( NULL != send_convertor ) OBJ_RELEASE( send_convertor );
    if( NULL != recv_convertor ) OBJ_RELEASE( recv_convertor );

#if defined (DDT_TEST_CUDA)
    if( NULL != pdst ) cudaFree( pdst );
    if( NULL != psrc ) cudaFree( psrc );
    if( NULL != ptemp ) cudaFreeHost( ptemp );
    if( NULL != phost ) cudaFreeHost( phost );
#else
    if( NULL != pdst ) free( pdst );
    if( NULL != psrc ) free( psrc );
    if( NULL != ptemp ) free( ptemp );
#endif
    return OMPI_SUCCESS;
}

/**
 * Main function. Call several tests and print-out the results. It try to stress the convertor
 * using difficult data-type constructions as well as strange segment sizes for the conversion.
 * Usually, it is able to detect most of the data-type and convertor problems. Any modifications
 * on the data-type engine should first pass all the tests from this file, before going into other
 * tests.
 */
int main( int argc, char* argv[] )
{
    ompi_datatype_t *pdt, *pdt1, *pdt2, *pdt3;
    int rc, length = 500, i;

#if defined (DDT_TEST_CUDA)
    opal_cuda_support = 1;
#endif
    opal_init_util(&argc, &argv);
#if defined (DDT_TEST_CUDA)
    mca_common_cuda_stage_one_init();
#endif
    ompi_datatype_init();

    /**
     * By default simulate homogeneous architectures.
     */
    remote_arch = opal_local_arch;
/*    printf( "\n\n#\n * TEST INVERSED VECTOR\n #\n\n" );
    pdt = create_inversed_vector( &ompi_mpi_int.dt, 10 );
    if( outputFlags & CHECK_PACK_UNPACK ) {
        local_copy_ddt_count(pdt, 100);
        local_copy_with_convertor(pdt, 100, 956);
    }
    OBJ_RELEASE( pdt ); assert( pdt == NULL );
    printf( "\n\n#\n * TEST STRANGE DATATYPE\n #\n\n" );
    pdt = create_strange_dt();
    if( outputFlags & CHECK_PACK_UNPACK ) {
        local_copy_ddt_count(pdt, 1);
        local_copy_with_convertor(pdt, 1, 956);
    }
    OBJ_RELEASE( pdt ); assert( pdt == NULL );
*/    
    printf("\n TEST STRUCT \n");
    pdt = create_struct_type(5);
    if( outputFlags & CHECK_PACK_UNPACK ) {
        for (i = 1; i <= 1; i++) {
  //          local_copy_with_convertor_2datatypes_struct(pdt, 1, pdt, 1, 1024*1024*100, 5);
        }
    }
    
    printf( "\n\n#\n * TEST UPPER TRIANGULAR MATRIX (size 100)\n #\n\n" );
    pdt = upper_matrix(1000);
    if( outputFlags & CHECK_PACK_UNPACK ) {
        for (i = 1; i <= 3; i++) {
//        local_copy_ddt_count(pdt, 1);
            local_copy_with_convertor(pdt, 1, 1024*1024*200, 1000);
        }
    }
    OBJ_RELEASE( pdt ); assert( pdt == NULL );
  /*  
    mpich_typeub();
    mpich_typeub2();
    mpich_typeub3();

    printf( "\n\n#\n * TEST UPPER MATRIX\n #\n\n" );
    rc = test_upper( length );
    if( rc == 0 )
        printf( "decode [PASSED]\n" );
    else
        printf( "decode [NOT PASSED]\n" );

    printf( "\n\n#\n * TEST MATRIX BORDERS\n #\n\n" );
    pdt = test_matrix_borders( length, 100 );
    if( outputFlags & DUMP_DATA_AFTER_COMMIT ) {
        ompi_datatype_dump( pdt );
    }
    OBJ_RELEASE( pdt ); assert( pdt == NULL );

    printf( "\n\n#\n * TEST CONTIGUOUS\n #\n\n" );
    pdt = test_contiguous();
    OBJ_RELEASE( pdt ); assert( pdt == NULL );
    printf( "\n\n#\n * TEST STRUCT\n #\n\n" );
    pdt = test_struct();
    OBJ_RELEASE( pdt ); assert( pdt == NULL );

    ompi_datatype_create_contiguous(0, &ompi_mpi_datatype_null.dt, &pdt1);
    ompi_datatype_create_contiguous(0, &ompi_mpi_datatype_null.dt, &pdt2);
    ompi_datatype_create_contiguous(0, &ompi_mpi_datatype_null.dt, &pdt3);
    ompi_datatype_add( pdt3, &ompi_mpi_int.dt, 10, 0, -1 );
    ompi_datatype_add( pdt3, &ompi_mpi_float.dt, 5, 10 * sizeof(int), -1 );

    ompi_datatype_add( pdt2, &ompi_mpi_float.dt, 1, 0, -1 );
    ompi_datatype_add( pdt2, pdt3, 3, sizeof(int) * 1, -1 );

    ompi_datatype_add( pdt1, &ompi_mpi_long_long_int.dt, 5, 0, -1 );
    ompi_datatype_add( pdt1, &ompi_mpi_long_double.dt, 2, sizeof(long long) * 5, -1 );

    printf( ">>--------------------------------------------<<\n" );
    if( outputFlags & DUMP_DATA_AFTER_COMMIT ) {
        ompi_datatype_dump( pdt1 );
    }
    printf( ">>--------------------------------------------<<\n" );
    if( outputFlags & DUMP_DATA_AFTER_COMMIT ) {
        ompi_datatype_dump( pdt2 );
    }
    printf( ">>--------------------------------------------<<\n" );
    if( outputFlags & DUMP_DATA_AFTER_COMMIT ) {
        ompi_datatype_dump( pdt3 );
    }

    OBJ_RELEASE( pdt1 ); assert( pdt1 == NULL );
    OBJ_RELEASE( pdt2 ); assert( pdt2 == NULL );
    OBJ_RELEASE( pdt3 ); assert( pdt3 == NULL );
    printf( ">>--------------------------------------------<<\n" );
    printf( " Contiguous data-type (MPI_DOUBLE)\n" );
    pdt = MPI_DOUBLE;
    if( outputFlags & CHECK_PACK_UNPACK ) {
        local_copy_ddt_count(pdt, 4500);
        local_copy_with_convertor( pdt, 4500, 12 );
        local_copy_with_convertor_2datatypes( pdt, 4500, pdt, 4500, 12 );
    }
    printf( ">>--------------------------------------------<<\n" );

    printf( ">>--------------------------------------------<<\n" );
    if( outputFlags & CHECK_PACK_UNPACK ) {
        printf( "Contiguous multiple data-type (4500*1)\n" );
        pdt = create_contiguous_type( MPI_DOUBLE, 4500 );
        local_copy_ddt_count(pdt, 1);
        local_copy_with_convertor( pdt, 1, 12 );
        local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 12 );
        OBJ_RELEASE( pdt ); assert( pdt == NULL );
        printf( "Contiguous multiple data-type (450*10)\n" );
        pdt = create_contiguous_type( MPI_DOUBLE, 450 );
        local_copy_ddt_count(pdt, 10);
        local_copy_with_convertor( pdt, 10, 12 );
        local_copy_with_convertor_2datatypes( pdt, 10, pdt, 10, 12 );
        OBJ_RELEASE( pdt ); assert( pdt == NULL );
        printf( "Contiguous multiple data-type (45*100)\n" );
        pdt = create_contiguous_type( MPI_DOUBLE, 45 );
        local_copy_ddt_count(pdt, 100);
        local_copy_with_convertor( pdt, 100, 12 );
        local_copy_with_convertor_2datatypes( pdt, 100, pdt, 100, 12 );
        OBJ_RELEASE( pdt ); assert( pdt == NULL );
        printf( "Contiguous multiple data-type (100*45)\n" );
        pdt = create_contiguous_type( MPI_DOUBLE, 100 );
        local_copy_ddt_count(pdt, 45);
        local_copy_with_convertor( pdt, 45, 12 );
        local_copy_with_convertor_2datatypes( pdt, 45, pdt, 45, 12 );
        OBJ_RELEASE( pdt ); assert( pdt == NULL );
        printf( "Contiguous multiple data-type (10*450)\n" );
        pdt = create_contiguous_type( MPI_DOUBLE, 10 );
        local_copy_ddt_count(pdt, 450);
        local_copy_with_convertor( pdt, 450, 12 );
        local_copy_with_convertor_2datatypes( pdt, 450, pdt, 450, 12 );
        OBJ_RELEASE( pdt ); assert( pdt == NULL );
        printf( "Contiguous multiple data-type (1*4500)\n" );
        pdt = create_contiguous_type( MPI_DOUBLE, 1 );
        local_copy_ddt_count(pdt, 4500);
        local_copy_with_convertor( pdt, 4500, 12 );
        local_copy_with_convertor_2datatypes( pdt, 4500, pdt, 4500, 12 );
        OBJ_RELEASE( pdt ); assert( pdt == NULL );
    }*/
    printf( ">>--------------------------------------------<<\n" );
    printf( "Vector data-type (4000 times 512 double stride 640)\n" );
#if 0
    pdt = create_vector_type( MPI_DOUBLE, 4000, 512, 640 );
    opal_datatype_resize(&pdt->super, 0, pdt->super.size+sizeof(MPI_DOUBLE));
    ompi_datatype_create_contiguous( 1, pdt, &pdt1 );
#else
    pdt = create_vector_type( MPI_DOUBLE, 4000, 512, 640 );
  //  opal_datatype_resize(&pdt->super, 0, pdt->super.size+sizeof(MPI_DOUBLE));
  //  ompi_datatype_create_contiguous( 4000, pdt, &pdt1 );
#endif
//    ompi_datatype_dump( pdt );
 //   ompi_datatype_commit(&pdt1);
    if( outputFlags & CHECK_PACK_UNPACK ) {
        for (i = 0; i < 10; i++) {
    //         local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 1024*1024*30 );
        }
    }
    printf( ">>--------------------------------------------<<\n" );
    printf( "Vector data-type (4000 times 384 double stride 512)\n" );
    pdt = create_vector_type( MPI_DOUBLE, 4000, 384, 512 );
    opal_datatype_resize(&pdt->super, 0, pdt->super.size+sizeof(MPI_DOUBLE));
    ompi_datatype_create_contiguous( 1, pdt, &pdt1 );
//    ompi_datatype_dump( pdt );
    if( outputFlags & CHECK_PACK_UNPACK ) {
        for (i = 0; i < 1; i++) {
       // local_copy_ddt_count(pdt, 1);
      //  local_copy_with_convertor( pdt, 1, 12 );
      //  local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 12 );
      //  local_copy_with_convertor( pdt, 1, 82 );
      //  local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 82 );
      //  local_copy_with_convertor( pdt, 1, 6000 );
      //  local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 6000 );
      //  local_copy_with_convertor( pdt, 1, 36000 );
           // local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 1024*1024*20 , 4000, 384, 512);
        }
    }
    printf( ">>--------------------------------------------<<\n" );
    OBJ_RELEASE( pdt ); assert( pdt == NULL );
    
    printf( "Vector data-type (4000 times 256 double stride 384)\n" );
    pdt = create_vector_type( MPI_DOUBLE, 4000, 256, 384 );
//    ompi_datatype_dump( pdt );
    if( outputFlags & CHECK_PACK_UNPACK ) {
        for (i = 0; i < 10; i++) {
       // local_copy_ddt_count(pdt, 1);
      //  local_copy_with_convertor( pdt, 1, 12 );
      //  local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 12 );
      //  local_copy_with_convertor( pdt, 1, 82 );
      //  local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 82 );
      //  local_copy_with_convertor( pdt, 1, 6000 );
      //  local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 6000 );
      //  local_copy_with_convertor( pdt, 1, 36000 );
  //        local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 1024*1024*10 );
        }
    }
    printf( ">>--------------------------------------------<<\n" );
    OBJ_RELEASE( pdt ); assert( pdt == NULL );
    
    printf( "Vector data-type (4000 times 128 double stride 256)\n" );
    pdt = create_vector_type( MPI_DOUBLE, 4000, 128, 256 );
//    ompi_datatype_dump( pdt );
    if( outputFlags & CHECK_PACK_UNPACK ) {
        for (i = 0; i < 10; i++) {
       // local_copy_ddt_count(pdt, 1);
      //  local_copy_with_convertor( pdt, 1, 12 );
      //  local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 12 );
      //  local_copy_with_convertor( pdt, 1, 82 );
      //  local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 82 );
      //  local_copy_with_convertor( pdt, 1, 6000 );
      //  local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 6000 );
      //  local_copy_with_convertor( pdt, 1, 36000 );
    //      local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 1024*1024*5 );
        }
    }
    printf( ">>--------------------------------------------<<\n" );
    OBJ_RELEASE( pdt ); assert( pdt == NULL );
    
    printf( "Vector data-type (2000 times 3 double stride 4)\n" );
    pdt = create_vector_type( MPI_DOUBLE, 2000, 3, 4 );
//    ompi_datatype_dump( pdt );
    if( outputFlags & CHECK_PACK_UNPACK ) {
        for (i = 0; i < 10; i++) {
       // local_copy_ddt_count(pdt, 1);
      //  local_copy_with_convertor( pdt, 1, 12 );
      //  local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 12 );
      //  local_copy_with_convertor( pdt, 1, 82 );
      //  local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 82 );
      //  local_copy_with_convertor( pdt, 1, 6000 );
      //  local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 6000 );
      //  local_copy_with_convertor( pdt, 1, 36000 );
      //  local_copy_with_convertor_2datatypes( pdt, 1, pdt, 1, 1024*1024*4 );
        }
    }
    printf( ">>--------------------------------------------<<\n" );
    OBJ_RELEASE( pdt ); assert( pdt == NULL );
    /*
    printf( ">>--------------------------------------------<<\n" );
    pdt = test_struct_char_double();
    if( outputFlags & CHECK_PACK_UNPACK ) {
        local_copy_ddt_count(pdt, 4500);
        local_copy_with_convertor( pdt, 4500, 12 );
        local_copy_with_convertor_2datatypes( pdt, 4500, pdt, 4500, 12 );
    }
    printf( ">>--------------------------------------------<<\n" );
    OBJ_RELEASE( pdt ); assert( pdt == NULL );

    printf( ">>--------------------------------------------<<\n" );
    pdt = test_create_twice_two_doubles();
    if( outputFlags & CHECK_PACK_UNPACK ) {
        local_copy_ddt_count(pdt, 4500);
        local_copy_with_convertor( pdt, 4500, 12 );
        local_copy_with_convertor_2datatypes( pdt, 4500, pdt, 4500, 12 );
    }
    printf( ">>--------------------------------------------<<\n" );
    OBJ_RELEASE( pdt ); assert( pdt == NULL );
    printf( ">>--------------------------------------------<<\n" );
    pdt = test_create_blacs_type();
    if( outputFlags & CHECK_PACK_UNPACK ) {
        ompi_datatype_dump( pdt );
        local_copy_ddt_count(pdt, 2);
        local_copy_ddt_count(pdt, 4500);
        local_copy_with_convertor( pdt, 4500, 956 );
        local_copy_with_convertor_2datatypes( pdt, 4500, pdt, 4500, 956 );
        local_copy_with_convertor( pdt, 4500, 16*1024 );
        local_copy_with_convertor_2datatypes( pdt, 4500, pdt, 4500, 16*1024 );
        local_copy_with_convertor( pdt, 4500, 64*1024 );
        local_copy_with_convertor_2datatypes( pdt, 4500, pdt, 4500, 64*1024 );
    }
    printf( ">>--------------------------------------------<<\n" );
    OBJ_RELEASE( pdt ); assert( pdt == NULL );
    printf( ">>--------------------------------------------<<\n" );
    pdt1 = test_create_blacs_type1( &ompi_mpi_int.dt );
    pdt2 = test_create_blacs_type2( &ompi_mpi_int.dt );
    if( outputFlags & CHECK_PACK_UNPACK ) {
        local_copy_with_convertor_2datatypes( pdt1, 1, pdt2, 1, 100 );
    }
    printf( ">>--------------------------------------------<<\n" );
    OBJ_RELEASE( pdt1 ); assert( pdt1 == NULL );
    OBJ_RELEASE( pdt2 ); assert( pdt2 == NULL );
*/
    /* clean-ups all data allocations */
    ompi_datatype_finalize();

    return OMPI_SUCCESS;
}
