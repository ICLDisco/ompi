/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2014-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef OPAL_DATATYPE_CUDA_INTERNAL_H_HAS_BEEN_INCLUDED
#define OPAL_DATATYPE_CUDA_INTERNAL_H_HAS_BEEN_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* OPAL_CUDA */
#define OPAL_DATATYPE_CUDA_DEBUG    1
#define OPAL_DATATYPE_CUDA_TIMING
#define OPAL_DATATYPE_USE_ZEROCPY   0
#define OPAL_DATATYPE_CUDA_IOV_CACHE    1

#define DT_CUDA_FREE_LIST_SIZE  50

#define THREAD_PER_BLOCK    32
#define CUDA_WARP_SIZE      32
#define NB_STREAMS          4
#define NB_PIPELINE_NON_CACHED_BLOCKS  4
#define NB_CACHED_BLOCKS    4
#define CUDA_MAX_NB_BLOCKS  1024
#define CUDA_IOV_MAX_TASK_PER_BLOCK 400
#define ALIGNMENT_DOUBLE    8
#define ALIGNMENT_FLOAT     4
#define ALIGNMENT_CHAR      1
#define NUM_CUDA_IOV_PER_DDT    150000
#define IOV_PIPELINE_SIZE   1000
#define KERNEL_UNROLL       16
#define UNROLL_16           16
#define UNROLL_8            8
#define UNROLL_4            4

#define TIMER_DATA_TYPE struct timeval
#define GET_TIME(TV)   gettimeofday( &(TV), NULL )
#define ELAPSED_TIME(TSTART, TEND)  (((TEND).tv_sec - (TSTART).tv_sec) * 1000000 + ((TEND).tv_usec - (TSTART).tv_usec))


typedef struct {
    cudaEvent_t cuda_event;
    int32_t event_type;
} ddt_cuda_event_t;

typedef struct {
    cudaStream_t ddt_cuda_stream[NB_STREAMS];
    int32_t current_stream_id;
} ddt_cuda_stream_t;

typedef struct {
    size_t ncontig_disp;
    size_t contig_disp;
} ddt_cuda_iov_dist_cached_t;

typedef struct {
    ddt_cuda_iov_dist_cached_t* cuda_iov_dist_d;
    uint32_t cuda_iov_count;
    uint32_t* nb_bytes_h;
    uint8_t cuda_iov_is_cached;
} ddt_cuda_iov_total_cached_t;

typedef struct {
    ddt_cuda_iov_dist_cached_t* cuda_iov_dist_non_cached_h;
    ddt_cuda_iov_dist_cached_t* cuda_iov_dist_non_cached_d;
    cudaStream_t cuda_stream;
    cudaEvent_t cuda_event;
} ddt_cuda_iov_pipeline_block_non_cached_t;

typedef struct {
    ddt_cuda_iov_dist_cached_t* cuda_iov_dist_cached_h;
    cudaStream_t cuda_stream;
    cudaEvent_t cuda_event;
} ddt_cuda_iov_process_block_cached_t;

typedef struct ddt_cuda_buffer{
    unsigned char* gpu_addr;
    size_t size;
    struct ddt_cuda_buffer *next;
    struct ddt_cuda_buffer *prev;
} ddt_cuda_buffer_t;

typedef struct {
    ddt_cuda_buffer_t *head;
    ddt_cuda_buffer_t *tail;
    size_t nb_elements;
} ddt_cuda_list_t;

typedef struct {
    int device_id;
    unsigned char* gpu_buffer;
    ddt_cuda_list_t buffer_free;
    ddt_cuda_list_t buffer_used;
    size_t buffer_free_size;
    size_t buffer_used_size;
    ddt_cuda_stream_t *cuda_streams;
    ddt_cuda_iov_pipeline_block_non_cached_t *cuda_iov_pipeline_block_non_cached[NB_PIPELINE_NON_CACHED_BLOCKS];
    ddt_cuda_iov_process_block_cached_t *cuda_iov_process_block_cached[NB_CACHED_BLOCKS];
    uint32_t cuda_iov_process_block_cached_first_avail;
    uint32_t cuda_iov_pipeline_block_non_cached_first_avail;
    cudaEvent_t memcpy_event;
} ddt_cuda_device_t;

extern ddt_cuda_list_t *cuda_free_list;
extern ddt_cuda_device_t *cuda_devices;
extern ddt_cuda_device_t *current_cuda_device;
extern uint32_t cuda_iov_cache_enabled;

extern int opal_datatype_cuda_output;
extern size_t opal_datatype_cuda_buffer_size;
      

__global__ void opal_generic_simple_pack_cuda_iov_cached_kernel( ddt_cuda_iov_dist_cached_t* cuda_iov_dist,
                                                                 uint32_t cuda_iov_pos,
                                                                 uint32_t cuda_iov_count,
                                                                 uint32_t ddt_extent,
                                                                 uint32_t current_count,
                                                                 int nb_blocks_used,
                                                                 unsigned char* source_base,
                                                                 unsigned char* destination_base);

__global__ void opal_generic_simple_unpack_cuda_iov_cached_kernel( ddt_cuda_iov_dist_cached_t* cuda_iov_dist,
                                                                   uint32_t cuda_iov_pos,
                                                                   uint32_t cuda_iov_count,
                                                                   uint32_t ddt_extent,
                                                                   uint32_t current_count,
                                                                   int nb_blocks_used,
                                                                   unsigned char* destination_base,
                                                                   unsigned char* source_base,
                                                                   size_t cuda_iov_partial_length_start,
                                                                   size_t cuda_iov_partial_length_end);

#define CUDA_ERROR_CHECK(err)                                                                                       \
    if (err != cudaSuccess) {                                                                                       \
        OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "CUDA calls error %s\n", cudaGetErrorString(err)));      \
    }                                                                                                               \

extern "C"
{
int32_t opal_convertor_set_position_nocheck( opal_convertor_t* convertor, size_t* position );

int32_t opal_convertor_raw( opal_convertor_t* pConvertor, 
		                    struct iovec* iov, uint32_t* iov_count,
		                    size_t* length );

int opal_convertor_raw_cached(struct opal_convertor_t *convertor,
                              const struct iovec **iov,
                              uint32_t* iov_count);
}

#endif  /* OPAL_DATATYPE_CUDA_INTERNAL_H_HAS_BEEN_INCLUDED */
