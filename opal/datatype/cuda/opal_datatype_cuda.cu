/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2014-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"
#include "opal/util/output.h"

#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"
#include <stdio.h>
#include <assert.h>
#include <stdarg.h>

ddt_cuda_list_t *cuda_free_list;
ddt_cuda_device_t *cuda_devices;
ddt_cuda_device_t *current_cuda_device;
uint32_t cuda_iov_cache_enabled;

extern size_t opal_datatype_cuda_buffer_size;

static inline ddt_cuda_buffer_t* obj_ddt_cuda_buffer_new()
{
    ddt_cuda_buffer_t *p = (ddt_cuda_buffer_t *)malloc(sizeof(ddt_cuda_buffer_t));
    p->next = NULL;
    p->prev = NULL;
    p->size = 0;
    p->gpu_addr = NULL;
    return p;
}

static inline void obj_ddt_cuda_buffer_chop(ddt_cuda_buffer_t *p)
{
    p->next = NULL;
    p->prev = NULL;
}

static inline void obj_ddt_cuda_buffer_reset(ddt_cuda_buffer_t *p)
{
    p->size = 0;
    p->gpu_addr = NULL;
}

static ddt_cuda_list_t* init_cuda_free_list()
{
    ddt_cuda_list_t *list = NULL;
    ddt_cuda_buffer_t *p, *prev;
    int i;
    list = (ddt_cuda_list_t *)malloc(sizeof(ddt_cuda_list_t));
    p = obj_ddt_cuda_buffer_new();
    list->head = p;
    prev = p;
    for (i = 1; i < DT_CUDA_FREE_LIST_SIZE; i++) {
        p = obj_ddt_cuda_buffer_new();
        prev->next = p;
        p->prev = prev;
        prev = p;
    }
    list->tail = p;
    list->nb_elements = DT_CUDA_FREE_LIST_SIZE;
    return list;
}

static inline ddt_cuda_buffer_t* cuda_list_pop_tail(ddt_cuda_list_t *list)
{
    ddt_cuda_buffer_t *p = list->tail;
    if (NULL != p) {
        list->nb_elements--;
        if (list->head == p) {
            list->head = NULL;
            list->tail = NULL;
        } else {
            list->tail = p->prev;
            p->prev->next = NULL;
            obj_ddt_cuda_buffer_chop(p);
        }
    }
    return p;
}

static inline void cuda_list_push_head(ddt_cuda_list_t *list, ddt_cuda_buffer_t *item)
{
    assert(item->next == NULL && item->prev == NULL);
    item->next = list->head;
    if (NULL == list->head) {
        list->tail = item;
    } else {
        list->head->prev = item;
    }
    list->head = item;
    list->nb_elements++;
}

static inline void cuda_list_push_tail(ddt_cuda_list_t *list, ddt_cuda_buffer_t *item)
{
    assert(item->next == NULL && item->prev == NULL);
    item->prev = list->tail;
    if (NULL == list->tail) {
        list->head = item;
    } else {
        list->tail->next = item;
    }
    list->tail = item;
    list->nb_elements++;
}

static inline void cuda_list_delete(ddt_cuda_list_t *list, ddt_cuda_buffer_t *item)
{
    if (item->prev == NULL && item->next == NULL) {
        list->head = NULL;
        list->tail = NULL;
    } else if (item->prev == NULL && item->next != NULL) {
        list->head = item->next;
        item->next->prev = NULL;
    } else if (item->next == NULL && item->prev != NULL) {
        list->tail = item->prev;
        item->prev->next = NULL;
    } else {
        item->prev->next = item->next;
        item->next->prev = item->prev;
    }
    list->nb_elements--;
    obj_ddt_cuda_buffer_chop(item);
}

static inline void cuda_list_insert_before(ddt_cuda_list_t *list, ddt_cuda_buffer_t *item, ddt_cuda_buffer_t *next)
{
    assert(item->next == NULL && item->prev == NULL);
    item->next = next;
    item->prev = next->prev;
    if (next->prev != NULL) {
        next->prev->next = item;
    }
    next->prev = item;
    if (list->head == next) {
        list->head = item;
    }
    list->nb_elements++;
}

/**
 * Collapse the list of free buffers by mergining consecutive buffers. As the property of this list
 * is continously maintained, we only have to parse it up to the newest inserted elements.
 */
static inline void cuda_list_item_merge_by_addr(ddt_cuda_list_t *list, ddt_cuda_buffer_t* last)
{
    ddt_cuda_buffer_t *current = list->head;
    ddt_cuda_buffer_t *next = NULL;
    void* stop_addr = last->gpu_addr;

    while(1) {  /* loop forever, the exit conditions are inside */
        if( NULL == (next = current->next) ) return;
        if ((current->gpu_addr + current->size) == next->gpu_addr) {
            current->size += next->size;
            cuda_list_delete(list, next);
            free(next);  /* release the element, and try to continue merging */
            continue;
        }
        current = current->next;
        if( NULL == current ) return;
        if( current->gpu_addr > stop_addr ) return;
    }
}

int32_t opal_datatype_cuda_kernel_init(void)
{
    uint32_t j;
    int device;
    cudaError cuda_err;

    cuda_err = cudaGetDevice(&device);
    if( cudaSuccess != cuda_err ) {
        OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "Cannot retrieve the device being used. Drop CUDA support!\n"));
        return OPAL_ERROR;
    }

    cuda_free_list = init_cuda_free_list();

    /* init cuda_iov */
    cuda_iov_cache_enabled = 1;

    /* init device */
    cuda_devices = (ddt_cuda_device_t *)malloc(sizeof(ddt_cuda_device_t));
    
    unsigned char *gpu_ptr = NULL;
    if (cudaMalloc((void **)(&gpu_ptr), sizeof(char) * opal_datatype_cuda_buffer_size) != cudaSuccess) {
        OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "cudaMalloc is failed in GPU %d\n", device));
        return OPAL_ERROR;
    }
    OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "DDT engine cudaMalloc buffer %p in GPU %d\n", gpu_ptr, device));
    cudaMemset(gpu_ptr, 0, sizeof(char) * opal_datatype_cuda_buffer_size);
    cuda_devices[0].gpu_buffer = gpu_ptr;

    cuda_devices[0].buffer_free_size = opal_datatype_cuda_buffer_size;
    ddt_cuda_buffer_t *p = obj_ddt_cuda_buffer_new();
    p->size = opal_datatype_cuda_buffer_size;
    p->gpu_addr = gpu_ptr;
    cuda_devices[0].buffer_free.head = p;
    cuda_devices[0].buffer_free.tail = cuda_devices[0].buffer_free.head;
    cuda_devices[0].buffer_free.nb_elements = 1;

    cuda_devices[0].buffer_used.head = NULL;
    cuda_devices[0].buffer_used.tail = NULL;
    cuda_devices[0].buffer_used_size = 0;
    cuda_devices[0].buffer_used.nb_elements = 0;

    cuda_devices[0].device_id = device;

    /* init cuda stream */
    ddt_cuda_stream_t *cuda_streams = (ddt_cuda_stream_t *)malloc(sizeof(ddt_cuda_stream_t));
    for (j = 0; j < NB_STREAMS; j++) {
        cuda_err = cudaStreamCreate(&(cuda_streams->ddt_cuda_stream[j]));
        CUDA_ERROR_CHECK(cuda_err);
    }

    cuda_streams->current_stream_id = 0;
    cuda_devices[0].cuda_streams = cuda_streams;
    cuda_err = cudaEventCreate(&(cuda_devices[0].memcpy_event), cudaEventDisableTiming);
    CUDA_ERROR_CHECK(cuda_err);

    /* init iov pipeline blocks */
    ddt_cuda_iov_pipeline_block_non_cached_t *cuda_iov_pipeline_block_non_cached = NULL;
    for (j = 0; j < NB_PIPELINE_NON_CACHED_BLOCKS; j++) {
        if (!cuda_iov_cache_enabled) {
            cuda_iov_pipeline_block_non_cached = (ddt_cuda_iov_pipeline_block_non_cached_t *)malloc(sizeof(ddt_cuda_iov_pipeline_block_non_cached_t));
            cuda_err = cudaMallocHost((void **)(&(cuda_iov_pipeline_block_non_cached->cuda_iov_dist_non_cached_h)), sizeof(ddt_cuda_iov_dist_cached_t) * CUDA_MAX_NB_BLOCKS * CUDA_IOV_MAX_TASK_PER_BLOCK);
            CUDA_ERROR_CHECK(cuda_err);
            cuda_err = cudaMalloc((void **)(&(cuda_iov_pipeline_block_non_cached->cuda_iov_dist_non_cached_d)), sizeof(ddt_cuda_iov_dist_cached_t) * CUDA_MAX_NB_BLOCKS * CUDA_IOV_MAX_TASK_PER_BLOCK);
            CUDA_ERROR_CHECK(cuda_err);
            cuda_err = cudaEventCreateWithFlags(&(cuda_iov_pipeline_block_non_cached->cuda_event), cudaEventDisableTiming);
            CUDA_ERROR_CHECK(cuda_err);
            cuda_iov_pipeline_block_non_cached->cuda_stream = NULL;
        }
        cuda_devices[0].cuda_iov_pipeline_block_non_cached[j] = cuda_iov_pipeline_block_non_cached;
        cuda_devices[0].cuda_iov_pipeline_block_non_cached_first_avail = 0;
    }

    /* init iov block for cached */
    ddt_cuda_iov_process_block_cached_t *cuda_iov_process_block_cached = NULL;
    for (j = 0; j < NB_CACHED_BLOCKS; j++) {
        if (cuda_iov_cache_enabled) {
            cuda_iov_process_block_cached = (ddt_cuda_iov_process_block_cached_t *)malloc(sizeof(ddt_cuda_iov_process_block_cached_t));
            cuda_iov_process_block_cached->cuda_iov_dist_cached_h = (ddt_cuda_iov_dist_cached_t *)malloc(sizeof(ddt_cuda_iov_dist_cached_t) * NUM_CUDA_IOV_PER_DDT);
            cuda_err = cudaEventCreateWithFlags(&(cuda_iov_process_block_cached->cuda_event), cudaEventDisableTiming);
            CUDA_ERROR_CHECK(cuda_err);
            cuda_iov_process_block_cached->cuda_stream = NULL;
        }
        cuda_devices[0].cuda_iov_process_block_cached[j] = cuda_iov_process_block_cached;
        cuda_devices[0].cuda_iov_process_block_cached_first_avail = 0;
    }
    current_cuda_device = &(cuda_devices[0]);

    cuda_err = cudaDeviceSynchronize();
    CUDA_ERROR_CHECK(cuda_err);
    return OPAL_SUCCESS;
}

int32_t opal_datatype_cuda_kernel_fini(void)
{
    uint32_t j;
    cudaError_t cuda_err;

    /* free gpu buffer */
    cuda_err = cudaFree(cuda_devices[0].gpu_buffer);
    CUDA_ERROR_CHECK(cuda_err);
    /* destory cuda stream and iov*/
    for (j = 0; j < NB_STREAMS; j++) {
        cuda_err = cudaStreamDestroy(cuda_devices[0].cuda_streams->ddt_cuda_stream[j]);
        CUDA_ERROR_CHECK(cuda_err);
    }
    free(cuda_devices[0].cuda_streams);

    ddt_cuda_iov_pipeline_block_non_cached_t *cuda_iov_pipeline_block_non_cached = NULL;
    for (j = 0; j < NB_PIPELINE_NON_CACHED_BLOCKS; j++) {
        if( NULL != (cuda_iov_pipeline_block_non_cached = cuda_devices[0].cuda_iov_pipeline_block_non_cached[j]) ) {
            if (cuda_iov_pipeline_block_non_cached->cuda_iov_dist_non_cached_d != NULL) {
                cuda_err = cudaFree(cuda_iov_pipeline_block_non_cached->cuda_iov_dist_non_cached_d);
                CUDA_ERROR_CHECK(cuda_err);
                cuda_iov_pipeline_block_non_cached->cuda_iov_dist_non_cached_d = NULL;
                cuda_iov_pipeline_block_non_cached->cuda_iov_dist_non_cached_h = NULL;
            }
            cuda_err = cudaEventDestroy(cuda_iov_pipeline_block_non_cached->cuda_event);
            CUDA_ERROR_CHECK(cuda_err);
            cuda_iov_pipeline_block_non_cached->cuda_stream = NULL;
            free(cuda_iov_pipeline_block_non_cached);
            cuda_iov_pipeline_block_non_cached = NULL;
        }
    }

    ddt_cuda_iov_process_block_cached_t *cuda_iov_process_block_cached = NULL;
    for (j = 0; j < NB_CACHED_BLOCKS; j++) {
        if( NULL != (cuda_iov_process_block_cached = cuda_devices[0].cuda_iov_process_block_cached[j]) ) {
            if (cuda_iov_process_block_cached->cuda_iov_dist_cached_h != NULL) {
                free(cuda_iov_process_block_cached->cuda_iov_dist_cached_h);
                cuda_iov_process_block_cached->cuda_iov_dist_cached_h = NULL;
            }
            cuda_err = cudaEventDestroy(cuda_iov_process_block_cached->cuda_event);
            CUDA_ERROR_CHECK(cuda_err);
            cuda_iov_process_block_cached->cuda_stream = NULL;
            free(cuda_iov_process_block_cached);
            cuda_iov_process_block_cached = NULL;
        }
    }
    cuda_devices[0].cuda_streams = NULL;
    cuda_err = cudaEventDestroy(cuda_devices[0].memcpy_event);
    CUDA_ERROR_CHECK(cuda_err);
    
    free(cuda_devices);
    cuda_devices = NULL;    
    current_cuda_device = NULL;

    return OPAL_SUCCESS;
}

void* opal_datatype_cuda_cached_cuda_iov_init(uint32_t size)
{
#if OPAL_DATATYPE_CUDA_IOV_CACHE
    ddt_cuda_iov_total_cached_t *tmp = (ddt_cuda_iov_total_cached_t *)malloc(sizeof(ddt_cuda_iov_total_cached_t) +
                                                                             size * sizeof(uint32_t));
    if( NULL != tmp ) {
        tmp->cuda_iov_dist_d    = NULL;
        tmp->cuda_iov_count     = size;
        tmp->cuda_iov_is_cached = 0;
        tmp->nb_bytes_h         = (uint32_t*)((char*)tmp + sizeof(ddt_cuda_iov_total_cached_t));
        OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "Malloc cuda_iov_dist_cached for ddt is successed, cached cuda iov %p, nb_bytes_h %p, size %d.\n", tmp, tmp->nb_bytes_h, size));
        return (void*)tmp;
    }
    OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "Malloc cuda_iov_dist_cached for ddt is failed.\n"));
#else
    OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "cuda iov cache is not enabled.\n"));
#endif /* OPAL_DATATYPE_CUDA_IOV_CACHE */
    return NULL;
}

void opal_datatype_cuda_cached_cuda_iov_fini(void* cached_cuda_iov)
{
#if OPAL_DATATYPE_CUDA_IOV_CACHE
    ddt_cuda_iov_total_cached_t *tmp = (ddt_cuda_iov_total_cached_t *) cached_cuda_iov;
    if (NULL != tmp) {
        OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "Free cuda_iov_dist for ddt is successed %p.\n", cached_cuda_iov));
        if (NULL != tmp->cuda_iov_dist_d) {
            cudaError_t cuda_err = cudaFree(tmp->cuda_iov_dist_d);
            CUDA_ERROR_CHECK(cuda_err);
            tmp->cuda_iov_dist_d = NULL;
        }
        tmp->nb_bytes_h = NULL;
        free(tmp);
    }
#endif /* OPAL_DATATYPE_CUDA_IOV_CACHE */
}

static inline int32_t
opal_datatype_cuda_cached_cuda_iov_isfull(ddt_cuda_iov_total_cached_t *cached_cuda_iov,
                                          ddt_cuda_iov_dist_cached_t **cuda_iov_dist_h,
                                          uint32_t nb_blocks_used)
{
    if (nb_blocks_used < cached_cuda_iov->cuda_iov_count) {
        return 0;
    }
realloc_cuda_iov:
    cached_cuda_iov->nb_bytes_h = (uint32_t *)realloc(cached_cuda_iov->nb_bytes_h, sizeof(uint32_t)*cached_cuda_iov->cuda_iov_count*2);
    assert(cached_cuda_iov->nb_bytes_h != NULL);
    cached_cuda_iov->cuda_iov_count *= 2;
    if (nb_blocks_used >= cached_cuda_iov->cuda_iov_count) {
        goto realloc_cuda_iov;
    }
    return 1;
}

/* cached_cuda_iov_d is not ready until explicitly sync with current cuda stream */
int32_t opal_datatype_cuda_cache_cuda_iov(opal_convertor_t* pConvertor, uint32_t *cuda_iov_count)
{
    uint32_t i, j;
    uint32_t count_desc, nb_blocks_per_description, residue_desc;
    uint32_t thread_per_block, nb_blocks_used;
    size_t length_per_iovec;
    uint32_t alignment;
    ddt_cuda_iov_process_block_cached_t *cuda_iov_process_block_cached = NULL;
    ddt_cuda_iov_total_cached_t* cached_cuda_iov = NULL;
    ddt_cuda_iov_dist_cached_t *cached_cuda_iov_dist_d = NULL;
    ddt_cuda_iov_dist_cached_t *cuda_iov_dist_h = NULL;
    cudaStream_t cuda_stream_iov = NULL;
    cudaError_t cuda_err;
    const struct iovec *ddt_iov = NULL;
    uint32_t ddt_iov_count = 0;
    size_t ncontig_disp_base;
    size_t contig_disp = 0;
    uint32_t *cached_cuda_iov_nb_bytes_list_h = NULL;
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;

    opal_datatype_t *datatype = (opal_datatype_t *)pConvertor->pDesc;

    opal_convertor_raw_cached( pConvertor, &ddt_iov, &ddt_iov_count);
    if (ddt_iov == NULL) {
        OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "Can not get ddt iov\n"));
        return OPAL_ERROR;
    }

    cached_cuda_iov = (ddt_cuda_iov_total_cached_t *)opal_datatype_cuda_cached_cuda_iov_init(NUM_CUDA_IOV_PER_DDT);
    if (cached_cuda_iov == NULL) {
        OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "Can not init cuda iov\n"));
        return OPAL_ERROR;
    }
    cached_cuda_iov_nb_bytes_list_h = cached_cuda_iov->nb_bytes_h;
    nb_blocks_used = 0;
    cuda_iov_process_block_cached = current_cuda_device->cuda_iov_process_block_cached[current_cuda_device->cuda_iov_process_block_cached_first_avail];
    current_cuda_device->cuda_iov_process_block_cached_first_avail ++;
    if (current_cuda_device->cuda_iov_process_block_cached_first_avail >= NB_CACHED_BLOCKS) {
        current_cuda_device->cuda_iov_process_block_cached_first_avail = 0;
    }
    cuda_err = cudaEventSynchronize(cuda_iov_process_block_cached->cuda_event);
    CUDA_ERROR_CHECK(cuda_err);

    if (pConvertor->stream == NULL) {
        cuda_iov_process_block_cached->cuda_stream = cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id];
    } else {
        cuda_iov_process_block_cached->cuda_stream = (cudaStream_t)pConvertor->stream;
    }
    cuda_iov_dist_h = cuda_iov_process_block_cached->cuda_iov_dist_cached_h;
    cuda_stream_iov = cuda_iov_process_block_cached->cuda_stream;
    thread_per_block = CUDA_WARP_SIZE * 64;

    for (i = 0; i < ddt_iov_count; i++) {
        length_per_iovec = ddt_iov[i].iov_len;
        ncontig_disp_base = (size_t)(ddt_iov[i].iov_base);

        /* block size is either multiple of ALIGNMENT_DOUBLE or residue */
        alignment = ALIGNMENT_DOUBLE * 1;

        count_desc = length_per_iovec / alignment;
        residue_desc = length_per_iovec % alignment;
        nb_blocks_per_description = (count_desc + thread_per_block - 1) / thread_per_block;
        OPAL_OUTPUT_VERBOSE((10, opal_datatype_cuda_output, "Cache cuda IOV description %d, size %d, residue %d, alignment %d, nb_block_aquired %d\n", i, count_desc, residue_desc, alignment, nb_blocks_per_description));
        if (opal_datatype_cuda_cached_cuda_iov_isfull(cached_cuda_iov, &(cuda_iov_process_block_cached->cuda_iov_dist_cached_h), nb_blocks_used + nb_blocks_per_description + 1)) {
            cached_cuda_iov_nb_bytes_list_h = cached_cuda_iov->nb_bytes_h;
            cuda_iov_dist_h = (ddt_cuda_iov_dist_cached_t *)realloc(cuda_iov_dist_h, sizeof(ddt_cuda_iov_dist_cached_t)*cached_cuda_iov->cuda_iov_count);
            assert(cuda_iov_dist_h != NULL);
            cuda_iov_process_block_cached->cuda_iov_dist_cached_h = cuda_iov_dist_h;
        }

        for (j = 0; j < nb_blocks_per_description; j++) {
            cuda_iov_dist_h[nb_blocks_used].ncontig_disp = ncontig_disp_base + j * thread_per_block * alignment;
            cuda_iov_dist_h[nb_blocks_used].contig_disp = contig_disp;
            if ( (j+1) * thread_per_block <= count_desc) {
                cached_cuda_iov_nb_bytes_list_h[nb_blocks_used] = thread_per_block * alignment;
            } else {
                cached_cuda_iov_nb_bytes_list_h[nb_blocks_used] = (count_desc - j*thread_per_block) * alignment; 
            }
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
            assert(cached_cuda_iov_nb_bytes_list_h[nb_blocks_used] > 0);
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
            contig_disp += cached_cuda_iov_nb_bytes_list_h[nb_blocks_used];
            OPAL_OUTPUT_VERBOSE((12, opal_datatype_cuda_output, "Cache cuda IOV \tblock %d, ncontig_disp %ld, contig_disp %ld, nb_bytes %d\n", nb_blocks_used, cuda_iov_dist_h[nb_blocks_used].ncontig_disp, cuda_iov_dist_h[nb_blocks_used].contig_disp, cached_cuda_iov_nb_bytes_list_h[nb_blocks_used]));
            nb_blocks_used ++;
        }

        /* handle residue */
        if (residue_desc != 0) {
            cuda_iov_dist_h[nb_blocks_used].ncontig_disp = ncontig_disp_base + length_per_iovec / alignment * alignment;
            cuda_iov_dist_h[nb_blocks_used].contig_disp = contig_disp;
            cached_cuda_iov_nb_bytes_list_h[nb_blocks_used] = length_per_iovec - length_per_iovec / alignment * alignment;
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
            assert(cached_cuda_iov_nb_bytes_list_h[nb_blocks_used] > 0);
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
            contig_disp += cached_cuda_iov_nb_bytes_list_h[nb_blocks_used];
            OPAL_OUTPUT_VERBOSE((12, opal_datatype_cuda_output, "Cache cuda IOV \tblock %d, ncontig_disp %ld, contig_disp %ld, nb_bytes %d\n", nb_blocks_used, cuda_iov_dist_h[nb_blocks_used].ncontig_disp, cuda_iov_dist_h[nb_blocks_used].contig_disp, cached_cuda_iov_nb_bytes_list_h[nb_blocks_used]));
            nb_blocks_used ++;
        }
    }
    /* use additional entry to store the size of entire contiguous buffer needed for one ddt */
    cuda_iov_dist_h[nb_blocks_used].contig_disp = contig_disp;
    cudaMalloc((void **)(&cached_cuda_iov_dist_d), sizeof(ddt_cuda_iov_dist_cached_t) * (nb_blocks_used+1));
    if (cached_cuda_iov_dist_d == NULL) {
        OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "Can not malloc cuda iov in GPU\n"));
        return OPAL_ERROR;
    }
    cuda_err = cudaMemcpyAsync(cached_cuda_iov_dist_d, cuda_iov_dist_h, sizeof(ddt_cuda_iov_dist_cached_t)*(nb_blocks_used+1),
                               cudaMemcpyHostToDevice, cuda_stream_iov);
    CUDA_ERROR_CHECK(cuda_err);
    cached_cuda_iov->cuda_iov_dist_d = cached_cuda_iov_dist_d;
    datatype->cached_iovec->cached_cuda_iov = (void*)cached_cuda_iov;
    *cuda_iov_count = nb_blocks_used;

    ddt_cuda_iov_total_cached_t *tmp = (ddt_cuda_iov_total_cached_t *)datatype->cached_iovec->cached_cuda_iov;
    tmp->cuda_iov_count = *cuda_iov_count;
    tmp->cuda_iov_is_cached = 1;

    cuda_err = cudaEventRecord(cuda_iov_process_block_cached->cuda_event, cuda_stream_iov);
    CUDA_ERROR_CHECK(cuda_err);
    return OPAL_SUCCESS;
}

uint8_t opal_datatype_cuda_iov_to_cuda_iov(opal_convertor_t* pConvertor,
                                           const struct iovec *ddt_iov,
                                           ddt_cuda_iov_dist_cached_t* cuda_iov_dist_h_current,
                                           uint32_t ddt_iov_start_pos, uint32_t ddt_iov_end_pos,
                                           size_t *buffer_size, uint32_t *nb_blocks_used,
                                           size_t *total_converted, size_t *contig_disp_out, uint32_t *current_ddt_iov_pos)
{
    size_t ncontig_disp_base, contig_disp = 0, current_cuda_iov_length = 0;
    uint32_t count_desc, nb_blocks_per_description, residue_desc, thread_per_block;
    uint8_t buffer_isfull = 0, alignment;
    size_t length_per_iovec;
    uint32_t i, j;

    thread_per_block = CUDA_WARP_SIZE * 5;

    for (i = ddt_iov_start_pos; i < ddt_iov_end_pos && !buffer_isfull; i++) {
        if (pConvertor->current_iov_partial_length > 0) {
            ncontig_disp_base = (size_t)(ddt_iov[i].iov_base) + ddt_iov[i].iov_len - pConvertor->current_iov_partial_length;
            length_per_iovec = pConvertor->current_iov_partial_length;
            pConvertor->current_iov_partial_length = 0;
        } else {
            ncontig_disp_base = (size_t)(ddt_iov[i].iov_base);
            length_per_iovec = ddt_iov[i].iov_len;
        }
        if (*buffer_size < length_per_iovec) {
            pConvertor->current_iov_pos = i;
            pConvertor->current_iov_partial_length = length_per_iovec - *buffer_size;
            length_per_iovec = *buffer_size;
            buffer_isfull = 1;
        }
        *buffer_size -= length_per_iovec;
        *total_converted += length_per_iovec;

        alignment = ALIGNMENT_DOUBLE;

        count_desc = length_per_iovec / alignment;
        residue_desc = length_per_iovec % alignment;
        nb_blocks_per_description = (count_desc + thread_per_block - 1) / thread_per_block;
        if ((*nb_blocks_used + nb_blocks_per_description + 1) > (CUDA_MAX_NB_BLOCKS*CUDA_IOV_MAX_TASK_PER_BLOCK)) {
            break;
        }
        OPAL_OUTPUT_VERBOSE((10, opal_datatype_cuda_output, "DDT IOV to CUDA IOV description %d, size %d, residue %d, alignment %d, nb_block_aquired %d\n", i, count_desc, residue_desc, alignment, nb_blocks_per_description));
        for (j = 0; j < nb_blocks_per_description; j++) {
            cuda_iov_dist_h_current[*nb_blocks_used].ncontig_disp = ncontig_disp_base + j * thread_per_block * alignment;
            cuda_iov_dist_h_current[*nb_blocks_used].contig_disp = contig_disp;
            if ( (j+1) * thread_per_block <= count_desc) {
                current_cuda_iov_length = thread_per_block * alignment;
            } else {
                current_cuda_iov_length = (count_desc - j*thread_per_block) * alignment;
            }
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
            assert(current_cuda_iov_length > 0);
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
            contig_disp += current_cuda_iov_length;
            OPAL_OUTPUT_VERBOSE((12, opal_datatype_cuda_output, "DDT IOV to CUDA IOV \tblock %d, ncontig_disp %ld, contig_disp %ld, nb_bytes %ld\n", *nb_blocks_used, cuda_iov_dist_h_current[*nb_blocks_used].ncontig_disp, cuda_iov_dist_h_current[*nb_blocks_used].contig_disp, current_cuda_iov_length));
            (*nb_blocks_used) ++;
            assert (*nb_blocks_used < CUDA_MAX_NB_BLOCKS*CUDA_IOV_MAX_TASK_PER_BLOCK);
        }

        /* handle residue */
        if (residue_desc != 0) {
            cuda_iov_dist_h_current[*nb_blocks_used].ncontig_disp = ncontig_disp_base + length_per_iovec / alignment * alignment;
            cuda_iov_dist_h_current[*nb_blocks_used].contig_disp = contig_disp;
            current_cuda_iov_length= length_per_iovec - length_per_iovec / alignment * alignment;
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
            assert(current_cuda_iov_length > 0);
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
            contig_disp += current_cuda_iov_length;
            OPAL_OUTPUT_VERBOSE((12, opal_datatype_cuda_output, "DDT IOV to CUDA IOV \tblock %d, ncontig_disp %ld, contig_disp %ld, nb_bytes %ld\n", *nb_blocks_used, cuda_iov_dist_h_current[*nb_blocks_used].ncontig_disp, cuda_iov_dist_h_current[*nb_blocks_used].contig_disp, current_cuda_iov_length));
            (*nb_blocks_used) ++;
            assert (*nb_blocks_used < CUDA_MAX_NB_BLOCKS*CUDA_IOV_MAX_TASK_PER_BLOCK);
        }
    }
    cuda_iov_dist_h_current[*nb_blocks_used].contig_disp = contig_disp;
    *contig_disp_out = contig_disp;
    *current_ddt_iov_pos = i;
    return buffer_isfull;
}

void opal_datatype_cuda_get_cached_cuda_iov(struct opal_convertor_t *convertor,
                                            ddt_cuda_iov_total_cached_t **cached_cuda_iov)
{
    *cached_cuda_iov = NULL;
    if (NULL != convertor->pDesc->cached_iovec) {
        *cached_cuda_iov = (ddt_cuda_iov_total_cached_t *)convertor->pDesc->cached_iovec->cached_cuda_iov;
    }
}

uint8_t opal_datatype_cuda_cuda_iov_is_cached(struct opal_convertor_t *convertor)
{
    opal_datatype_t *datatype = (opal_datatype_t *)convertor->pDesc;
    if (NULL == datatype->cached_iovec) {
        return 0;
    }
    if (NULL == datatype->cached_iovec->cached_cuda_iov) {
        return 0;
    }
    ddt_cuda_iov_total_cached_t *tmp = (ddt_cuda_iov_total_cached_t *)datatype->cached_iovec->cached_cuda_iov;
    return tmp->cuda_iov_is_cached;
}

void opal_datatype_cuda_set_cuda_iov_position(struct opal_convertor_t *convertor,
                                              size_t ddt_offset,
                                              const uint32_t *cached_cuda_iov_nb_bytes_list_h,
                                              const uint32_t cuda_iov_count)
{
    size_t iov_size = 0, ddt_size;
    uint32_t i;

    convertor->current_iov_partial_length = 0;
    convertor->current_cuda_iov_pos = 0;
    convertor->current_count = 0;
    if (ddt_offset == 0)
       return;

    opal_datatype_type_size(convertor->pDesc, &ddt_size);
    convertor->current_count = ddt_offset / ddt_size;
    ddt_offset = ddt_offset % ddt_size;
    for(i = 0; i < cuda_iov_count; i++) {
        iov_size += cached_cuda_iov_nb_bytes_list_h[i];
        if (iov_size >= ddt_offset) {
            convertor->current_iov_partial_length = iov_size - ddt_offset;
            convertor->current_cuda_iov_pos = i;
            if (iov_size == ddt_offset)
                convertor->current_cuda_iov_pos++;
            return;
        }
    }
}

void opal_datatype_cuda_set_ddt_iov_position(struct opal_convertor_t *convertor,
                                             size_t ddt_offset,
                                             const struct iovec *ddt_iov,
                                             const uint32_t ddt_iov_count)
{
    size_t iov_size = 0, ddt_size;
    uint32_t i;

    convertor->current_iov_partial_length = 0;
    convertor->current_iov_pos = 0;
    convertor->current_count = 0;
    if (ddt_offset == 0)
       return;

    opal_datatype_type_size(convertor->pDesc, &ddt_size);
    convertor->current_count = ddt_offset / ddt_size;
    ddt_offset = ddt_offset % ddt_size;
    for(i = 0; i < ddt_iov_count; i++) {
        iov_size += ddt_iov[i].iov_len;
        if (iov_size >= ddt_offset) {
            convertor->current_iov_partial_length = iov_size - ddt_offset;
            convertor->current_iov_pos = i;
            if (iov_size == ddt_offset)
                convertor->current_iov_pos++;
            return;
        }
    }
}

/* following function will be called outside the cuda kernel lib */
int32_t opal_datatype_cuda_is_gpu_buffer(const void *ptr)
{
    CUmemorytype memType;
    CUdeviceptr dbuf = (CUdeviceptr)ptr;
    int res;

    res = cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dbuf);
    if (res != CUDA_SUCCESS) {
        /* If we cannot determine it is device pointer,
         * just assume it is not. */
        OPAL_OUTPUT_VERBOSE((1, opal_datatype_cuda_output, "!!!!!!! %p is not a gpu buffer. Take no-CUDA path!\n", ptr));
        return 0;
    }
    /* Anything but CU_MEMORYTYPE_DEVICE is not a GPU memory */
    return (memType == CU_MEMORYTYPE_DEVICE) ? 1 : 0;
}

void* opal_datatype_cuda_malloc_gpu_buffer(size_t size, int gpu_id)
{
    ddt_cuda_device_t *device = &cuda_devices[gpu_id];
    int dev_id = device->device_id;
    if (device->buffer_free_size < size) {
        OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "No GPU buffer for pack/unpack at device %d, if program crashes, please set --mca opal_opal_opal_datatype_cuda_buffer_size to larger size\n", dev_id));
        return NULL;
    }
    ddt_cuda_buffer_t *ptr = device->buffer_free.head;
    while (ptr != NULL) {
        if (ptr->size < size) {  /* Not enough room in this buffer, check next */
            ptr = ptr->next;
            continue;
        }
        void *addr = ptr->gpu_addr;
        ptr->size -= size;
        if (ptr->size == 0) {
            cuda_list_delete(&device->buffer_free, ptr);
            obj_ddt_cuda_buffer_reset(ptr);
            /* hold on this ptr object, we will reuse it right away */
        } else {
            ptr->gpu_addr += size;
            ptr = cuda_list_pop_tail(cuda_free_list);
            if( NULL == ptr )
                ptr = obj_ddt_cuda_buffer_new();
        }
        assert(NULL != ptr);
        ptr->size = size;
        ptr->gpu_addr = (unsigned char*)addr;
        cuda_list_push_head(&device->buffer_used, ptr);
        device->buffer_used_size += size;
        device->buffer_free_size -= size;
        OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "Malloc GPU buffer %p, size %lu, dev_id %d.\n", addr, size, dev_id));
        return addr;
    }
    return NULL;
}

void opal_datatype_cuda_free_gpu_buffer(void *addr, int gpu_id)
{
    ddt_cuda_device_t *device = &cuda_devices[gpu_id];
    ddt_cuda_buffer_t *ptr = device->buffer_used.head;

    /* Find the holder of this GPU allocation */
    for( ; (NULL != ptr) && (ptr->gpu_addr != addr); ptr = ptr->next );
    if (NULL == ptr) {  /* we could not find it. something went wrong */
        OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "addr %p is not managed.\n", addr));
        return;
    }
    cuda_list_delete(&device->buffer_used, ptr);
    /* Insert the element in the list of free buffers ordered by the addr */
    ddt_cuda_buffer_t *ptr_next = device->buffer_free.head;
    while (ptr_next != NULL) {
        if (ptr_next->gpu_addr > addr) {
            break;
        }
        ptr_next = ptr_next->next;
    }
    if (ptr_next == NULL) {  /* buffer_free is empty, or insert to last one */
        cuda_list_push_tail(&device->buffer_free, ptr);
    } else {
        cuda_list_insert_before(&device->buffer_free, ptr, ptr_next);
    }
    size_t size = ptr->size;
    cuda_list_item_merge_by_addr(&device->buffer_free, ptr);
    device->buffer_free_size += size;
    device->buffer_used_size -= size;
    OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "Free GPU buffer %p, size %lu\n", addr, size));
}

void opal_datatype_cuda_d2dcpy_async(void* dst, const void* src, size_t count, void* stream)
{
    cudaError_t cuda_err = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    CUDA_ERROR_CHECK(cuda_err);
}

void opal_datatype_cuda_d2dcpy(void* dst, const void* src, size_t count, void* stream)
{
    cudaError_t cuda_err = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    CUDA_ERROR_CHECK(cuda_err);
    cuda_err = cudaStreamSynchronize((cudaStream_t)stream);
    CUDA_ERROR_CHECK(cuda_err);
}

void* opal_datatype_cuda_get_cuda_stream_by_id(int stream_id)
{
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    return (void*)cuda_streams->ddt_cuda_stream[stream_id];
}

void *opal_datatype_cuda_get_current_cuda_stream()
{
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    return (void*)cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id];
}

void opal_datatype_cuda_sync_current_cuda_stream()
{
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    cudaError_t cuda_err = cudaStreamSynchronize(cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
    CUDA_ERROR_CHECK(cuda_err);
}

void opal_datatype_cuda_sync_cuda_stream(int stream_id)
{
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    cudaError cuda_err = cudaStreamSynchronize(cuda_streams->ddt_cuda_stream[stream_id]);
    CUDA_ERROR_CHECK(cuda_err);
}

void* opal_datatype_cuda_alloc_event(int32_t nb_events, int32_t *loc)
{
    *loc = 0;
    ddt_cuda_event_t *event_list = (ddt_cuda_event_t *)malloc(sizeof(ddt_cuda_event_t) * nb_events);
    cudaError_t cuda_err;
    for (int i = 0; i < nb_events; i++) {
        cuda_err = cudaEventCreateWithFlags(&(event_list[i].cuda_event), cudaEventDisableTiming);
        CUDA_ERROR_CHECK(cuda_err);
    }
    return (void*)event_list;
}

void opal_datatype_cuda_free_event(void *cuda_event_list, int32_t nb_events)
{
    ddt_cuda_event_t *event_list = (ddt_cuda_event_t *)cuda_event_list;
    cudaError_t cuda_err;
    for (int i = 0; i < nb_events; i++) {
        cuda_err = cudaEventDestroy(event_list[i].cuda_event);
        CUDA_ERROR_CHECK(cuda_err);
    }
    free (event_list);
    return;
}

int32_t opal_datatype_cuda_event_query(void *cuda_event_list, int32_t i)
{
    ddt_cuda_event_t *event_list = (ddt_cuda_event_t *)cuda_event_list;
    cudaError_t rv = cudaEventQuery(event_list[i].cuda_event);
    if (rv == cudaSuccess) {
        return 1;
    } else if (rv == cudaErrorNotReady) {
        return 0;
    } else {
        OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "cuda event query error.\n"));
        return -1;
    }
}

int32_t opal_datatype_cuda_event_sync(void *cuda_event_list, int32_t i)
{
    ddt_cuda_event_t *event_list = (ddt_cuda_event_t *)cuda_event_list;
    cudaError_t rv = cudaEventSynchronize(event_list[i].cuda_event);
    if (rv == cudaSuccess) {
        return 1;
    }
    OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "cuda event sync error.\n"));
    return -1;
}

int32_t opal_datatype_cuda_event_record(void *cuda_event_list, int32_t i, void* stream)
{
    ddt_cuda_event_t *event_list = (ddt_cuda_event_t *)cuda_event_list;
    cudaError_t rv = cudaEventRecord(event_list[i].cuda_event, (cudaStream_t)stream);
    if (rv == cudaSuccess) {
        return 1;
    }
    OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "cuda event record error.\n"));
    return -1;
}

void opal_dump_cuda_list(ddt_cuda_list_t *list)
{
    ddt_cuda_buffer_t *ptr = NULL;
    ptr = list->head;
    OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "DUMP cuda list %p, nb_elements %zu\n", list, list->nb_elements));
    while (ptr != NULL) {
        OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "\titem addr %p, size %ld.\n", ptr->gpu_addr, ptr->size));
        ptr = ptr->next;
    }
}

