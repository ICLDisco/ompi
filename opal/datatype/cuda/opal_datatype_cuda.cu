#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <stdarg.h> 


ddt_cuda_list_t *cuda_free_list;
ddt_cuda_device_t *cuda_devices;
ddt_cuda_device_t *current_cuda_device;
struct iovec cuda_iov[CUDA_NB_IOV];
uint32_t cuda_iov_count;

//uint8_t ALIGNMENT_DOUBLE, ALIGNMENT_FLOAT, ALIGNMENT_CHAR;


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
    ddt_cuda_buffer_t *p = NULL;
    p = list->tail;
    if (p == NULL) {
        return p;
    } else {
        list->nb_elements --;
        if (list->head == p) {
            list->head = NULL;
            list->tail = NULL;
        } else {
            list->tail = p->prev;
            p->prev->next = NULL;
            obj_ddt_cuda_buffer_chop(p);
        }
        return p;
    }
}

static inline void cuda_list_push_head(ddt_cuda_list_t *list, ddt_cuda_buffer_t *item)
{
    ddt_cuda_buffer_t * orig_head = list->head;
    assert(item->next == NULL && item->prev == NULL);
    list->head = item;
    item->next = orig_head;
    if (orig_head == NULL) {
        list->tail = item;
    } else {
        orig_head->prev = item;
    }
    list->nb_elements ++;
}

static inline void cuda_list_push_tail(ddt_cuda_list_t *list, ddt_cuda_buffer_t *item)
{
    ddt_cuda_buffer_t * orig_tail = list->tail;
    assert(item->next == NULL && item->prev == NULL);
    list->tail = item;
    item->prev = orig_tail;
    if (orig_tail == NULL) {
        list->head = item;
    } else {
        orig_tail->next = item;
    }
    list->nb_elements ++;
}

static inline void cuda_list_delete(ddt_cuda_list_t *list, ddt_cuda_buffer_t *item)
{
    if (item->prev == NULL && item->next == NULL) {
        list->head = NULL;
        list->tail = NULL;
    }else if (item->prev == NULL && item->next != NULL) {
        list->head = item->next;
        item->next->prev = NULL;
    } else if (item->next == NULL && item->prev != NULL) {
        list->tail = item->prev;
        item->prev->next = NULL;
    } else {
        item->prev->next = item->next;
        item->next->prev = item->prev;
    }
    list->nb_elements --;
    obj_ddt_cuda_buffer_chop(item);
}

static inline void cuda_list_insert_before(ddt_cuda_list_t *list, ddt_cuda_buffer_t *item, ddt_cuda_buffer_t *next)
{
    assert(item->next == NULL && item->prev == NULL);
    item->next = next;
    item->prev = next->prev;
    next->prev = item;
    if (list->head == next) {
        list->head = item;
    }
    list->nb_elements ++;
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

void opal_cuda_output(int output_id, const char *format, ...)
{
    if (output_id >= 0 && output_id <= OPAL_DATATYPE_CUDA_DEBUG_LEVEL) {
        va_list arglist;
        fprintf( stderr, "[Debug %d]: ", output_id );
        va_start(arglist, format);
        vfprintf(stderr, format, arglist);
        va_end(arglist);
    }
}

int32_t opal_datatype_cuda_init(void)
{
    uint32_t i, j;
    int device;
    cudaError res;

    res = cudaGetDevice(&device);
    if( cudaSuccess != res ) {
        opal_cuda_output(0, "Cannot retrieve the device being used. Drop CUDA support!\n");
        return OPAL_ERROR;
    }    

    cuda_free_list = init_cuda_free_list();
    
    /* init device */
    cuda_devices = (ddt_cuda_device_t *)malloc(sizeof(ddt_cuda_device_t)*NB_GPUS);
    for (i = 0; i < NB_GPUS; i++) {
        unsigned char *gpu_ptr = NULL;
        if (cudaMalloc((void **)(&gpu_ptr), sizeof(char)*DT_CUDA_BUFFER_SIZE) != cudaSuccess) {
            DT_CUDA_DEBUG( opal_cuda_output( 0, "cudaMalloc is failed in GPU %d\n", i); );
            return OPAL_ERROR;
        }
        DT_CUDA_DEBUG ( opal_cuda_output(2, "DDT engine cudaMalloc buffer %p in GPU %d\n", gpu_ptr, i););
        cudaMemset(gpu_ptr, 0, sizeof(char)*DT_CUDA_BUFFER_SIZE);
        cuda_devices[i].gpu_buffer = gpu_ptr;
        
        cuda_devices[i].buffer_free_size = DT_CUDA_BUFFER_SIZE;
        ddt_cuda_buffer_t *p = obj_ddt_cuda_buffer_new();
        p->size = DT_CUDA_BUFFER_SIZE;
        p->gpu_addr = gpu_ptr;
        cuda_devices[i].buffer_free.head = p;
        cuda_devices[i].buffer_free.tail = cuda_devices[i].buffer_free.head;
        cuda_devices[i].buffer_free.nb_elements = 1;
        
        cuda_devices[i].buffer_used.head = NULL;
        cuda_devices[i].buffer_used.tail = NULL;
        cuda_devices[i].buffer_used_size = 0;
        cuda_devices[i].buffer_used.nb_elements = 0;
    
        /* init cuda stream */
        ddt_cuda_stream_t *cuda_streams = (ddt_cuda_stream_t *)malloc(sizeof(ddt_cuda_stream_t));
        ddt_cuda_iov_pipeline_block_t *cuda_iov_pipeline_block = NULL;
        for (j = 0; j < NB_STREAMS; j++) {
            cudaStreamCreate(&(cuda_streams->opal_cuda_stream[j]));
            cuda_iov_pipeline_block = (ddt_cuda_iov_pipeline_block_t *)malloc(sizeof(ddt_cuda_iov_pipeline_block_t));
            cudaMallocHost((void **)(&(cuda_iov_pipeline_block->cuda_iov_dist_h)), sizeof(ddt_cuda_iov_dist_t) * CUDA_MAX_NB_BLOCKS * CUDA_IOV_MAX_TASK_PER_BLOCK);
            cudaMalloc((void **)(&(cuda_iov_pipeline_block->cuda_iov_dist_d)), sizeof(ddt_cuda_iov_dist_t) * CUDA_MAX_NB_BLOCKS * CUDA_IOV_MAX_TASK_PER_BLOCK);
            cuda_iov_pipeline_block->cuda_stream = &(cuda_streams->opal_cuda_stream[0]);
            cuda_iov_pipeline_block->cuda_stream_id = 0;
            cudaEventCreate(&(cuda_iov_pipeline_block->cuda_event), cudaEventDisableTiming);
            cuda_devices[i].cuda_iov_pipeline_block[j] = cuda_iov_pipeline_block;
        }
        cuda_streams->current_stream_id = 0;
        cuda_devices[i].cuda_streams = cuda_streams;
        cudaEventCreate(&(cuda_devices[i].memcpy_event), cudaEventDisableTiming);
    }
    current_cuda_device = &(cuda_devices[0]);
    
    /* init cuda_iov */
    cuda_iov_count = CUDA_NB_IOV;
    
    // /* init size for double, float, char */
    // ALIGNMENT_DOUBLE = sizeof(double);
    // ALIGNMENT_FLOAT = sizeof(float);
    // ALIGNMENT_CHAR = sizeof(char);
    
    cudaDeviceSynchronize();
    return OPAL_SUCCESS;
}

int32_t opal_datatype_cuda_fini(void)
{
    uint32_t i, j;
    
    for (i = 0; i < NB_GPUS; i++) {
        /* free gpu buffer */
        cudaFree(cuda_devices[i].gpu_buffer);   
        /* destory cuda stream and iov*/
        ddt_cuda_iov_pipeline_block_t *cuda_iov_pipeline_block = NULL;
        for (j = 0; j < NB_STREAMS; j++) {
            cudaStreamDestroy(cuda_devices[i].cuda_streams->opal_cuda_stream[j]);
            cuda_iov_pipeline_block = cuda_devices[i].cuda_iov_pipeline_block[j];
            if (cuda_iov_pipeline_block != NULL) {
                cudaFreeHost(cuda_iov_pipeline_block->cuda_iov_dist_h);
                cudaFree(cuda_iov_pipeline_block->cuda_iov_dist_d);
                cudaEventDestroy(cuda_iov_pipeline_block->cuda_event);
                cuda_iov_pipeline_block->cuda_stream = NULL;
                cuda_iov_pipeline_block->cuda_stream_id = -1;
                free(cuda_iov_pipeline_block);
                cuda_iov_pipeline_block = NULL;
            }
        }
        free(cuda_devices[i].cuda_streams);
        cuda_devices[i].cuda_streams = NULL;
        cudaEventDestroy(cuda_devices[i].memcpy_event);
    }
    current_cuda_device = NULL;
    return OPAL_SUCCESS;
}

int32_t opal_cuda_is_gpu_buffer(const void *ptr)
{
    int res;
    CUmemorytype memType;
    CUdeviceptr dbuf = (CUdeviceptr)ptr;
    res = cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dbuf);
    if (res != CUDA_SUCCESS) {
        /* If we cannot determine it is device pointer,
         * just assume it is not. */
        printf("!!!!!!! %p is not a gpu buffer. Take no-CUDA path!\n", ptr);
        return 0;
    }
    /* Anything but CU_MEMORYTYPE_DEVICE is not a GPU memory */
    return (memType == CU_MEMORYTYPE_DEVICE) ? 1 : 0;
}

void* opal_cuda_malloc_gpu_buffer(size_t size, int gpu_id)
{
    int dev_id;
    cudaGetDevice(&dev_id);
    ddt_cuda_device_t *device = &cuda_devices[gpu_id];
    if (device->buffer_free_size < size) {
        DT_CUDA_DEBUG( opal_cuda_output( 0, "No GPU buffer at dev_id %d.\n", dev_id); );
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
        DT_CUDA_DEBUG( opal_cuda_output( 2, "Malloc GPU buffer %p, dev_id %d.\n", addr, dev_id); );
        return addr;
    }
    return NULL;
}

void opal_cuda_free_gpu_buffer(void *addr, int gpu_id)
{
    ddt_cuda_device_t *device = &cuda_devices[gpu_id];
    ddt_cuda_buffer_t *ptr = device->buffer_used.head;

    /* Find the holder of this GPU allocation */
    for( ; (NULL != ptr) && (ptr->gpu_addr != addr); ptr = ptr->next );
    if (NULL == ptr) {  /* we could not find it. something went wrong */
        DT_CUDA_DEBUG( opal_cuda_output( 0, "addr %p is not managed.\n", addr); );
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
    DT_CUDA_DEBUG( opal_cuda_output( 2, "Free GPU buffer %p.\n", addr); );
}

void opal_cuda_check_error(cudaError_t err)
{
    if (err != cudaSuccess) {
        DT_CUDA_DEBUG( opal_cuda_output(0, "CUDA calls error %s\n", cudaGetErrorString(err)); );
    }
}

void opal_cuda_d2dcpy_async(void* dst, const void* src, size_t count)
{
    cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, current_cuda_device->cuda_streams->opal_cuda_stream[0]);
}

void opal_cuda_d2dcpy(void* dst, const void* src, size_t count)
{
    cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, current_cuda_device->cuda_streams->opal_cuda_stream[0]);
    cudaStreamSynchronize(current_cuda_device->cuda_streams->opal_cuda_stream[0]);
}

void opal_dump_cuda_list(ddt_cuda_list_t *list)
{
    ddt_cuda_buffer_t *ptr = NULL;
    ptr = list->head;
    DT_CUDA_DEBUG( opal_cuda_output( 2, "DUMP cuda list %p, nb_elements %d\n", list, list->nb_elements); );
    while (ptr != NULL) {
        DT_CUDA_DEBUG( opal_cuda_output( 2, "\titem addr %p, size %ld.\n", ptr->gpu_addr, ptr->size); );
        ptr = ptr->next;
    }
}
