#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <stdarg.h> 

/*
 * NOTE: The order of this array *MUST* match what is listed in datatype.h
 * (use of designated initializers should relax this restrictions some)
 */
/*
OPAL_DECLSPEC const size_t opal_datatype_basicDatatypesSize[OPAL_DATATYPE_MAX_PREDEFINED] = {
    OPAL_DATATYPE_LOOP_SIZE,
    OPAL_DATATYPE_END_LOOP_SIZE,
    OPAL_DATATYPE_LB_SIZE,
    OPAL_DATATYPE_UB_SIZE,
    OPAL_DATATYPE_INT1_SIZE,
    OPAL_DATATYPE_INT2_SIZE,
    OPAL_DATATYPE_INT4_SIZE,
    OPAL_DATATYPE_INT8_SIZE,
    OPAL_DATATYPE_INT16_SIZE,   
    OPAL_DATATYPE_UINT1_SIZE,
    OPAL_DATATYPE_UINT2_SIZE,
    OPAL_DATATYPE_UINT4_SIZE,
    OPAL_DATATYPE_UINT8_SIZE,
    OPAL_DATATYPE_UINT16_SIZE,  
    OPAL_DATATYPE_FLOAT2_SIZE,
    OPAL_DATATYPE_FLOAT4_SIZE,
    OPAL_DATATYPE_FLOAT8_SIZE,
    OPAL_DATATYPE_FLOAT12_SIZE,
    OPAL_DATATYPE_FLOAT16_SIZE,
    OPAL_DATATYPE_FLOAT_COMPLEX_SIZE,
    OPAL_DATATYPE_DOUBLE_COMPLEX_SIZE,
    OPAL_DATATYPE_LONG_DOUBLE_COMPLEX_SIZE,
    OPAL_DATATYPE_BOOL_SIZE,
    OPAL_DATATYPE_WCHAR_SIZE,
    OPAL_DATATYPE_UNAVAILABLE_SIZE,
};
*/
/***** my variables ********/


ddt_cuda_list_t *cuda_free_list;
ddt_cuda_device_t *cuda_device;
ddt_cuda_desc_t *cuda_desc_d, *cuda_desc_h;
unsigned char *pBaseBuf_GPU, *gpu_src_const, *gpu_dest_const;
unsigned char *ddt_cuda_pack_buffer, *ddt_cuda_unpack_buffer;
ddt_cuda_stream_t* cuda_streams;
struct iovec cuda_iov[CUDA_NB_IOV];
uint32_t cuda_iov_count;
ddt_cuda_description_dist_t description_dist_h[CUDA_MAX_NB_BLOCKS];
ddt_cuda_description_dist_t* description_dist_d;
ddt_cuda_iov_dist_t cuda_iov_dist_h[NB_STREAMS][CUDA_MAX_NB_BLOCKS];
ddt_cuda_iov_dist_t* cuda_iov_dist_d[NB_STREAMS];
dt_elem_desc_t* description_d;
uint8_t opal_datatype_cuda_debug;

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

static inline void cuda_list_item_merge_by_addr(ddt_cuda_list_t *list)
{
    ddt_cuda_buffer_t *ptr = NULL;
    ddt_cuda_buffer_t *next = NULL;
    ptr = list->head;
    while(ptr != NULL) {
        next = ptr->next;
        if (next == NULL) {
            break;
        } else if ((ptr->gpu_addr + ptr->size) == next->gpu_addr) {
            ptr->size += next->size;
            cuda_list_delete(list, next);
        } else {
            ptr = ptr->next;
        }
    }
}

void opal_datatype_cuda_init(void)
{
    uint32_t i;
    int device;
    cudaError res;

    res = cudaGetDevice(&device);
    if( cudaSuccess != res ) {
        opal_cuda_output(0, "Cannot retrieve the device being used. Drop CUDA support!\n");
        return;
    }    
    printf("current device %d\n", device);

    cuda_free_list = init_cuda_free_list();
    
    /* init device */
    cuda_device = (ddt_cuda_device_t *)malloc(sizeof(ddt_cuda_device_t)*1);
    for (i = 0; i < 1; i++) {
        unsigned char *gpu_ptr = NULL;
        if (cudaMalloc((void **)(&gpu_ptr), sizeof(char)*DT_CUDA_BUFFER_SIZE) != cudaSuccess) {
            DT_CUDA_DEBUG( opal_cuda_output( 0, "cudaMalloc is failed in GPU %d\n", i); );
        }
        cudaMemset(gpu_ptr, 0, sizeof(char)*DT_CUDA_BUFFER_SIZE);
        cuda_device[i].gpu_buffer = gpu_ptr;
        
        cuda_device[i].buffer_free_size = DT_CUDA_BUFFER_SIZE;
        ddt_cuda_buffer_t *p = obj_ddt_cuda_buffer_new();
        p->size = DT_CUDA_BUFFER_SIZE;
        p->gpu_addr = gpu_ptr;
        cuda_device[i].buffer_free.head = p;
        cuda_device[i].buffer_free.tail = cuda_device[i].buffer_free.head;
        cuda_device[i].buffer_free.nb_elements = 1;
        
        cuda_device[i].buffer_used.head = NULL;
        cuda_device[i].buffer_used.tail = NULL;
        cuda_device[i].buffer_used_size = 0;
        cuda_device[i].buffer_used.nb_elements = 0;
    }
    
    cudaMalloc((void **)&cuda_desc_d, sizeof(ddt_cuda_desc_t));
    cudaMallocHost((void **)&cuda_desc_h, sizeof(ddt_cuda_desc_t));
    printf("size cuda_desc %d\n", sizeof(ddt_cuda_desc_t));
    
    // printf("malloc iov\n");
    // for (i = 0; i < IOV_ARRAY_SIZE; i++) {
    //     void* iov_base;
    //     cudaMalloc( (void **)&iov_base, sizeof(char)*IOV_LEN);
    //     cuda_desc_h->iov[i].iov_base = iov_base;
    //     cuda_desc_h->iov[i].iov_len = IOV_LEN;
    // }
    
    cudaMalloc((void **)(&ddt_cuda_pack_buffer), sizeof(char)*DT_CUDA_BUFFER_SIZE);
    printf("malloc cuda packing buffer, %p\n", ddt_cuda_pack_buffer);
    cudaMalloc((void **)(&ddt_cuda_unpack_buffer), sizeof(char)*DT_CUDA_BUFFER_SIZE);
    printf("malloc cuda unpacking buffer, %p\n", ddt_cuda_unpack_buffer);

    cuda_desc_h->iov[0].iov_base = ddt_cuda_pack_buffer;
    cuda_desc_h->iov[0].iov_len = DT_CUDA_BUFFER_SIZE;
    
    cudaMalloc((void **)(&pBaseBuf_GPU), sizeof(char)*DT_CUDA_BUFFER_SIZE);
    gpu_src_const = pBaseBuf_GPU;
    gpu_dest_const = (unsigned char*)cuda_desc_h->iov[0].iov_base; 
    
    cuda_desc_h->description_max_count = 0;
    cuda_desc_h->description_count = 0;
    
    /* init cuda stream */
    cuda_streams = (ddt_cuda_stream_t*)malloc(sizeof(ddt_cuda_stream_t));
    for (i = 0; i < NB_STREAMS; i++) {
        cudaStreamCreate(&(cuda_streams->opal_cuda_stream[i]));
    }
    cuda_streams->current_stream_id = 0;
    
    /* init cuda_iov */
    cuda_iov_count = CUDA_NB_IOV;
    
    /* init description dist array */
    cudaMalloc((void **)(&description_dist_d), sizeof(ddt_cuda_description_dist_t)*CUDA_MAX_NB_BLOCKS);
    cuda_desc_h->description_dist = description_dist_d;
    
    /* only for iov version */
    for (i = 0; i < NB_STREAMS; i++) {
        cudaMalloc((void **)(&cuda_iov_dist_d[i]), sizeof(ddt_cuda_iov_dist_t)*CUDA_MAX_NB_BLOCKS);
    }
    
    opal_datatype_cuda_debug = 1;
    
    // /* init size for double, float, char */
    // ALIGNMENT_DOUBLE = sizeof(double);
    // ALIGNMENT_FLOAT = sizeof(float);
    // ALIGNMENT_CHAR = sizeof(char);
    
    cudaDeviceSynchronize();
}

void opal_datatype_cuda_fini(void)
{
    uint32_t i;
    
    if (cuda_desc_d != NULL) {
        cudaFree(cuda_desc_d);
        cuda_desc_d = NULL;
    }
    if (cuda_desc_h->description != NULL) {
        cudaFree(cuda_desc_h->description);
        cuda_desc_h->description = NULL;
    }
    if (cuda_desc_h->description_dist != NULL) {
        cudaFree(cuda_desc_h->description_dist);
        cuda_desc_h->description_dist = NULL;
    }
    printf("free iov\n");
    if (cuda_desc_h != NULL) {    
        for (i = 0; i < IOV_ARRAY_SIZE; i++) {
            cudaFree(cuda_desc_h->iov[i].iov_base);
            cuda_desc_h->iov[i].iov_base = NULL;
        }
    
        cudaFreeHost(cuda_desc_h);
        cuda_desc_h = NULL;
    }
    
    /* destory cuda stream */
    for (i = 0; i < NB_STREAMS; i++) {
        cudaStreamDestroy(cuda_streams->opal_cuda_stream[i]);
    }
    free(cuda_streams);
    
    /* only for iov version */
    for (i = 0; i < NB_STREAMS; i++) {
        cudaFree(cuda_iov_dist_d[i]);
    }
}

void opal_cuda_sync_device(void)
{
    cudaDeviceSynchronize();
    pBaseBuf_GPU = gpu_src_const;
    cuda_desc_h->iov[0].iov_base = (void*)gpu_dest_const;
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

unsigned char* opal_cuda_get_gpu_pack_buffer()
{
    if (ddt_cuda_pack_buffer != NULL) {
        return ddt_cuda_pack_buffer;
    } else {
        return NULL;
    }
}

void* opal_cuda_malloc_gpu_buffer(size_t size, int gpu_id)
{
    int dev_id;
    cudaGetDevice(&dev_id);
    ddt_cuda_device_t *device = &cuda_device[gpu_id];
    if (device->buffer_free_size < size) {
        DT_CUDA_DEBUG( opal_cuda_output( 0, "No GPU buffer at dev_id %d.\n", dev_id); );
        return NULL;
    }
    ddt_cuda_buffer_t *ptr = NULL;
    void *addr = NULL;
    ptr = device->buffer_free.head;
    while (ptr != NULL) {
        if (ptr->size >= size) {
            addr = ptr->gpu_addr;
            ptr->size -= size;
            if (ptr->size == 0) {
                cuda_list_delete(&device->buffer_free, ptr);
                obj_ddt_cuda_buffer_reset(ptr);
                cuda_list_push_head(cuda_free_list, ptr);
            } else {
                ptr->gpu_addr += size;
            }
            break;
        }
        ptr = ptr->next;
    }
    
    if (ptr == NULL) {
        return NULL;
    } else {    
        ddt_cuda_buffer_t *p = cuda_list_pop_tail(cuda_free_list);
        if (p == NULL) {
            p = obj_ddt_cuda_buffer_new();
        }
        p->size = size;
        p->gpu_addr = (unsigned char*)addr;
        cuda_list_push_head(&device->buffer_used, p);
        device->buffer_used_size += size;
        device->buffer_free_size -= size;
        DT_CUDA_DEBUG( opal_cuda_output( 0, "Malloc GPU buffer %p, dev_id %d.\n", addr, dev_id); );
        return addr;
    }
}

void opal_cuda_free_gpu_buffer(void *addr, int gpu_id)
{
    ddt_cuda_device_t *device = &cuda_device[gpu_id];
    ddt_cuda_buffer_t *ptr = NULL;
    ddt_cuda_buffer_t *ptr_next = NULL;
    ptr = device->buffer_used.head;
    while (ptr != NULL) {
        if (ptr->gpu_addr == addr) {
            cuda_list_delete(&device->buffer_used, ptr);
            ptr_next = device->buffer_free.head;
            while (ptr_next != NULL) {
                if (ptr_next->gpu_addr > addr) {
                    break;
                }
                ptr_next = ptr_next->next;
            }
            if (ptr_next == NULL) {
                /* buffer_free is empty, or insert to last one */
                cuda_list_push_tail(&device->buffer_free, ptr);
            } else {
                cuda_list_insert_before(&device->buffer_free, ptr, ptr_next);
            }
            cuda_list_item_merge_by_addr(&device->buffer_free);
            device->buffer_free_size += ptr->size;
            break;
        }
        ptr = ptr->next;
    }
    if (ptr == NULL) {
        DT_CUDA_DEBUG( opal_cuda_output( 0, "addr %p is not managed.\n", addr); );
    }
    size_t size = ptr->size;
    cuda_list_item_merge_by_addr(&device->buffer_free, ptr);
    device->buffer_free_size += size;
    device->buffer_used_size -= size;
    DT_CUDA_DEBUG( opal_cuda_output( 0, "Free GPU buffer %p.\n", addr); );
}

void opal_dump_cuda_list(ddt_cuda_list_t *list)
{
    ddt_cuda_buffer_t *ptr = NULL;
    ptr = list->head;
    DT_CUDA_DEBUG( opal_cuda_output( 0, "DUMP cuda list %p, nb_elements %d\n", list, list->nb_elements); );
    while (ptr != NULL) {
        DT_CUDA_DEBUG( opal_cuda_output( 0, "\titem addr %p, size %ld.\n", ptr->gpu_addr, ptr->size); );
        ptr = ptr->next;
    }
}

/* from internal.h*/
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
