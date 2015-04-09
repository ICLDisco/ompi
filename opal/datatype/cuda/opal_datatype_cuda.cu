#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <stdarg.h> 

/*
 * NOTE: The order of this array *MUST* match what is listed in datatype.h
 * (use of designated initializers should relax this restrictions some)
 */
OPAL_DECLSPEC const size_t opal_datatype_basicDatatypesSize[OPAL_DATATYPE_MAX_PREDEFINED] = {
    OPAL_DATATYPE_LOOP_SIZE,
    OPAL_DATATYPE_END_LOOP_SIZE,
    OPAL_DATATYPE_LB_SIZE,
    OPAL_DATATYPE_UB_SIZE,
    OPAL_DATATYPE_INT1_SIZE,
    OPAL_DATATYPE_INT2_SIZE,
    OPAL_DATATYPE_INT4_SIZE,
    OPAL_DATATYPE_INT8_SIZE,
    OPAL_DATATYPE_INT16_SIZE,       /* Yes, double-machine word integers are available */
    OPAL_DATATYPE_UINT1_SIZE,
    OPAL_DATATYPE_UINT2_SIZE,
    OPAL_DATATYPE_UINT4_SIZE,
    OPAL_DATATYPE_UINT8_SIZE,
    OPAL_DATATYPE_UINT16_SIZE,      /* Yes, double-machine word integers are available */
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

/***** my variables ********/

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

void opal_datatype_cuda_init(void)
{
    uint32_t i;
    
    int cuda_device = OPAL_GPU_INDEX;
    cudaSetDevice(cuda_device);
    
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
    printf("malloc cuda packing buffer\n");
    cudaMalloc((void **)(&ddt_cuda_pack_buffer), sizeof(char)*DT_CUDA_BUFFER_SIZE);
    cudaMemset(ddt_cuda_pack_buffer, 0, sizeof(char)*DT_CUDA_BUFFER_SIZE);
    printf("malloc cuda unpacking buffer\n");
    cudaMalloc((void **)(&ddt_cuda_unpack_buffer), sizeof(char)*DT_CUDA_BUFFER_SIZE);
    cudaMemset(ddt_cuda_unpack_buffer, 0, sizeof(char)*DT_CUDA_BUFFER_SIZE);

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
        printf("!!!!!!!is gpu buffer error\n");
        return 0;
    } 
    if (memType == CU_MEMORYTYPE_DEVICE) {
        return 1;
    } else if (memType == CU_MEMORYTYPE_HOST){
        return 0;
    } else if (memType == 0) {
        return 0;
    } else {
        return 0;
    }
}

unsigned char* opal_cuda_get_gpu_pack_buffer()
{
    if (ddt_cuda_pack_buffer != NULL) {
        return ddt_cuda_pack_buffer;
    } else {
        return NULL;
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
