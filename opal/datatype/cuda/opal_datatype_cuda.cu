#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"
#include <cuda_runtime_api.h>
#include <stdio.h>

ddt_cuda_desc_t *cuda_desc_d, *cuda_desc_h;
unsigned char *pBaseBuf_GPU, *gpu_src_const, *gpu_dest_const;
ddt_cuda_stream_t* cuda_streams;

void opal_datatype_cuda_init(void)
{
    uint32_t i;
    
    int cuda_device = OPAL_GPU_INDEX;
    cudaSetDevice(cuda_device);
    
    cudaMalloc((void **)&cuda_desc_d, sizeof(ddt_cuda_desc_t));
    cudaMallocHost((void **)&cuda_desc_h, sizeof(ddt_cuda_desc_t));
    printf("size cuda_desc %d\n", sizeof(ddt_cuda_desc_t));
    
    printf("malloc iov\n");
    for (i = 0; i < IOV_ARRAY_SIZE; i++) {
        void* iov_base;
        cudaMalloc( (void **)&iov_base, sizeof(char)*IOV_LEN);
        cuda_desc_h->iov[i].iov_base = iov_base;
        cuda_desc_h->iov[i].iov_len = IOV_LEN;
    }
    cudaMalloc((void **)(&pBaseBuf_GPU), sizeof(char)*IOV_LEN);
    gpu_src_const = pBaseBuf_GPU;
    gpu_dest_const = (unsigned char*)cuda_desc_h->iov[0].iov_base; 
    
    cuda_desc_h->description_max_count = 0;
    cuda_desc_h->description_count = 0;
    
    cuda_streams = (ddt_cuda_stream_t*)malloc(sizeof(ddt_cuda_stream_t));
    /* init cuda stream */
    for (i = 0; i < NB_STREAMS; i++) {
        cudaStreamCreate(&(cuda_streams->opal_cuda_stream[i]));
    }
    cuda_streams->current_stream_id = 0;
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
}

void opal_cuda_sync_device(void)
{
    cudaDeviceSynchronize();
    pBaseBuf_GPU = gpu_src_const;
    cuda_desc_h->iov[0].iov_base = (void*)gpu_dest_const;
}