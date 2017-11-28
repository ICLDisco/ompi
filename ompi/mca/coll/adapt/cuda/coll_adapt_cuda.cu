#include "ompi_config.h"
#include "coll_adapt_cuda.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <assert.h>
#include <stdarg.h>

static int coll_adapt_cuda_kernel_enabled = 0;
cudaStream_t op_internal_stream; 
cublasHandle_t cublas_handle;

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

int coll_adapt_cuda_init(void)
{
    int device;
    cudaError cuda_err;

    cuda_err = cudaGetDevice(&device);
    if( cudaSuccess != cuda_err ) {
       // OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "Cannot retrieve the device being used. Drop CUDA support!\n"));
        return -1;
    }
    //cudaStreamCreate(&op_internal_stream);
    cublasStatus_t stat;
    stat = cublasCreate(&cublas_handle); 
    if (stat != CUBLAS_STATUS_SUCCESS) { 
        printf("CUBLAS initialization failed\n");
        return -1; 
    }
    cudaStreamCreate(&op_internal_stream);
    coll_adapt_cuda_kernel_enabled = 1;
    cudaDeviceSynchronize();
    printf("CUBLAS initialization done device %d\n", device);
    return 0;
}

int coll_adapt_cuda_fini(void)
{
    coll_adapt_cuda_kernel_enabled = 0;
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(op_internal_stream);
    op_internal_stream = NULL;
    return 0;
}

void* coll_adapt_cuda_malloc(size_t size)
{
    cudaError cuda_err;
    void *ptr = NULL;
    cuda_err = cudaMalloc((void**)&ptr, size);
    if( cudaSuccess != cuda_err ) {
       // OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "Cannot retrieve the device being used. Drop CUDA support!\n"));
        return NULL;
    } else {
        return ptr;
    }
}

int coll_adapt_cuda_is_gpu_buffer(const void *ptr)
{
    CUmemorytype memType;
    CUdeviceptr dbuf = (CUdeviceptr)ptr;
    int res;

    res = cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dbuf);
    if (res != CUDA_SUCCESS) {
        /* If we cannot determine it is device pointer,
         * just assume it is not. */
      //  OPAL_OUTPUT_VERBOSE((1, opal_datatype_cuda_output, "!!!!!!! %p is not a gpu buffer. Take no-CUDA path!\n", ptr));
        return 0;
    }
    /* Anything but CU_MEMORYTYPE_DEVICE is not a GPU memory */
    return (memType == CU_MEMORYTYPE_DEVICE) ? 1 : 0;
}

int coll_adapt_cuda_op_sum_float(void *source, void *target, int count, void *op_stream)
{
    int is_sync = 0;
    float alpha_f = 1.0;

    if (op_stream == NULL) {
        cublasSetStream(cublas_handle, op_internal_stream);
        is_sync = 1;
    } else {
        cublasSetStream(cublas_handle, (cudaStream_t)op_stream);
    }
    //stat = cublasDaxpy(cublas_handle, count, &alpha, (const double *)source, 1, (double *)target, 1);
    cublasStatus_t stat = cublasSaxpy(cublas_handle, count, &alpha_f, (const float *)source, 1, (float *)target, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
        printf("cublasSaxpy error %s. src %p, targrt %p, count %d\n", _cudaGetErrorEnum(stat), source, target, count);
        return -1; 
    }
    if (is_sync) {
        cudaStreamSynchronize(op_internal_stream);
    }
    return 1;
}