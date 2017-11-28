#include "coll_adapt.h"
#include "coll_adapt_cuda.h"
#include "coll_adapt_cuda_mpool.h"
#include "coll_adapt_context.h"
#include "opal/mca/common/cuda/common_cuda.h"
#include "opal/mca/installdirs/installdirs.h"
#include <dlfcn.h>
#include <unistd.h>

static coll_adapt_cuda_function_table_t coll_adapt_cuda_table;
static void *coll_adapt_cuda_kernel_handle = NULL;
static char *coll_adapt_cuda_kernel_lib = NULL;

#define COLL_ADAPT_CUDA_FIND_CUDA_FUNCTION_OR_RETURN(handle, fname)            \
    do {                                                                            \
        char* _error;                                                               \
        *(void **)(&(coll_adapt_cuda_table.fname ## _p)) = dlsym((handle), # fname);    \
        if(NULL != (_error = dlerror()) )  {                                        \
            opal_output(0, "Finding %s error: %s\n", # fname, _error);              \
            coll_adapt_cuda_table.fname ## _p = NULL;                                   \
            return OMPI_ERROR;                                                      \
        }                                                                           \
    } while (0)

int coll_adapt_cuda_init(void)
{
    if (coll_adapt_cuda_kernel_handle == NULL) {
        if( NULL != coll_adapt_cuda_kernel_lib )
            free(coll_adapt_cuda_kernel_lib);
        asprintf(&coll_adapt_cuda_kernel_lib, "%s/%s", opal_install_dirs.libdir, "mca_coll_adapt_cuda_kernel.so");

        coll_adapt_cuda_kernel_handle = dlopen(coll_adapt_cuda_kernel_lib , RTLD_LAZY);
        if (!coll_adapt_cuda_kernel_handle) {
            opal_output( 0, "Failed to load %s library: error %s\n", coll_adapt_cuda_kernel_lib, dlerror());
            coll_adapt_cuda_kernel_handle = NULL;
            return OMPI_ERROR;
        }
    
        COLL_ADAPT_CUDA_FIND_CUDA_FUNCTION_OR_RETURN( coll_adapt_cuda_kernel_handle, coll_adapt_cuda_init );
        COLL_ADAPT_CUDA_FIND_CUDA_FUNCTION_OR_RETURN( coll_adapt_cuda_kernel_handle, coll_adapt_cuda_fini );
        COLL_ADAPT_CUDA_FIND_CUDA_FUNCTION_OR_RETURN( coll_adapt_cuda_kernel_handle, coll_adapt_cuda_is_gpu_buffer );
        COLL_ADAPT_CUDA_FIND_CUDA_FUNCTION_OR_RETURN( coll_adapt_cuda_kernel_handle, coll_adapt_cuda_malloc);
        COLL_ADAPT_CUDA_FIND_CUDA_FUNCTION_OR_RETURN( coll_adapt_cuda_kernel_handle, coll_adapt_cuda_op_sum_float );
    
        coll_adapt_cuda_table.coll_adapt_cuda_init_p();
        mca_coll_adapt_component.coll_adapt_cuda_enabled = 1;
        
        mca_coll_adapt_component.pined_cpu_mpool = NULL;
        mca_coll_adapt_component.pined_gpu_mpool = NULL;
        
        opal_output( 0, "coll_adapt_cuda_init done\n");
    }
    
    return OMPI_SUCCESS;
}

int coll_adapt_cuda_fini(void)
{
    if (coll_adapt_cuda_kernel_handle != NULL) {
        coll_adapt_cuda_table.coll_adapt_cuda_fini_p();
        
        coll_adapt_cuda_table.coll_adapt_cuda_init_p = NULL;
        coll_adapt_cuda_table.coll_adapt_cuda_fini_p = NULL;
        coll_adapt_cuda_table.coll_adapt_cuda_is_gpu_buffer_p = NULL;
        
        dlclose(coll_adapt_cuda_kernel_handle);
        coll_adapt_cuda_kernel_handle = NULL;
        if( NULL != coll_adapt_cuda_kernel_lib ) {
            free(coll_adapt_cuda_kernel_lib);
        }
        coll_adapt_cuda_kernel_lib = NULL;
        mca_coll_adapt_component.coll_adapt_cuda_enabled = 0;
        mca_coll_adapt_component.pined_cpu_mpool = NULL;
        mca_coll_adapt_component.pined_gpu_mpool = NULL;
        opal_output( 0, "coll_adapt_cuda_fini done\n");
    }
    
    return OMPI_SUCCESS;
}

int coll_adapt_cuda_is_gpu_buffer(const void *ptr)
{
    return coll_adapt_cuda_table.coll_adapt_cuda_is_gpu_buffer_p(ptr);
}

void* coll_adapt_cuda_malloc(size_t size)
{
    return coll_adapt_cuda_table.coll_adapt_cuda_malloc_p(size);
}

int coll_adapt_cuda_op_reduce(ompi_op_t * op, void *source, void *target, int count, ompi_datatype_t * dtype)
{
    void *op_cuda_stream = NULL;
    op_cuda_stream = mca_common_cuda_get_op_stream(0);
    coll_adapt_cuda_table.coll_adapt_cuda_op_sum_float_p(source, target, count, op_cuda_stream);
    return OMPI_SUCCESS;
}

int coll_adapt_cuda_get_gpu_topo(ompi_coll_topo_gpu_t *gpu_topo)
{
    int i;
    int nb_gpus = 0;
    mca_common_cuda_get_device(&(gpu_topo->gpu_id));
    mca_common_cuda_get_device_count(&nb_gpus);
    int *gpu_numa = (int *)malloc(sizeof(int) * nb_gpus * 2);
    for (i = 0; i < nb_gpus; i++) {
        gpu_numa[i] = i;
    }
    int start_gpu = 0;
    gpu_numa[nb_gpus + start_gpu] = 0;
    int peer_access = 0;
    for (i = 1; i < nb_gpus; i++) {
        mca_common_cuda_device_can_access_peer(&peer_access, start_gpu, i);
        if (1 == peer_access) {
            gpu_numa[nb_gpus + i] = gpu_numa[nb_gpus + start_gpu];
        } else {
            gpu_numa[nb_gpus + i] = gpu_numa[nb_gpus + start_gpu] + 1;
            start_gpu = i;
        }
    }
    gpu_topo->nb_gpus = nb_gpus;
    gpu_topo->gpu_numa = gpu_numa;
    
    return OMPI_SUCCESS;
}

int coll_adapt_cuda_free_gpu_topo(ompi_coll_topo_gpu_t *gpu_topo)
{
    free(gpu_topo->gpu_numa);
    return OMPI_SUCCESS;
}

int coll_adapt_cuda_progress(void)
{
    char *context;
    while (1 == progress_one_cuda_op_event((void **)&context)) {
        if (context != NULL) {
            int *flag = (int *)(context + sizeof(opal_free_list_item_t));
            if (*flag == COLL_ADAPT_CONTEXT_FLAGS_CUDA_REDUCE) {
              //  opal_output(0, "reduce call back\n");
                mca_coll_adapt_reduce_context_t *reduce_context = (mca_coll_adapt_reduce_context_t *)context;
                assert(reduce_context->cuda_callback != NULL);
                reduce_context->cuda_callback(reduce_context);
            }
        }
    }
    while (1 == progress_one_cuda_memcpy_event((void **)&context)) {
        if (context != NULL) {
            int *flag = (int *)(context + sizeof(opal_free_list_item_t));
            if (*flag == COLL_ADAPT_CONTEXT_FLAGS_CUDA_BCAST) {
        //        opal_output(0, "bcast call back\n");
                mca_coll_adapt_bcast_context_t *bcast_context = (mca_coll_adapt_bcast_context_t *)context;
                assert(bcast_context->cuda_callback != NULL);
                bcast_context->cuda_callback(bcast_context);
            }
        }
    }
    return OMPI_SUCCESS;
}

