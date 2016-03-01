/*
 * Copyright (c) 2011-2014 NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"

#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>

#include "opal/align.h"
#include "opal/util/output.h"
#include "opal/datatype/opal_convertor.h"
#include "opal/datatype/opal_datatype_cuda.h"
#include "opal/mca/installdirs/installdirs.h"

static bool initialized = false;
int opal_cuda_verbose = 0;
static int opal_cuda_enabled = 0; /* Starts out disabled */
static int opal_cuda_output = 0;
static void opal_cuda_support_init(void);
static int (*common_cuda_initialization_function)(opal_common_cuda_function_table_t *) = NULL;
static opal_common_cuda_function_table_t ftable;

/* folowing variables are used for cuda ddt kernel support */
static opal_datatype_cuda_kernel_function_table_t cuda_kernel_table;
static void *opal_datatype_cuda_kernel_handle = NULL;
static char *opal_datatype_cuda_kernel_lib = NULL;
int32_t opal_datatype_cuda_kernel_support = 0;

#define OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN(handle, fname)            \
    do {                                                                            \
        char* _error;                                                               \
        *(void **)(&(cuda_kernel_table.fname ## _p)) = dlsym((handle), # fname);    \
        if(NULL != (_error = dlerror()) )  {                                        \
            opal_output(0, "Finding %s error: %s\n", # fname, _error);              \
            cuda_kernel_table.fname ## _p = NULL;                                   \
            return OPAL_ERROR;                                                      \
        }                                                                           \
    } while (0)


/* This function allows the common cuda code to register an
 * initialization function that gets called the first time an attempt
 * is made to send or receive a GPU pointer.  This allows us to delay
 * some CUDA initialization until after MPI_Init().
 */
void opal_cuda_add_initialization_function(int (*fptr)(opal_common_cuda_function_table_t *)) {
    common_cuda_initialization_function = fptr;
}

/**
 * This function is called when a convertor is instantiated.  It has to call
 * the opal_cuda_support_init() function once to figure out if CUDA support
 * is enabled or not.  If CUDA is not enabled, then short circuit out
 * for all future calls.
 */
void mca_cuda_convertor_init(opal_convertor_t* convertor, const void *pUserBuf, const struct opal_datatype_t* datatype)
{
    /* Only do the initialization on the first GPU access */
    if (!initialized) {
        opal_cuda_support_init();
    }

    /* This is needed to handle case where convertor is not fully initialized
     * like when trying to do a sendi with convertor on the statck */
    convertor->cbmemcpy = (memcpy_fct_t)&opal_cuda_memcpy;

    /* If not enabled, then nothing else to do */
    if (!opal_cuda_enabled) {
        return;
    }

    if (ftable.gpu_is_gpu_buffer(pUserBuf, convertor)) {
        convertor->flags |= CONVERTOR_CUDA;
    }
    
    if (OPAL_SUCCESS != opal_cuda_kernel_support_init()) {
        opal_cuda_kernel_support_fini();    
    }

    convertor->current_cuda_iov_pos = 0;
    convertor->current_iov_pos = 0;
    convertor->current_iov_partial_length = 0;
    convertor->current_count = 0;
}

/* Checks the type of pointer
 *
 * @param dest   One pointer to check
 * @param source Another pointer to check
 */
bool opal_cuda_check_bufs(char *dest, char *src)
{
    /* Only do the initialization on the first GPU access */
    if (!initialized) {
        opal_cuda_support_init();
    }

    if (!opal_cuda_enabled) {
        return false;
    }

    if (ftable.gpu_is_gpu_buffer(dest, NULL) || ftable.gpu_is_gpu_buffer(src, NULL)) {
        return true;
    }
    return false;
}

/*
 * With CUDA enabled, all contiguous copies will pass through this function.
 * Therefore, the first check is to see if the convertor is a GPU buffer.
 * Note that if there is an error with any of the CUDA calls, the program
 * aborts as there is no recovering.
 */
void *opal_cuda_memcpy(void *dest, const void *src, size_t size, opal_convertor_t* convertor)
{
    int res;

    if (!(convertor->flags & CONVERTOR_CUDA)) {
        return memcpy(dest, src, size);
    }

    if (convertor->flags & CONVERTOR_CUDA_ASYNC) {
        res = ftable.gpu_cu_memcpy_async(dest, (void *)src, size, convertor);
    } else {
        res = ftable.gpu_cu_memcpy(dest, (void *)src, size);
    }

    if (res != 0) {
        opal_output(0, "CUDA: Error in cuMemcpy: res=%d, dest=%p, src=%p, size=%d",
                    res, dest, src, (int)size);
        abort();
    }
    return dest;
}

/*
 * This function is needed in cases where we do not have contiguous
 * datatypes.  The current code has macros that cannot handle a convertor
 * argument to the memcpy call.
 */
void *opal_cuda_memcpy_sync(void *dest, const void *src, size_t size)
{
    int res;
    res = ftable.gpu_cu_memcpy(dest, src, size);
    if (res != 0) {
        opal_output(0, "CUDA: Error in cuMemcpy: res=%d, dest=%p, src=%p, size=%d",
                    res, dest, src, (int)size);
        abort();
    }
    return dest;
}

/*
 * In some cases, need an implementation of memmove.  This is not fast, but
 * it is not often needed.
 */
void *opal_cuda_memmove(void *dest, void *src, size_t size)
{
    int res;

    res = ftable.gpu_memmove(dest, src, size);
    if(res != 0){
        opal_output(0, "CUDA: Error in gpu memmove: res=%d, dest=%p, src=%p, size=%d",
                    res, dest, src, (int)size);
        abort();
    }
    return dest;
}

/**
 * This function gets called once to check if the program is running in a cuda
 * environment.
 */
static void opal_cuda_support_init(void)
{
    if (initialized) {
        return;
    }

    /* Set different levels of verbosity in the cuda related code. */
    opal_cuda_output = opal_output_open(NULL);
    opal_output_set_verbosity(opal_cuda_output, opal_cuda_verbose);

    /* Callback into the common cuda initialization routine. This is only
     * set if some work had been done already in the common cuda code.*/
    if (NULL != common_cuda_initialization_function) {
        if (0 == common_cuda_initialization_function(&ftable)) {
            opal_cuda_enabled = 1;
        }
    }

    if (1 == opal_cuda_enabled) {
        opal_output_verbose(10, opal_cuda_output,
                            "CUDA: enabled successfully, CUDA device pointers will work");
    } else {
        opal_output_verbose(10, opal_cuda_output,
                            "CUDA: not enabled, CUDA device pointers will not work");
    }

    initialized = true;
    
}

/**
 * Tell the convertor that copies will be asynchronous CUDA copies.  The
 * flags are cleared when the convertor is reinitialized.
 */
void opal_cuda_set_copy_function_async(opal_convertor_t* convertor, void *stream)
{
    convertor->flags |= CONVERTOR_CUDA_ASYNC;
    convertor->stream = stream;
}

/* following functions are used for cuda ddt kernel support */
int32_t opal_cuda_kernel_support_init(void)
{
    if (opal_datatype_cuda_kernel_handle ==  NULL) {

        /* If the library name was initialized but the load failed, we have another chance to change it */
        if( NULL != opal_datatype_cuda_kernel_lib )
            free(opal_datatype_cuda_kernel_lib);
        asprintf(&opal_datatype_cuda_kernel_lib, "%s/%s", opal_install_dirs.libdir, "opal_datatype_cuda_kernel.so");

        opal_datatype_cuda_kernel_handle = dlopen(opal_datatype_cuda_kernel_lib , RTLD_LAZY);
        if (!opal_datatype_cuda_kernel_handle) {
            opal_output( 0, "Failed to load %s library: error %s\n", opal_datatype_cuda_kernel_lib, dlerror());
            opal_datatype_cuda_kernel_handle = NULL;
            return OPAL_ERROR;
        }
        
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_kernel_init );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_kernel_fini );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_generic_simple_pack_function_cuda_iov );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_generic_simple_unpack_function_cuda_iov );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_generic_simple_pack_function_cuda_vector );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_generic_simple_unpack_function_cuda_vector );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_free_gpu_buffer );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_malloc_gpu_buffer );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_d2dcpy_async );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_d2dcpy );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cached_cuda_iov_fini );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_set_cuda_stream );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_get_cuda_stream );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_get_current_cuda_stream );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_sync_current_cuda_stream );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_sync_cuda_stream );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_set_outer_cuda_stream );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_set_callback_current_stream );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_alloc_event );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_free_event );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_event_query );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_event_sync );
        OPAL_DATATYPE_FIND_CUDA_KERNEL_FUNCTION_OR_RETURN( opal_datatype_cuda_kernel_handle, opal_ddt_cuda_event_record );
        
        if (OPAL_SUCCESS != cuda_kernel_table.opal_ddt_cuda_kernel_init_p()) {
            return OPAL_ERROR;
        }
        opal_datatype_cuda_kernel_support = 1;
        opal_output( 0, "opal_cuda_kernel_support_init done\n");
    }
    return OPAL_SUCCESS;
}

int32_t opal_cuda_kernel_support_fini(void)
{
    if (opal_datatype_cuda_kernel_handle != NULL) {
        cuda_kernel_table.opal_ddt_cuda_kernel_fini_p();
        /* Reset all functions to NULL */
        cuda_kernel_table.opal_ddt_cuda_kernel_init_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_kernel_fini_p = NULL;
        cuda_kernel_table.opal_ddt_generic_simple_pack_function_cuda_iov_p = NULL;
        cuda_kernel_table.opal_ddt_generic_simple_unpack_function_cuda_iov_p = NULL;
        cuda_kernel_table.opal_ddt_generic_simple_pack_function_cuda_vector_p = NULL;
        cuda_kernel_table.opal_ddt_generic_simple_unpack_function_cuda_vector_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_free_gpu_buffer_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_malloc_gpu_buffer_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_d2dcpy_async_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_d2dcpy_p = NULL;
        cuda_kernel_table.opal_ddt_cached_cuda_iov_fini_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_set_cuda_stream_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_get_cuda_stream_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_get_current_cuda_stream_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_sync_current_cuda_stream_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_sync_cuda_stream_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_set_outer_cuda_stream_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_set_callback_current_stream_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_alloc_event_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_free_event_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_event_query_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_event_sync_p = NULL;
        cuda_kernel_table.opal_ddt_cuda_event_record_p = NULL;

        dlclose(opal_datatype_cuda_kernel_handle);
        opal_datatype_cuda_kernel_handle = NULL;

        if( NULL != opal_datatype_cuda_kernel_lib )
            free(opal_datatype_cuda_kernel_lib);
        opal_datatype_cuda_kernel_lib = NULL;
        opal_datatype_cuda_kernel_support = 0;
        opal_output( 0, "opal_cuda_kernel_support_fini done\n");
    }
    return OPAL_SUCCESS;
}

int32_t opal_cuda_sync_all_events(void *cuda_event_list, int32_t nb_events)
{
    int i;
    for (i = 0; i < nb_events; i++) {
        opal_cuda_event_sync(cuda_event_list, i);
    }
    return OPAL_SUCCESS;
}

int32_t opal_generic_simple_pack_function_cuda_iov( opal_convertor_t* pConvertor, struct iovec* iov, uint32_t* out_size, size_t* max_data )
{
    if (cuda_kernel_table.opal_ddt_generic_simple_pack_function_cuda_iov_p != NULL) {
        return cuda_kernel_table.opal_ddt_generic_simple_pack_function_cuda_iov_p(pConvertor, iov, out_size, max_data);
    } else {
        opal_output(0, "opal_ddt_generic_simple_pack_function_cuda_iov function pointer is NULL\n");
        return -1;
    }
}

int32_t opal_generic_simple_unpack_function_cuda_iov( opal_convertor_t* pConvertor, struct iovec* iov, uint32_t* out_size, size_t* max_data )
{
    if (cuda_kernel_table.opal_ddt_generic_simple_unpack_function_cuda_iov_p != NULL) {
        return cuda_kernel_table.opal_ddt_generic_simple_unpack_function_cuda_iov_p(pConvertor, iov, out_size, max_data);
    } else {
        opal_output(0, "opal_ddt_generic_simple_unpack_function_cuda_iov function pointer is NULL\n");
        return -1;
    }
}

int32_t opal_generic_simple_pack_function_cuda_vector( opal_convertor_t* pConvertor, struct iovec* iov, uint32_t* out_size, size_t* max_data )
{
    if (cuda_kernel_table.opal_ddt_generic_simple_pack_function_cuda_vector_p != NULL) {
        return cuda_kernel_table.opal_ddt_generic_simple_pack_function_cuda_vector_p(pConvertor, iov, out_size, max_data);
    } else {
        opal_output(0, "opal_ddt_generic_simple_pack_function_cuda_vector function pointer is NULL\n");
        return -1;
    }
}

int32_t opal_generic_simple_unpack_function_cuda_vector( opal_convertor_t* pConvertor, struct iovec* iov, uint32_t* out_size, size_t* max_data )
{
    if (cuda_kernel_table.opal_ddt_generic_simple_unpack_function_cuda_vector_p != NULL) {
        return cuda_kernel_table.opal_ddt_generic_simple_unpack_function_cuda_vector_p(pConvertor, iov, out_size, max_data);
    } else {
        opal_output(0, "opal_ddt_generic_simple_unpack_function_cuda_vector function pointer is NULL\n");
        return -1;
    }
}

void* opal_cuda_malloc_gpu_buffer(size_t size, int gpu_id)
{
    if (cuda_kernel_table.opal_ddt_cuda_malloc_gpu_buffer_p != NULL) {
        return cuda_kernel_table.opal_ddt_cuda_malloc_gpu_buffer_p(size, gpu_id);
    } else {
        opal_output(0, "opal_ddt_cuda_malloc_gpu_buffer function pointer is NULL\n");
        return NULL;
    }
}

void opal_cuda_free_gpu_buffer(void *addr, int gpu_id)
{
    if (cuda_kernel_table.opal_ddt_cuda_free_gpu_buffer_p != NULL) {
        cuda_kernel_table.opal_ddt_cuda_free_gpu_buffer_p(addr, gpu_id);
    } else {
        opal_output(0, "opal_ddt_cuda_free_gpu_buffer function pointer is NULL\n");
    }
}

void opal_cuda_d2dcpy(void* dst, const void* src, size_t count)
{
    if (cuda_kernel_table.opal_ddt_cuda_d2dcpy_p != NULL) {
        cuda_kernel_table.opal_ddt_cuda_d2dcpy_p(dst, src, count);
    } else {
        opal_output(0, "opal_ddt_cuda_d2dcpy function pointer is NULL\n");
    }
}

void opal_cuda_d2dcpy_async(void* dst, const void* src, size_t count)
{
    if (cuda_kernel_table.opal_ddt_cuda_d2dcpy_async_p != NULL) {
        cuda_kernel_table.opal_ddt_cuda_d2dcpy_async_p(dst, src, count);
    } else {
        opal_output(0, "opal_ddt_cuda_d2dcpy_async function pointer is NULL\n");
    }
}

void opal_cached_cuda_iov_fini(void *cached_cuda_iov)
{
    if (cuda_kernel_table.opal_ddt_cached_cuda_iov_fini_p != NULL) {
        cuda_kernel_table.opal_ddt_cached_cuda_iov_fini_p(cached_cuda_iov);
    } else {
        opal_output(0, "opal_ddt_cached_cuda_iov_fini function pointer is NULL\n");
    }
}

void opal_cuda_set_cuda_stream(int stream_id)
{
    if (cuda_kernel_table.opal_ddt_cuda_set_cuda_stream_p != NULL) {
        cuda_kernel_table.opal_ddt_cuda_set_cuda_stream_p(stream_id);
    } else {
        opal_output(0, "opal_ddt_cuda_set_cuda_stream function pointer is NULL\n");
    }
}

int32_t opal_cuda_get_cuda_stream(void)
{
    if (cuda_kernel_table.opal_ddt_cuda_get_cuda_stream_p != NULL) {
        return cuda_kernel_table.opal_ddt_cuda_get_cuda_stream_p();
    } else {
        opal_output(0, "opal_ddt_cuda_get_cuda_stream function pointer is NULL\n");
        return -2;
    }
}

void* opal_cuda_get_current_cuda_stream(void)
{
    if (cuda_kernel_table.opal_ddt_cuda_get_current_cuda_stream_p != NULL) {
        return cuda_kernel_table.opal_ddt_cuda_get_current_cuda_stream_p();
    } else {
        opal_output(0, "opal_ddt_cuda_get_current_cuda_stream function pointer is NULL\n");
        return NULL;
    }
}

void opal_cuda_sync_current_cuda_stream(void)
{
    if (cuda_kernel_table.opal_ddt_cuda_sync_current_cuda_stream_p != NULL) {
        cuda_kernel_table.opal_ddt_cuda_sync_current_cuda_stream_p();
    } else {
        opal_output(0, "opal_ddt_cuda_sync_current_cuda_stream function pointer is NULL\n");
    }
}

void opal_cuda_sync_cuda_stream(int stream_id)
{
    if (cuda_kernel_table.opal_ddt_cuda_sync_cuda_stream_p != NULL) {
        cuda_kernel_table.opal_ddt_cuda_sync_cuda_stream_p(stream_id);
    } else {
        opal_output(0, "opal_ddt_cuda_sync_cuda_stream function pointer is NULL\n");
    }
}

void opal_cuda_set_outer_cuda_stream(void *stream)
{
    if (cuda_kernel_table.opal_ddt_cuda_set_outer_cuda_stream_p != NULL) {
        cuda_kernel_table.opal_ddt_cuda_set_outer_cuda_stream_p(stream);
    } else {
        opal_output(0, "opal_ddt_cuda_set_outer_cuda_stream function pointer is NULL\n");
    }
}

void opal_cuda_set_callback_current_stream(void *callback_func, void *callback_data)
{
    if (cuda_kernel_table.opal_ddt_cuda_set_callback_current_stream_p != NULL) {
        cuda_kernel_table.opal_ddt_cuda_set_callback_current_stream_p(callback_func, callback_data);
    } else {
        opal_output(0, "opal_ddt_cuda_set_callback_current_stream function pointer is NULL\n");
    }
}

void* opal_cuda_alloc_event(int32_t nb_events, int32_t *loc)
{
    if (cuda_kernel_table.opal_ddt_cuda_alloc_event_p != NULL) {
        return cuda_kernel_table.opal_ddt_cuda_alloc_event_p(nb_events, loc);
    } else {
        opal_output(0, "opal_ddt_cuda_alloc_event function pointer is NULL\n");
        return NULL;
    }
}

void opal_cuda_free_event(int32_t loc)
{
    if (cuda_kernel_table.opal_ddt_cuda_free_event_p != NULL) {
        cuda_kernel_table.opal_ddt_cuda_free_event_p(loc);
    } else {
        opal_output(0, "opal_ddt_cuda_free_event function pointer is NULL\n");
    }
}

int32_t opal_cuda_event_query(void *cuda_event_list, int32_t i)
{
    if (cuda_kernel_table.opal_ddt_cuda_event_query_p != NULL) {
        return cuda_kernel_table.opal_ddt_cuda_event_query_p(cuda_event_list, i);
    } else {
        opal_output(0, "opal_ddt_cuda_event_query function pointer is NULL\n");
        return -2;
    }
}

int32_t opal_cuda_event_sync(void *cuda_event_list, int32_t i)
{
    if (cuda_kernel_table.opal_ddt_cuda_event_sync_p != NULL) {
        return cuda_kernel_table.opal_ddt_cuda_event_sync_p(cuda_event_list, i);
    } else {
        opal_output(0, "opal_ddt_cuda_event_sync function pointer is NULL\n");
        return -2;
    }
}

int32_t opal_cuda_event_record(void *cuda_event_list, int32_t i)
{
    if (cuda_kernel_table.opal_ddt_cuda_event_record_p != NULL) {
        return cuda_kernel_table.opal_ddt_cuda_event_record_p(cuda_event_list, i);
    } else {
        opal_output(0, "opal_ddt_cuda_event_record function pointer is NULL\n");
        return -2;
    }
}
