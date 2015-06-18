/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2013 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"

#include <stddef.h>
#include <dlfcn.h>

#include "opal/mca/installdirs/installdirs.h"
#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#if OPAL_ENABLE_DEBUG
#include "opal/util/output.h"

#define DO_DEBUG(INST)  if( opal_pack_debug ) { INST }
#else
#define DO_DEBUG(INST)
#endif  /* OPAL_ENABLE_DEBUG */

#include "opal/datatype/opal_datatype_gpu.h"

static void *opal_datatype_cuda_handle = NULL;
static char *opal_datatype_cuda_lib = NULL;

void (*opal_datatype_cuda_init_p)(void) = NULL;

void (*opal_datatype_cuda_fini_p)(void) = NULL;

int32_t (*opal_generic_simple_pack_function_cuda_p)( opal_convertor_t* pConvertor,
                                                     struct iovec* iov,
                                                     uint32_t* out_size,
                                                     size_t* max_data ) = NULL;

int32_t (*opal_generic_simple_unpack_function_cuda_p)( opal_convertor_t* pConvertor,
                                                       struct iovec* iov,
                                                       uint32_t* out_size,
                                                       size_t* max_data ) = NULL;

int32_t (*opal_generic_simple_pack_function_cuda_iov_p)( opal_convertor_t* pConvertor,
                                                        struct iovec* iov,
                                                        uint32_t* out_size,
                                                        size_t* max_data ) = NULL;

int32_t (*opal_generic_simple_unpack_function_cuda_iov_p)( opal_convertor_t* pConvertor,
                                                        struct iovec* iov,
                                                        uint32_t* out_size,
                                                        size_t* max_data ) = NULL;

int32_t (*opal_generic_simple_pack_function_cuda_vector_p)( opal_convertor_t* pConvertor,
                                                            struct iovec* iov,
                                                            uint32_t* out_size,
                                                            size_t* max_data ) = NULL;

int32_t (*opal_generic_simple_unpack_function_cuda_vector_p)( opal_convertor_t* pConvertor,
                                                              struct iovec* iov,
                                                              uint32_t* out_size,
                                                              size_t* max_data ) = NULL;

void (*pack_contiguous_loop_cuda_p)( dt_elem_desc_t* ELEM,
                                     uint32_t* COUNT,
                                     unsigned char** SOURCE,
                                     unsigned char** DESTINATION,
                                     size_t* SPACE ) = NULL;

void (*unpack_contiguous_loop_cuda_p)( dt_elem_desc_t* ELEM,
                                       uint32_t* COUNT,
                                       unsigned char** SOURCE,
                                       unsigned char** DESTINATION,
                                       size_t* SPACE ) = NULL;

void (*pack_predefined_data_cuda_p)( dt_elem_desc_t* ELEM,
                                     uint32_t* COUNT,
                                     unsigned char** SOURCE,
                                     unsigned char** DESTINATION,
                                     size_t* SPACE ) = NULL;

void (*opal_cuda_sync_device_p)(void) = NULL;

unsigned char* (*opal_cuda_get_gpu_pack_buffer_p)(void) = NULL;

void (*opal_cuda_free_gpu_buffer_p)(void *addr, int gpu_id) = NULL;

void* (*opal_cuda_malloc_gpu_buffer_p)(size_t size, int gpu_id) = NULL;

#define OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN(handle, fname)       \
    do {                                                                \
        char* _error;                                                   \
        *(void **)(&(fname ## _p)) = dlsym((handle), # fname);          \
        if(NULL != (_error = dlerror()) )  {                            \
            opal_output(0, "Finding %s error: %s\n", # fname, _error);  \
            fname ## _p = NULL;                                         \
            return OPAL_ERROR;                                          \
        }                                                               \
    } while (0)

int32_t opal_datatype_gpu_init(void)
{
    if (opal_datatype_cuda_handle ==  NULL) {

        /* If the library name was initialized but the load failed, we have another chance to change it */
        if( NULL != opal_datatype_cuda_lib )
            free(opal_datatype_cuda_lib);
        asprintf(&opal_datatype_cuda_lib, "%s/%s", opal_install_dirs.libdir, "opal_datatype_cuda.so");

        opal_datatype_cuda_handle = dlopen(opal_datatype_cuda_lib , RTLD_LAZY);
        if (!opal_datatype_cuda_handle) {
            opal_output( 0, "Failed to load %s library: error %s\n", opal_datatype_cuda_lib, dlerror());
            opal_datatype_cuda_handle = NULL;
            return OPAL_ERROR;
        }
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, opal_datatype_cuda_init );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, opal_datatype_cuda_fini );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, opal_generic_simple_pack_function_cuda );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, opal_generic_simple_unpack_function_cuda );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, opal_generic_simple_pack_function_cuda_iov );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, opal_generic_simple_unpack_function_cuda_iov );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, opal_generic_simple_pack_function_cuda_vector );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, opal_generic_simple_unpack_function_cuda_vector );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, pack_contiguous_loop_cuda );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, unpack_contiguous_loop_cuda );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, pack_predefined_data_cuda );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, opal_cuda_sync_device );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, opal_cuda_get_gpu_pack_buffer );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, opal_cuda_free_gpu_buffer );
        OPAL_DATATYPE_FIND_CUDA_FUNCTION_OR_RETURN( opal_datatype_cuda_handle, opal_cuda_malloc_gpu_buffer );

        (*opal_datatype_cuda_init_p)();
        printf("cuda init done\n");
    }
    return OPAL_SUCCESS;
}

int32_t opal_datatype_gpu_fini(void)
{
    if (opal_datatype_cuda_handle != NULL) {
        (*opal_datatype_cuda_fini_p)();
        /* Reset all functions to NULL */
        opal_datatype_cuda_init_p = NULL;
        opal_datatype_cuda_fini_p = NULL;
        opal_generic_simple_pack_function_cuda_p = NULL;
        opal_generic_simple_unpack_function_cuda_p = NULL;
        opal_generic_simple_pack_function_cuda_iov_p = NULL;
        opal_generic_simple_unpack_function_cuda_iov_p = NULL;
        opal_generic_simple_pack_function_cuda_vector_p = NULL;
        opal_generic_simple_unpack_function_cuda_vector_p = NULL;
        pack_contiguous_loop_cuda_p = NULL;
        unpack_contiguous_loop_cuda_p = NULL;
        pack_predefined_data_cuda_p = NULL;
        opal_cuda_sync_device_p = NULL;
        opal_cuda_get_gpu_pack_buffer_p = NULL;
        opal_cuda_free_gpu_buffer_p = NULL;
        opal_cuda_malloc_gpu_buffer_p = NULL;

        dlclose(opal_datatype_cuda_handle);
        opal_datatype_cuda_handle = NULL;

        if( NULL != opal_datatype_cuda_lib )
            free(opal_datatype_cuda_lib);
        opal_datatype_cuda_lib = NULL;
        printf("cuda fini done\n");
    }
    return OPAL_SUCCESS;
}

unsigned char* opal_datatype_get_gpu_buffer(void)
{
#if OPAL_DATATYPE_CUDA_KERNEL
    if (opal_datatype_gpu_init() != OPAL_SUCCESS) {
        opal_datatype_gpu_fini();
        return NULL;
    }
    return (*opal_cuda_get_gpu_pack_buffer_p)();
#else
    return NULL;
#endif /* defined OPAL_DATATYPE_CUDA_KERNEL */
    
}
