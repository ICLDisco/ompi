/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2014 The University of Tennessee and The University
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

int32_t opal_datatype_gpu_init(void)
{
    char *error;
    char *lib = "/home/wwu12/ompi/ompi-gpu/opal/datatype/cuda/opal_datatype_cuda.so";
    
    if (opal_datatype_cuda_handle ==  NULL) {
        opal_datatype_cuda_handle = dlopen(lib, RTLD_LAZY);
        if (!opal_datatype_cuda_handle) {
            fprintf(stderr, "%s\n", dlerror());
            opal_datatype_cuda_handle = NULL;
            return OPAL_ERROR;
        }
        
        *(void **)(&opal_datatype_cuda_init_p) = dlsym(opal_datatype_cuda_handle, "opal_datatype_cuda_init");
        if ((error = dlerror()) != NULL)  {
            fprintf(stderr, "opal_datatype_cuda_init error: %s\n", error);
            opal_datatype_cuda_init_p = NULL;
            return OPAL_ERROR;
        }
        
        *(void **)(&opal_datatype_cuda_fini_p) = dlsym(opal_datatype_cuda_handle, "opal_datatype_cuda_fini");
        if ((error = dlerror()) != NULL)  {
            fprintf(stderr, "opal_datatype_cuda_fini error: %s\n", error);
            opal_datatype_cuda_fini_p = NULL;
            return OPAL_ERROR;
        }
        
        *(void **)(&opal_generic_simple_pack_function_cuda_p) = dlsym(opal_datatype_cuda_handle, "opal_generic_simple_pack_function_cuda");
        if ((error = dlerror()) != NULL)  {
            fprintf(stderr, "opal_generic_simple_pack_function_cuda error: %s\n", error);
            opal_generic_simple_pack_function_cuda_p = NULL;
            return OPAL_ERROR;
        }
        
        *(void **)(&opal_generic_simple_unpack_function_cuda_p) = dlsym(opal_datatype_cuda_handle, "opal_generic_simple_unpack_function_cuda");
        if ((error = dlerror()) != NULL)  {
            fprintf(stderr, "opal_generic_simple_unpack_function_cuda error: %s\n", error);
            opal_generic_simple_unpack_function_cuda_p = NULL;
            return OPAL_ERROR;
        }
        
        *(void **)(&opal_generic_simple_pack_function_cuda_iov_p) = dlsym(opal_datatype_cuda_handle, "opal_generic_simple_pack_function_cuda_iov");
        if ((error = dlerror()) != NULL)  {
            fprintf(stderr, "opal_generic_simple_pack_function_cuda_iov error: %s\n", error);
            opal_generic_simple_pack_function_cuda_iov_p = NULL;
            return OPAL_ERROR;
        }
        
        *(void **)(&opal_generic_simple_unpack_function_cuda_iov_p) = dlsym(opal_datatype_cuda_handle, "opal_generic_simple_unpack_function_cuda_iov");
        if ((error = dlerror()) != NULL)  {
            fprintf(stderr, "opal_generic_simple_unpack_function_cuda_iov error: %s\n", error);
            opal_generic_simple_unpack_function_cuda_iov_p = NULL;
            return OPAL_ERROR;
        }
        
        *(void **)(&pack_contiguous_loop_cuda_p) = dlsym(opal_datatype_cuda_handle, "pack_contiguous_loop_cuda");
        if ((error = dlerror()) != NULL)  {
            fprintf(stderr, "pack_contiguous_loop_cuda error: %s\n", error);
            pack_contiguous_loop_cuda_p = NULL;
            return OPAL_ERROR;
        }
        
        *(void **)(&unpack_contiguous_loop_cuda_p) = dlsym(opal_datatype_cuda_handle, "unpack_contiguous_loop_cuda");
        if ((error = dlerror()) != NULL)  {
            fprintf(stderr, "unpack_contiguous_loop_cuda error: %s\n", error);
            unpack_contiguous_loop_cuda_p = NULL;
            return OPAL_ERROR;
        }
        
        *(void **)(&pack_predefined_data_cuda_p) = dlsym(opal_datatype_cuda_handle, "pack_predefined_data_cuda");
        if ((error = dlerror()) != NULL)  {
            fprintf(stderr, "pack_predefined_data_cuda error: %s\n", error);
            pack_predefined_data_cuda_p = NULL;
            return OPAL_ERROR;
        }
        
        *(void **)(&opal_cuda_sync_device_p) = dlsym(opal_datatype_cuda_handle, "opal_cuda_sync_device");
        if ((error = dlerror()) != NULL)  {
            fprintf(stderr, "opal_cuda_sync_device error: %s\n", error);
            opal_cuda_sync_device_p = NULL;
            return OPAL_ERROR;
        }
        
        *(void **)(&opal_cuda_get_gpu_pack_buffer_p) = dlsym(opal_datatype_cuda_handle, "opal_cuda_get_gpu_pack_buffer");
        if ((error = dlerror()) != NULL)  {
            fprintf(stderr, "opal_cuda_get_gpu_pack_buffer error: %s\n", error);
            opal_cuda_get_gpu_pack_buffer_p = NULL;
            return OPAL_ERROR;
        }
        
        (*opal_datatype_cuda_init_p)();
        printf("cuda init done\n");   
    }
    return OPAL_SUCCESS;
}

int32_t opal_datatype_gpu_fini(void)
{
    if (opal_datatype_cuda_handle != NULL) {
        (*opal_datatype_cuda_fini_p)();
        dlclose(opal_datatype_cuda_handle);
        opal_datatype_cuda_handle = NULL;
        opal_datatype_cuda_init_p = NULL;
        opal_datatype_cuda_fini_p = NULL;
        opal_generic_simple_pack_function_cuda_p = NULL;
        opal_generic_simple_unpack_function_cuda_p = NULL;
        opal_generic_simple_pack_function_cuda_iov_p = NULL;
        opal_generic_simple_unpack_function_cuda_iov_p = NULL;
        pack_contiguous_loop_cuda_p = NULL;
        unpack_contiguous_loop_cuda_p = NULL;
        pack_predefined_data_cuda_p = NULL;
        opal_cuda_sync_device_p = NULL;
        opal_cuda_get_gpu_pack_buffer_p = NULL;
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