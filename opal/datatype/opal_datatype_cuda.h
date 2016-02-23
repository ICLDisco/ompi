/*
 * Copyright (c) 2011-2014 NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef _OPAL_DATATYPE_CUDA_H
#define _OPAL_DATATYPE_CUDA_H

/* Structure to hold CUDA support functions that gets filled in when the
 * common cuda code is initialized.  This removes any dependency on <cuda.h>
 * in the opal cuda datatype code. */
struct opal_common_cuda_function_table {
    int (*gpu_is_gpu_buffer)(const void*, opal_convertor_t*);
    int (*gpu_cu_memcpy_async)(void*, const void*, size_t, opal_convertor_t*);
    int (*gpu_cu_memcpy)(void*, const void*, size_t);
    int (*gpu_memmove)(void*, void*, size_t);
};
typedef struct opal_common_cuda_function_table opal_common_cuda_function_table_t;

struct opal_datatype_cuda_kernel_function_table {
    int32_t (*opal_ddt_cuda_kernel_init_p)(void);
    int32_t (*opal_ddt_cuda_kernel_fini_p)(void);
    void (*opal_ddt_cuda_free_gpu_buffer_p)(void *addr, int gpu_id);
    void* (*opal_ddt_cuda_malloc_gpu_buffer_p)(size_t size, int gpu_id);
    void (*opal_ddt_cuda_d2dcpy_async_p)(void* dst, const void* src, size_t count);
    void (*opal_ddt_cuda_d2dcpy_p)(void* dst, const void* src, size_t count);
    void (*opal_ddt_cached_cuda_iov_fini_p)(void *cached_cuda_iov);
    void (*opal_ddt_cuda_set_cuda_stream_p)(void);
    int32_t (*opal_ddt_cuda_get_cuda_stream_p)(void);
    int32_t (*opal_ddt_generic_simple_pack_function_cuda_iov_p)( opal_convertor_t* pConvertor, struct iovec* iov, uint32_t* out_size, size_t* max_data );
    int32_t (*opal_ddt_generic_simple_unpack_function_cuda_iov_p)( opal_convertor_t* pConvertor, struct iovec* iov, uint32_t* out_size, size_t* max_data );
    int32_t (*opal_ddt_generic_simple_pack_function_cuda_vector_p)( opal_convertor_t* pConvertor, struct iovec* iov, uint32_t* out_size, size_t* max_data );
    int32_t (*opal_ddt_generic_simple_unpack_function_cuda_vector_p)( opal_convertor_t* pConvertor, struct iovec* iov, uint32_t* out_size, size_t* max_data );                                                         
};
typedef struct opal_datatype_cuda_kernel_function_table opal_datatype_cuda_kernel_function_table_t;
extern int32_t opal_datatype_cuda_kernel_support;

void mca_cuda_convertor_init(opal_convertor_t* convertor, const void *pUserBuf, const struct opal_datatype_t* datatype);
bool opal_cuda_check_bufs(char *dest, char *src);
void* opal_cuda_memcpy(void * dest, const void * src, size_t size, opal_convertor_t* convertor);
void* opal_cuda_memcpy_sync(void * dest, const void * src, size_t size);
void* opal_cuda_memmove(void * dest, void * src, size_t size);
void opal_cuda_add_initialization_function(int (*fptr)(opal_common_cuda_function_table_t *));
void opal_cuda_set_copy_function_async(opal_convertor_t* convertor, void *stream);

int32_t opal_cuda_kernel_support_init(void);
int32_t opal_cuda_kernel_support_fini(void);
int32_t opal_generic_simple_pack_function_cuda_iov( opal_convertor_t* pConvertor, struct iovec* iov, uint32_t* out_size, size_t* max_data );
int32_t opal_generic_simple_unpack_function_cuda_iov( opal_convertor_t* pConvertor, struct iovec* iov, uint32_t* out_size, size_t* max_data );
int32_t opal_generic_simple_pack_function_cuda_vector( opal_convertor_t* pConvertor, struct iovec* iov, uint32_t* out_size, size_t* max_data );
int32_t opal_generic_simple_unpack_function_cuda_vector( opal_convertor_t* pConvertor, struct iovec* iov, uint32_t* out_size, size_t* max_data ); 
void* opal_cuda_malloc_gpu_buffer(size_t size, int gpu_id);
void opal_cuda_free_gpu_buffer(void *addr, int gpu_id);
void opal_cuda_d2dcpy(void* dst, const void* src, size_t count);
void opal_cuda_d2dcpy_async(void* dst, const void* src, size_t count);
void* opal_cached_cuda_iov_init(void);
void opal_cached_cuda_iov_fini(void *cached_cuda_iov);
void opal_cuda_set_cuda_stream(void);
int32_t opal_cuda_get_cuda_stream(void);

#endif
