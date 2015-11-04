#ifndef OPAL_DATATYPE_CUDA_H_HAS_BEEN_INCLUDED
#define OPAL_DATATYPE_CUDA_H_HAS_BEEN_INCLUDED

extern "C"
{
    
int32_t opal_ddt_cuda_kernel_init(void);

int32_t opal_ddt_cuda_kernel_fini(void);
                                
                                                
int32_t opal_ddt_generic_simple_pack_function_cuda_vector( opal_convertor_t* pConvertor,
                                                           struct iovec* iov, 
                                                           uint32_t* out_size,
                                                           size_t* max_data );
                                                
int32_t opal_ddt_generic_simple_pack_function_cuda_iov( opal_convertor_t* pConvertor,
                                                        struct iovec* iov, 
                                                        uint32_t* out_size,
                                                        size_t* max_data );                                              
                                                  
int32_t opal_ddt_generic_simple_unpack_function_cuda_iov( opal_convertor_t* pConvertor,
                                                          struct iovec* iov, 
                                                          uint32_t* out_size,
                                                          size_t* max_data );  
                                                
int32_t opal_ddt_generic_simple_unpack_function_cuda_vector( opal_convertor_t* pConvertor,
                                                             struct iovec* iov, 
                                                             uint32_t* out_size,
                                                             size_t* max_data );

void pack_contiguous_loop_cuda( dt_elem_desc_t* ELEM,
                                uint32_t* COUNT,
                                unsigned char** SOURCE,
                                unsigned char** DESTINATION,
                                size_t* SPACE );
                                
void pack_contiguous_loop_cuda_memcpy2d_d2h( dt_elem_desc_t* ELEM,
                                         uint32_t* COUNT,
                                         unsigned char** SOURCE,
                                         unsigned char** DESTINATION,
                                         size_t* SPACE );
                                         
void pack_contiguous_loop_cuda_zerocopy( dt_elem_desc_t* ELEM,
                                         uint32_t* COUNT,
                                         unsigned char** SOURCE,
                                         unsigned char** DESTINATION,
                                         size_t* SPACE );
                                         
void pack_contiguous_loop_cuda_pipeline( dt_elem_desc_t* ELEM,
                                         uint32_t* COUNT,
                                         unsigned char** SOURCE,
                                         unsigned char** DESTINATION,
                                         size_t* SPACE, unsigned char* gpu_buffer );
                                
void unpack_contiguous_loop_cuda( dt_elem_desc_t* ELEM,
                                  uint32_t* COUNT,
                                  unsigned char** SOURCE,
                                  unsigned char** DESTINATION,
                                  size_t* SPACE );

void unpack_contiguous_loop_cuda_memcpy2d_d2h( dt_elem_desc_t* ELEM,
                                           uint32_t* COUNT,
                                           unsigned char** SOURCE,
                                           unsigned char** DESTINATION,
                                           size_t* SPACE );

void unpack_contiguous_loop_cuda_zerocopy( dt_elem_desc_t* ELEM,
                                           uint32_t* COUNT,
                                           unsigned char** SOURCE,
                                           unsigned char** DESTINATION,
                                           size_t* SPACE);
                                  
void pack_predefined_data_cuda( dt_elem_desc_t* ELEM,
                                uint32_t* COUNT,
                                unsigned char** SOURCE,
                                unsigned char** DESTINATION,
                                size_t* SPACE );
                                
void unpack_predefined_data_cuda( dt_elem_desc_t* ELEM,
                                  uint32_t* COUNT,
                                  unsigned char** SOURCE,
                                  unsigned char** DESTINATION,
                                  size_t* SPACE );

int32_t opal_ddt_cuda_is_gpu_buffer(const void *ptr);

void* opal_ddt_cuda_malloc_gpu_buffer(size_t size, int gpu_id);

void opal_ddt_cuda_free_gpu_buffer(void *addr, int gpu_id);

void opal_ddt_cuda_d2dcpy_async(void* dst, const void* src, size_t count);

void opal_ddt_cuda_d2dcpy(void* dst, const void* src, size_t count);

void opal_dump_cuda_list(ddt_cuda_list_t *list);

void* opal_ddt_cuda_iov_dist_init(void);

void opal_ddt_cuda_iov_dist_fini(void *cuda_iov_dist);

void pack_iov_cached(opal_convertor_t* pConvertor, unsigned char *destination);

}
                            
#endif  /* OPAL_DATATYPE_CUDA_H_HAS_BEEN_INCLUDED */