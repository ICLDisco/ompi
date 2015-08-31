#ifndef OPAL_DATATYPE_CUDA_H_HAS_BEEN_INCLUDED
#define OPAL_DATATYPE_CUDA_H_HAS_BEEN_INCLUDED

extern "C"
{
    
void opal_datatype_cuda_init(void);

void opal_datatype_cuda_fini(void);
                                
int32_t opal_generic_simple_pack_function_cuda( opal_convertor_t* pConvertor,
                                                struct iovec* iov, 
                                                uint32_t* out_size,
                                                size_t* max_data );
                                                
int32_t opal_generic_simple_pack_function_cuda_vector( opal_convertor_t* pConvertor,
                                                       struct iovec* iov, 
                                                       uint32_t* out_size,
                                                       size_t* max_data );
                                                
int32_t opal_generic_simple_pack_function_cuda_iov( opal_convertor_t* pConvertor,
                                                    struct iovec* iov, 
                                                    uint32_t* out_size,
                                                    size_t* max_data );                                              

int32_t opal_generic_simple_unpack_function_cuda( opal_convertor_t* pConvertor,
                                                  struct iovec* iov, 
                                                  uint32_t* out_size,
                                                  size_t* max_data );
                                                  
int32_t opal_generic_simple_unpack_function_cuda_iov( opal_convertor_t* pConvertor,
                                                  struct iovec* iov, 
                                                  uint32_t* out_size,
                                                  size_t* max_data );  
                                                
int32_t opal_generic_simple_unpack_function_cuda_vector( opal_convertor_t* pConvertor,
                                                         struct iovec* iov, 
                                                         uint32_t* out_size,
                                                         size_t* max_data );

void pack_contiguous_loop_cuda( dt_elem_desc_t* ELEM,
                                uint32_t* COUNT,
                                unsigned char** SOURCE,
                                unsigned char** DESTINATION,
                                size_t* SPACE );
                                
void pack_contiguous_loop_cuda_memcpy2d( dt_elem_desc_t* ELEM,
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

void unpack_contiguous_loop_cuda_memcpy2d( dt_elem_desc_t* ELEM,
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

void opal_cuda_sync_device(void);

int32_t opal_cuda_is_gpu_buffer(const void *ptr);

void* opal_cuda_malloc_gpu_buffer(size_t size, int gpu_id);

void opal_cuda_free_gpu_buffer(void *addr, int gpu_id);

void opal_dump_cuda_list(ddt_cuda_list_t *list);

unsigned char* opal_cuda_get_gpu_pack_buffer();
}
                            
#endif  /* OPAL_DATATYPE_CUDA_H_HAS_BEEN_INCLUDED */