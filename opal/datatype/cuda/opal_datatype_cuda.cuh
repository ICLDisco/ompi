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
                                                
int32_t opal_ddt_generic_simple_unpack_function_cuda_vector( opal_convertor_t* pConvertor,
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
                                                          
int32_t opal_ddt_generic_simple_pack_function_cuda_iov_non_cached( opal_convertor_t* pConvertor, unsigned char *destination, size_t buffer_size, size_t *total_packed);                                        

int32_t opal_ddt_generic_simple_unpack_function_cuda_iov_non_cached( opal_convertor_t* pConvertor, unsigned char *source, size_t buffer_size, size_t *total_unpacked);
                                                                                                                    
int32_t opal_ddt_generic_simple_pack_function_cuda_iov_cached( opal_convertor_t* pConvertor, unsigned char *destination, size_t buffer_size, size_t *total_packed);                                        

int32_t opal_ddt_generic_simple_unpack_function_cuda_iov_cached( opal_convertor_t* pConvertor, unsigned char *source, size_t buffer_size, size_t *total_unpacked);

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

void* opal_ddt_cached_cuda_iov_init(void);

void opal_ddt_cached_cuda_iov_fini(void *cached_cuda_iov);

void pack_iov_cached(opal_convertor_t* pConvertor, unsigned char *destination);

void opal_ddt_get_cached_cuda_iov(struct opal_convertor_t *convertor, ddt_cuda_iov_total_cached_t **cached_cuda_iov);
                                  
void opal_ddt_set_cuda_iov_cached(struct opal_convertor_t *convertor, uint32_t cuda_iov_count);

uint8_t opal_ddt_cuda_iov_is_cached(struct opal_convertor_t *convertor);

void opal_ddt_check_cuda_iov_is_full(struct opal_convertor_t *convertor, uint32_t cuda_iov_count);

void opal_ddt_set_cuda_iov_position(struct opal_convertor_t *convertor, size_t ddt_offset, const uint32_t *cached_cuda_iov_nb_bytes_list_h, const uint32_t cuda_iov_count);

void opal_ddt_set_ddt_iov_position(struct opal_convertor_t *convertor, size_t ddt_offset, const struct iovec *ddt_iov,  const uint32_t ddt_iov_count);

int32_t opal_ddt_cache_cuda_iov(opal_convertor_t* pConvertor, uint32_t *cuda_iov_count);

uint8_t opal_ddt_iov_to_cuda_iov(opal_convertor_t* pConvertor, const struct iovec *ddt_iov, ddt_cuda_iov_dist_cached_t* cuda_iov_dist_h_current, uint32_t ddt_iov_start_pos, uint32_t ddt_iov_end_pos, size_t *buffer_size, uint32_t *nb_blocks_used, size_t *total_packed, size_t *contig_disp_out, uint32_t *current_ddt_iov_pos);

}
                            
#endif  /* OPAL_DATATYPE_CUDA_H_HAS_BEEN_INCLUDED */
