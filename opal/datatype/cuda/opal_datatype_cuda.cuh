/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2014-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef OPAL_DATATYPE_CUDA_H_HAS_BEEN_INCLUDED
#define OPAL_DATATYPE_CUDA_H_HAS_BEEN_INCLUDED

BEGIN_C_DECLS

/* init functions of GPU datatype engine */    
int32_t opal_datatype_cuda_kernel_init(void);

/* fini function of GPU datatype engine */
int32_t opal_datatype_cuda_kernel_fini(void);

/* iov pack function */                                
int32_t opal_datatype_cuda_generic_simple_pack_function_iov( opal_convertor_t* pConvertor,
                                                                  struct iovec* iov, 
                                                                  uint32_t* out_size,
                                                                  size_t* max_data );                                              

/* iov unpack function */ 
int32_t opal_datatype_cuda_generic_simple_unpack_function_iov( opal_convertor_t* pConvertor,
                                                               struct iovec* iov, 
                                                               uint32_t* out_size,
                                                               size_t* max_data ); 

/* iov pack without cache */                                                          
int32_t opal_datatype_cuda_generic_simple_pack_function_iov_non_cached( opal_convertor_t* pConvertor,
                                                                        unsigned char *destination,
                                                                        size_t buffer_size,
                                                                        size_t *total_packed);

/* iov unpack without cache */
int32_t opal_datatype_cuda_generic_simple_unpack_function_iov_non_cached( opal_convertor_t* pConvertor, unsigned char *source, size_t buffer_size, size_t *total_unpacked);

/* iov pack with cache */                                                                                                                    
int32_t opal_datatype_cuda_generic_simple_pack_function_iov_cached( opal_convertor_t* pConvertor, unsigned char *destination, size_t buffer_size, size_t *total_packed);                                        

/* iov unpack with cache */
int32_t opal_datatype_cuda_generic_simple_unpack_function_iov_cached( opal_convertor_t* pConvertor, unsigned char *source, size_t buffer_size, size_t *total_unpacked);

/* check if ptr is gpu memory */
int32_t opal_datatype_cuda_is_gpu_buffer(const void *ptr);

/* malloc gpu buffer for pack/unpack */
void* opal_datatype_cuda_malloc_gpu_buffer(size_t size, int gpu_id);

/* free gpu buffer used for pack/unpack */
void opal_datatype_cuda_free_gpu_buffer(void *addr, int gpu_id);

/* async cuda memory movement */
void opal_datatype_cuda_d2dcpy_async(void* dst, const void* src, size_t count, void* stream);

/* sync cuda memory movement */
void opal_datatype_cuda_d2dcpy(void* dst, const void* src, size_t count, void* stream);

void opal_dump_cuda_list(ddt_cuda_list_t *list);

/* init the cuda iov used for caching */
void* opal_datatype_cuda_cached_cuda_iov_init(void);

/* clean up cached cuda iov */ 
void opal_datatype_cuda_cached_cuda_iov_fini(void *cached_cuda_iov);

/* get cached cuda iov */
void opal_datatype_cuda_get_cached_cuda_iov(struct opal_convertor_t *convertor, ddt_cuda_iov_total_cached_t **cached_cuda_iov);                                

/* check if cuda iov is cached or not */
uint8_t opal_datatype_cuda_cuda_iov_is_cached(struct opal_convertor_t *convertor);

/* move cuda iov position */
void opal_datatype_cuda_set_cuda_iov_position(struct opal_convertor_t *convertor, size_t ddt_offset, const uint32_t *cached_cuda_iov_nb_bytes_list_h, const uint32_t cuda_iov_count);

/* move cpu iov position */
void opal_datatype_cuda_set_ddt_iov_position(struct opal_convertor_t *convertor, size_t ddt_offset, const struct iovec *ddt_iov,  const uint32_t ddt_iov_count);

/* cache cuda iov */
int32_t opal_datatype_cuda_cache_cuda_iov(opal_convertor_t* pConvertor, uint32_t *cuda_iov_count);

/* convertor cpu iov to cuda iov */
uint8_t opal_datatype_cuda_iov_to_cuda_iov(opal_convertor_t* pConvertor, const struct iovec *ddt_iov,
                                           ddt_cuda_iov_dist_cached_t* cuda_iov_dist_h_current,
                                           uint32_t ddt_iov_start_pos,
                                           uint32_t ddt_iov_end_pos,
                                           size_t *buffer_size,
                                           uint32_t *nb_blocks_used,
                                           size_t *total_packed,
                                           size_t *contig_disp_out,
                                           uint32_t *current_ddt_iov_pos);

/* get cuda stream whose id is stream_id */
void* opal_datatype_cuda_get_cuda_stream_by_id(int stream_id);

/* get current cuda stream */
void *opal_datatype_cuda_get_current_cuda_stream();

/* sync current cuda stream */
void opal_datatype_cuda_sync_current_cuda_stream();

/* sync cuda stream (id) */
void opal_datatype_cuda_sync_cuda_stream(int stream_id);

/* alloc event for smcuda pack/unpack */
void* opal_datatype_cuda_alloc_event(int32_t nb_events, int32_t *loc);

/* free events used for smcuda pack/unpack */
void opal_datatype_cuda_free_event(void *cuda_event_list, int32_t nb_events);

/* query the event i */
int32_t opal_datatype_cuda_event_query(void *cuda_event_list, int32_t i);

/* sync the event i */
int32_t opal_datatype_cuda_event_sync(void *cuda_event_list, int32_t i);

/* record the event i */
int32_t opal_datatype_cuda_event_record(void *cuda_event_list, int32_t i, void* stream);

END_C_DECLS
                            
#endif  /* OPAL_DATATYPE_CUDA_H_HAS_BEEN_INCLUDED */
