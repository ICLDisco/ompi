#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"

#include <stdio.h>
#include <assert.h>

int32_t opal_datatype_cuda_generic_simple_unpack_function_iov( opal_convertor_t* pConvertor,
                                                               struct iovec* iov,
                                                               uint32_t* out_size,
                                                               size_t* max_data )
{
    size_t buffer_size;
    unsigned char *source;
    size_t total_unpacked;
    uint8_t free_required = 0;
    uint8_t gpu_rdma = 0;
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    cudaStream_t working_stream;

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time, move_time;
#endif
    
//    printf("buffer size %d, max_data %d\n", iov[0].iov_len, *max_data);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start_total);
#endif
    
    if (cuda_outer_stream == NULL) {
        working_stream = cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id];
    } else {
        working_stream = cuda_outer_stream;
    }

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    if (opal_datatype_cuda_is_gpu_buffer(iov[0].iov_base)) {
        source = (unsigned char*)iov[0].iov_base;
        free_required = 0;
        gpu_rdma = 1;
    } else {
        if (OPAL_DATATYPE_USE_ZEROCPY) {
            cudaHostGetDevicePointer((void **)&source, (void *)iov[0].iov_base, 0);
            pConvertor->gpu_buffer_ptr = NULL;
            free_required = 0;
        } else {
            if (pConvertor->gpu_buffer_ptr == NULL) {
                pConvertor->gpu_buffer_ptr = (unsigned char*)opal_datatype_cuda_malloc_gpu_buffer(iov[0].iov_len, 0);
                pConvertor->gpu_buffer_size = iov[0].iov_len;
            }
            source = pConvertor->gpu_buffer_ptr + pConvertor->pipeline_size * pConvertor->pipeline_seq;
            cudaMemcpyAsync(source, iov[0].iov_base, iov[0].iov_len, cudaMemcpyHostToDevice, working_stream);
            if (!(pConvertor->flags & CONVERTOR_CUDA_ASYNC)) {
                cudaStreamSynchronize(working_stream);
            }
       //     cudaStreamSynchronize(working_stream);
            free_required = 1;
        }
    }

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME( end );
    move_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: HtoD memcpy in %ld microsec, free required %d, pipeline_size %lu, pipeline_seq %lu\n", move_time, free_required, pConvertor->pipeline_size, pConvertor->pipeline_seq); );
#endif


    buffer_size = iov[0].iov_len;
    total_unpacked = 0;
    
    /* start unpack */
    if (cuda_iov_cache_enabled) {
        opal_datatype_cuda_generic_simple_unpack_function_iov_cached(pConvertor, source, buffer_size, &total_unpacked);
    } else {
        opal_datatype_cuda_generic_simple_unpack_function_iov_non_cached(pConvertor, source, buffer_size, &total_unpacked);
    }
    
    pConvertor->bConverted += total_unpacked;
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Unpack total unpacked %d\n", total_unpacked); );

    iov[0].iov_len = total_unpacked;
    *max_data = total_unpacked;
    *out_size = 1;

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME( end_total );
    total_time = ELAPSED_TIME( start_total, end_total );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: total unpacking in %ld microsec, kernel %ld microsec\n", total_time, total_time-move_time ); );
#endif
    
    if (gpu_rdma == 0 && !(pConvertor->flags & CONVERTOR_CUDA_ASYNC)) {
        DT_CUDA_DEBUG ( opal_cuda_output(2, "Unpack sync cuda stream\n"); );
        cudaStreamSynchronize(working_stream);
    }

    if( pConvertor->bConverted == pConvertor->local_size ) {
        pConvertor->flags |= CONVERTOR_COMPLETED;
        if (pConvertor->gpu_buffer_ptr != NULL && free_required && !(pConvertor->flags & CONVERTOR_CUDA_ASYNC)) {
            DT_CUDA_DEBUG ( opal_cuda_output(2, "Unpack free buffer %p\n", pConvertor->gpu_buffer_ptr); );
            opal_datatype_cuda_free_gpu_buffer(pConvertor->gpu_buffer_ptr, 0);
            pConvertor->gpu_buffer_ptr = NULL;
        }
        return 1;
    }
    return 0;
}

int32_t opal_datatype_cuda_generic_simple_unpack_function_iov_non_cached( opal_convertor_t* pConvertor, unsigned char *source, size_t buffer_size, size_t *total_unpacked)
{
    uint32_t nb_blocks, thread_per_block, nb_blocks_used;
    unsigned char *source_base, *destination_base;
    uint8_t buffer_isfull = 0;
    cudaError_t cuda_err;
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    ddt_cuda_iov_dist_cached_t* cuda_iov_dist_h_current;
    ddt_cuda_iov_dist_cached_t* cuda_iov_dist_d_current;
    ddt_cuda_iov_pipeline_block_non_cached_t *cuda_iov_pipeline_block_non_cached;
    cudaStream_t cuda_stream_iov = NULL;
    const struct iovec *ddt_iov = NULL;
    uint32_t ddt_iov_count = 0;
    size_t contig_disp = 0;
    uint32_t ddt_iov_start_pos, ddt_iov_end_pos, current_ddt_iov_pos;
    OPAL_PTRDIFF_TYPE ddt_extent;

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    TIMER_DATA_TYPE start, end;
    long total_time;
#endif
    
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Unpack using IOV non cached, convertor %p, GPU base %p, unpack from buffer %p, total size %ld\n",
                                     pConvertor, pConvertor->pBaseBuf, source, buffer_size); );
    
    opal_convertor_raw_cached( pConvertor, &ddt_iov, &ddt_iov_count);
    if (ddt_iov == NULL) {
        DT_CUDA_DEBUG ( opal_cuda_output(0, "Can not get ddt iov\n"););
        return OPAL_ERROR;
    }
    
  //  cuda_streams->current_stream_id = 0;
    thread_per_block = CUDA_WARP_SIZE * 5;
    nb_blocks = 256;
    source_base = source;
    opal_datatype_type_extent(pConvertor->pDesc, &ddt_extent);
    opal_datatype_cuda_set_ddt_iov_position(pConvertor, pConvertor->bConverted, ddt_iov, ddt_iov_count);
    destination_base = (unsigned char*)pConvertor->pBaseBuf + pConvertor->current_count * ddt_extent;

    while( pConvertor->current_count < pConvertor->count && !buffer_isfull) {

        nb_blocks_used = 0;
        ddt_iov_start_pos = pConvertor->current_iov_pos;
        ddt_iov_end_pos = ddt_iov_start_pos + IOV_PIPELINE_SIZE;
        if (ddt_iov_end_pos > ddt_iov_count) {
            ddt_iov_end_pos = ddt_iov_count;
        }
        cuda_iov_pipeline_block_non_cached = current_cuda_device->cuda_iov_pipeline_block_non_cached[current_cuda_device->cuda_iov_pipeline_block_non_cached_first_avail];
        cuda_iov_pipeline_block_non_cached->cuda_stream = cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id];
        cuda_iov_dist_h_current = cuda_iov_pipeline_block_non_cached->cuda_iov_dist_non_cached_h;
        cuda_iov_dist_d_current = cuda_iov_pipeline_block_non_cached->cuda_iov_dist_non_cached_d;
        cuda_stream_iov = cuda_iov_pipeline_block_non_cached->cuda_stream;
        cuda_err = cudaEventSynchronize(cuda_iov_pipeline_block_non_cached->cuda_event);
        opal_cuda_check_error(cuda_err);
#if OPAL_DATATYPE_IOV_UNIFIED_MEM
        cuda_err = cudaStreamAttachMemAsync(cuda_stream_iov, cuda_iov_dist_h_current, 0, cudaMemAttachHost);
        opal_cuda_check_error(cuda_err);
        cudaStreamSynchronize(cuda_stream_iov);
#endif        

#if defined (OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif

        buffer_isfull = opal_datatype_cuda_iov_to_cuda_iov(pConvertor, ddt_iov, cuda_iov_dist_h_current, ddt_iov_start_pos, ddt_iov_end_pos, &buffer_size, &nb_blocks_used, total_unpacked, &contig_disp, &current_ddt_iov_pos);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: Unpack src %p to dest %p, iov is prepared in %ld microsec, kernel submitted to CUDA stream %d, nb_blocks_used %d\n", source_base, destination_base, total_time,  cuda_streams->current_stream_id, nb_blocks_used); );
#endif
#if OPAL_DATATYPE_IOV_UNIFIED_MEM
        //cuda_err = cudaStreamAttachMemAsync(cuda_stream_iov, cuda_iov_dist_d_current);
        //cudaStreamSynchronize(cuda_stream_iov);
#else
        cudaMemcpyAsync(cuda_iov_dist_d_current, cuda_iov_dist_h_current, sizeof(ddt_cuda_iov_dist_cached_t)*(nb_blocks_used+1), cudaMemcpyHostToDevice, cuda_stream_iov);
#endif
        opal_generic_simple_unpack_cuda_iov_cached_kernel<<<nb_blocks, thread_per_block, 0, cuda_stream_iov>>>(cuda_iov_dist_d_current, 0, nb_blocks_used, 0, 0, nb_blocks_used, destination_base, source_base, 0, 0);
        //cudaStreamSynchronize(*cuda_stream_iov);
        cuda_err = cudaEventRecord(cuda_iov_pipeline_block_non_cached->cuda_event, cuda_stream_iov);
        opal_cuda_check_error(cuda_err);
        current_cuda_device->cuda_iov_pipeline_block_non_cached_first_avail ++;
        if (current_cuda_device->cuda_iov_pipeline_block_non_cached_first_avail >= NB_PIPELINE_NON_CACHED_BLOCKS) {
            current_cuda_device->cuda_iov_pipeline_block_non_cached_first_avail = 0;
        }
        source_base += contig_disp;
        if (!buffer_isfull) {
            pConvertor->current_iov_pos = current_ddt_iov_pos;
            if (current_ddt_iov_pos == ddt_iov_count) {
                pConvertor->current_count ++;
                pConvertor->current_iov_pos = 0;
                destination_base += ddt_extent;
            }
        }
    }

 //   cudaStreamSynchronize(cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);

    return OPAL_SUCCESS;
}

int32_t opal_datatype_cuda_generic_simple_unpack_function_iov_cached( opal_convertor_t* pConvertor, unsigned char *source, size_t buffer_size, size_t *total_unpacked)
{
    uint32_t i;
    uint32_t nb_blocks, thread_per_block, nb_blocks_used;
    unsigned char *source_base, *destination_base;
    uint8_t buffer_isfull = 0;
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    cudaStream_t cuda_stream_iov = NULL;
    uint32_t cuda_iov_start_pos, cuda_iov_end_pos;
    ddt_cuda_iov_total_cached_t* cached_cuda_iov = NULL;
    ddt_cuda_iov_dist_cached_t* cached_cuda_iov_dist_d = NULL;
    uint32_t *cached_cuda_iov_nb_bytes_list_h = NULL;
    uint32_t cached_cuda_iov_count = 0;
    size_t cuda_iov_partial_length_start = 0;
    size_t cuda_iov_partial_length_end = 0;
    opal_datatype_count_t convertor_current_count;
    OPAL_PTRDIFF_TYPE ddt_extent;

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    TIMER_DATA_TYPE start, end;
    long total_time;
#endif
    
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Unpack using IOV cached, convertor %p, GPU base %p, unpack from buffer %p, total size %ld\n",
                                     pConvertor, pConvertor->pBaseBuf, source, buffer_size); );

#if defined (OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif

 //   cuda_streams->current_stream_id = 0;
    source_base = source;
    thread_per_block = CUDA_WARP_SIZE * 8;
    nb_blocks = 64;
    destination_base = (unsigned char*)pConvertor->pBaseBuf;
    
    /* cuda iov is not cached, start to cache iov */
    if(opal_datatype_cuda_cuda_iov_is_cached(pConvertor) == 0) {
#if defined (OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        if (opal_datatype_cuda_cache_cuda_iov(pConvertor, &nb_blocks_used) == OPAL_SUCCESS) {
            DT_CUDA_DEBUG ( opal_cuda_output(2, "Unpack cuda iov is cached, count %d\n", nb_blocks_used););
        }
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: Unpack cuda iov is cached in %ld microsec, nb_blocks_used %d\n", total_time, nb_blocks_used); );
#endif
    }
      
    /* now we use cached cuda iov */
    opal_datatype_cuda_get_cached_cuda_iov(pConvertor, &cached_cuda_iov);
    cached_cuda_iov_dist_d = cached_cuda_iov->cuda_iov_dist_d;
    assert(cached_cuda_iov_dist_d != NULL);
    cached_cuda_iov_nb_bytes_list_h = cached_cuda_iov->nb_bytes_h;
    assert(cached_cuda_iov_nb_bytes_list_h != NULL);
    
    cached_cuda_iov_count = cached_cuda_iov->cuda_iov_count;
    opal_datatype_cuda_set_cuda_iov_position(pConvertor, pConvertor->bConverted, cached_cuda_iov_nb_bytes_list_h, cached_cuda_iov_count);
    cuda_iov_start_pos = pConvertor->current_cuda_iov_pos;
    cuda_iov_end_pos = cached_cuda_iov_count;
    nb_blocks_used = 0;
    if (cuda_outer_stream == NULL) {
        cuda_stream_iov = cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id];
    } else {
        cuda_stream_iov = cuda_outer_stream;
    }
    convertor_current_count = pConvertor->current_count;
    
    if (pConvertor->current_iov_partial_length > 0) {
        cuda_iov_partial_length_start = pConvertor->current_iov_partial_length;
        *total_unpacked += cuda_iov_partial_length_start;
        buffer_size -= cuda_iov_partial_length_start;
        pConvertor->current_iov_partial_length = 0;
        cuda_iov_start_pos ++;
        nb_blocks_used ++;
    }
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    while( pConvertor->current_count < pConvertor->count && !buffer_isfull) {
        for (i = cuda_iov_start_pos; i < cuda_iov_end_pos && !buffer_isfull; i++) {
            if (buffer_size >= cached_cuda_iov_nb_bytes_list_h[i]) {
                *total_unpacked += cached_cuda_iov_nb_bytes_list_h[i];
                buffer_size -= cached_cuda_iov_nb_bytes_list_h[i];
                nb_blocks_used ++;
            } else {
                if (buffer_size > 0) {
                    cuda_iov_partial_length_end = buffer_size;
                    *total_unpacked += cuda_iov_partial_length_end;
                    nb_blocks_used ++;
                }
                buffer_size = 0;
                buffer_isfull = 1;
                break;
            }
        }
        if (!buffer_isfull) {
            pConvertor->current_count ++;
            cuda_iov_start_pos = 0;
            cuda_iov_end_pos = cached_cuda_iov_count;
        }
    }
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: Unpack src %p, cached cuda iov is prepared in %ld microsec, kernel submitted to CUDA stream %d, nb_blocks %d\n", source_base, total_time,  cuda_streams->current_stream_id, nb_blocks_used); );
#endif
    opal_datatype_type_extent(pConvertor->pDesc, &ddt_extent);
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Unpack kernel launched src_base %p, dst_base %p, nb_blocks %ld\n", source_base, destination_base, nb_blocks_used ); );

//   cudaStreamSynchronize(cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    
    opal_generic_simple_unpack_cuda_iov_cached_kernel<<<nb_blocks, thread_per_block, 0, cuda_stream_iov>>>(cached_cuda_iov_dist_d, pConvertor->current_cuda_iov_pos, cached_cuda_iov_count, ddt_extent, convertor_current_count, nb_blocks_used, destination_base, source_base, cuda_iov_partial_length_start, cuda_iov_partial_length_end);

//   cudaStreamSynchronize(cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: Unpack kernel %ld microsec\n", total_time); );
#endif

    return OPAL_SUCCESS;
}