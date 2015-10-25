#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"

#include <stdio.h>
#include <assert.h>


int32_t opal_generic_simple_unpack_function_cuda_vector2( opal_convertor_t* pConvertor,
                                                         struct iovec* iov, uint32_t* out_size,
                                                         size_t* max_data )
{
    dt_stack_t* pStack;                /* pointer to the position on the stack */
    uint32_t pos_desc;                 /* actual position in the description of the derived datatype */
    uint32_t count_desc;               /* the number of items already done in the actual pos_desc */
    size_t total_unpacked = 0;         /* total size unpacked this time */
    dt_elem_desc_t* description;
    dt_elem_desc_t* pElem;
    const opal_datatype_t *pData = pConvertor->pDesc;
    unsigned char *conv_ptr, *iov_ptr;
    size_t iov_len_local;
    uint32_t iov_count;
    uint8_t free_required;
    uint32_t count_desc_tmp;

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    TIMER_DATA_TYPE start, end;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 1, "opal_convertor_generic_simple_unpack( %p, {%p, %lu}, %u , %u)\n",
                                     (void*)pConvertor, iov[0].iov_base, (unsigned long)iov[0].iov_len, *out_size, *max_data ); )

    description = pConvertor->use_desc->desc;

    /* For the first step we have to add both displacement to the source. After in the
     * main while loop we will set back the source_base to the correct value. This is
     * due to the fact that the convertor can stop in the middle of a data with a count
     */
    pStack     = pConvertor->pStack + pConvertor->stack_pos;
    pos_desc   = pStack->index;
    conv_ptr   = pConvertor->pBaseBuf + pStack->disp;
    count_desc = (uint32_t)pStack->count;
    pStack--;
    pConvertor->stack_pos--;
    pElem = &(description[pos_desc]);

    DT_CUDA_DEBUG( opal_cuda_output( 1, "unpack start pos_desc %d count_desc %d disp %ld\n"
                           "stack_pos %d pos_desc %d count_desc %d disp %ld\n",
                           pos_desc, count_desc, (long)(conv_ptr - pConvertor->pBaseBuf),
                           pConvertor->stack_pos, pStack->index, (int)pStack->count, (long)(pStack->disp) ); );

    for( iov_count = 0; iov_count < (*out_size); iov_count++ ) {
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        if (opal_cuda_is_gpu_buffer(iov[iov_count].iov_base)) {
            iov_ptr = (unsigned char*)iov[iov_count].iov_base;
            free_required = 0;
        } else {
            if (OPAL_DATATYPE_VECTOR_USE_MEMCPY2D || OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
                iov_ptr = (unsigned char*)iov[iov_count].iov_base;
                pConvertor->gpu_buffer_ptr = NULL;
                free_required = 0;
            } else {
                if (pConvertor->gpu_buffer_ptr == NULL) {
                    pConvertor->gpu_buffer_ptr = (unsigned char*)opal_cuda_malloc_gpu_buffer(iov[iov_count].iov_len, 0);
                }
                iov_ptr = pConvertor->gpu_buffer_ptr;
                cudaMemcpy(iov_ptr, iov[iov_count].iov_base, iov[iov_count].iov_len, cudaMemcpyHostToDevice);
                free_required = 1;
            }
        } 
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        printf( "[Timing]: HtoD memcpy in %ld microsec, free required %d\n", total_time, free_required );
#endif
        iov_len_local = iov[iov_count].iov_len;
        if( 0 != pConvertor->partial_length ) {
            /* not support yet */
        }
        while( 1 ) {
            while( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {
                /* now here we have a basic datatype */
                /* should not go to here */
                unpack_predefined_data_cuda( pElem, &count_desc, &iov_ptr, &conv_ptr, &iov_len_local );
                if( 0 == count_desc ) {  /* completed */
                    conv_ptr = pConvertor->pBaseBuf + pStack->disp;
                    pos_desc++;  /* advance to the next data */
                    UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                    continue;
                }
                assert( pElem->elem.common.type < OPAL_DATATYPE_MAX_PREDEFINED );
                if( 0 != iov_len_local ) {
                    assert(0);
                }
                goto complete_loop;
            }
            if( OPAL_DATATYPE_END_LOOP == pElem->elem.common.type ) { /* end of the current loop */
                DT_CUDA_DEBUG( opal_cuda_output( 2, "unpack end_loop count %d stack_pos %d pos_desc %d disp %ld space %lu\n",
                                                 (int)pStack->count, pConvertor->stack_pos, pos_desc,
                                                 (long)pStack->disp, (unsigned long)iov_len_local ); );
                if( --(pStack->count) == 0 ) { /* end of loop */
                    if( 0 == pConvertor->stack_pos ) {
                        /* Do the same thing as when the loop is completed */
                        iov[iov_count].iov_len -= iov_len_local;  /* update the amount of valid data */
                        total_unpacked += iov[iov_count].iov_len;
                        iov_count++;  /* go to the next */
                        goto complete_conversion;
                    }
                    pConvertor->stack_pos--;
                    pStack--;
                    pos_desc++;
                } else {
                    pos_desc = pStack->index + 1;
                    if( pStack->index == -1 ) {
                        pStack->disp += (pData->ub - pData->lb);
                    } else {
                        assert( OPAL_DATATYPE_LOOP == description[pStack->index].loop.common.type );
                        pStack->disp += description[pStack->index].loop.extent;
                    }
                }
                conv_ptr = pConvertor->pBaseBuf + pStack->disp;
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                DT_CUDA_DEBUG( opal_cuda_output( 2, "unpack new_loop count %d stack_pos %d pos_desc %d disp %ld space %lu\n",
                                                 (int)pStack->count, pConvertor->stack_pos, pos_desc,
                                                 (long)pStack->disp, (unsigned long)iov_len_local ); );
            }
            if( OPAL_DATATYPE_LOOP == pElem->elem.common.type ) {
                OPAL_PTRDIFF_TYPE local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr;
                if( pElem->loop.common.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                    if (OPAL_DATATYPE_VECTOR_USE_MEMCPY2D) {
                        unpack_contiguous_loop_cuda_memcpy2d(pElem, &count_desc, &iov_ptr, &conv_ptr, &iov_len_local);
                    } else if (OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
                        unpack_contiguous_loop_cuda_zerocopy(pElem, &count_desc, &iov_ptr, &conv_ptr, &iov_len_local);
                    } else {
                        unpack_contiguous_loop_cuda(pElem, &count_desc, &iov_ptr, &conv_ptr, &iov_len_local);
                    }
                    if( 0 == count_desc ) {  /* completed */
                        pos_desc += pElem->loop.items + 1;
                        goto update_loop_description;
                    }
                    /* Save the stack with the correct last_count value. */
                }
                local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr - local_disp;
                PUSH_STACK( pStack, pConvertor->stack_pos, pos_desc, OPAL_DATATYPE_LOOP, count_desc,
                            pStack->disp + local_disp);
                pos_desc++;
            update_loop_description:  /* update the current state */
                conv_ptr = pConvertor->pBaseBuf + pStack->disp;
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                continue;
            }
        }
    complete_loop:
        iov[iov_count].iov_len -= iov_len_local;  /* update the amount of valid data */
        total_unpacked += iov[iov_count].iov_len;
    }
 complete_conversion:
    cudaDeviceSynchronize();
    *max_data = total_unpacked;
    pConvertor->bConverted += total_unpacked;  /* update the already converted bytes */
    *out_size = iov_count;
    if( pConvertor->bConverted == pConvertor->remote_size ) {
        pConvertor->flags |= CONVERTOR_COMPLETED;
        DT_CUDA_DEBUG( opal_cuda_output( 0, "Total unpacked %lu\n", pConvertor->bConverted); );
        if (pConvertor->gpu_buffer_ptr != NULL && free_required == 1) {
            opal_cuda_free_gpu_buffer(pConvertor->gpu_buffer_ptr, 0);
            pConvertor->gpu_buffer_ptr = NULL;
        }
        return 1;
    }
    /* Save the global position for the next round */
    PUSH_STACK( pStack, pConvertor->stack_pos, pos_desc, pElem->elem.common.type, count_desc,
                conv_ptr - pConvertor->pBaseBuf );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "unpack save stack stack_pos %d pos_desc %d count_desc %d disp %ld\n",
                                     pConvertor->stack_pos, pStack->index, (int)pStack->count, (long)pStack->disp ); );
    return 0;
}

int32_t opal_generic_simple_unpack_function_cuda_vector( opal_convertor_t* pConvertor,
                                                         struct iovec* iov, uint32_t* out_size,
                                                         size_t* max_data )
{
    dt_stack_t* pStack;                /* pointer to the position on the stack */
    uint32_t pos_desc;                 /* actual position in the description of the derived datatype */
    uint32_t count_desc;               /* the number of items already done in the actual pos_desc */
    size_t total_unpacked = 0;         /* total size unpacked this time */
    dt_elem_desc_t* description;
    dt_elem_desc_t* pElem;
    const opal_datatype_t *pData = pConvertor->pDesc;
    unsigned char *conv_ptr, *iov_ptr;
    size_t iov_len_local;
    uint32_t iov_count;
    uint8_t free_required;
    uint32_t count_desc_tmp;

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    TIMER_DATA_TYPE start, end;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 2, "opal_convertor_generic_simple_unpack_vector( %p, {%p, %lu}, %u , %u)\n",
                                     (void*)pConvertor, iov[0].iov_base, (unsigned long)iov[0].iov_len, *out_size, *max_data ); )

    description = pConvertor->use_desc->desc;

    /* For the first step we have to add both displacement to the source. After in the
     * main while loop we will set back the source_base to the correct value. This is
     * due to the fact that the convertor can stop in the middle of a data with a count
     */
    pStack     = pConvertor->pStack + pConvertor->stack_pos;
    pos_desc   = pStack->index;
    conv_ptr   = pConvertor->pBaseBuf + pStack->disp;
    count_desc = (uint32_t)pStack->count;
    pStack--;
    pConvertor->stack_pos--;
    pElem = &(description[pos_desc]);

    DT_CUDA_DEBUG( opal_cuda_output( 4, "unpack start pos_desc %d count_desc %d disp %ld\n"
                           "stack_pos %d pos_desc %d count_desc %d disp %ld\n",
                           pos_desc, count_desc, (long)(conv_ptr - pConvertor->pBaseBuf),
                           pConvertor->stack_pos, pStack->index, (int)pStack->count, (long)(pStack->disp) ); );

    for( iov_count = 0; iov_count < (*out_size); iov_count++ ) {
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        if (opal_cuda_is_gpu_buffer(iov[iov_count].iov_base)) {
            iov_ptr = (unsigned char*)iov[iov_count].iov_base;
            free_required = 0;
        } else {
            if (OPAL_DATATYPE_VECTOR_USE_MEMCPY2D || OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
                iov_ptr = (unsigned char*)iov[iov_count].iov_base;
                pConvertor->gpu_buffer_ptr = NULL;
                free_required = 0;
            } else {
                if (pConvertor->gpu_buffer_ptr == NULL) {
                    pConvertor->gpu_buffer_ptr = (unsigned char*)opal_cuda_malloc_gpu_buffer(iov[iov_count].iov_len, 0);
                }
                iov_ptr = pConvertor->gpu_buffer_ptr;
                cudaMemcpy(iov_ptr, iov[iov_count].iov_base, iov[iov_count].iov_len, cudaMemcpyHostToDevice);
                free_required = 1;
            }
        }
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: HtoD memcpy in %ld microsec, free required %d\n", total_time, free_required ); );
#endif
        cudaStreamSynchronize(cuda_streams->opal_cuda_stream[0]);
        iov_len_local = iov[iov_count].iov_len;
        if( 0 != pConvertor->partial_length ) {
            /* not support yet */
        }
        while( 1 ) {
            while( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {
                /* now here we have a basic datatype */
                /* should not go to here */
                pStack--;
                pConvertor->stack_pos--;
                pos_desc --;
                pElem = &(description[pos_desc]);
                count_desc = count_desc_tmp;
                goto complete_loop;
            }
            if( OPAL_DATATYPE_END_LOOP == pElem->elem.common.type ) { /* end of the current loop */
                DT_CUDA_DEBUG( opal_cuda_output( 4, "unpack end_loop count %d stack_pos %d pos_desc %d disp %ld space %lu\n",
                                                 (int)pStack->count, pConvertor->stack_pos, pos_desc,
                                                 (long)pStack->disp, (unsigned long)iov_len_local ); );
                if( --(pStack->count) == 0 ) { /* end of loop */
                    if( 0 == pConvertor->stack_pos ) {
                        /* Do the same thing as when the loop is completed */
                        iov[iov_count].iov_len -= iov_len_local;  /* update the amount of valid data */
                        total_unpacked += iov[iov_count].iov_len;
                        iov_count++;  /* go to the next */
                        goto complete_conversion;
                    }
                    pConvertor->stack_pos--;
                    pStack--;
                    pos_desc++;
                } else {
                    pos_desc = pStack->index + 1;
                    if( pStack->index == -1 ) {
                        pStack->disp += (pData->ub - pData->lb);
                    } else {
                        assert( OPAL_DATATYPE_LOOP == description[pStack->index].loop.common.type );
                        pStack->disp += description[pStack->index].loop.extent;
                    }
                }
                conv_ptr = pConvertor->pBaseBuf + pStack->disp;
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                DT_CUDA_DEBUG( opal_cuda_output( 4, "unpack new_loop count %d stack_pos %d pos_desc %d disp %ld space %lu\n",
                                                 (int)pStack->count, pConvertor->stack_pos, pos_desc,
                                                 (long)pStack->disp, (unsigned long)iov_len_local ); );
            }
            if( OPAL_DATATYPE_LOOP == pElem->elem.common.type ) {
                OPAL_PTRDIFF_TYPE local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr;
                if( pElem->loop.common.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                    if (OPAL_DATATYPE_VECTOR_USE_MEMCPY2D) {
                        unpack_contiguous_loop_cuda_memcpy2d(pElem, &count_desc, &iov_ptr, &conv_ptr, &iov_len_local);
                    } else if (OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
                        unpack_contiguous_loop_cuda_zerocopy(pElem, &count_desc, &iov_ptr, &conv_ptr, &iov_len_local);
                    } else {
                        unpack_contiguous_loop_cuda(pElem, &count_desc, &iov_ptr, &conv_ptr, &iov_len_local);
                    }
                    if( 0 == count_desc ) {  /* completed */
                        pos_desc += pElem->loop.items + 1;
                        goto update_loop_description;
                    }
                    /* Save the stack with the correct last_count value. */
                }
                local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr - local_disp;
                PUSH_STACK( pStack, pConvertor->stack_pos, pos_desc, OPAL_DATATYPE_LOOP, count_desc,
                            pStack->disp + local_disp);
                pos_desc++;
            update_loop_description:  /* update the current state */
            //    conv_ptr = pConvertor->pBaseBuf + pStack->disp;
                count_desc_tmp = count_desc;
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                continue;
            }
        }
    complete_loop:
        iov[iov_count].iov_len -= iov_len_local;  /* update the amount of valid data */
        total_unpacked += iov[iov_count].iov_len;
    }
 complete_conversion:
    *max_data = total_unpacked;
    pConvertor->bConverted += total_unpacked;  /* update the already converted bytes */
    *out_size = iov_count;
    DT_CUDA_DEBUG( opal_cuda_output( 2, "Unpack total unpacked %lu\n", total_unpacked); );
    if( pConvertor->bConverted == pConvertor->remote_size ) {
        pConvertor->flags |= CONVERTOR_COMPLETED;
        if (pConvertor->gpu_buffer_ptr != NULL && free_required == 1) {
            opal_cuda_free_gpu_buffer(pConvertor->gpu_buffer_ptr, 0);
            pConvertor->gpu_buffer_ptr = NULL;
        }
        return 1;
    }
    /* Save the global position for the next round */
    PUSH_STACK( pStack, pConvertor->stack_pos, pos_desc, pElem->elem.common.type, count_desc,
                conv_ptr - pConvertor->pBaseBuf );
    DT_CUDA_DEBUG( opal_cuda_output( 4, "unpack save stack stack_pos %d pos_desc %d count_desc %d disp %ld\n",
                                     pConvertor->stack_pos, pStack->index, (int)pStack->count, (long)pStack->disp ); );
    return 0;
}

int32_t opal_generic_simple_unpack_function_cuda_iov( opal_convertor_t* pConvertor,
                                                      struct iovec* iov,
                                                      uint32_t* out_size,
                                                      size_t* max_data )
{
    uint32_t i, j;
    uint32_t count_desc, nb_blocks_per_description, dst_offset, residue_desc;
    uint32_t nb_blocks, thread_per_block, nb_blocks_used;
    size_t length, buffer_size, length_per_iovec;
    unsigned char *source, *source_base, *destination_base;
    size_t total_unpacked, total_converted;
    int32_t complete_flag = 0;
    uint8_t buffer_isfull = 0;
    uint8_t free_required = 0;
    uint32_t convertor_flags;
//    dt_elem_desc_t* description;
//    dt_elem_desc_t* pElem;
//    dt_stack_t* pStack;
    uint8_t alignment, orig_alignment;
//    int32_t orig_stack_index;

    ddt_cuda_iov_dist_t* cuda_iov_dist_h_current;
    ddt_cuda_iov_dist_t* cuda_iov_dist_d_current;

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time, move_time;
#endif
    
    cudaStreamSynchronize(cuda_streams->opal_cuda_stream[0]);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start_total);
#endif

/*    description = pConvertor->use_desc->desc;
    pStack = pConvertor->pStack + pConvertor->stack_pos;
    pElem = &(description[pStack->index]);
    printf("size elem %d, size %lu\n", pElem->elem.common.type, opal_datatype_basicDatatypes[pElem->elem.common.type]->size);
*/

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    if (opal_cuda_is_gpu_buffer(iov[0].iov_base)) {
        source = (unsigned char*)iov[0].iov_base;
        free_required = 0;
    } else {
        if (OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
            cudaHostGetDevicePointer((void **)&source, (void *)iov[0].iov_base, 0);
            pConvertor->gpu_buffer_ptr = NULL;
            free_required = 0;
        } else {
            if (pConvertor->gpu_buffer_ptr == NULL) {
                pConvertor->gpu_buffer_ptr = (unsigned char*)opal_cuda_malloc_gpu_buffer(iov[0].iov_len, 0);
            }
            source = pConvertor->gpu_buffer_ptr;
            cudaMemcpy(source, iov[0].iov_base, iov[0].iov_len, cudaMemcpyHostToDevice);
            free_required = 1;
        }
    }

    DT_CUDA_DEBUG ( opal_cuda_output(2, "Unpack using IOV, GPU base %p, unpack from buffer %p, total size %ld\n",
                                     pConvertor->pBaseBuf, source, iov[0].iov_len); );
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME( end );
    move_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: HtoD memcpy in %ld microsec, free required %d\n", move_time, free_required ); );
#endif


#if defined (OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    buffer_size = iov[0].iov_len;
    cuda_iov_count = 1000;
    total_unpacked = 0;
    total_converted = pConvertor->bConverted;
    cuda_streams->current_stream_id = 0;
    convertor_flags = pConvertor->flags;
//    orig_stack_index = pStack->index;
    source_base = source;
    complete_flag = opal_convertor_raw( pConvertor, cuda_iov, &cuda_iov_count, &length );
    DT_CUDA_DEBUG ( opal_cuda_output(4, "Unpack complete flag %d, iov count %d, length %d, submit to CUDA stream %d\n", complete_flag, cuda_iov_count, length, cuda_streams->current_stream_id); );

#if defined (OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: ddt to iov in %ld microsec\n", total_time ); );
#endif

    dst_offset = 0;
    thread_per_block = CUDA_WARP_SIZE * 5;
    nb_blocks = 256;

    while (cuda_iov_count > 0) {

        nb_blocks_used = 0;
        cuda_iov_dist_h_current = cuda_iov_dist_h[cuda_streams->current_stream_id];
        cuda_iov_dist_d_current = cuda_iov_dist_d[cuda_streams->current_stream_id];
        destination_base = (unsigned char*)cuda_iov[0].iov_base;

#if defined (OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif

        for (i = 0; i < cuda_iov_count; i++) {
//            pElem = &(description[orig_stack_index+i]);
            if (buffer_size >= cuda_iov[i].iov_len) {
                length_per_iovec = cuda_iov[i].iov_len;
            } else {
              /*  orig_alignment = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;*/
                orig_alignment = ALIGNMENT_CHAR;
                length_per_iovec = buffer_size / orig_alignment * orig_alignment;
                buffer_isfull = 1;
            }
            buffer_size -= length_per_iovec;
            total_unpacked += length_per_iovec;

            /* check alignment */
            if ((uintptr_t)(cuda_iov[i].iov_base) % ALIGNMENT_DOUBLE == 0 && (uintptr_t)source % ALIGNMENT_DOUBLE == 0 && length_per_iovec >= ALIGNMENT_DOUBLE) {
                alignment = ALIGNMENT_DOUBLE;
            } else if ((uintptr_t)(cuda_iov[i].iov_base) % ALIGNMENT_FLOAT == 0 && (uintptr_t)source % ALIGNMENT_FLOAT == 0 && length_per_iovec >= ALIGNMENT_FLOAT) {
                alignment = ALIGNMENT_FLOAT;
            } else {
                alignment = ALIGNMENT_CHAR;
            }

            //alignment = ALIGNMENT_DOUBLE;

            count_desc = length_per_iovec / alignment;
            residue_desc = length_per_iovec % alignment;
            nb_blocks_per_description = (count_desc + thread_per_block - 1) / thread_per_block;
            DT_CUDA_DEBUG ( opal_cuda_output(10, "Unpack description %d, size %d, residue %d, alignment %d\n", i, count_desc, residue_desc, alignment); );
            for (j = 0; j < nb_blocks_per_description; j++) {
                cuda_iov_dist_h_current[nb_blocks_used].dst_offset = (unsigned char *)(cuda_iov[i].iov_base) + j * thread_per_block * alignment - destination_base;
                cuda_iov_dist_h_current[nb_blocks_used].src_offset = source - source_base;
                cuda_iov_dist_h_current[nb_blocks_used].element_alignment = alignment;
                if ( (j+1) * thread_per_block <= count_desc) {
                    cuda_iov_dist_h_current[nb_blocks_used].nb_elements = thread_per_block;// * sizeof(double);
                } else {
                    cuda_iov_dist_h_current[nb_blocks_used].nb_elements = (thread_per_block - ((j+1)*thread_per_block - count_desc));// * sizeof(double);
                }
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
                assert (cuda_iov_dist_h_current[nb_blocks_used].nb_elements > 0); 
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
                source += cuda_iov_dist_h_current[nb_blocks_used].nb_elements * alignment;
                DT_CUDA_DEBUG( opal_cuda_output(12, "Unpack \tblock %d, src %p, dst %p, nb_elements %d, alignment %d\n", nb_blocks_used, cuda_iov_dist_h_current[nb_blocks_used].src_offset, cuda_iov_dist_h_current[nb_blocks_used].dst_offset, cuda_iov_dist_h_current[nb_blocks_used].nb_elements, cuda_iov_dist_h_current[nb_blocks_used].element_alignment); );
                nb_blocks_used ++;
            }

            /* handle residue */
            if (residue_desc != 0) {
               /* orig_alignment = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;*/
                orig_alignment = ALIGNMENT_CHAR;
                cuda_iov_dist_h_current[nb_blocks_used].dst_offset = (unsigned char *)(cuda_iov[i].iov_base) + length_per_iovec / alignment * alignment - destination_base;
                cuda_iov_dist_h_current[nb_blocks_used].src_offset = source - source_base;
                cuda_iov_dist_h_current[nb_blocks_used].element_alignment = orig_alignment;
                cuda_iov_dist_h_current[nb_blocks_used].nb_elements = (length_per_iovec - length_per_iovec / alignment * alignment) / orig_alignment;
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
                assert (cuda_iov_dist_h_current[nb_blocks_used].nb_elements > 0);
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
                source += cuda_iov_dist_h_current[nb_blocks_used].nb_elements * orig_alignment;
                DT_CUDA_DEBUG( opal_cuda_output(12, "Unpack \tblock %d, src %p, dst %p, nb_elements %d, alignment %d\n", nb_blocks_used, cuda_iov_dist_h_current[nb_blocks_used].src_offset, cuda_iov_dist_h_current[nb_blocks_used].dst_offset, cuda_iov_dist_h_current[nb_blocks_used].nb_elements, cuda_iov_dist_h_current[nb_blocks_used].element_alignment); );
                nb_blocks_used ++;
            }

            if (buffer_isfull) {
                break;
            }
        }

#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: Unpack src %p, iov is prepared in %ld microsec, kernel submitted to CUDA stream %d\n", source_base, total_time,  cuda_streams->current_stream_id); );
#endif

        cudaMemcpyAsync(cuda_iov_dist_d_current, cuda_iov_dist_h_current, sizeof(ddt_cuda_iov_dist_t)*(nb_blocks_used), cudaMemcpyHostToDevice, cuda_streams->opal_cuda_stream[cuda_streams->current_stream_id]);
        opal_generic_simple_unpack_cuda_iov_kernel<<<nb_blocks, thread_per_block, 0, cuda_streams->opal_cuda_stream[cuda_streams->current_stream_id]>>>(cuda_iov_dist_d_current, nb_blocks_used, source_base, destination_base);
        cuda_streams->current_stream_id ++;
        cuda_streams->current_stream_id = cuda_streams->current_stream_id % NB_STREAMS;

        /* buffer is full */
        if (buffer_isfull) {
            size_t total_converted_tmp = total_converted;
            pConvertor->flags = convertor_flags;
            total_converted += total_unpacked;
            opal_convertor_set_position_nocheck(pConvertor, &total_converted);
            total_unpacked = total_converted - total_converted_tmp;
            break;
        }
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        convertor_flags = pConvertor->flags;
//        orig_stack_index = pStack->index;
        complete_flag = opal_convertor_raw( pConvertor, cuda_iov, &cuda_iov_count, &length );
        DT_CUDA_DEBUG ( opal_cuda_output(4, "Unpack complete flag %d, iov count %d, length %d, submit to CUDA stream %d, nb_blocks %d\n", complete_flag, cuda_iov_count, length, cuda_streams->current_stream_id, nb_blocks_used); );
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: ddt to iov in %ld microsec\n", total_time ); );
#endif

    }
   // cudaDeviceSynchronize();
    for (i = 0; i < NB_STREAMS; i++) {
        cudaStreamSynchronize(cuda_streams->opal_cuda_stream[i]);
    }

    iov[0].iov_len = total_unpacked;
    *max_data = total_unpacked;
    *out_size = 1;
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Unpack total unpacked %d\n", total_unpacked); );

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME( end_total );
    total_time = ELAPSED_TIME( start_total, end_total );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: total unpacking in %ld microsec, kernel %ld microsec\n", total_time, total_time-move_time ); );
#endif

    if( pConvertor->bConverted == pConvertor->local_size ) {
        pConvertor->flags |= CONVERTOR_COMPLETED;
        if (pConvertor->gpu_buffer_ptr != NULL && free_required) {
            opal_cuda_free_gpu_buffer(pConvertor->gpu_buffer_ptr, 0);
            pConvertor->gpu_buffer_ptr = NULL;
        }
        return 1;
    }
    return 0;
}

void unpack_contiguous_loop_cuda( dt_elem_desc_t* ELEM,
                                  uint32_t* COUNT,
                                  unsigned char** SOURCE,
                                  unsigned char** DESTINATION,
                                  size_t* SPACE )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _destination = (*DESTINATION) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _source = *(SOURCE);

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif
    
    DT_CUDA_DEBUG( opal_cuda_output( 2, "Unpack using contiguous_loop_cuda\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
//    tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
//    num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;
//    unpack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK>>>(_copy_loops, _end_loop->size, _loop->extent, _source, _destination);
     cudaMemcpy2D(_destination, _loop->extent, _source, _end_loop->size, _end_loop->size, _copy_loops, cudaMemcpyDeviceToDevice);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)
    *(DESTINATION) = _destination + _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(SOURCE) = *(SOURCE)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif

    cudaDeviceSynchronize();
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector unpacking in %ld microsec\n", total_time ); );
#endif
}

void unpack_contiguous_loop_cuda_memcpy2d( dt_elem_desc_t* ELEM,
                                  uint32_t* COUNT,
                                  unsigned char** SOURCE,
                                  unsigned char** DESTINATION,
                                  size_t* SPACE )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _destination = (*DESTINATION) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _source = *(SOURCE);

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif
    
    DT_CUDA_DEBUG( opal_cuda_output( 2, "Unpack using contiguous_loop_cuda_memcpy2d\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    cudaMemcpy2D(_destination, _loop->extent, _source, _end_loop->size, _end_loop->size, _copy_loops, cudaMemcpyHostToDevice);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)
    *(DESTINATION) = _destination + _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(SOURCE) = *(SOURCE)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    

#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector unpacking with memcpy2d in %ld microsec\n", total_time ); );
#endif
}

void unpack_contiguous_loop_cuda_zerocopy( dt_elem_desc_t* ELEM,
                                           uint32_t* COUNT,
                                           unsigned char** SOURCE,
                                           unsigned char** DESTINATION,
                                           size_t* SPACE)
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _destination = (*DESTINATION) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _source = *(SOURCE);
    unsigned char* _source_dev;

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif
    
    DT_CUDA_DEBUG( opal_cuda_output( 2, "Unpack using contiguous_loop_cuda_zerocopy\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
//    tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
//    num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;
//    cudaHostRegister(_source, _copy_loops*_end_loop->size, cudaHostRegisterMapped);
    cudaError_t reg_rv = cudaHostGetDevicePointer((void **)&_source_dev, (void *) _source, 0);
    if (reg_rv != cudaSuccess) {
        const char *cuda_err = cudaGetErrorString(reg_rv);
        printf("can not get dev mem, %s\n", cuda_err);
    }
    //cudaMemcpy2D(_destination, _loop->extent, _source_dev, _end_loop->size, _end_loop->size, _copy_loops, cudaMemcpyDeviceToDevice);
    unpack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK>>>(_copy_loops, _end_loop->size, _loop->extent, _source_dev, _destination);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)
    *(DESTINATION) = _destination + _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(SOURCE) = *(SOURCE)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif

    cudaDeviceSynchronize();
  //  cudaHostUnregister(_source);
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector unpacking in %ld microsec\n", total_time ); );
#endif
}

void unpack_predefined_data_cuda( dt_elem_desc_t* ELEM,
                                  uint32_t* COUNT,
                                  unsigned char** SOURCE,
                                  unsigned char** DESTINATION,
                                  size_t* SPACE )
{
    uint32_t _copy_count = *(COUNT);
    size_t _copy_blength;
    ddt_elem_desc_t* _elem = &((ELEM)->elem);
    unsigned char* _source = (*SOURCE);
    uint32_t nb_blocks, tasks_per_block, thread_per_block;
    unsigned char* _destination = *(DESTINATION) + _elem->disp;;

    _copy_blength = 8;//opal_datatype_basicDatatypes[_elem->common.type]->size;
    if( (_copy_count * _copy_blength) > *(SPACE) ) {
        _copy_count = (uint32_t)(*(SPACE) / _copy_blength);
        if( 0 == _copy_count ) return;  /* nothing to do */
    }
    
    
    if (*COUNT / TASK_PER_THREAD < CUDA_WARP_SIZE) {
        thread_per_block = CUDA_WARP_SIZE;
    } else if (*COUNT / TASK_PER_THREAD < CUDA_WARP_SIZE * 2) {
        thread_per_block = CUDA_WARP_SIZE * 2;
    } else if (*COUNT / TASK_PER_THREAD < CUDA_WARP_SIZE * 3) {
        thread_per_block = CUDA_WARP_SIZE * 3;
    } else {
        thread_per_block = CUDA_WARP_SIZE * 5;
    }
    tasks_per_block = thread_per_block * TASK_PER_THREAD;
    nb_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;

 //   DBGPRINT("num_blocks %d, thread %d\n", nb_blocks, tasks_per_block);
 //   DBGPRINT( "GPU pack 1. memcpy( %p, %p, %lu ) => space %lu\n", _destination, _source, (unsigned long)_copy_count, (unsigned long)(*(SPACE)) );
    
    unpack_contiguous_loop_cuda_kernel_global<<<nb_blocks, thread_per_block, 0, cuda_streams->opal_cuda_stream[cuda_streams->current_stream_id]>>>(_copy_count, _copy_blength, _elem->extent, _source, _destination);
    cuda_streams->current_stream_id ++;
    cuda_streams->current_stream_id = cuda_streams->current_stream_id % NB_STREAMS;
    
#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)  
    _copy_blength *= _copy_count;
    *(DESTINATION)  = _destination + _elem->extent*_copy_count - _elem->disp;
    *(SOURCE) += _copy_blength;
    *(SPACE)  -= _copy_blength;
    *(COUNT)  -= _copy_count;
#endif
    
}
