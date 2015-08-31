#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"

#include <stdio.h>
#include <assert.h>

int32_t opal_generic_simple_unpack_function_cuda( opal_convertor_t* pConvertor,
                                                  struct iovec* iov, 
                                                  uint32_t* out_size,
                                                  size_t* max_data )
{
    uint32_t i;
    dt_elem_desc_t* description;
    const opal_datatype_t *pData = pConvertor->pDesc;
    uint32_t tasks_per_block, num_blocks, thread_per_block;
    dt_stack_t* pStack;
    
    return -99;
    description = pConvertor->use_desc->desc;
    
    cuda_desc_h->stack_pos = pConvertor->stack_pos;
#if defined(OPAL_DATATYPE_CUDA_DRY_RUN)
    cuda_desc_h->pBaseBuf = pConvertor->pBaseBuf;
#else
    cuda_desc_h->pBaseBuf = pBaseBuf_GPU;
#endif /* OPAL_DATATYPE_CUDA_DRY_RUN */
    cuda_desc_h->lb = pData->lb;
    cuda_desc_h->ub = pData->ub;
    cuda_desc_h->out_size = *out_size;
    cuda_desc_h->max_data = *max_data;
    cuda_desc_h->bConverted = pConvertor->bConverted;
    cuda_desc_h->local_size = pConvertor->local_size;
    cuda_desc_h->stack_size = pConvertor->stack_size;
    
    for (i = 0; i < pConvertor->stack_size; i++) {
        cuda_desc_h->pStack[i] = pConvertor->pStack[i];
    }
    if (cuda_desc_h->description_max_count != 0) {
        if (cuda_desc_h->description_max_count >= (pConvertor->use_desc->used+1)) {
            cuda_desc_h->description_count = pConvertor->use_desc->used+1;
        } else {
            cudaFree(cuda_desc_h->description);
            cuda_desc_h->description = NULL;
            cudaMalloc((void **)&(cuda_desc_h->description), sizeof(dt_elem_desc_t)*(pConvertor->use_desc->used+1));
            cuda_desc_h->description_max_count = pConvertor->use_desc->used+1;
            cuda_desc_h->description_count = pConvertor->use_desc->used+1;
        }
        
    } else {
        cudaMalloc((void **)&(cuda_desc_h->description), sizeof(dt_elem_desc_t)*(pConvertor->use_desc->used+1));
        cuda_desc_h->description_max_count = pConvertor->use_desc->used+1;
        cuda_desc_h->description_count = pConvertor->use_desc->used+1;
    }
    cudaMemcpy(cuda_desc_h->description, description, sizeof(dt_elem_desc_t)*(pConvertor->use_desc->used+1), cudaMemcpyHostToDevice);
    
    DBGPRINT("stack_size %d\n", pConvertor->stack_size);

    DBGPRINT("flags %d, types %d, count %d\n", description->elem.common.flags, description->elem.common.type, description->elem.count);
    
    for (i = 0; i < *out_size; i++) {
#if defined (OPAL_DATATYPE_CUDA_DRY_RUN)
        cuda_desc_h->iov[i].iov_base = iov[i].iov_base;
#endif /* OPAL_DATATYPE_CUDA_DRY_RUN */
        cuda_desc_h->iov[i].iov_len = iov[i].iov_len;
    }
    
    cudaMemcpy(cuda_desc_d, cuda_desc_h, sizeof(ddt_cuda_desc_t), cudaMemcpyHostToDevice);
    
    pStack = pConvertor->pStack + pConvertor->stack_pos;
    thread_per_block = CUDA_WARP_SIZE * 3;
    tasks_per_block = thread_per_block * TASK_PER_THREAD;
    num_blocks = ((uint32_t)pStack->count + tasks_per_block - 1) / tasks_per_block;
    printf("launch unpack kernel, count %d, num_blocks %d, total threads %d\n", (uint32_t)pStack->count, num_blocks, num_blocks*thread_per_block);
    opal_generic_simple_unpack_cuda_kernel<<<192, thread_per_block>>>(cuda_desc_d);
#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)
    size_t position = pConvertor->pDesc->size;
    opal_convertor_set_position_nocheck(pConvertor, &position);
#endif
    cudaDeviceSynchronize();
    
#if defined(OPAL_DATATYPE_CUDA_DRY_RUN)
    return -99;
#else
    // /* copy stack and description data back to CPU */
    // cudaMemcpy(cuda_desc_h, cuda_desc_d, sizeof(ddt_cuda_desc_t), cudaMemcpyDeviceToHost);
    //
    // for (i = 0; i < pConvertor->stack_size; i++) {
    //     pConvertor->pStack[i] = cuda_desc_h->pStack[i];
    // }
    //
    // pConvertor->stack_pos = cuda_desc_h->stack_pos;
    // *out_size = cuda_desc_h->out_size;
    // *max_data = cuda_desc_h->max_data;
    // pConvertor->bConverted = cuda_desc_h->bConverted;
    // pConvertor->local_size = cuda_desc_h->local_size;
    //
    // for (i = 0; i < *out_size; i++) {
    //     iov[i].iov_len = cuda_desc_h->iov[i].iov_len;
    // }
    //
    if( pConvertor->bConverted == pConvertor->local_size ) {
        // pConvertor->flags |= CONVERTOR_COMPLETED;
        return 1;
    }

    return 0;
#endif /* OPAL_DATATYPE_CUDA_DRY_RUN */
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
    TIMER_DATA_TYPE start, end, start_total, end_total;
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
                pStack--;
                pConvertor->stack_pos--;
                pos_desc --;
                pElem = &(description[pos_desc]);
                count_desc = count_desc_tmp;
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

int32_t opal_generic_simple_unpack_function_cuda_iov( opal_convertor_t* pConvertor,
                                                  struct iovec* iov, 
                                                  uint32_t* out_size,
                                                  size_t* max_data )
{
    uint32_t i, j;
    uint32_t count_desc, current_block, task_iteration, nb_blocks_per_description, dst_offset, residue_desc;
    uint32_t nb_blocks, thread_per_block;
    size_t length, buffer_size, length_per_iovec;
    unsigned char *source, *source_tmp;
    size_t total_unpacked, total_converted;
    int32_t complete_flag = 0;
    uint8_t buffer_isfull = 0;
    uint8_t free_required = 0;
    uint32_t convertor_flags;
    dt_elem_desc_t* description;
    dt_elem_desc_t* pElem;
    dt_stack_t* pStack;
    uint8_t alignment, orig_alignment;
    int32_t orig_stack_index;

    ddt_cuda_iov_dist_t* cuda_iov_dist_h_current;
    ddt_cuda_iov_dist_t* cuda_iov_dist_d_current;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start_total);
#endif
    
    description = pConvertor->use_desc->desc;
    pStack = pConvertor->pStack + pConvertor->stack_pos;
    pElem = &(description[pStack->index]);
    DT_CUDA_DEBUG ( opal_cuda_output(0, "GPU datatype UNpacking using iovec\n"); );
    
    // double *vtmp = (double *)iov[0].iov_base;
    // for (uint32_t i = 0; i < iov[0].iov_len/sizeof(double); i++) {
    //     printf(" %1.f ", *vtmp);
    //     vtmp ++;
    // }
    // printf("\n");
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    if (opal_cuda_is_gpu_buffer(iov[0].iov_base)) {
        source = (unsigned char*)iov[0].iov_base;
        free_required = 0;
    } else {  
#if defined(OPAL_DATATYPE_CUDA_DRY_RUN)
        source = (unsigned char*)iov[0].iov_base;
#else
        if (pConvertor->gpu_buffer_ptr == NULL) {
            pConvertor->gpu_buffer_ptr = (unsigned char*)opal_cuda_malloc_gpu_buffer(iov[0].iov_len, 0);
        }
        source = pConvertor->gpu_buffer_ptr;
#endif /* OPAL_DATATYPE_CUDA_DRY_RUN */  
        cudaMemcpy(source, iov[0].iov_base, iov[0].iov_len, cudaMemcpyHostToDevice);
        free_required = 1;
    }
    
    source_tmp = source;

    DT_CUDA_DEBUG ( opal_cuda_output(0, "UNpack GPU base %p, unpack from buffer %p, total size %ld\n", pConvertor->pBaseBuf, source, iov[0].iov_len); );
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    printf( "[Timing]: HtoD memcpy in %ld microsec, free required %d\n", total_time, free_required );
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
    orig_stack_index = pStack->index;
    complete_flag = opal_convertor_raw( pConvertor, cuda_iov, &cuda_iov_count, &length );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "UNPACKING complete flag %d, iov count %d, length %d, submit to CUDA stream %d\n", complete_flag, cuda_iov_count, length, cuda_streams->current_stream_id); );

#if defined (OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    printf( "[Timing]: ddt to iov in %ld microsec\n", total_time );
#endif
    
    dst_offset = 0;
    thread_per_block = CUDA_WARP_SIZE * 4;
    nb_blocks = 256;
    
    while (cuda_iov_count > 0) {
        
        current_block = 0;
        task_iteration = 0;
        cuda_iov_dist_h_current = cuda_iov_dist_h[cuda_streams->current_stream_id];
        cuda_iov_dist_d_current = cuda_iov_dist_d[cuda_streams->current_stream_id]; 
        
#if defined (OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        for (i = 0; i < nb_blocks; i++) {
            cuda_iov_dist_h_current[i].nb_tasks = 0;
        }
        
        for (i = 0; i < cuda_iov_count; i++) {
            pElem = &(description[orig_stack_index+i]);
            if (buffer_size >= cuda_iov[i].iov_len) {
                length_per_iovec = cuda_iov[i].iov_len;
            } else {
                orig_alignment = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;
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

            count_desc = length_per_iovec / alignment;
            residue_desc = length_per_iovec % alignment;
            nb_blocks_per_description = (count_desc + thread_per_block - 1) / thread_per_block;
            DT_CUDA_DEBUG ( opal_cuda_output(10, "UNPACKING description %d, size %d, residue %d, alignment %d\n", i, count_desc, residue_desc, alignment); );
            for (j = 0; j < nb_blocks_per_description; j++) {
                cuda_iov_dist_h_current[current_block].dst[task_iteration] = (unsigned char *)(cuda_iov[i].iov_base) + j * thread_per_block * alignment;
                cuda_iov_dist_h_current[current_block].src[task_iteration] = source;
                cuda_iov_dist_h_current[current_block].element_alignment[task_iteration] = alignment;
                cuda_iov_dist_h_current[current_block].nb_tasks = task_iteration + 1;
                if ( (j+1) * thread_per_block <= count_desc) {
                    cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] = thread_per_block;// * sizeof(double);
                } else {
                    cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] = (thread_per_block - ((j+1)*thread_per_block - count_desc));// * sizeof(double);
                }
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
                assert (cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] > 0); 
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
                source += cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] * alignment;
                DT_CUDA_DEBUG( opal_cuda_output(12, "UNPACKING \tblock %d, src %p, dst %p, nb_elements %d, alignment %d\n", current_block, cuda_iov_dist_h_current[current_block].src[task_iteration], cuda_iov_dist_h_current[current_block].dst[task_iteration], cuda_iov_dist_h_current[current_block].nb_elements[task_iteration], cuda_iov_dist_h_current[current_block].element_alignment[task_iteration]); );
                current_block += 1;
                if (current_block >= nb_blocks) {
                    current_block = 0;
                    task_iteration ++;
                    assert(task_iteration < CUDA_IOV_MAX_TASK_PER_BLOCK);
                }
            }
            
            /* handle residue */
            if (residue_desc != 0) {
                orig_alignment = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;
                cuda_iov_dist_h_current[current_block].dst[task_iteration] = (unsigned char *)(cuda_iov[i].iov_base) + length_per_iovec / alignment * alignment;
                cuda_iov_dist_h_current[current_block].src[task_iteration] = source;
                cuda_iov_dist_h_current[current_block].element_alignment[task_iteration] = orig_alignment;
                cuda_iov_dist_h_current[current_block].nb_tasks = task_iteration + 1;
                cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] = (length_per_iovec - length_per_iovec / alignment * alignment) / orig_alignment;
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
                assert (cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] > 0);
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
                source += cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] * orig_alignment;
                DT_CUDA_DEBUG( opal_cuda_output(12, "UNPACKING \tblock %d, src %p, dst %p, nb_elements %d, alignment %d\n", current_block, cuda_iov_dist_h_current[current_block].src[task_iteration], cuda_iov_dist_h_current[current_block].dst[task_iteration], cuda_iov_dist_h_current[current_block].nb_elements[task_iteration], cuda_iov_dist_h_current[current_block].element_alignment[task_iteration]); );
                current_block += 1;
                if (current_block >= nb_blocks) {
                    current_block = 0;
                    task_iteration ++;
                    assert(task_iteration < CUDA_IOV_MAX_TASK_PER_BLOCK);
                }
            }
            
            if (buffer_isfull) {
                break;
            }
        }

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        printf( "[Timing]: UNpack src %p, iov is prepared in %ld microsec, kernel submitted to CUDA stream %d\n", source_tmp, total_time,  cuda_streams->current_stream_id);
#endif
                
        cudaMemcpyAsync(cuda_iov_dist_d_current, cuda_iov_dist_h_current, sizeof(ddt_cuda_iov_dist_t)*(nb_blocks), cudaMemcpyHostToDevice, cuda_streams->opal_cuda_stream[cuda_streams->current_stream_id]);
        opal_generic_simple_unpack_cuda_iov_kernel<<<nb_blocks, thread_per_block, 0, cuda_streams->opal_cuda_stream[cuda_streams->current_stream_id]>>>(cuda_iov_dist_d_current);
        cuda_streams->current_stream_id ++;
        cuda_streams->current_stream_id = cuda_streams->current_stream_id % NB_STREAMS;    
        
        /* buffer is full */
        if (buffer_isfull) {
            pConvertor->flags = convertor_flags;
            total_converted += total_unpacked;
            opal_convertor_set_position_nocheck(pConvertor, &total_converted);
            break;
        }
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
        convertor_flags = pConvertor->flags;     
#endif
        convertor_flags = pConvertor->flags;
        orig_stack_index = pStack->index;
        complete_flag = opal_convertor_raw( pConvertor, cuda_iov, &cuda_iov_count, &length );
        DT_CUDA_DEBUG ( opal_cuda_output(8, "UNPACKING complete flag %d, iov count %d, length %d, submit to CUDA stream %d\n", complete_flag, cuda_iov_count, length, cuda_streams->current_stream_id); );
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        printf( "[Timing]: ddt to iov in %ld microsec\n", total_time );
#endif

    }
    for (i = 0; i < NB_STREAMS; i++) {
        cudaStreamSynchronize(cuda_streams->opal_cuda_stream[i]);
    }

    iov[0].iov_len = total_unpacked;
    *max_data = total_unpacked;
    *out_size = 1;
    DT_CUDA_DEBUG ( opal_cuda_output(0, "UNPACKING total unpacked %d\n", total_unpacked); );

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME( end_total );
    total_time = ELAPSED_TIME( start_total, end_total );
    printf( "[Timing]: total unpacking in %ld microsec\n", total_time );
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
    
    DT_CUDA_DEBUG( opal_cuda_output( 0, "I am in unpack_contiguous_loop_cuda\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
//    tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
//    num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;
    unpack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK>>>(_copy_loops, _end_loop->size, _loop->extent, _source, _destination);
//     cudaMemcpy2D(_destination, _loop->extent, _source, _end_loop->size, _end_loop->size, _copy_loops, cudaMemcpyDeviceToDevice);

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
    printf( "[Timing]: vector unpacking in %ld microsec\n", total_time );
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
    
    DT_CUDA_DEBUG( opal_cuda_output( 0, "I am in unpack_contiguous_loop_cuda_memcpy2d\n"); );

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
    printf( "[Timing]: vector unpacking with memcpy2d in %ld microsec\n", total_time );
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
    
    DT_CUDA_DEBUG( opal_cuda_output( 0, "I am in unpack_contiguous_loop_cuda_zerocopy\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
//    tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
//    num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;
//    cudaHostRegister(_source, _copy_loops*_end_loop->size, cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&_source_dev, (void *) _source, 0);
    unpack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK>>>(_copy_loops, _end_loop->size, _loop->extent, _source, _destination);

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
    printf( "[Timing]: vector unpacking in %ld microsec\n", total_time );
#endif
}
