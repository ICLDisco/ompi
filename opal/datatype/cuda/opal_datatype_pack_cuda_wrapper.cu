#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"

#include <stdio.h>
#include <assert.h>

int32_t opal_generic_simple_pack_function_cuda( opal_convertor_t* pConvertor,
                                                struct iovec* iov, 
                                                uint32_t* out_size,
                                                size_t* max_data )
{
    uint32_t i;
    dt_elem_desc_t* description;
    dt_elem_desc_t* pElem;
    const opal_datatype_t *pData = pConvertor->pDesc;
    uint32_t tasks_per_block, num_blocks, thread_per_block;
    dt_stack_t* pStack;
    
    //return -99;

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
    cudaMemcpy(cuda_desc_h->description, description, sizeof(dt_elem_desc_t)*(cuda_desc_h->description_count), cudaMemcpyHostToDevice);
    printf("description ct %d\n", cuda_desc_h->description_count);
    
    // for (i = 0; i < pConvertor->use_desc->used+1; i++) {
    //     cuda_desc_h->description[i] = description[i];
    // }
    
    DBGPRINT("stack_size %d\n", pConvertor->stack_size);

    DBGPRINT("flags %d, types %d, count %d\n", description->elem.common.flags, description->elem.common.type, description->elem.count);
    
    for (i = 0; i < *out_size; i++) {
#if defined (OPAL_DATATYPE_CUDA_DRY_RUN)
        cuda_desc_h->iov[i].iov_base = iov[i].iov_base;
#endif /* OPAL_DATATYPE_CUDA_DRY_RUN */
        cuda_desc_h->iov[i].iov_len = iov[i].iov_len;
    }
    
    pStack = pConvertor->pStack + pConvertor->stack_pos;
    thread_per_block = CUDA_WARP_SIZE * 5;
    tasks_per_block = thread_per_block * TASK_PER_THREAD;
    num_blocks = ((uint32_t)pStack->count + tasks_per_block - 1) / tasks_per_block;
    num_blocks = 512;

    /***/
    uint32_t pos_desc, count_desc, current_block, task_iteration, nb_blocks_per_description, j, dst_offset;
    pos_desc   = pStack->index;
    pElem = &(description[pos_desc]);
    count_desc = (uint32_t)pStack->count;
    current_block = 0;
    task_iteration = 0;
    dst_offset = 0;
    while( 1 ) {
        while( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {
            nb_blocks_per_description = (count_desc + thread_per_block - 1) / thread_per_block;
            for (i = 0; i < nb_blocks_per_description; i++) {
                description_dist_h[current_block].description_index[task_iteration] = pos_desc;
                description_dist_h[current_block].description_local_index[task_iteration] = i;
                description_dist_h[current_block].dst_offset[task_iteration] = dst_offset;
                description_dist_h[current_block].description_used = task_iteration + 1;
                if ( (i+1) * thread_per_block <= count_desc) {
                    dst_offset += thread_per_block;
                } else {
                    dst_offset += thread_per_block - ((i+1)*thread_per_block - count_desc);
                }
                current_block += 1;
                if (current_block >= num_blocks) {
                    current_block = 0;
                    task_iteration ++;
                }
            }
            pos_desc ++;
            pElem = &(description[pos_desc]);
            count_desc = pElem->elem.count;
        }
        if( OPAL_DATATYPE_END_LOOP == pElem->elem.common.type ) {
            break;
        }
    }

    // for (i = 0; i < num_blocks; i++) {
    //     printf("block %d\t, used %d\n", i, description_dist_h[i].description_used);
    //     for (j = 0; j < description_dist_h[i].description_used; j++) {
    //         pos_desc = description_dist_h[i].description_index[j];
    //         pElem = &(description[pos_desc]);
    //         printf("i %d\t, descp_pos %d\t, local_index %d\t, count %d\t, dst offset %d\n", j, description_dist_h[i].description_index[j], description_dist_h[i].description_local_index[j], pElem->elem.count, description_dist_h[i].dst_offset[j]);
    //     }
    // }

    cudaMemcpy(cuda_desc_h->description_dist, description_dist_h, sizeof(ddt_cuda_description_dist_t)*(num_blocks), cudaMemcpyHostToDevice);
    /***/
    
    cudaMemcpy(cuda_desc_d, cuda_desc_h, sizeof(ddt_cuda_desc_t), cudaMemcpyHostToDevice);
      
    printf("launch pack kernel, count %d, num_blocks %d, total threads %d\n", (uint32_t)pStack->count, num_blocks, num_blocks*thread_per_block);
    opal_generic_simple_pack_cuda_kernel_v2<<<num_blocks, thread_per_block>>>(cuda_desc_d);
#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)
    size_t position = pConvertor->pDesc->size;
//    opal_convertor_set_position_nocheck(pConvertor, &position);
#endif
    cudaDeviceSynchronize();
    
   return 1;
    
    
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

int32_t opal_generic_simple_pack_function_cuda_vector2(opal_convertor_t* pConvertor,
                                                      struct iovec* iov,
                                                      uint32_t* out_size,
                                                      size_t* max_data )
{
    dt_stack_t* pStack;       /* pointer to the position on the stack */
    uint32_t pos_desc;        /* actual position in the description of the derived datatype */
    uint32_t count_desc;      /* the number of items already done in the actual pos_desc */
    size_t total_packed = 0;  /* total amount packed this time */
    dt_elem_desc_t* description;
    dt_elem_desc_t* pElem;
    const opal_datatype_t *pData = pConvertor->pDesc;
    unsigned char *conv_ptr, *iov_ptr;
    size_t iov_len_local;
    uint32_t iov_count;
    uint8_t transfer_required;
    uint8_t free_required;
    uint32_t count_desc_tmp;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif
    
    DT_CUDA_DEBUG( opal_cuda_output( 1, "opal_convertor_generic_simple_pack_cuda_vector( %p:%p, {%p, %lu}, %u, %u )\n",
                                (void*)pConvertor, (void*)pConvertor->pBaseBuf,
                                iov[0].iov_base, (unsigned long)iov[0].iov_len, *out_size, *max_data ); );

    description = pConvertor->use_desc->desc;
    
    /* For the first step we have to add both displacement to the source. After in the
     * main while loop we will set back the conv_ptr to the correct value. This is
     * due to the fact that the convertor can stop in the middle of a data with a count
     */
    pStack = pConvertor->pStack + pConvertor->stack_pos;
    pos_desc   = pStack->index;
    conv_ptr   = pConvertor->pBaseBuf + pStack->disp;
    count_desc = (uint32_t)pStack->count;
    pStack--;
    pConvertor->stack_pos--;
    pElem = &(description[pos_desc]);

    DT_CUDA_DEBUG( opal_cuda_output( 1, "pack start pos_desc %d count_desc %d disp %ld\n"
                           "stack_pos %d pos_desc %d count_desc %d disp %ld\n",
                           pos_desc, count_desc, (long)(conv_ptr - pConvertor->pBaseBuf),
                           pConvertor->stack_pos, pStack->index, (int)pStack->count, (long)pStack->disp ); );
    
    
    for( iov_count = 0; iov_count < (*out_size); iov_count++ ) {
        if ((iov[iov_count].iov_base == NULL) || opal_cuda_is_gpu_buffer(iov[iov_count].iov_base)) {
            if (iov[iov_count].iov_len == 0) {
                iov_len_local = DT_CUDA_BUFFER_SIZE;
            } else {
                iov_len_local = iov[iov_count].iov_len;
            }
        
            if (iov[iov_count].iov_base == NULL) {
                iov[iov_count].iov_base = (unsigned char *)opal_cuda_malloc_gpu_buffer(iov_len_local, 0);
                iov_ptr = (unsigned char *)iov[iov_count].iov_base;
                pConvertor->gpu_buffer_ptr = iov_ptr;
                free_required = 1;
            } else {
                iov_ptr = (unsigned char *)iov[iov_count].iov_base;
                free_required = 0;
            }
            transfer_required = 0;
        } else {
            if (OPAL_DATATYPE_VECTOR_USE_MEMCPY2D || OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
                pConvertor->gpu_buffer_ptr = NULL;
                transfer_required = 0;
                free_required = 0;
                iov_ptr = (unsigned char*)iov[iov_count].iov_base;
                iov_len_local = iov[iov_count].iov_len;
            } else if (OPAL_DATATYPE_VECTOR_USE_PIPELINE){
                iov_len_local = iov[iov_count].iov_len;
                if (pConvertor->gpu_buffer_ptr == NULL) {
                    pConvertor->gpu_buffer_ptr = (unsigned char*)opal_cuda_malloc_gpu_buffer(iov_len_local, 0);
                }
                transfer_required = 0;
                free_required = 1;
                iov_ptr = (unsigned char*)iov[iov_count].iov_base;
            } else {
                iov_len_local = iov[iov_count].iov_len;
                if (pConvertor->gpu_buffer_ptr == NULL) {
                    pConvertor->gpu_buffer_ptr = (unsigned char*)opal_cuda_malloc_gpu_buffer(iov_len_local, 0);
                }
                transfer_required = 1;
                free_required = 1;
                iov_ptr = pConvertor->gpu_buffer_ptr;
            }
        }
        while( 1 ) {
            while( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {
                /* now here we have a basic datatype */
                /* should not go into here */
                pack_predefined_data_cuda( pElem, &count_desc, &conv_ptr, &iov_ptr, &iov_len_local );
                if( 0 == count_desc ) {  /* completed */
                    conv_ptr = pConvertor->pBaseBuf + pStack->disp;
                    pos_desc++;  /* advance to the next data */
                    UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                    continue;
                }
                goto complete_loop;
            }
            if( OPAL_DATATYPE_END_LOOP == pElem->elem.common.type ) { /* end of the current loop */
                DT_CUDA_DEBUG( opal_cuda_output( 2, "pack end_loop count %d stack_pos %d"
                                                 " pos_desc %d disp %ld space %lu\n",
                                                 (int)pStack->count, pConvertor->stack_pos,
                                                 pos_desc, (long)pStack->disp, (unsigned long)iov_len_local ); );
                if( --(pStack->count) == 0 ) { /* end of loop */
                    if( 0 == pConvertor->stack_pos ) {
                        /* we lie about the size of the next element in order to
                         * make sure we exit the main loop.
                         */
                        *out_size = iov_count;
                        goto complete_loop;  /* completed */
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
                DT_CUDA_DEBUG( opal_cuda_output( 2, "pack new_loop count %d stack_pos %d pos_desc %d count_desc %d disp %ld space %lu\n",
                                                 (int)pStack->count, pConvertor->stack_pos, pos_desc,
                                                 count_desc, (long)pStack->disp, (unsigned long)iov_len_local ); );
            }
            if( OPAL_DATATYPE_LOOP == pElem->elem.common.type ) {
                OPAL_PTRDIFF_TYPE local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr;
                if( pElem->loop.common.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                    if (OPAL_DATATYPE_VECTOR_USE_MEMCPY2D) {
                        pack_contiguous_loop_cuda_memcpy2d(pElem, &count_desc, &conv_ptr, &iov_ptr, &iov_len_local);
                    } else if (OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
                        pack_contiguous_loop_cuda_zerocopy(pElem, &count_desc, &conv_ptr, &iov_ptr, &iov_len_local);
                    } else if (OPAL_DATATYPE_VECTOR_USE_PIPELINE) {
                        pack_contiguous_loop_cuda_pipeline(pElem, &count_desc, &conv_ptr, &iov_ptr, &iov_len_local, pConvertor->gpu_buffer_ptr);
                    } else {
                        pack_contiguous_loop_cuda(pElem, &count_desc, &conv_ptr, &iov_ptr, &iov_len_local);
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
        total_packed += iov[iov_count].iov_len;
 //       printf("iov_len %d, local %d\n", iov[iov_count].iov_len, iov_len_local);
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        if (transfer_required) {
            cudaMemcpy(iov[iov_count].iov_base, pConvertor->gpu_buffer_ptr, total_packed, cudaMemcpyDeviceToHost);
        } 
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        printf( "[Timing]: DtoH memcpy in %ld microsec, transfer required %d\n", total_time, transfer_required );
#endif
    }
    cudaDeviceSynchronize();
    *max_data = total_packed;
    pConvertor->bConverted += total_packed;  /* update the already converted bytes */
    *out_size = iov_count;
    if( pConvertor->bConverted == pConvertor->local_size ) {
        pConvertor->flags |= CONVERTOR_COMPLETED;
        DT_CUDA_DEBUG( opal_cuda_output( 0, "Total packed %lu\n", pConvertor->bConverted); );
        if (pConvertor->gpu_buffer_ptr != NULL && free_required == 1) {
            printf("free\n");
           opal_cuda_free_gpu_buffer(pConvertor->gpu_buffer_ptr, 0);
           pConvertor->gpu_buffer_ptr = NULL;
        }
        return 1;
    }
    /* Save the global position for the next round */
    PUSH_STACK( pStack, pConvertor->stack_pos, pos_desc, pElem->elem.common.type, count_desc,
                conv_ptr - pConvertor->pBaseBuf );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "pack save stack stack_pos %d pos_desc %d count_desc %d disp %ld\n",
                                     pConvertor->stack_pos, pStack->index, (int)pStack->count, (long)pStack->disp ); );
    return 0;
}

int32_t opal_generic_simple_pack_function_cuda_vector(opal_convertor_t* pConvertor,
                                                      struct iovec* iov,
                                                      uint32_t* out_size,
                                                      size_t* max_data )
{
    dt_stack_t* pStack;       /* pointer to the position on the stack */
    uint32_t pos_desc;        /* actual position in the description of the derived datatype */
    uint32_t count_desc;      /* the number of items already done in the actual pos_desc */
    size_t total_packed = 0;  /* total amount packed this time */
    dt_elem_desc_t* description;
    dt_elem_desc_t* pElem;
    const opal_datatype_t *pData = pConvertor->pDesc;
    unsigned char *conv_ptr, *iov_ptr;
    size_t iov_len_local;
    uint32_t iov_count;
    uint8_t transfer_required;
    uint8_t free_required;
    uint32_t count_desc_tmp;

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 1, "opal_convertor_generic_simple_pack_cuda_vector( %p:%p, {%p, %lu}, %u, %u )\n",
                                (void*)pConvertor, (void*)pConvertor->pBaseBuf,
                                iov[0].iov_base, (unsigned long)iov[0].iov_len, *out_size, *max_data ); );

    description = pConvertor->use_desc->desc;

    /* For the first step we have to add both displacement to the source. After in the
     * main while loop we will set back the conv_ptr to the correct value. This is
     * due to the fact that the convertor can stop in the middle of a data with a count
     */
    pStack = pConvertor->pStack + pConvertor->stack_pos;
    pos_desc   = pStack->index;
    conv_ptr   = pConvertor->pBaseBuf + pStack->disp;
    count_desc = (uint32_t)pStack->count;
    pStack--;
    pConvertor->stack_pos--;
    pElem = &(description[pos_desc]);

    DT_CUDA_DEBUG( opal_cuda_output( 1, "pack start pos_desc %d count_desc %d disp %ld\n"
                           "stack_pos %d pos_desc %d count_desc %d disp %ld\n",
                           pos_desc, count_desc, (long)(conv_ptr - pConvertor->pBaseBuf),
                           pConvertor->stack_pos, pStack->index, (int)pStack->count, (long)pStack->disp ); );


    for( iov_count = 0; iov_count < (*out_size); iov_count++ ) {
        if ((iov[iov_count].iov_base == NULL) || opal_cuda_is_gpu_buffer(iov[iov_count].iov_base)) {
            if (iov[iov_count].iov_len == 0) {
                iov_len_local = DT_CUDA_BUFFER_SIZE;
            } else {
                iov_len_local = iov[iov_count].iov_len;
            }

            if (iov[iov_count].iov_base == NULL) {
                iov[iov_count].iov_base = (unsigned char *)opal_cuda_malloc_gpu_buffer(iov_len_local, 0);
                iov_ptr = (unsigned char *)iov[iov_count].iov_base;
                pConvertor->gpu_buffer_ptr = iov_ptr;
                free_required = 1;
            } else {
                iov_ptr = (unsigned char *)iov[iov_count].iov_base;
                free_required = 0;
            }
            transfer_required = 0;
        } else {
            if (OPAL_DATATYPE_VECTOR_USE_MEMCPY2D || OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
                pConvertor->gpu_buffer_ptr = NULL;
                transfer_required = 0;
                free_required = 0;
                iov_ptr = (unsigned char*)iov[iov_count].iov_base;
                iov_len_local = iov[iov_count].iov_len;
            } else if (OPAL_DATATYPE_VECTOR_USE_PIPELINE){
                iov_len_local = iov[iov_count].iov_len;
                if (pConvertor->gpu_buffer_ptr == NULL) {
                    pConvertor->gpu_buffer_ptr = (unsigned char*)opal_cuda_malloc_gpu_buffer(iov_len_local, 0);
                }
                transfer_required = 0;
                free_required = 1;
                iov_ptr = (unsigned char*)iov[iov_count].iov_base;
            } else {
                iov_len_local = iov[iov_count].iov_len;
                if (pConvertor->gpu_buffer_ptr == NULL) {
                    pConvertor->gpu_buffer_ptr = (unsigned char*)opal_cuda_malloc_gpu_buffer(iov_len_local, 0);
                }
                transfer_required = 1;
                free_required = 1;
                iov_ptr = pConvertor->gpu_buffer_ptr;
            }
        }
        while( 1 ) {
            while( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {
                /* now here we have a basic datatype */
                /* should not go into here */
                pStack--;
                pConvertor->stack_pos--;
                pos_desc --;
                pElem = &(description[pos_desc]);
                count_desc = count_desc_tmp;
                goto complete_loop;
            }
            if( OPAL_DATATYPE_END_LOOP == pElem->elem.common.type ) { /* end of the current loop */
                DT_CUDA_DEBUG( opal_cuda_output( 2, "pack end_loop count %d stack_pos %d"
                                                 " pos_desc %d disp %ld space %lu\n",
                                                 (int)pStack->count, pConvertor->stack_pos,
                                                 pos_desc, (long)pStack->disp, (unsigned long)iov_len_local ); );
                if( --(pStack->count) == 0 ) { /* end of loop */
                    if( 0 == pConvertor->stack_pos ) {
                        /* we lie about the size of the next element in order to
                         * make sure we exit the main loop.
                         */
                        *out_size = iov_count;
                        goto complete_loop;  /* completed */
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
                DT_CUDA_DEBUG( opal_cuda_output( 2, "pack new_loop count %d stack_pos %d pos_desc %d count_desc %d disp %ld space %lu\n",
                                                 (int)pStack->count, pConvertor->stack_pos, pos_desc,
                                                 count_desc, (long)pStack->disp, (unsigned long)iov_len_local ); );
            }
            if( OPAL_DATATYPE_LOOP == pElem->elem.common.type ) {
                OPAL_PTRDIFF_TYPE local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr;
                if( pElem->loop.common.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                    if (OPAL_DATATYPE_VECTOR_USE_MEMCPY2D) {
                        pack_contiguous_loop_cuda_memcpy2d(pElem, &count_desc, &conv_ptr, &iov_ptr, &iov_len_local);
                    } else if (OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
                        pack_contiguous_loop_cuda_zerocopy(pElem, &count_desc, &conv_ptr, &iov_ptr, &iov_len_local);
                    } else if (OPAL_DATATYPE_VECTOR_USE_PIPELINE) {
                        pack_contiguous_loop_cuda_pipeline(pElem, &count_desc, &conv_ptr, &iov_ptr, &iov_len_local, pConvertor->gpu_buffer_ptr);
                    } else {
                        pack_contiguous_loop_cuda(pElem, &count_desc, &conv_ptr, &iov_ptr, &iov_len_local);
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
              //  conv_ptr = pConvertor->pBaseBuf + pStack->disp;
                count_desc_tmp = count_desc;
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                continue;
            }
        }
    complete_loop:
        iov[iov_count].iov_len -= iov_len_local;  /* update the amount of valid data */
        total_packed += iov[iov_count].iov_len;
 //       printf("iov_len %d, local %d\n", iov[iov_count].iov_len, iov_len_local);
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        if (transfer_required) {
            cudaMemcpy(iov[iov_count].iov_base, pConvertor->gpu_buffer_ptr, total_packed, cudaMemcpyDeviceToHost);
        }
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        printf( "[Timing]: DtoH memcpy in %ld microsec, transfer required %d\n", total_time, transfer_required );
#endif
    }
    *max_data = total_packed;
    pConvertor->bConverted += total_packed;  /* update the already converted bytes */
    *out_size = iov_count;
    if( pConvertor->bConverted == pConvertor->local_size ) {
        pConvertor->flags |= CONVERTOR_COMPLETED;
        DT_CUDA_DEBUG( opal_cuda_output( 0, "Total packed %lu\n", pConvertor->bConverted); );
        if (pConvertor->gpu_buffer_ptr != NULL && free_required == 1) {
            printf("free\n");
           opal_cuda_free_gpu_buffer(pConvertor->gpu_buffer_ptr, 0);
           pConvertor->gpu_buffer_ptr = NULL;
        }
        return 1;
    }
    /* Save the global position for the next round */
    PUSH_STACK( pStack, pConvertor->stack_pos, pos_desc, pElem->elem.common.type, count_desc,
                conv_ptr - pConvertor->pBaseBuf );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "pack save stack stack_pos %d pos_desc %d count_desc %d disp %ld\n",
                                     pConvertor->stack_pos, pStack->index, (int)pStack->count, (long)pStack->disp ); );
    return 0;
}

void pack_contiguous_loop_cuda( dt_elem_desc_t* ELEM,
                                uint32_t* COUNT,
                                unsigned char** SOURCE,
                                unsigned char** DESTINATION,
                                size_t* SPACE )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _source = (*SOURCE) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _destination = *(DESTINATION);
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 0, "I am in pack_contiguous_loop_cuda\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)
 //   _source = pBaseBuf_GPU;
 //   _destination = (unsigned char*)cuda_desc_h->iov[0].iov_base;
#endif

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    
 //   tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
 //   num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;
 //    cudaMemcpy2D(_destination, _end_loop->size, _source, _loop->extent, _end_loop->size, _copy_loops, cudaMemcpyDeviceToDevice);
//    pack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK>>>(_copy_loops, _end_loop->size, _loop->extent, _source, _destination);
//    int i;
//    for (i = 0; i < 4; i++) {
//     opal_empty_kernel<<<192, 4*THREAD_PER_BLOCK>>>(_copy_loops, _end_loop->size, _loop->extent, _source, _destination);
     pack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK>>>(_copy_loops, _end_loop->size, _loop->extent, _source, _destination);
//     }

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)    
    *(SOURCE) = _source +  _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(DESTINATION) = *(DESTINATION)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    
    cudaDeviceSynchronize();
    
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    printf( "[Timing]: vector packing in %ld microsec\n", total_time );
#endif
}

void pack_contiguous_loop_cuda_pipeline( dt_elem_desc_t* ELEM,
                                         uint32_t* COUNT,
                                         unsigned char** SOURCE,
                                         unsigned char** DESTINATION,
                                         size_t* SPACE, unsigned char* gpu_buffer )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _source = (*SOURCE) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _destination_host = *(DESTINATION);
    unsigned char* _destination_dev = gpu_buffer;
    int i, pipeline_blocks;
    uint32_t _copy_loops_per_pipeline; 
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 0, "I am in pack_contiguous_loop_cuda_pipeline\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)
 //   _source = pBaseBuf_GPU;
 //   _destination = (unsigned char*)cuda_desc_h->iov[0].iov_base;
#endif

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    
 //   tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
 //   num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;
//    cudaMemcpy2D(_destination, _end_loop->size, _source, _loop->extent, _end_loop->size, _copy_loops, cudaMemcpyDeviceToDevice);
    pipeline_blocks = 4;
    cuda_streams->current_stream_id = 0;
    _copy_loops_per_pipeline = (_copy_loops + pipeline_blocks -1 )/ pipeline_blocks;
    pack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK, 0, cuda_streams->opal_cuda_stream[cuda_streams->current_stream_id]>>>(_copy_loops_per_pipeline, _end_loop->size, _loop->extent, _source, _destination_dev);
    for (i = 1; i <= pipeline_blocks; i++) {
        cudaMemcpyAsync(_destination_host, _destination_dev, _end_loop->size * _copy_loops_per_pipeline, cudaMemcpyDeviceToHost, cuda_streams->opal_cuda_stream[cuda_streams->current_stream_id]);
        cuda_streams->current_stream_id ++;
        cuda_streams->current_stream_id = cuda_streams->current_stream_id % NB_STREAMS;
        _source += _loop->extent * _copy_loops_per_pipeline;
        _destination_dev += _end_loop->size * _copy_loops_per_pipeline;
        _destination_host += _end_loop->size * _copy_loops_per_pipeline;
        if (i == pipeline_blocks) {
            _copy_loops_per_pipeline = _copy_loops - _copy_loops_per_pipeline * (pipeline_blocks - 1);
        }
        pack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK, 0, cuda_streams->opal_cuda_stream[cuda_streams->current_stream_id]>>>(_copy_loops_per_pipeline, _end_loop->size, _loop->extent, _source, _destination_dev);
    }
    cudaMemcpyAsync(_destination_host, _destination_dev, _end_loop->size * _copy_loops_per_pipeline, cudaMemcpyDeviceToHost, cuda_streams->opal_cuda_stream[cuda_streams->current_stream_id]);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)    
    *(SOURCE) = _source +  _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(DESTINATION) = *(DESTINATION)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    
    cudaDeviceSynchronize();
    
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    printf( "[Timing]: vector packing in %ld microsec\n", total_time );
#endif
}

void pack_contiguous_loop_cuda_memcpy2d( dt_elem_desc_t* ELEM,
                                uint32_t* COUNT,
                                unsigned char** SOURCE,
                                unsigned char** DESTINATION,
                                size_t* SPACE )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _source = (*SOURCE) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _destination = *(DESTINATION);
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 0, "I am in pack_contiguous_loop_cuda_memcpy2d\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    

    cudaMemcpy2D(_destination, _end_loop->size, _source, _loop->extent, _end_loop->size, _copy_loops, cudaMemcpyDeviceToHost);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)    
    *(SOURCE) = _source +  _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(DESTINATION) = *(DESTINATION)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    
//    cudaDeviceSynchronize();
    
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    printf( "[Timing]: vector packing with memcpy2d in %ld microsec\n", total_time );
#endif
}

void pack_contiguous_loop_cuda_zerocopy( dt_elem_desc_t* ELEM,
                                uint32_t* COUNT,
                                unsigned char** SOURCE,
                                unsigned char** DESTINATION,
                                size_t* SPACE )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _source = (*SOURCE) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _destination = *(DESTINATION);
    unsigned char* _destination_dev;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 0, "I am in pack_contiguous_loop_cuda_zerocopy\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);


#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    
 //   tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
 //   num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;
 //   cudaHostRegister(_destination, _copy_loops*_end_loop->size, cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void **)&_destination_dev, (void *) _destination, 0);
    pack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK>>>(_copy_loops, _end_loop->size, _loop->extent, _source, _destination_dev);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)    
    *(SOURCE) = _source +  _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(DESTINATION) = *(DESTINATION)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    
    cudaDeviceSynchronize();
 //   cudaHostUnregister(_destination);
    
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    printf( "[Timing]: vector packing in %ld microsec\n", total_time );
#endif
}

int32_t opal_generic_simple_pack_function_cuda_iov( opal_convertor_t* pConvertor,
                                                    struct iovec* iov,
                                                    uint32_t* out_size,
                                                    size_t* max_data )
{
    uint32_t i, j;
    uint32_t count_desc, current_block, task_iteration, nb_blocks_per_description, residue_desc;
    uint32_t nb_blocks, thread_per_block, nb_blocks_used;
    size_t length, buffer_size, length_per_iovec, dst_offset;
    unsigned char *destination, *destination_tmp;
    size_t total_packed, total_converted;
    int32_t complete_flag = 0;
    uint8_t buffer_isfull = 0, transfer_required, free_required;
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
    long total_time, move_time;
#endif
    
    DT_CUDA_DEBUG ( opal_cuda_output(0, "GPU datatype PACKING using iovec\n"); );
    
    description = pConvertor->use_desc->desc;
    pStack = pConvertor->pStack + pConvertor->stack_pos;
    pElem = &(description[pStack->index]);
//    printf("size elem %lu, size %d\n", pElem->elem.common.type, opal_datatype_basicDatatypes[pElem->elem.common.type]->size);
    
//    assert(opal_datatype_basicDatatypes[pElem->elem.common.type]->size != 0);

 //   printf("buffer size %d, max_data %d\n", iov[0].iov_len, *max_data);
    if ((iov[0].iov_base == NULL) || opal_cuda_is_gpu_buffer(iov[0].iov_base)) {
        if (iov[0].iov_len == 0) {
            buffer_size = DT_CUDA_BUFFER_SIZE;
        } else {
            buffer_size = iov[0].iov_len;
        }
        
        if (iov[0].iov_base == NULL) {
            iov[0].iov_base = (unsigned char *)opal_cuda_malloc_gpu_buffer(buffer_size, 0);
            destination = (unsigned char *)iov[0].iov_base;
            pConvertor->gpu_buffer_ptr = destination;
            free_required = 1;
        } else {
            destination = (unsigned char *)iov[0].iov_base;
            free_required = 0;
        }
        transfer_required = 0;
    } else {
        if (OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
            pConvertor->gpu_buffer_ptr = NULL;
            transfer_required = 0;
            free_required = 0;
            cudaHostGetDevicePointer((void **)&destination, (void *)iov[0].iov_base, 0);
        } else {
            buffer_size = iov[0].iov_len;
            if (pConvertor->gpu_buffer_ptr == NULL) {
                pConvertor->gpu_buffer_ptr = (unsigned char*)opal_cuda_malloc_gpu_buffer(buffer_size, 0);
            }
            transfer_required = 1;
            free_required = 1;
#if defined(OPAL_DATATYPE_CUDA_DRY_RUN)
            destination = (unsigned char*)iov[0].iov_base;
#else
            destination = pConvertor->gpu_buffer_ptr;
#endif /* OPAL_DATATYPE_CUDA_DRY_RUN */
        }
    }
    
    destination_tmp = destination;
    
    DT_CUDA_DEBUG ( opal_cuda_output(0, "Pack GPU base %p, pack to buffer %p\n", pConvertor->pBaseBuf, destination););

    cuda_iov_count = 1000;//CUDA_NB_IOV;
    total_packed = 0;
    total_converted = pConvertor->bConverted;
    cuda_streams->current_stream_id = 0;
    convertor_flags = pConvertor->flags;
    orig_stack_index = pStack->index;

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start_total);
#endif
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    complete_flag = opal_convertor_raw( pConvertor, cuda_iov, &cuda_iov_count, &length );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "PACKING complete flag %d, iov count %d, length %d, submit to CUDA stream %d\n", complete_flag, cuda_iov_count, length, cuda_streams->current_stream_id); );

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    printf( "[Timing]: ddt to iov in %ld microsec\n", total_time );
#endif
    
    dst_offset = 0;
    thread_per_block = CUDA_WARP_SIZE * 5;
    nb_blocks = 256;
    nb_blocks_used = 0;
    
    while (cuda_iov_count > 0) {
        
        current_block = 0;
        task_iteration = 0;
        cuda_iov_dist_h_current = cuda_iov_dist_h[cuda_streams->current_stream_id];
        cuda_iov_dist_d_current = cuda_iov_dist_d[cuda_streams->current_stream_id]; 

#if defined(OPAL_DATATYPE_CUDA_TIMING)
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
            total_packed += length_per_iovec;
            
            /* check alignment */
            if ((uintptr_t)(cuda_iov[i].iov_base) % ALIGNMENT_DOUBLE == 0 && (uintptr_t)destination % ALIGNMENT_DOUBLE == 0 && length_per_iovec >= ALIGNMENT_DOUBLE) {
                alignment = ALIGNMENT_DOUBLE;
            } else if ((uintptr_t)(cuda_iov[i].iov_base) % ALIGNMENT_FLOAT == 0 && (uintptr_t)destination % ALIGNMENT_FLOAT == 0 && length_per_iovec >= ALIGNMENT_FLOAT) {
                alignment = ALIGNMENT_FLOAT;
            } else {
                alignment = ALIGNMENT_CHAR;
            }
            
           // alignment = ALIGNMENT_DOUBLE;

            count_desc = length_per_iovec / alignment;
            residue_desc = length_per_iovec % alignment;
            nb_blocks_per_description = (count_desc + thread_per_block - 1) / thread_per_block;
            DT_CUDA_DEBUG ( opal_cuda_output(10, "PACKING description %d, size %d, residue %d, alignment %d, nb_block_aquired %d\n", i, count_desc, residue_desc, alignment, nb_blocks_per_description); );
            for (j = 0; j < nb_blocks_per_description; j++) {
                cuda_iov_dist_h_current[current_block].src[task_iteration] = (unsigned char *)(cuda_iov[i].iov_base) + j * thread_per_block * alignment;
                cuda_iov_dist_h_current[current_block].dst[task_iteration] = destination;
                cuda_iov_dist_h_current[current_block].element_alignment[task_iteration] = alignment;
                cuda_iov_dist_h_current[current_block].nb_tasks = task_iteration + 1;
                if ( (j+1) * thread_per_block <= count_desc) {
                    cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] = thread_per_block;// * sizeof(double);
                } else {
                    cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] = count_desc - j*thread_per_block; //(thread_per_block - ((j+1)*thread_per_block - count_desc));// * sizeof(double);
                }
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
                assert(cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] > 0);
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
                destination += cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] * alignment;
                DT_CUDA_DEBUG( opal_cuda_output(12, "PACKING \tblock %d, src %p, dst %p, nb_elements %d, alignment %d\n", current_block, cuda_iov_dist_h_current[current_block].src[task_iteration], cuda_iov_dist_h_current[current_block].dst[task_iteration], cuda_iov_dist_h_current[current_block].nb_elements[task_iteration], cuda_iov_dist_h_current[current_block].element_alignment[task_iteration]); );
                current_block += 1;
                nb_blocks_used ++;
                if (current_block >= nb_blocks) {
                    current_block = 0;
                    task_iteration ++;
                    assert(task_iteration < CUDA_IOV_MAX_TASK_PER_BLOCK);
                }
            }
            
            /* handle residue */
            if (residue_desc != 0) {
                orig_alignment = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;
                cuda_iov_dist_h_current[current_block].src[task_iteration] = (unsigned char *)(cuda_iov[i].iov_base) + length_per_iovec / alignment * alignment;
                cuda_iov_dist_h_current[current_block].dst[task_iteration] = destination;
                cuda_iov_dist_h_current[current_block].element_alignment[task_iteration] = orig_alignment;
                cuda_iov_dist_h_current[current_block].nb_tasks = task_iteration + 1;
                cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] = (length_per_iovec - length_per_iovec / alignment * alignment) / orig_alignment;
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
                assert(cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] > 0);
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
                destination += cuda_iov_dist_h_current[current_block].nb_elements[task_iteration] * orig_alignment;
                DT_CUDA_DEBUG( opal_cuda_output(12, "PACKING \tblock %d, src %p, dst %p, nb_elements %d, alignment %d\n", current_block, cuda_iov_dist_h_current[current_block].src[task_iteration], cuda_iov_dist_h_current[current_block].dst[task_iteration], cuda_iov_dist_h_current[current_block].nb_elements[task_iteration], cuda_iov_dist_h_current[current_block].element_alignment[task_iteration]); );
                current_block += 1;
                nb_blocks_used ++;
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
        printf( "[Timing]: Pack to dest %p, iov is prepared in %ld microsec, kernel submitted to CUDA stream %d, nb_blocks %d\n", destination_tmp, total_time,  cuda_streams->current_stream_id, nb_blocks_used);
#endif

        cudaMemcpyAsync(cuda_iov_dist_d_current, cuda_iov_dist_h_current, sizeof(ddt_cuda_iov_dist_t)*(nb_blocks), cudaMemcpyHostToDevice, cuda_streams->opal_cuda_stream[cuda_streams->current_stream_id]);
        opal_generic_simple_pack_cuda_iov_kernel<<<nb_blocks, thread_per_block, 0, cuda_streams->opal_cuda_stream[cuda_streams->current_stream_id]>>>(cuda_iov_dist_d_current);
        cuda_streams->current_stream_id ++;
        cuda_streams->current_stream_id = cuda_streams->current_stream_id % NB_STREAMS;
        
        /* buffer is full */
        if (buffer_isfull) {
            pConvertor->flags = convertor_flags;
            total_converted += total_packed;
            opal_convertor_set_position_nocheck(pConvertor, &total_converted);
            break;
        }
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        convertor_flags = pConvertor->flags;
        orig_stack_index = pStack->index;
        complete_flag = opal_convertor_raw( pConvertor, cuda_iov, &cuda_iov_count, &length );
        DT_CUDA_DEBUG ( opal_cuda_output(2, "PACKING complete flag %d, iov count %d, length %d, submit to CUDA stream %d\n", complete_flag, cuda_iov_count, length, cuda_streams->current_stream_id); );
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        printf( "[Timing]: ddt to iov in %ld microsec\n", total_time );
#endif
    }
    

    cudaDeviceSynchronize();
 /*   for (i = 0; i < NB_STREAMS; i++) {
        cudaStreamSynchronize(cuda_streams->opal_cuda_stream[i]);
    }*/
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    if (transfer_required) {
        cudaMemcpy(iov[0].iov_base, pConvertor->gpu_buffer_ptr, total_packed, cudaMemcpyDeviceToHost);
    } 
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    move_time = ELAPSED_TIME( start, end );
    printf( "[Timing]: DtoH memcpy in %ld microsec, transfer required %d\n", move_time, transfer_required );
#endif
    // float *vtmp = (float *)iov[0].iov_base;
    // DT_CUDA_DEBUG ( opal_cuda_output(0, "packed iov buffer, total packed %d\n", total_packed); );
    // for (uint32_t i = 0; i < total_packed/sizeof(float); i++) {
    //     printf(" %1.f ", *vtmp);
    //     vtmp ++;
    // }
    // printf("\n");
    iov[0].iov_len = total_packed;
    *max_data = total_packed;
    *out_size = 1;
    DT_CUDA_DEBUG ( opal_cuda_output(0, "PACKING total packed %d\n", total_packed); );
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    GET_TIME( end_total );
    total_time = ELAPSED_TIME( start_total, end_total );
    printf( "[Timing]: total packing in %ld microsec, kernel %ld microsec\n", total_time, total_time-move_time );
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


void pack_predefined_data_cuda( dt_elem_desc_t* ELEM,
                                uint32_t* COUNT,
                                unsigned char** SOURCE,
                                unsigned char** DESTINATION,
                                size_t* SPACE )
{
    uint32_t _copy_count = *(COUNT);
    size_t _copy_blength;
    ddt_elem_desc_t* _elem = &((ELEM)->elem);
    unsigned char* _source = (*SOURCE) + _elem->disp;
    uint32_t nb_blocks, tasks_per_block, thread_per_block;
    unsigned char* _destination = *(DESTINATION);

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
    
    pack_contiguous_loop_cuda_kernel_global<<<nb_blocks, thread_per_block, 0, cuda_streams->opal_cuda_stream[cuda_streams->current_stream_id]>>>(_copy_count, _copy_blength, _elem->extent, _source, _destination);
    cuda_streams->current_stream_id ++;
    cuda_streams->current_stream_id = cuda_streams->current_stream_id % NB_STREAMS;
    
#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)  
    _copy_blength *= _copy_count;
    *(SOURCE)  = _source + _elem->extent*_copy_count - _elem->disp;
    *(DESTINATION) += _copy_blength;
    *(SPACE)  -= _copy_blength;
    *(COUNT)  -= _copy_count;
#endif
    
}

