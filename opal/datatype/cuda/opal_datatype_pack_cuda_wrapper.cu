#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"

#include <stdio.h>
#include <assert.h>


int32_t opal_ddt_generic_simple_pack_function_cuda_vector(opal_convertor_t* pConvertor,
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
    
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    int contiguous_loop_flag = 0;
    int i;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif
    
    DT_CUDA_DEBUG( opal_cuda_output( 2, "opal_convertor_generic_simple_pack_cuda_vector( %p:%p, {%p, %lu}, %u, %u )\n",
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

    DT_CUDA_DEBUG( opal_cuda_output( 4, "pack start pos_desc %d count_desc %d disp %ld\n"
                           "stack_pos %d pos_desc %d count_desc %d disp %ld\n",
                           pos_desc, count_desc, (long)(conv_ptr - pConvertor->pBaseBuf),
                           pConvertor->stack_pos, pStack->index, (int)pStack->count, (long)pStack->disp ); );
    
    
    for( iov_count = 0; iov_count < (*out_size); iov_count++ ) {
        if ((iov[iov_count].iov_base == NULL) || opal_ddt_cuda_is_gpu_buffer(iov[iov_count].iov_base)) {
            if (iov[iov_count].iov_len == 0) {
                iov_len_local = DT_CUDA_BUFFER_SIZE;
            } else {
                iov_len_local = iov[iov_count].iov_len;
            }
        
            if (iov[iov_count].iov_base == NULL) {
                iov[iov_count].iov_base = (unsigned char *)opal_ddt_cuda_malloc_gpu_buffer(iov_len_local, 0);
                iov_ptr = (unsigned char *)iov[iov_count].iov_base;
                pConvertor->gpu_buffer_ptr = iov_ptr;
                free_required = 1;
            } else {
                iov_ptr = (unsigned char *)iov[iov_count].iov_base;
                free_required = 0;
            }
            transfer_required = 0;
        } else {
            if (OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_D2H || OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
                pConvertor->gpu_buffer_ptr = NULL;
                transfer_required = 0;
                free_required = 0;
                iov_ptr = (unsigned char*)iov[iov_count].iov_base;
                iov_len_local = iov[iov_count].iov_len;
            } else if (OPAL_DATATYPE_VECTOR_USE_PIPELINE){
                iov_len_local = iov[iov_count].iov_len;
                if (pConvertor->gpu_buffer_ptr == NULL) {
                    pConvertor->gpu_buffer_ptr = (unsigned char*)opal_ddt_cuda_malloc_gpu_buffer(iov_len_local, 0);
                }
                transfer_required = 0;
                free_required = 1;
                iov_ptr = (unsigned char*)iov[iov_count].iov_base;
            } else {
                iov_len_local = iov[iov_count].iov_len;
                if (pConvertor->gpu_buffer_ptr == NULL) {
                    pConvertor->gpu_buffer_ptr = (unsigned char*)opal_ddt_cuda_malloc_gpu_buffer(iov_len_local, 0);
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
                if (contiguous_loop_flag) {
                    pStack--;
                    pConvertor->stack_pos--;
                    pos_desc --;
                    pElem = &(description[pos_desc]);
                    count_desc = count_desc_tmp;
                }
                goto complete_loop;
            }
            if( OPAL_DATATYPE_END_LOOP == pElem->elem.common.type ) { /* end of the current loop */
                DT_CUDA_DEBUG( opal_cuda_output( 4, "pack end_loop count %d stack_pos %d"
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
                DT_CUDA_DEBUG( opal_cuda_output( 4, "pack new_loop count %d stack_pos %d pos_desc %d count_desc %d disp %ld space %lu\n",
                                                 (int)pStack->count, pConvertor->stack_pos, pos_desc,
                                                 count_desc, (long)pStack->disp, (unsigned long)iov_len_local ); );
            }
            if( OPAL_DATATYPE_LOOP == pElem->elem.common.type ) {
                OPAL_PTRDIFF_TYPE local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr;
                if( pElem->loop.common.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                    if (OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_D2H) {
                        pack_contiguous_loop_cuda_memcpy2d_d2h(pElem, &count_desc, &conv_ptr, &iov_ptr, &iov_len_local);
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
                    } else {
                        contiguous_loop_flag = 1;
                    }
                    /* Save the stack with the correct last_count value. */
                }
                local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr - local_disp;
                PUSH_STACK( pStack, pConvertor->stack_pos, pos_desc, OPAL_DATATYPE_LOOP, count_desc,
                            pStack->disp + local_disp);
                pos_desc++;
            update_loop_description:  /* update the current state */
                if (contiguous_loop_flag) {
                    count_desc_tmp = count_desc;
                } else {
                    conv_ptr = pConvertor->pBaseBuf + pStack->disp;
                }
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                continue;
            }
        }
    complete_loop:
        iov[iov_count].iov_len -= iov_len_local;  /* update the amount of valid data */
        total_packed += iov[iov_count].iov_len;
 //       printf("iov_len %d, local %d\n", iov[iov_count].iov_len, iov_len_local);
        for (i = 0; i < NB_STREAMS; i++) {
            cudaStreamSynchronize(cuda_streams->opal_cuda_stream[i]);
        }
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        if (transfer_required) {
            cudaMemcpy(iov[iov_count].iov_base, pConvertor->gpu_buffer_ptr, total_packed, cudaMemcpyDeviceToHost);
        } 
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: DtoH memcpy in %ld microsec, transfer required %d\n", total_time, transfer_required ); );
#endif
    }
    *max_data = total_packed;
    pConvertor->bConverted += total_packed;  /* update the already converted bytes */
    *out_size = iov_count;
    if( pConvertor->bConverted == pConvertor->local_size ) {
        pConvertor->flags |= CONVERTOR_COMPLETED;
        DT_CUDA_DEBUG( opal_cuda_output( 0, "Pack total packed %lu\n", pConvertor->bConverted); );
        if (pConvertor->gpu_buffer_ptr != NULL && free_required == 1) {
            printf("free\n");
           opal_ddt_cuda_free_gpu_buffer(pConvertor->gpu_buffer_ptr, 0);
           pConvertor->gpu_buffer_ptr = NULL;
        }
        return 1;
    }
    /* Save the global position for the next round */
    PUSH_STACK( pStack, pConvertor->stack_pos, pos_desc, pElem->elem.common.type, count_desc,
                conv_ptr - pConvertor->pBaseBuf );
    DT_CUDA_DEBUG( opal_cuda_output( 4, "pack save stack stack_pos %d pos_desc %d count_desc %d disp %ld\n",
                                     pConvertor->stack_pos, pStack->index, (int)pStack->count, (long)pStack->disp ); );
    return 0;
}

int32_t opal_ddt_generic_simple_pack_function_cuda_vector2(opal_convertor_t* pConvertor,
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
    
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 2, "opal_convertor_generic_simple_pack_cuda_vector( %p:%p, {%p, %lu}, %u, %u )\n",
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

    DT_CUDA_DEBUG( opal_cuda_output( 4, "pack start pos_desc %d count_desc %d disp %ld\n"
                           "stack_pos %d pos_desc %d count_desc %d disp %ld\n",
                           pos_desc, count_desc, (long)(conv_ptr - pConvertor->pBaseBuf),
                           pConvertor->stack_pos, pStack->index, (int)pStack->count, (long)pStack->disp ); );


    for( iov_count = 0; iov_count < (*out_size); iov_count++ ) {
        if ((iov[iov_count].iov_base == NULL) || opal_ddt_cuda_is_gpu_buffer(iov[iov_count].iov_base)) {
            if (iov[iov_count].iov_len == 0) {
                iov_len_local = DT_CUDA_BUFFER_SIZE;
            } else {
                iov_len_local = iov[iov_count].iov_len;
            }

            if (iov[iov_count].iov_base == NULL) {
                iov[iov_count].iov_base = (unsigned char *)opal_ddt_cuda_malloc_gpu_buffer(iov_len_local, 0);
                iov_ptr = (unsigned char *)iov[iov_count].iov_base;
                pConvertor->gpu_buffer_ptr = iov_ptr;
                free_required = 1;
            } else {
                iov_ptr = (unsigned char *)iov[iov_count].iov_base;
                free_required = 0;
            }
            transfer_required = 0;
        } else {
            if (OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_D2H || OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
                pConvertor->gpu_buffer_ptr = NULL;
                transfer_required = 0;
                free_required = 0;
                iov_ptr = (unsigned char*)iov[iov_count].iov_base;
                iov_len_local = iov[iov_count].iov_len;
            } else if (OPAL_DATATYPE_VECTOR_USE_PIPELINE){
                iov_len_local = iov[iov_count].iov_len;
                if (pConvertor->gpu_buffer_ptr == NULL) {
                    pConvertor->gpu_buffer_ptr = (unsigned char*)opal_ddt_cuda_malloc_gpu_buffer(iov_len_local, 0);
                }
                transfer_required = 0;
                free_required = 1;
                iov_ptr = (unsigned char*)iov[iov_count].iov_base;
            } else {
                iov_len_local = iov[iov_count].iov_len;
                if (pConvertor->gpu_buffer_ptr == NULL) {
                    pConvertor->gpu_buffer_ptr = (unsigned char*)opal_ddt_cuda_malloc_gpu_buffer(iov_len_local, 0);
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
                DT_CUDA_DEBUG( opal_cuda_output( 4, "pack end_loop count %d stack_pos %d"
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
                DT_CUDA_DEBUG( opal_cuda_output( 4, "pack new_loop count %d stack_pos %d pos_desc %d count_desc %d disp %ld space %lu\n",
                                                 (int)pStack->count, pConvertor->stack_pos, pos_desc,
                                                 count_desc, (long)pStack->disp, (unsigned long)iov_len_local ); );
            }
            if( OPAL_DATATYPE_LOOP == pElem->elem.common.type ) {
                OPAL_PTRDIFF_TYPE local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr;
                if( pElem->loop.common.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                    if (OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_D2H) {
                        pack_contiguous_loop_cuda_memcpy2d_d2h(pElem, &count_desc, &conv_ptr, &iov_ptr, &iov_len_local);
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
           //     conv_ptr = pConvertor->pBaseBuf + pStack->disp;
                count_desc_tmp = count_desc;
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                continue;
            }
        }
    complete_loop:
        iov[iov_count].iov_len -= iov_len_local;  /* update the amount of valid data */
        total_packed += iov[iov_count].iov_len;
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        if (transfer_required) {
            cudaMemcpy(iov[iov_count].iov_base, pConvertor->gpu_buffer_ptr, total_packed, cudaMemcpyDeviceToHost);
        }
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: DtoH memcpy in %ld microsec, transfer required %d\n", total_time, transfer_required ); );
#endif
    }
    *max_data = total_packed;
    pConvertor->bConverted += total_packed;  /* update the already converted bytes */
    *out_size = iov_count;
    DT_CUDA_DEBUG( opal_cuda_output( 2, "Pack total packed %lu\n", total_packed); );
    if( pConvertor->bConverted == pConvertor->local_size ) {
        pConvertor->flags |= CONVERTOR_COMPLETED;
        if (pConvertor->gpu_buffer_ptr != NULL && free_required == 1) {
            printf("free\n");
           opal_ddt_cuda_free_gpu_buffer(pConvertor->gpu_buffer_ptr, 0);
           pConvertor->gpu_buffer_ptr = NULL;
        }
        return 1;
    }
    /* Save the global position for the next round */
    PUSH_STACK( pStack, pConvertor->stack_pos, pos_desc, pElem->elem.common.type, count_desc,
                conv_ptr - pConvertor->pBaseBuf );
    DT_CUDA_DEBUG( opal_cuda_output( 4, "pack save stack stack_pos %d pos_desc %d count_desc %d disp %ld\n",
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
    
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 2, "Pack using contiguous_loop_cuda\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);


#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    
 //   tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
 //   num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;
//    printf("extent %ld, size %ld, count %ld\n", _loop->extent, _end_loop->size, _copy_loops);
#if OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_AS_KERNEL
    cudaMemcpy2DAsync(_destination, _end_loop->size, _source, _loop->extent, _end_loop->size, _copy_loops, cudaMemcpyDeviceToDevice, cuda_streams->opal_cuda_stream[0]);
#else
    pack_contiguous_loop_cuda_kernel_global<<<32, 8*THREAD_PER_BLOCK, 0, cuda_streams->opal_cuda_stream[0]>>>(_copy_loops, _end_loop->size, _loop->extent, _source, _destination);
#endif /* OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_AS_KERNEL */

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)    
    *(SOURCE) = _source +  _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(DESTINATION) = *(DESTINATION)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    
    cudaStreamSynchronize(cuda_streams->opal_cuda_stream[0]);
    
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector packing in %ld microsec\n", total_time ); );
#endif
}

/* this function will not be used */
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
    
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 2, "Pack using contiguous_loop_cuda_pipeline\n"); );

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
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector packing in %ld microsec\n", total_time ); );
#endif
} 

void pack_contiguous_loop_cuda_memcpy2d_d2h( dt_elem_desc_t* ELEM,
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
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 2, "Pack using contiguous_loop_cuda_memcpy2d\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    

    cudaMemcpy2DAsync(_destination, _end_loop->size, _source, _loop->extent, _end_loop->size, _copy_loops, cudaMemcpyDeviceToHost, cuda_streams->opal_cuda_stream[0]);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)    
    *(SOURCE) = _source +  _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(DESTINATION) = *(DESTINATION)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    
    cudaStreamSynchronize(cuda_streams->opal_cuda_stream[0]);
    
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector packing with memcpy2d in %ld microsec\n", total_time ); );
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
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 2, "Pack using contiguous_loop_cuda_zerocopy\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);


#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    

    cudaError_t reg_rv = cudaHostGetDevicePointer((void **)&_destination_dev, (void *) _destination, 0);
    if (reg_rv != cudaSuccess) {
        const char *cuda_err = cudaGetErrorString(reg_rv);
        printf("can not get dev  mem, %s\n", cuda_err);
    }
#if OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_AS_KERNEL
    cudaMemcpy2DAsync(_destination_dev, _end_loop->size, _source, _loop->extent, _end_loop->size, _copy_loops, cudaMemcpyDeviceToDevice, cuda_streams->opal_cuda_stream[0]);
#else
    pack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK, 0, cuda_streams->opal_cuda_stream[0]>>>(_copy_loops, _end_loop->size, _loop->extent, _source, _destination_dev);
#endif /* OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_AS_KERNEL */

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)    
    *(SOURCE) = _source +  _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(DESTINATION) = *(DESTINATION)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    
    cudaStreamSynchronize(cuda_streams->opal_cuda_stream[0]);
    
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector packing in %ld microsec\n", total_time ); );
#endif
}

int32_t opal_ddt_generic_simple_pack_function_cuda_iov( opal_convertor_t* pConvertor,
                                                        struct iovec* iov,
                                                        uint32_t* out_size,
                                                        size_t* max_data )
{
    size_t buffer_size;
    unsigned char *destination;
    size_t total_packed;
    uint8_t transfer_required, free_required;

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time, move_time;
#endif

 //   printf("buffer size %d, max_data %d\n", iov[0].iov_len, *max_data);
    if ((iov[0].iov_base == NULL) || opal_ddt_cuda_is_gpu_buffer(iov[0].iov_base)) {
        if (iov[0].iov_len == 0) {
            buffer_size = DT_CUDA_BUFFER_SIZE;
        } else {
            buffer_size = iov[0].iov_len;
        }
        
        if (iov[0].iov_base == NULL) {
            iov[0].iov_base = (unsigned char *)opal_ddt_cuda_malloc_gpu_buffer(buffer_size, 0);
            destination = (unsigned char *)iov[0].iov_base;
            pConvertor->gpu_buffer_ptr = destination;
            free_required = 1;
        } else {
            destination = (unsigned char *)iov[0].iov_base;
            free_required = 0;
        }
        transfer_required = 0;
    } else {
        buffer_size = iov[0].iov_len;
        if (OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
            pConvertor->gpu_buffer_ptr = NULL;
            transfer_required = 0;
            free_required = 0;
            cudaHostGetDevicePointer((void **)&destination, (void *)iov[0].iov_base, 0);
        } else {
            if (pConvertor->gpu_buffer_ptr == NULL) {
                pConvertor->gpu_buffer_ptr = (unsigned char*)opal_ddt_cuda_malloc_gpu_buffer(buffer_size, 0);
            }
            transfer_required = 1;
            free_required = 1;
            destination = pConvertor->gpu_buffer_ptr;
        }
    }   

    total_packed = 0;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start_total);
#endif
    
    /* start pack */
    if (cuda_iov_cache_enabled) {
        opal_ddt_generic_simple_pack_function_cuda_iov_cached(pConvertor, destination, buffer_size, &total_packed);
    } else {
        opal_ddt_generic_simple_pack_function_cuda_iov_non_cached(pConvertor, destination, buffer_size, &total_packed);
    }

    pConvertor->bConverted += total_packed;
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Pack total packed %d\n", total_packed); );
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    if (transfer_required) {
        cudaMemcpy(iov[0].iov_base, pConvertor->gpu_buffer_ptr, total_packed, cudaMemcpyDeviceToHost);
    } 
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    move_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: DtoH memcpy in %ld microsec, transfer required %d\n", move_time, transfer_required ); );
#endif

    iov[0].iov_len = total_packed;
    *max_data = total_packed;
    *out_size = 1;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    GET_TIME( end_total );
    total_time = ELAPSED_TIME( start_total, end_total );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: total packing in %ld microsec, kernel %ld microsec\n", total_time, total_time-move_time ); );
#endif
    
    if( pConvertor->bConverted == pConvertor->local_size ) {
        pConvertor->flags |= CONVERTOR_COMPLETED;
        if (pConvertor->gpu_buffer_ptr != NULL && free_required) {
           opal_ddt_cuda_free_gpu_buffer(pConvertor->gpu_buffer_ptr, 0);
           pConvertor->gpu_buffer_ptr = NULL;
        }
        return 1;
    }        
    return 0; 
}

#if 0

int32_t opal_ddt_generic_simple_pack_function_cuda_iov_non_cached( opal_convertor_t* pConvertor,
                                                                   struct iovec* iov,
                                                                   uint32_t* out_size,
                                                                   size_t* max_data )
{
    uint32_t i, j;
    uint32_t count_desc, nb_blocks_per_description, residue_desc;
    uint32_t nb_blocks, thread_per_block, nb_blocks_used;
    size_t length, buffer_size, length_per_iovec, dst_offset;
    unsigned char *destination, *destination_base;
    size_t total_packed, total_converted;
    int32_t complete_flag = 0;
    uint8_t buffer_isfull = 0, transfer_required, free_required;
    uint32_t convertor_flags;
//    dt_elem_desc_t* description;
//    dt_elem_desc_t* pElem;
//    dt_stack_t* pStack;
    uint8_t alignment, orig_alignment;
//    int32_t orig_stack_index;
    cudaError_t cuda_err;
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    ddt_cuda_iov_dist_non_cached_t* cuda_iov_dist_h_current;
    ddt_cuda_iov_dist_non_cached_t* cuda_iov_dist_d_current;
    ddt_cuda_iov_pipeline_block_t *cuda_iov_pipeline_block;
    int iov_pipeline_block_id = 0;
    cudaStream_t *cuda_stream_iov = NULL;

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time, move_time;
#endif
    
    /*description = pConvertor->use_desc->desc;
    pStack = pConvertor->pStack + pConvertor->stack_pos;
    pElem = &(description[pStack->index]);
    printf("size elem %lu, size %d\n", pElem->elem.common.type, opal_datatype_basicDatatypes[pElem->elem.common.type]->size);
    */
    
//    assert(opal_datatype_basicDatatypes[pElem->elem.common.type]->size != 0);

 //   printf("buffer size %d, max_data %d\n", iov[0].iov_len, *max_data);
    if ((iov[0].iov_base == NULL) || opal_ddt_cuda_is_gpu_buffer(iov[0].iov_base)) {
        if (iov[0].iov_len == 0) {
            buffer_size = DT_CUDA_BUFFER_SIZE;
        } else {
            buffer_size = iov[0].iov_len;
        }
        
        if (iov[0].iov_base == NULL) {
            iov[0].iov_base = (unsigned char *)opal_ddt_cuda_malloc_gpu_buffer(buffer_size, 0);
            destination = (unsigned char *)iov[0].iov_base;
            pConvertor->gpu_buffer_ptr = destination;
            free_required = 1;
        } else {
            destination = (unsigned char *)iov[0].iov_base;
            free_required = 0;
        }
        transfer_required = 0;
    } else {
        buffer_size = iov[0].iov_len;
        if (OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
            pConvertor->gpu_buffer_ptr = NULL;
            transfer_required = 0;
            free_required = 0;
            cudaHostGetDevicePointer((void **)&destination, (void *)iov[0].iov_base, 0);
        } else {
            if (pConvertor->gpu_buffer_ptr == NULL) {
                pConvertor->gpu_buffer_ptr = (unsigned char*)opal_ddt_cuda_malloc_gpu_buffer(buffer_size, 0);
            }
            transfer_required = 1;
            free_required = 1;
            destination = pConvertor->gpu_buffer_ptr;
        }
    }   
    
    destination_base = destination;
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Pack using IOV non cached, GPU base %p, pack to buffer %p\n", pConvertor->pBaseBuf, destination););

    cuda_iov_count = 1000;//CUDA_NB_IOV;
    total_packed = 0;
    total_converted = pConvertor->bConverted;
    cuda_streams->current_stream_id = 0;
    convertor_flags = pConvertor->flags;
  //  orig_stack_index = pStack->index;

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start_total);
#endif
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    complete_flag = opal_convertor_raw( pConvertor, cuda_iov, &cuda_iov_count, &length );
    DT_CUDA_DEBUG ( opal_cuda_output(4, "Pack complete flag %d, iov count %d, length %d, submit to CUDA stream %d\n", complete_flag, cuda_iov_count, length, cuda_streams->current_stream_id); );

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: ddt to iov in %ld microsec\n", total_time ); );
#endif
    
    dst_offset = 0;
    thread_per_block = CUDA_WARP_SIZE * 5;
    nb_blocks = 256;
    
    while (cuda_iov_count > 0) {
        
        nb_blocks_used = 0;
        cuda_iov_pipeline_block = current_cuda_device->cuda_iov_pipeline_block[iov_pipeline_block_id];
        cuda_iov_dist_h_current = cuda_iov_pipeline_block->cuda_iov_dist_non_cached_h;
        cuda_iov_dist_d_current = cuda_iov_pipeline_block->cuda_iov_dist_non_cached_d;
        cuda_stream_iov = cuda_iov_pipeline_block->cuda_stream;
        cuda_err = cudaStreamWaitEvent(*cuda_stream_iov, cuda_iov_pipeline_block->cuda_event, 0);
        opal_cuda_check_error(cuda_err);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif

        for (i = 0; i < cuda_iov_count; i++) {
          /*  pElem = &(description[orig_stack_index+i]);*/
            if (buffer_size >= cuda_iov[i].iov_len) {
                length_per_iovec = cuda_iov[i].iov_len;
            } else {
                /*orig_alignment = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;*/
                orig_alignment = ALIGNMENT_CHAR;
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

            count_desc = length_per_iovec / alignment;
            residue_desc = length_per_iovec % alignment;
            nb_blocks_per_description = (count_desc + thread_per_block - 1) / thread_per_block;
            DT_CUDA_DEBUG ( opal_cuda_output(10, "Pack description %d, size %d, residue %d, alignment %d, nb_block_aquired %d\n", i, count_desc, residue_desc, alignment, nb_blocks_per_description); );
            for (j = 0; j < nb_blocks_per_description; j++) {
                cuda_iov_dist_h_current[nb_blocks_used].src = (unsigned char *)(cuda_iov[i].iov_base) + j * thread_per_block * alignment;
                cuda_iov_dist_h_current[nb_blocks_used].dst = destination;
                cuda_iov_dist_h_current[nb_blocks_used].element_alignment = alignment;
                if ( (j+1) * thread_per_block <= count_desc) {
                    cuda_iov_dist_h_current[nb_blocks_used].nb_elements = thread_per_block;
                } else {
                    cuda_iov_dist_h_current[nb_blocks_used].nb_elements = count_desc - j*thread_per_block; 
                }
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
                assert(cuda_iov_dist_h_current[nb_blocks_used].nb_elements > 0);
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
                destination += cuda_iov_dist_h_current[nb_blocks_used].nb_elements * alignment;
                DT_CUDA_DEBUG( opal_cuda_output(12, "Pack \tblock %d, src %p, dst %p, nb_elements %d, alignment %d\n", nb_blocks_used, cuda_iov_dist_h_current[nb_blocks_used].src, cuda_iov_dist_h_current[nb_blocks_used].dst, cuda_iov_dist_h_current[nb_blocks_used].nb_elements, cuda_iov_dist_h_current[nb_blocks_used].element_alignment); );
                nb_blocks_used ++;
                assert (nb_blocks_used < CUDA_MAX_NB_BLOCKS*CUDA_IOV_MAX_TASK_PER_BLOCK);
            }
            
            /* handle residue */
            if (residue_desc != 0) {
                /*orig_alignment = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;*/
                orig_alignment = ALIGNMENT_CHAR;
                cuda_iov_dist_h_current[nb_blocks_used].src = (unsigned char *)(cuda_iov[i].iov_base) + length_per_iovec / alignment * alignment;
                cuda_iov_dist_h_current[nb_blocks_used].dst = destination;
                cuda_iov_dist_h_current[nb_blocks_used].element_alignment = orig_alignment;
                cuda_iov_dist_h_current[nb_blocks_used].nb_elements = (length_per_iovec - length_per_iovec / alignment * alignment) / orig_alignment;
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
                assert(cuda_iov_dist_h_current[nb_blocks_used].nb_elements > 0);
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
                destination += cuda_iov_dist_h_current[nb_blocks_used].nb_elements * orig_alignment;
                DT_CUDA_DEBUG( opal_cuda_output(12, "Pack \tblock %d, src %p, dst %p, nb_elements %d, alignment %d\n", nb_blocks_used, cuda_iov_dist_h_current[nb_blocks_used].src, cuda_iov_dist_h_current[nb_blocks_used].dst, cuda_iov_dist_h_current[nb_blocks_used].nb_elements, cuda_iov_dist_h_current[nb_blocks_used].element_alignment); );
                nb_blocks_used ++;
                assert (nb_blocks_used < CUDA_MAX_NB_BLOCKS*CUDA_IOV_MAX_TASK_PER_BLOCK);
            }
            
            if (buffer_isfull) {
                break;
            }
        }

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: Pack to dest %p, iov is prepared in %ld microsec, kernel submitted to CUDA stream %d, nb_blocks %d\n", destination_base, total_time,  cuda_iov_pipeline_block->cuda_stream_id, nb_blocks_used); );
#endif

        cudaMemcpyAsync(cuda_iov_dist_d_current, cuda_iov_dist_h_current, sizeof(ddt_cuda_iov_dist_non_cached_t)*(nb_blocks_used), cudaMemcpyHostToDevice, *cuda_stream_iov);
        opal_generic_simple_pack_cuda_iov_non_cached_kernel<<<nb_blocks, thread_per_block, 0, *cuda_stream_iov>>>(cuda_iov_dist_d_current, nb_blocks_used);
        cuda_err = cudaEventRecord(cuda_iov_pipeline_block->cuda_event, *cuda_stream_iov);
        opal_cuda_check_error(cuda_err);
        iov_pipeline_block_id ++;
        iov_pipeline_block_id = iov_pipeline_block_id % NB_STREAMS;
        
        /* buffer is full */
        if (buffer_isfull) {
            size_t total_converted_tmp = total_converted;
            pConvertor->flags = convertor_flags;
            total_converted += total_packed;
            opal_convertor_set_position_nocheck(pConvertor, &total_converted);
            total_packed = total_converted - total_converted_tmp;
            break;
        }
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        convertor_flags = pConvertor->flags;
//        orig_stack_index = pStack->index;
        complete_flag = opal_convertor_raw( pConvertor, cuda_iov, &cuda_iov_count, &length );
        DT_CUDA_DEBUG ( opal_cuda_output(4, "Pack complete flag %d, iov count %d, length %d, submit to CUDA stream %d\n", complete_flag, cuda_iov_count, length, cuda_streams->current_stream_id); );
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: ddt to iov in %ld microsec\n", total_time ); );
#endif
    }
    

    for (i = 0; i < NB_STREAMS; i++) {
        cudaStreamSynchronize(cuda_streams->opal_cuda_stream[i]);
    }
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    if (transfer_required) {
        cudaMemcpy(iov[0].iov_base, pConvertor->gpu_buffer_ptr, total_packed, cudaMemcpyDeviceToHost);
    } 
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    move_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: DtoH memcpy in %ld microsec, transfer required %d\n", move_time, transfer_required ); );
#endif

    iov[0].iov_len = total_packed;
    *max_data = total_packed;
    *out_size = 1;
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Pack total packed %d\n", total_packed); );
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    GET_TIME( end_total );
    total_time = ELAPSED_TIME( start_total, end_total );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: total packing in %ld microsec, kernel %ld microsec\n", total_time, total_time-move_time ); );
#endif
    
    if( pConvertor->bConverted == pConvertor->local_size ) {
        pConvertor->flags |= CONVERTOR_COMPLETED;
        if (pConvertor->gpu_buffer_ptr != NULL && free_required) {
           opal_ddt_cuda_free_gpu_buffer(pConvertor->gpu_buffer_ptr, 0);
           pConvertor->gpu_buffer_ptr = NULL;
        }
        return 1;
    }        
    return 0;
}

#endif


int32_t opal_ddt_generic_simple_pack_function_cuda_iov_non_cached( opal_convertor_t* pConvertor, unsigned char *destination, size_t buffer_size, size_t *total_packed)
{
    uint32_t i;
    uint32_t nb_blocks, thread_per_block, nb_blocks_used;
    unsigned char *destination_base, *source_base;
    uint8_t buffer_isfull = 0;
    cudaError_t cuda_err;
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    ddt_cuda_iov_dist_cached_t* cuda_iov_dist_h_current;
    ddt_cuda_iov_dist_cached_t* cuda_iov_dist_d_current;
    ddt_cuda_iov_pipeline_block_t *cuda_iov_pipeline_block;
    int iov_pipeline_block_id = 0;
    cudaStream_t *cuda_stream_iov = NULL;
    const struct iovec *ddt_iov = NULL;
    uint32_t ddt_iov_count = 0;
    size_t contig_disp = 0;
    uint32_t ddt_iov_start_pos, ddt_iov_end_pos, current_ddt_iov_pos;
    OPAL_PTRDIFF_TYPE ddt_extent;

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end;
    long total_time;
#endif
    
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Pack using IOV non cached, GPU base %p, pack to buffer %p\n", pConvertor->pBaseBuf, destination););
    
    opal_convertor_raw_cached( pConvertor, &ddt_iov, &ddt_iov_count);
    if (ddt_iov == NULL) {
        DT_CUDA_DEBUG ( opal_cuda_output(0, "Can not get ddt iov\n"););
        return OPAL_ERROR;
    }
    
    cuda_streams->current_stream_id = 0;
    thread_per_block = CUDA_WARP_SIZE * 5;
    nb_blocks = 256;
    opal_datatype_type_extent(pConvertor->pDesc, &ddt_extent);
    source_base = (unsigned char*)pConvertor->pBaseBuf + pConvertor->current_count * ddt_extent; 
    destination_base = destination;
    
    for (i = 0; i < NB_STREAMS; i++) {
        cudaStreamSynchronize(cuda_streams->opal_cuda_stream[i]);
    }
    
    while( pConvertor->current_count < pConvertor->count && !buffer_isfull) {
        
        nb_blocks_used = 0;
        ddt_iov_start_pos = pConvertor->current_iov_pos;
        ddt_iov_end_pos = ddt_iov_start_pos + IOV_PIPELINE_SIZE;
        if (ddt_iov_end_pos > ddt_iov_count) {
            ddt_iov_end_pos = ddt_iov_count;
        }
        cuda_iov_pipeline_block = current_cuda_device->cuda_iov_pipeline_block[iov_pipeline_block_id];
        cuda_iov_dist_h_current = cuda_iov_pipeline_block->cuda_iov_dist_non_cached_h;
        cuda_iov_dist_d_current = cuda_iov_pipeline_block->cuda_iov_dist_non_cached_d;
        cuda_stream_iov = cuda_iov_pipeline_block->cuda_stream;
        cuda_err = cudaStreamWaitEvent(*cuda_stream_iov, cuda_iov_pipeline_block->cuda_event, 0);
        opal_cuda_check_error(cuda_err);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif

        buffer_isfull = opal_ddt_iov_to_cuda_iov(pConvertor, ddt_iov, cuda_iov_dist_h_current, ddt_iov_start_pos, ddt_iov_end_pos, &buffer_size, &nb_blocks_used, total_packed, &contig_disp, &current_ddt_iov_pos);

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: Pack src %p to dest %p, iov is prepared in %ld microsec, kernel submitted to CUDA stream %d, nb_blocks %d\n", source_base, destination_base, total_time,  cuda_iov_pipeline_block->cuda_stream_id, nb_blocks_used); );
#endif

        cudaMemcpyAsync(cuda_iov_dist_d_current, cuda_iov_dist_h_current, sizeof(ddt_cuda_iov_dist_cached_t)*(nb_blocks_used+1), cudaMemcpyHostToDevice, *cuda_stream_iov);
        opal_generic_simple_pack_cuda_iov_cached_kernel<<<nb_blocks, thread_per_block, 0, *cuda_stream_iov>>>(cuda_iov_dist_d_current, 0, nb_blocks_used, 0, 0, nb_blocks_used, source_base, destination_base);
        //cudaStreamSynchronize(*cuda_stream_iov);
        cuda_err = cudaEventRecord(cuda_iov_pipeline_block->cuda_event, *cuda_stream_iov);
        opal_cuda_check_error(cuda_err);
        iov_pipeline_block_id ++;
        iov_pipeline_block_id = iov_pipeline_block_id % NB_STREAMS;
        destination_base += contig_disp;
        
        if (!buffer_isfull) {
            pConvertor->current_iov_pos = current_ddt_iov_pos;
            if (current_ddt_iov_pos == ddt_iov_count) {
                pConvertor->current_count ++;
                pConvertor->current_iov_pos = 0;
                source_base += ddt_extent;
            }
        }
        
    }
    
    for (i = 0; i < NB_STREAMS; i++) {
        cudaStreamSynchronize(cuda_streams->opal_cuda_stream[i]);
    }
        
    return OPAL_SUCCESS;
}

int32_t opal_ddt_generic_simple_pack_function_cuda_iov_cached( opal_convertor_t* pConvertor, unsigned char *destination, size_t buffer_size, size_t *total_packed)
{
    uint32_t i;
    uint32_t nb_blocks, thread_per_block, nb_blocks_used;
    unsigned char *destination_base, *source_base;
    uint8_t buffer_isfull = 0;
    cudaError_t cuda_err;
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    ddt_cuda_iov_pipeline_block_t *cuda_iov_pipeline_block = NULL;
    cudaStream_t *cuda_stream_iov = NULL;
    uint32_t cuda_iov_start_pos, cuda_iov_end_pos;
    ddt_cuda_iov_total_cached_t* cached_cuda_iov = NULL;
    ddt_cuda_iov_dist_cached_t* cached_cuda_iov_dist_d = NULL;
    uint32_t *cached_cuda_iov_nb_bytes_list_h = NULL;
    uint32_t cached_cuda_iov_count = 0;
    opal_datatype_count_t convertor_current_count;
    OPAL_PTRDIFF_TYPE ddt_extent;

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end;
    long total_time;
#endif
    
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Pack using IOV cached, GPU base %p, pack to buffer %p\n", pConvertor->pBaseBuf, destination););

    cuda_streams->current_stream_id = 0;
    destination_base = destination;
    thread_per_block = CUDA_WARP_SIZE * 8;
    nb_blocks = 4;
    source_base = (unsigned char*)pConvertor->pBaseBuf; 
    
    /* cuda iov is not cached, start to cache iov */
    if(opal_ddt_cuda_iov_is_cached(pConvertor) == 0) {
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        if (opal_ddt_cache_cuda_iov(pConvertor, &nb_blocks_used) == OPAL_SUCCESS) {
            opal_ddt_set_cuda_iov_cached(pConvertor, nb_blocks_used);
            DT_CUDA_DEBUG ( opal_cuda_output(2, "Pack cuda iov is cached, count %d\n", nb_blocks_used););
        } else {
            DT_CUDA_DEBUG ( opal_cuda_output(0, "Pack cache cuda iov is failed\n"););
            return OPAL_ERROR;
        }
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: Pack cuda iov is cached in %ld microsec, nb_blocks %d\n", total_time, nb_blocks_used); );
#endif
    }
    
    /* now we use cached cuda iov */
    opal_ddt_get_cached_cuda_iov(pConvertor, &cached_cuda_iov);
    cached_cuda_iov_dist_d = cached_cuda_iov->cuda_iov_dist_d;
    assert(cached_cuda_iov_dist_d != NULL);
    cached_cuda_iov_nb_bytes_list_h = cached_cuda_iov->nb_bytes_h;
    assert(cached_cuda_iov_nb_bytes_list_h != NULL);
    
    cached_cuda_iov_count = cached_cuda_iov->cuda_iov_count;
    cuda_iov_start_pos = pConvertor->current_cuda_iov_pos;
    cuda_iov_end_pos = cached_cuda_iov_count;
    nb_blocks_used = 0;
    cuda_iov_pipeline_block = current_cuda_device->cuda_iov_pipeline_block[0];
    cuda_stream_iov = cuda_iov_pipeline_block->cuda_stream;
    convertor_current_count = pConvertor->current_count;
   
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    while( pConvertor->current_count < pConvertor->count && !buffer_isfull) {
        for (i = cuda_iov_start_pos; i < cuda_iov_end_pos && !buffer_isfull; i++) {
            if (buffer_size >= cached_cuda_iov_nb_bytes_list_h[i]) {
                *total_packed += cached_cuda_iov_nb_bytes_list_h[i];
                buffer_size -= cached_cuda_iov_nb_bytes_list_h[i];
                nb_blocks_used++;
            } else {
                buffer_isfull = 1;
                break;
            }
        }
        if (!buffer_isfull) {
            pConvertor->current_count ++;
            cuda_iov_start_pos = 0;
            cuda_iov_end_pos = cached_cuda_iov->cuda_iov_count;
        }
    }
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: Pack to dest %p, cached cuda iov is prepared in %ld microsec, kernel submitted to CUDA stream %d, nb_blocks %d\n", destination_base, total_time,  cuda_iov_pipeline_block->cuda_stream_id, nb_blocks_used); );
#endif
    opal_datatype_type_extent(pConvertor->pDesc, &ddt_extent);
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Pack kernel launched src_base %p, dst_base %p, nb_blocks %ld, extent %ld\n", source_base, destination_base, nb_blocks_used, ddt_extent ); );
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    
    opal_generic_simple_pack_cuda_iov_cached_kernel<<<nb_blocks, thread_per_block, 0, *cuda_stream_iov>>>(cached_cuda_iov_dist_d, pConvertor->current_cuda_iov_pos, cached_cuda_iov_count, ddt_extent, convertor_current_count, nb_blocks_used, source_base, destination_base);
    pConvertor->current_cuda_iov_pos += nb_blocks_used;
    pConvertor->current_cuda_iov_pos = pConvertor->current_cuda_iov_pos % cached_cuda_iov->cuda_iov_count;

    cudaStreamSynchronize(cuda_streams->opal_cuda_stream[0]);
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: Pack kernel %ld microsec\n", total_time); );
#endif    
    return OPAL_SUCCESS;
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
    
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;

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

