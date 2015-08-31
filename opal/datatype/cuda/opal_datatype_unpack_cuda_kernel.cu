#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#include "opal_datatype_cuda_internal.cuh"
#include <cuda.h>
#include <stdio.h> 

__device__ void unpack_contiguous_loop_cuda_kernel( dt_elem_desc_t* ELEM,
                                                    uint32_t* COUNT,
                                                    unsigned char** SOURCE,
                                                    unsigned char** DESTINATION,
                                                    size_t* SPACE )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _dst_disp = (*DESTINATION) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t _i, tid, num_threads;
    unsigned char* _source = *SOURCE;
//    unsigned char* _source = _src_disp;
    uint32_t gap, nb_elements;
    double *_source_tmp, *_destination_tmp, *_dst_disp_tmp;;
    
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    num_threads = gridDim.x * blockDim.x;

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);
    
    gap = (_loop->extent - _end_loop->size) / 8;
    nb_elements = _end_loop->size / 8;
    _dst_disp_tmp = (double*)_dst_disp;
    _source_tmp = (double*)_source;
    _destination_tmp = _dst_disp_tmp + tid;
    _source_tmp += tid;

    __syncthreads();
    for (_i = tid; _i < _copy_loops*nb_elements; _i+=num_threads) {
        _destination_tmp = _dst_disp_tmp + tid + _i/num_threads*num_threads + _i/nb_elements * gap;
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
        if (_i % nb_elements == 0 ) {
            DBGPRINT("tid %d, pack 3. memcpy( %p, %p, %lu ) => space %lu, _i %d, actual _i %d\n",
                                            tid, _destination_tmp, _source_tmp, (unsigned long)_end_loop->size, (unsigned long)(*(SPACE) - _i/nb_elements * _end_loop->size), _i/nb_elements, _i );
        }
        // if (_i / nb_elements ==1 && tid == 0 ) {
        //     DBGPRINT("tid %d, pack 3. memcpy( %p, %p, %lu ) => space %lu, _i %d, actual _i %d\n",
        //                                     tid, _destination_tmp, _source_tmp, (unsigned long)_end_loop->size, (unsigned long)(*(SPACE) - _i/nb_elements * _end_loop->size), _i/nb_elements, _i );
        // }
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
#if !defined (OPAL_DATATYPE_CUDA_DRY_RUN)
        *_destination_tmp = *_source_tmp;
#endif /* ! OPAL_DATATYPE_CUDA_DRY_RUN */
        _source_tmp += num_threads;
//        _source_tmp += num_threads;

    }
    *(DESTINATION) = _dst_disp + _copy_loops*_loop->extent - _end_loop->first_elem_disp;
    *(SOURCE) = *(SOURCE)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;

    __syncthreads();
}

__global__ void opal_generic_simple_unpack_cuda_kernel(ddt_cuda_desc_t* cuda_desc)
{
    dt_stack_t* pStack;                /* pointer to the position on the stack */
    uint32_t pos_desc;                 /* actual position in the description of the derived datatype */
    uint32_t count_desc;               /* the number of items already done in the actual pos_desc */
    size_t total_unpacked = 0;         /* total size unpacked this time */
    dt_elem_desc_t* description;
    dt_elem_desc_t* pElem;
    unsigned char *conv_ptr, *iov_ptr, *pBaseBuf;
    size_t iov_len_local;
    uint32_t iov_count;
    uint32_t stack_pos;
    struct iovec* iov;

    OPAL_PTRDIFF_TYPE lb; 
    OPAL_PTRDIFF_TYPE ub;
    uint32_t out_size;
    uint32_t tid;

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    
 //   __shared__ ddt_cuda_desc_t cuda_desc_b;
    __shared__ dt_stack_t shared_pStack[DT_STATIC_STACK_SIZE];

    if (threadIdx.x < DT_STATIC_STACK_SIZE) {
        shared_pStack[threadIdx.x] = cuda_desc->pStack[threadIdx.x];
    }
    __syncthreads();
    
    // load cuda descriptor from constant memory
    iov = cuda_desc->iov;
    pStack = shared_pStack;
    description = cuda_desc->description;
    stack_pos = cuda_desc->stack_pos;
    pBaseBuf = cuda_desc->pBaseBuf;
    lb = cuda_desc->lb;
    ub = cuda_desc->ub;
    out_size = cuda_desc->out_size;

    /* For the first step we have to add both displacement to the source. After in the
     * main while loop we will set back the source_base to the correct value. This is
     * due to the fact that the convertor can stop in the middle of a data with a count
     */
    pStack     = pStack + stack_pos;
    pos_desc   = pStack->index;
    conv_ptr   = pBaseBuf + pStack->disp;
    count_desc = (uint32_t)pStack->count;
    pStack--;
    stack_pos--;
    pElem = &(description[pos_desc]);


    for( iov_count = 0; iov_count < out_size; iov_count++ ) {
        iov_ptr = (unsigned char *) iov[iov_count].iov_base;
        iov_len_local = iov[iov_count].iov_len;
        // if( 0 != pConvertor->partial_length ) {
        //     size_t element_length = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;
        //     size_t missing_length = element_length - pConvertor->partial_length;
        //
        //     assert( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA );
        //     COMPUTE_CSUM( iov_ptr, missing_length, pConvertor );
        //     opal_unpack_partial_datatype( pConvertor, pElem,
        //                                   iov_ptr,
        //                                   pConvertor->partial_length, element_length - pConvertor->partial_length,
        //                                   &conv_ptr );
        //     --count_desc;
        //     if( 0 == count_desc ) {
        //         conv_ptr = pConvertor->pBaseBuf + pStack->disp;
        //         pos_desc++;  /* advance to the next data */
        //         UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
        //     }
        //     iov_ptr       += missing_length;
        //     iov_len_local -= missing_length;
        //     pConvertor->partial_length = 0;  /* nothing more inside */
        // }
        while( 1 ) {
            while( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {
                /* now here we have a basic datatype */
                // UNPACK_PREDEFINED_DATATYPE( pConvertor, pElem, count_desc,
                //                             iov_ptr, conv_ptr, iov_len_local );
                if( 0 == count_desc ) {  /* completed */
                    conv_ptr = pBaseBuf + pStack->disp;
                    pos_desc++;  /* advance to the next data */
                    UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                    continue;
                }
                // assert( pElem->elem.common.type < OPAL_DATATYPE_MAX_PREDEFINED );
                if( 0 != iov_len_local ) {
                    unsigned char* temp = conv_ptr;
                    /* We have some partial data here. Let's copy it into the convertor
                     * and keep it hot until the next round.
                     */
                    // assert( iov_len_local < opal_datatype_basicDatatypes[pElem->elem.common.type]->size );
                    // COMPUTE_CSUM( iov_ptr, iov_len_local, pConvertor );
                    //
                    // opal_unpack_partial_datatype( pConvertor, pElem,
                    //                               iov_ptr, 0, iov_len_local,
                    //                               &temp );
                    //
                    // pConvertor->partial_length = (uint32_t)iov_len_local;
                    iov_len_local = 0;
                }
                goto complete_loop;
            }
            if( OPAL_DATATYPE_END_LOOP == pElem->elem.common.type ) { /* end of the current loop */
                // DO_DEBUG( opal_output( 0, "unpack end_loop count %d stack_pos %d pos_desc %d disp %ld space %lu\n",
                //                        (int)pStack->count, pConvertor->stack_pos, pos_desc,
                //                        (long)pStack->disp, (unsigned long)iov_len_local ); );
                if (threadIdx.x == 0) {
                    (pStack->count)--;
                }
                __syncthreads();
                
                if( pStack->count == 0 ) { /* end of loop */
                    if( 0 == stack_pos ) {
                        /* Do the same thing as when the loop is completed */
                        iov[iov_count].iov_len -= iov_len_local;  /* update the amount of valid data */
                        total_unpacked += iov[iov_count].iov_len;
                        iov_count++;  /* go to the next */
                        goto complete_conversion;
                    }
                    stack_pos--;
                    pStack--;
                    pos_desc++;
                } else {
                    pos_desc = pStack->index + 1;
                    if (threadIdx.x == 0) {
                        if( pStack->index == -1 ) {
                            pStack->disp += (ub - lb);
                        } else {
                            //assert( OPAL_DATATYPE_LOOP == description[pStack->index].loop.common.type );
                            pStack->disp += description[pStack->index].loop.extent;
                        }
                    }
                    __syncthreads();
                }
                conv_ptr = pBaseBuf + pStack->disp;
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                // DO_DEBUG( opal_output( 0, "unpack new_loop count %d stack_pos %d pos_desc %d disp %ld space %lu\n",
                //                        (int)pStack->count, pConvertor->stack_pos, pos_desc,
                //                        (long)pStack->disp, (unsigned long)iov_len_local ); );
            }
            if( OPAL_DATATYPE_LOOP == pElem->elem.common.type ) {
                OPAL_PTRDIFF_TYPE local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr;
                if( pElem->loop.common.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                    unpack_contiguous_loop_cuda_kernel( pElem, &count_desc,
                                                        &iov_ptr, &conv_ptr, &iov_len_local );
                    count_desc = 0;
                    if( 0 == count_desc ) {  /* completed */
                        pos_desc += pElem->loop.items + 1;
                        goto update_loop_description;
                    }
                    /* Save the stack with the correct last_count value. */
                }
                local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr - local_disp;
                PUSH_STACK( pStack, stack_pos, pos_desc, OPAL_DATATYPE_LOOP, count_desc,
                            pStack->disp + local_disp);
                pos_desc++;
            update_loop_description:  /* update the current state */
                conv_ptr = pBaseBuf + pStack->disp;
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                // DDT_DUMP_STACK( pConvertor->pStack, pConvertor->stack_pos, pElem, "advance loop" );
                continue;
            }
        }
    complete_loop:
        if (threadIdx.x == 0) { 
            iov[iov_count].iov_len -= iov_len_local;  /* update the amount of valid data */
        }
        __syncthreads();
        total_unpacked += iov[iov_count].iov_len;
    }
 complete_conversion:
    if (tid == 0) {
        cuda_desc->max_data = total_unpacked;
    //    pConvertor->bConverted += total_unpacked;  /* update the already converted bytes */
        cuda_desc->out_size = iov_count;
        // if( pConvertor->bConverted == pConvertor->remote_size ) {
        //     pConvertor->flags |= CONVERTOR_COMPLETED;
        //     return 1;
        // }
        // /* Save the global position for the next round */
        // PUSH_STACK( pStack, pConvertor->stack_pos, pos_desc, OPAL_DATATYPE_UINT1, count_desc,
        //             conv_ptr - pConvertor->pBaseBuf );
        // DO_DEBUG( opal_output( 0, "unpack save stack stack_pos %d pos_desc %d count_desc %d disp %ld\n",
        //                        pConvertor->stack_pos, pStack->index, (int)pStack->count, (long)pStack->disp ); );
    }
}


__global__ void opal_generic_simple_unpack_cuda_iov_kernel( ddt_cuda_iov_dist_t* cuda_iov_dist)
{
    uint32_t i, _copy_count;
    unsigned char *src, *dst;
    uint8_t alignment;
    unsigned char *_source_tmp, *_destination_tmp;
    
    __shared__ uint32_t nb_tasks;
    
    if (threadIdx.x == 0) {
        nb_tasks = cuda_iov_dist[blockIdx.x].nb_tasks;
    }
    __syncthreads();
    
    for (i = 0; i < nb_tasks; i++) {
        src = cuda_iov_dist[blockIdx.x].src[i];
        dst = cuda_iov_dist[blockIdx.x].dst[i];
        _copy_count = cuda_iov_dist[blockIdx.x].nb_elements[i];
        alignment = cuda_iov_dist[blockIdx.x].element_alignment[i];
        
        if (threadIdx.x < _copy_count) {
            _source_tmp = src + threadIdx.x * alignment;
            _destination_tmp = dst + threadIdx.x * alignment;
#if !defined (OPAL_DATATYPE_CUDA_DRY_RUN)
                if (alignment == ALIGNMENT_DOUBLE) {
                    *((long *)_destination_tmp) = *((long *)_source_tmp);
                } else if (alignment == ALIGNMENT_FLOAT) {
                    *((int *)_destination_tmp) = *((int *)_source_tmp);
                } else {
                    * _destination_tmp = *_source_tmp;
                }
        //   printf("src %p, %1.f | dst %p, %1.f\n", _source_tmp, *_source_tmp, _destination_tmp, *_destination_tmp);
#endif /* ! OPAL_DATATYPE_CUDA_DRY_RUN */
        }
    }
}
__global__ void unpack_contiguous_loop_cuda_kernel_global( uint32_t copy_loops,
                                                           size_t size,
                                                           OPAL_PTRDIFF_TYPE extent,
                                                           unsigned char* source,
                                                           unsigned char* destination )
{
    uint32_t _i, tid, num_threads;
    uint32_t gap, nb_elements;
    double *_source_tmp, *_destination_tmp, *_dst_disp_tmp;;
    
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    num_threads = gridDim.x * blockDim.x;
    
    gap = (extent - size) / 8;
    nb_elements = size / 8;
    _dst_disp_tmp = (double*)destination;
    _source_tmp = (double*)source;
    _destination_tmp = _dst_disp_tmp + tid;
    _source_tmp += tid;

    for (_i = tid; _i < copy_loops*nb_elements; _i+=num_threads) {
        _destination_tmp = _dst_disp_tmp + tid + _i/num_threads*num_threads + _i/nb_elements * gap;
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
        // if (_i % nb_elements == 0 ) {
        //     DBGPRINT("tid %d, pack 3. memcpy( %p, %p, %lu ) => _i %d, actual _i %d\n",
        //                                     tid, _destination_tmp, _source_tmp, (unsigned long)size,  _i/nb_elements, _i );
        // }
        // if (_i / nb_elements ==1 && tid == 0 ) {
        //     DBGPRINT("tid %d, pack 3. memcpy( %p, %p, %lu ) => space %lu, _i %d, actual _i %d\n",
        //                                     tid, _destination_tmp, _source_tmp, (unsigned long)_end_loop->size, (unsigned long)(*(SPACE) - _i/nb_elements * _end_loop->size), _i/nb_elements, _i );
        // }
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
#if !defined (OPAL_DATATYPE_CUDA_DRY_RUN)
        *_destination_tmp = *_source_tmp;
#endif /* ! OPAL_DATATYPE_CUDA_DRY_RUN */
        _source_tmp += num_threads;
    }
}
