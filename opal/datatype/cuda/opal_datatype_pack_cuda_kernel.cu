#include "opal_datatype_cuda_internal.cuh"
#include <stdio.h> 
#include <time.h>

__device__ void pack_contiguous_loop_cuda_kernel( dt_elem_desc_t* ELEM,
                                                  uint32_t* COUNT,
                                                  unsigned char** SOURCE,
                                                  unsigned char** DESTINATION,
                                                  size_t* SPACE )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _src_disp = (*SOURCE) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t _i, tid, num_threads;
    unsigned char* _destination = *DESTINATION;
//    unsigned char* _source = _src_disp;
    uint32_t gap, nb_elements;
    double *_source_tmp, *_destination_tmp, *_src_disp_tmp;
    
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    num_threads = gridDim.x * blockDim.x;

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);
    
//     num_task_per_thread = _copy_loops / num_threads;
//     residue = _copy_loops % num_threads;
//     if ( ((tid < residue) && (residue != 0)) || (residue == 0) ) {
//         num_task_per_thread += residue == 0 ? 0 : 1;
//         start_index = tid * num_task_per_thread;
//     } else {
//         start_index = residue * (num_task_per_thread+1) + (tid-residue) * num_task_per_thread;
//     }
//
//     end_index = start_index + num_task_per_thread;
//     DBGPRINT("tid %d, start %d, end %d, num_task_per_thread %d, copy_loops %d\n", tid, start_index, end_index, num_task_per_thread, _copy_loops);
//     for( _i = start_index; _i < end_index; _i++ ) {
//         // OPAL_DATATYPE_SAFEGUARD_POINTER( _source, _loop->extent, (CONVERTOR)->pBaseBuf,
//         //                             (CONVERTOR)->pDesc, (CONVERTOR)->count );
//         _source = _src_disp + _i * _loop->extent;
//         _destination = *DESTINATION + _i * _end_loop->size;
//         DBGPRINT("tid %d, pack 3. memcpy( %p, %p, %lu ) => space %lu, _i %d\n",
//                                tid, _destination, _source, (unsigned long)_end_loop->size, (unsigned long)(*(SPACE) - _i * _end_loop->size), _i );
//     //    MEMCPY_CSUM( *(DESTINATION), _source, _end_loop->size, (CONVERTOR) );
// #if !defined (OPAL_DATATYPE_CUDA_DRY_RUN)
//  //       memcpy(_destination, _source, _end_loop->size);
//         _source_tmp = (double *)_source;
//         _destination_tmp = (double *)_destination;
//         for (_j = 0; _j < _end_loop->size/8; _j++)
//         {
//             *_destination_tmp = *_source_tmp;
//             _destination_tmp ++;
//             _source_tmp ++;
//         }
// #endif /* OPAL_DATATYPE_CUDA_DRY_RUN */
//     }
    
    gap = (_loop->extent - _end_loop->size) / 8;
    nb_elements = _end_loop->size / 8;
    _src_disp_tmp = (double*)_src_disp;
    _destination_tmp = (double*)_destination;
    _destination_tmp += tid;

    __syncthreads();

    for (_i = tid; _i < _copy_loops*nb_elements; _i+=num_threads) {
        _source_tmp = _src_disp_tmp + tid + _i/num_threads*num_threads + _i/nb_elements * gap;
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
        _destination_tmp += num_threads;

    }
    *(SOURCE) = _src_disp + _copy_loops*_loop->extent - _end_loop->first_elem_disp;
    *(DESTINATION) = *(DESTINATION)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;

    __syncthreads();
}

__device__ void pack_predefined_data_cuda_kernel( dt_elem_desc_t* ELEM,
                                                  uint32_t* COUNT,
                                                  unsigned char** SOURCE,
                                                  unsigned char** DESTINATION,
                                                  size_t* SPACE )
{
    uint32_t _copy_count = *(COUNT);
    size_t _copy_blength;
    ddt_elem_desc_t* _elem = &((ELEM)->elem);
    unsigned char* _src_disp = (*SOURCE) + _elem->disp;
    uint32_t _i, tid, num_threads;
    unsigned char* _destination = *DESTINATION;
    uint32_t gap, nb_elements;
    double *_source_tmp, *_destination_tmp, *_src_disp_tmp;;

    _copy_blength = 8;//opal_datatype_basicDatatypes[_elem->common.type]->size;
    if( (_copy_count * _copy_blength) > *(SPACE) ) {
        _copy_count = (uint32_t)(*(SPACE) / _copy_blength);
        if( 0 == _copy_count ) return;  /* nothing to do */
    }
    
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    num_threads = gridDim.x * blockDim.x;
    
    gap = (_elem->extent - _copy_blength) / 8;
    nb_elements = _copy_blength / 8;
    _src_disp_tmp = (double*)_src_disp;
    _destination_tmp = (double*)_destination;
    _source_tmp = _src_disp_tmp + tid;
    _destination_tmp += tid;
    
    __syncthreads();
    
    for (_i = tid; _i < _copy_count*nb_elements; _i+=num_threads) {
        _source_tmp = _src_disp_tmp + tid + _i/num_threads*num_threads + _i/nb_elements * gap;
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
        if (_i == 0 ) {
            DBGPRINT("tid %d, pack 3. memcpy( %p, %p, %lu ) => space %lu, _i %d, actual _i %d\n",
                                            tid, _destination_tmp, _source_tmp, (unsigned long)_copy_blength*_copy_count, (unsigned long)(*(SPACE) - _i/nb_elements * _copy_blength), _i/nb_elements, _i );
        }
        // if (_i / nb_elements ==1 && tid == 0 ) {
        //     DBGPRINT("tid %d, pack 3. memcpy( %p, %p, %lu ) => space %lu, _i %d, actual _i %d\n",
        //                                     tid, _destination_tmp, _source_tmp, (unsigned long)_end_loop->size, (unsigned long)(*(SPACE) - _i/nb_elements * _end_loop->size), _i/nb_elements, _i );
        // }
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
#if !defined (OPAL_DATATYPE_CUDA_DRY_RUN)
        *_destination_tmp = *_source_tmp;
#endif /* ! OPAL_DATATYPE_CUDA_DRY_RUN */
        _destination_tmp += num_threads;

    }
    
    _copy_blength *= _copy_count;
    *(SOURCE)  = _src_disp + _elem->extent*_copy_count - _elem->disp;
    *(DESTINATION) += _copy_blength;
    *(SPACE)  -= _copy_blength;
    *(COUNT)  -= _copy_count;
    
    __syncthreads();
}

__global__ void opal_generic_simple_pack_cuda_kernel(ddt_cuda_desc_t* cuda_desc)
{
    dt_stack_t *pStack, *pStack_head;       /* pointer to the position on the stack */
    uint32_t pos_desc;        /* actual position in the description of the derived datatype */
    uint32_t count_desc;      /* the number of items already done in the actual pos_desc */
    size_t total_packed = 0;  /* total amount packed this time */
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

    __shared__ ddt_cuda_desc_t cuda_desc_b;

    if (threadIdx.x == 0) {
        memcpy(&cuda_desc_b, cuda_desc, sizeof(ddt_cuda_desc_t));
    }
    __syncthreads();

    // load cuda descriptor from constant memory
    iov = cuda_desc_b.iov;
    pStack_head = cuda_desc_b.pStack;
    pStack = pStack_head;
    description = cuda_desc_b.description;
    stack_pos = cuda_desc_b.stack_pos;
    pBaseBuf = cuda_desc_b.pBaseBuf;
    lb = cuda_desc_b.lb;
    ub = cuda_desc_b.ub;
    out_size = cuda_desc_b.out_size;

    pStack = pStack + stack_pos;
    pos_desc   = pStack->index;
    conv_ptr   = pBaseBuf + pStack->disp;
    count_desc = (uint32_t)pStack->count;
    pStack--;
    stack_pos--;
    pElem = &(description[pos_desc]);

//    printf("pack start pos_desc %d count_desc %d disp %ld, stack_pos %d pos_desc %d count_desc %d disp %ld\n",
//            pos_desc, count_desc, (long)(conv_ptr - pBaseBuf), stack_pos, pStack->index, (int)pStack->count, (long)pStack->disp);

    for( iov_count = 0; iov_count < out_size; iov_count++ ) {
        iov_ptr = (unsigned char *) iov[iov_count].iov_base;
        iov_len_local = iov[iov_count].iov_len;
        DBGPRINT("iov_len_local %lu, flags %d, types %d, count %d\n", iov_len_local, description->elem.common.flags, description->elem.common.type, description->elem.count);
        while( 1 ) {
            while( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {
                /* now here we have a basic datatype */
                // PACK_PREDEFINED_DATATYPE( pConvertor, pElem, count_desc,
                //                           conv_ptr, iov_ptr, iov_len_local );
                pack_predefined_data_cuda_kernel(pElem, &count_desc, &conv_ptr, &iov_ptr, &iov_len_local);
                if( 0 == count_desc ) {  /* completed */
                    conv_ptr = pBaseBuf + pStack->disp;
                    pos_desc++;  /* advance to the next data */
                    UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                    continue;
                }
                goto complete_loop;
            }
            if( OPAL_DATATYPE_END_LOOP == pElem->elem.common.type ) { /* end of the current loop */
                // DO_DEBUG( opal_output( 0, "pack end_loop count %d stack_pos %d"
                //                        " pos_desc %d disp %ld space %lu\n",
                //                        (int)pStack->count, pConvertor->stack_pos,
                //                        pos_desc, (long)pStack->disp, (unsigned long)iov_len_local ); );
                if (threadIdx.x == 0) {
                    (pStack->count)--;
                }
                __syncthreads();

                if( (pStack->count) == 0 ) { /* end of loop */
                    if( 0 == stack_pos ) {
                        /* we lie about the size of the next element in order to
                         * make sure we exit the main loop.
                         */
                        out_size = iov_count;
                        goto complete_loop;  /* completed */
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
                            // assert( OPAL_DATATYPE_LOOP == description[pStack->index].loop.common.type );
                            pStack->disp += description[pStack->index].loop.extent;
                        }
                    }
                    __syncthreads();
                }
                conv_ptr = pBaseBuf + pStack->disp;
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                // DO_DEBUG( opal_output( 0, "pack new_loop count %d stack_pos %d pos_desc %d count_desc %d disp %ld space %lu\n",
                //                        (int)pStack->count, pConvertor->stack_pos, pos_desc,
                //                        count_desc, (long)pStack->disp, (unsigned long)iov_len_local ); );
            }
            if( OPAL_DATATYPE_LOOP == pElem->elem.common.type ) {
                OPAL_PTRDIFF_TYPE local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr;
                if( pElem->loop.common.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                    pack_contiguous_loop_cuda_kernel( pElem, &count_desc,
                                          &conv_ptr, &iov_ptr, &iov_len_local );
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
        total_packed += iov[iov_count].iov_len;
    }

    if (tid == 0) {
        cuda_desc->max_data = total_packed;
        cuda_desc->out_size = iov_count;
        // cuda_desc->bConverted += total_packed;  /* update the already converted bytes */
        // if( cuda_desc->bConverted == cuda_desc->local_size ) {
        //     cuda_desc->stack_pos = stack_pos;
        //     memcpy(cuda_desc->pStack, pStack_head, sizeof(dt_stack_t)*cuda_desc->stack_size);
        //     return;
        // }
        // /* Save the global position for the next round */
        // PUSH_STACK( pStack, stack_pos, pos_desc, OPAL_DATATYPE_INT8, count_desc,
        //             conv_ptr - pBaseBuf );
        // memcpy(cuda_desc->pStack, pStack_head, sizeof(dt_stack_t)*cuda_desc->stack_size);
        // cuda_desc->stack_pos = stack_pos;
    }
    __syncthreads();

    return;
}

// __global__ void opal_generic_simple_pack_cuda_kernel(ddt_cuda_desc_t* cuda_desc)
// {
//     dt_stack_t *pStack, *pStack_head;       /* pointer to the position on the stack */
//     uint32_t pos_desc;        /* actual position in the description of the derived datatype */
//     uint32_t count_desc;      /* the number of items already done in the actual pos_desc */
//     size_t total_packed = 0;  /* total amount packed this time */
//     dt_elem_desc_t* description;
//     dt_elem_desc_t* pElem;
//     unsigned char *conv_ptr, *iov_ptr, *pBaseBuf;
//     size_t iov_len_local;
//     uint32_t iov_count;
//     uint32_t stack_pos;
//     struct iovec* iov;
//
//     OPAL_PTRDIFF_TYPE lb;
//     OPAL_PTRDIFF_TYPE ub;
//     uint32_t out_size;
//     uint32_t tid;
//
//     tid = threadIdx.x + blockIdx.x * blockDim.x;
//
//     __shared__ ddt_cuda_desc_t cuda_desc_b;
//
//     if (threadIdx.x == 0) {
//         memcpy(&cuda_desc_b, cuda_desc, sizeof(ddt_cuda_desc_t));
//     }
//     __syncthreads();
//
//
//     // load cuda descriptor from constant memory
//     iov = cuda_desc_b.iov;
//     pStack_head = cuda_desc_b.pStack;
//     pStack = pStack_head;
//     description = cuda_desc_b.description;
//     stack_pos = cuda_desc_b.stack_pos;
//     pBaseBuf = cuda_desc_b.pBaseBuf;
//     lb = cuda_desc_b.lb;
//     ub = cuda_desc_b.ub;
//     out_size = cuda_desc_b.out_size;
//
//     pStack = pStack + stack_pos;
//     pos_desc   = pStack->index;
//     conv_ptr   = pBaseBuf + pStack->disp;
//     count_desc = (uint32_t)pStack->count;
//     pStack--;
//     stack_pos--;
//     pElem = &(description[pos_desc]);
//
// //    printf("pack start pos_desc %d count_desc %d disp %ld, stack_pos %d pos_desc %d count_desc %d disp %ld\n",
// //            pos_desc, count_desc, (long)(conv_ptr - pBaseBuf), stack_pos, pStack->index, (int)pStack->count, (long)pStack->disp);
//
//     if (threadIdx.x == 0) {
//     for( iov_count = 0; iov_count < out_size; iov_count++ ) {
//         iov_ptr = (unsigned char *) iov[iov_count].iov_base;
//         iov_len_local = iov[iov_count].iov_len;
//         DBGPRINT("iov_len_local %lu, flags %d, types %d, count %d\n", iov_len_local, description->elem.common.flags, description->elem.common.type, description->elem.count);
//         while( 1 ) {
//             while( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {
//                 /* now here we have a basic datatype */
//                 // PACK_PREDEFINED_DATATYPE( pConvertor, pElem, count_desc,
//                 //                           conv_ptr, iov_ptr, iov_len_local );
//                 if( 0 == count_desc ) {  /* completed */
//                     conv_ptr = pBaseBuf + pStack->disp;
//                     pos_desc++;  /* advance to the next data */
//                     UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
//                     continue;
//                 }
//                 goto complete_loop;
//             }
//             if( OPAL_DATATYPE_END_LOOP == pElem->elem.common.type ) { /* end of the current loop */
//                 // DO_DEBUG( opal_output( 0, "pack end_loop count %d stack_pos %d"
//                 //                        " pos_desc %d disp %ld space %lu\n",
//                 //                        (int)pStack->count, pConvertor->stack_pos,
//                 //                        pos_desc, (long)pStack->disp, (unsigned long)iov_len_local ); );
//
//                 if( --(pStack->count) == 0 ) { /* end of loop */
//                     if( 0 == stack_pos ) {
//                         /* we lie about the size of the next element in order to
//                          * make sure we exit the main loop.
//                          */
//                         out_size = iov_count;
//                         goto complete_loop;  /* completed */
//                     }
//                     stack_pos--;
//                     pStack--;
//                     pos_desc++;
//                 } else {
//                     pos_desc = pStack->index + 1;
//                     if( pStack->index == -1 ) {
//                         pStack->disp += (ub - lb);
//                     } else {
//                         // assert( OPAL_DATATYPE_LOOP == description[pStack->index].loop.common.type );
//                         pStack->disp += description[pStack->index].loop.extent;
//                     }
//
//                 }
//                 conv_ptr = pBaseBuf + pStack->disp;
//                 UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
//                 // DO_DEBUG( opal_output( 0, "pack new_loop count %d stack_pos %d pos_desc %d count_desc %d disp %ld space %lu\n",
//                 //                        (int)pStack->count, pConvertor->stack_pos, pos_desc,
//                 //                        count_desc, (long)pStack->disp, (unsigned long)iov_len_local ); );
//             }
//             if( OPAL_DATATYPE_LOOP == pElem->elem.common.type ) {
//                 OPAL_PTRDIFF_TYPE local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr;
//                 if( pElem->loop.common.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
//                     // pack_contiguous_loop_cuda_kernel( pElem, &count_desc,
//                     //                       &conv_ptr, &iov_ptr, &iov_len_local );
//                     count_desc = 0;
//                     if( 0 == count_desc ) {  /* completed */
//                         pos_desc += pElem->loop.items + 1;
//                         goto update_loop_description;
//                     }
//                     /* Save the stack with the correct last_count value. */
//                 }
//                 local_disp = (OPAL_PTRDIFF_TYPE)conv_ptr - local_disp;
//
//                 PUSH_STACK( pStack, stack_pos, pos_desc, OPAL_DATATYPE_LOOP, count_desc,
//                             pStack->disp + local_disp);
//
//                 pos_desc++;
//             update_loop_description:  /* update the current state */
//                 conv_ptr = pBaseBuf + pStack->disp;
//                 UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
//                 // DDT_DUMP_STACK( pConvertor->pStack, pConvertor->stack_pos, pElem, "advance loop" );
//                 continue;
//             }
//         }
//     complete_loop:
//         iov[iov_count].iov_len -= iov_len_local;  /* update the amount of valid data */
//         total_packed += iov[iov_count].iov_len;
//     }
//
//     }
//     __syncthreads();
//     if (tid == 0) {
//         cuda_desc->max_data = total_packed;
//         cuda_desc->out_size = iov_count;
//         // cuda_desc->bConverted += total_packed;  /* update the already converted bytes */
//         // if( cuda_desc->bConverted == cuda_desc->local_size ) {
//         //     cuda_desc->stack_pos = stack_pos;
//         //     memcpy(cuda_desc->pStack, pStack_head, sizeof(dt_stack_t)*cuda_desc->stack_size);
//         //     return;
//         // }
//         // /* Save the global position for the next round */
//         // PUSH_STACK( pStack, stack_pos, pos_desc, OPAL_DATATYPE_INT8, count_desc,
//         //             conv_ptr - pBaseBuf );
//         // memcpy(cuda_desc->pStack, pStack_head, sizeof(dt_stack_t)*cuda_desc->stack_size);
//         // cuda_desc->stack_pos = stack_pos;
//     }
//     return;
// }

__global__ void pack_contiguous_loop_cuda_kernel_global( uint32_t copy_loops,
                                                         size_t size,
                                                         OPAL_PTRDIFF_TYPE extent,
                                                         unsigned char* source,
                                                         unsigned char* destination )
{
    uint32_t _i, tid, num_threads;
    uint32_t gap, nb_elements;
    double *_source_tmp, *_destination_tmp, *_src_disp_tmp;;
    
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    num_threads = gridDim.x * blockDim.x;
    
    gap = (extent - size) / 8;
    nb_elements = size / 8;
    _src_disp_tmp = (double*)source;
    _destination_tmp = (double*)destination;
    _source_tmp = _src_disp_tmp + tid;
    _destination_tmp += tid;

    for (_i = tid; _i < copy_loops*nb_elements; _i+=num_threads) {
        _source_tmp = _src_disp_tmp + tid + _i/num_threads*num_threads + _i/nb_elements * gap;
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
        if (_i % nb_elements == 0 ) {
            DBGPRINT("tid %d, pack 3. memcpy( %p, %p, %lu ) => _i %d, actual _i %d, count %d\n",
                                            tid, _destination_tmp, _source_tmp, (unsigned long)size, _i/nb_elements, _i, copy_loops );
        }
        // if (_i / nb_elements ==1 && tid == 0 ) {
        //     DBGPRINT("tid %d, pack 3. memcpy( %p, %p, %lu ) => space %lu, _i %d, actual _i %d\n",
        //                                     tid, _destination_tmp, _source_tmp, (unsigned long)_end_loop->size, (unsigned long)(*(SPACE) - _i/nb_elements * _end_loop->size), _i/nb_elements, _i );
        // }
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
#if !defined (OPAL_DATATYPE_CUDA_DRY_RUN)
        *_destination_tmp = *_source_tmp;
#endif /* ! OPAL_DATATYPE_CUDA_DRY_RUN */
        _destination_tmp += num_threads;
    }
}