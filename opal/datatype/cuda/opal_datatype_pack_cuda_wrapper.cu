#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"

#include <stdio.h>

int32_t opal_generic_simple_pack_function_cuda( opal_convertor_t* pConvertor,
                                                struct iovec* iov, 
                                                uint32_t* out_size,
                                                size_t* max_data )
{
    uint32_t i;
    dt_elem_desc_t* description;
    const opal_datatype_t *pData = pConvertor->pDesc;
    uint32_t tasks_per_block, num_blocks;
    dt_stack_t* pStack;
    
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
    
    cudaMemcpy(cuda_desc_d, cuda_desc_h, sizeof(ddt_cuda_desc_t), cudaMemcpyHostToDevice);
    
    pStack = pConvertor->pStack + pConvertor->stack_pos;
    tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
    num_blocks = ((uint32_t)pStack->count + tasks_per_block - 1) / tasks_per_block;
    printf("launch kernel, count %d, num_blocks %d, total threads %d\n", (uint32_t)pStack->count, num_blocks, num_blocks*2*THREAD_PER_BLOCK);
    opal_generic_simple_pack_cuda_kernel<<<192,4*THREAD_PER_BLOCK>>>(cuda_desc_d);
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

    printf("I am in pack_contiguous_loop_cuda\n");

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)
    _source = pBaseBuf_GPU;
    _destination = (unsigned char*)cuda_desc_h->iov[0].iov_base;
#endif
    
    tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
    num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;
    pack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK>>>(_copy_loops, _end_loop->size, _loop->extent, _source, _destination);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)    
    *(SOURCE) = _source +  _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(DESTINATION) = *(DESTINATION)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    
    cudaDeviceSynchronize();
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
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _destination = *(DESTINATION);

    _copy_blength = 8;//opal_datatype_basicDatatypes[_elem->common.type]->size;
    if( (_copy_count * _copy_blength) > *(SPACE) ) {
        _copy_count = (uint32_t)(*(SPACE) / _copy_blength);
        if( 0 == _copy_count ) return;  /* nothing to do */
    }
    
#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)
    _source = pBaseBuf_GPU;
    _destination = (unsigned char*)cuda_desc_h->iov[0].iov_base;
#endif
    
    tasks_per_block = THREAD_PER_BLOCK*4;
    num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;

    DBGPRINT("num_blocks %d, thread %d\n", num_blocks, tasks_per_block);
    DBGPRINT( "GPU pack 1. memcpy( %p, %p, %lu ) => space %lu\n", _destination, _source, (unsigned long)_copy_count, (unsigned long)(*(SPACE)) );
    
    pack_contiguous_loop_cuda_kernel_global<<<1, THREAD_PER_BLOCK, 0, cuda_streams->opal_cuda_stream[cuda_streams->current_stream_id]>>>(_copy_count, _copy_blength, _elem->extent, _source, _destination);
    cuda_streams->current_stream_id ++;
    cuda_streams->current_stream_id = cuda_streams->current_stream_id % NB_STREAMS;
    
#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)  
    _copy_blength *= _copy_count;
    *(SOURCE)  = _source + _elem->extent*_copy_count - _elem->disp;
    *(DESTINATION) += _copy_blength;
    *(SPACE)  -= _copy_blength;
    *(COUNT)  -= _copy_count;
#endif
    
    pBaseBuf_GPU += _elem->extent*_copy_count;
    cuda_desc_h->iov[0].iov_base = (unsigned char*)cuda_desc_h->iov[0].iov_base + _copy_blength;
 //   cudaDeviceSynchronize();
}

