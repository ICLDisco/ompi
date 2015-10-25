#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#include "opal_datatype_cuda_internal.cuh"
#include <stdio.h> 
#include <time.h>

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
    _destination_tmp += tid;

    for (_i = tid; _i < copy_loops*nb_elements; _i+=num_threads) {
        _source_tmp = _src_disp_tmp + tid + _i/num_threads*num_threads + _i/nb_elements * gap;
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
        // if (_i % nb_elements == 0 ) {
        //     DBGPRINT("tid %d, pack 3. memcpy( %p, %p, %lu ) => _i %d, actual _i %d, count %d\n",
        //                                     tid, _destination_tmp, _source_tmp, (unsigned long)size, _i/nb_elements, _i, copy_loops );
        // }
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

// __global__ void opal_generic_simple_pack_cuda_iov_kernel( ddt_cuda_description_dist_t* desc_dist_d,
//                                                         dt_elem_desc_t* desc_d,
//                                                         uint32_t required_blocks, struct iovec* iov, unsigned char* pBaseBuf)
// {
//     uint32_t i;
//     dt_elem_desc_t* pElem;
//     unsigned char *conv_ptr, *iov_ptr;
//     uint32_t local_index, dst_offset, pos_desc, count_desc;
//     size_t iov_len_local;
//
//     iov_ptr = (unsigned char *) iov[0].iov_base;
//     iov_len_local = iov[0].iov_len;
//     conv_ptr = pBaseBuf;
//     for (i = 0; i < desc_dist_d[blockIdx.x].description_used; i++) {
//         pos_desc = desc_dist_d[blockIdx.x].description_index[i];
//         local_index = desc_dist_d[blockIdx.x].description_local_index[i];
//         dst_offset = desc_dist_d[blockIdx.x].dst_offset[i];
//         pElem = &(desc_d[pos_desc]);
//         count_desc = pElem->elem.count;
//
//   //      if ( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {
//             pack_predefined_data_cuda_kernel_v2(pElem, &count_desc, conv_ptr, iov_ptr, &iov_len_local, local_index, dst_offset);
// //        }
//     }
//
// }

__global__ void opal_generic_simple_pack_cuda_iov_kernel( ddt_cuda_iov_dist_t* cuda_iov_dist, int nb_blocks_used, unsigned char* source_base, unsigned char* destination_base)
{
    uint32_t i, _copy_count;
    size_t src_offset, dst_offset;
    uint8_t alignment;
    unsigned char *_source_tmp, *_destination_tmp;
    
    __shared__ uint32_t nb_tasks;
    
    if (threadIdx.x == 0) {
        //printf("iov pack kernel \n");
        nb_tasks = nb_blocks_used / gridDim.x;
        if (blockIdx.x < (nb_blocks_used % gridDim.x)) {
            nb_tasks ++;
        }
   //     printf("nb_tasks %d, griddim %d, nb_blocks_used %d, bloid %d \n", nb_tasks, gridDim.x, nb_blocks_used, blockIdx.x);
    }
    __syncthreads();
    
    for (i = 0; i < nb_tasks; i++) {
        src_offset = cuda_iov_dist[blockIdx.x + i * gridDim.x].src_offset;
        dst_offset = cuda_iov_dist[blockIdx.x + i * gridDim.x].dst_offset;
        _copy_count = cuda_iov_dist[blockIdx.x + i * gridDim.x].nb_elements;
        alignment = cuda_iov_dist[blockIdx.x + i * gridDim.x].element_alignment;
        
        // if (threadIdx.x == 0) {
        //     printf("block %d, ali %d, nb_element %d\n", blockIdx.x, cuda_iov_dist[blockIdx.x].element_alignment[i], _copy_count);
        // }
        
        if (threadIdx.x < _copy_count) {
            _source_tmp = source_base + src_offset + threadIdx.x * alignment;
            _destination_tmp = destination_base + dst_offset + threadIdx.x * alignment;
#if !defined (OPAL_DATATYPE_CUDA_DRY_RUN)
            if (alignment == ALIGNMENT_DOUBLE) {
                *((long *)_destination_tmp) = *((long *)_source_tmp);
            } else if (alignment == ALIGNMENT_FLOAT) {
                *((int *)_destination_tmp) = *((int *)_source_tmp);
            } else {
                * _destination_tmp = *_source_tmp;
            }
#endif /* ! OPAL_DATATYPE_CUDA_DRY_RUN */
        }
    }
}

__global__ void opal_empty_kernel(uint32_t copy_loops,
                                  size_t size,
                                  OPAL_PTRDIFF_TYPE extent,
                                  unsigned char* source,
                                  unsigned char* destination)
{
    
}

__global__ void opal_empty_kernel_noargs()
{
    
}
