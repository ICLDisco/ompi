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

__global__ void opal_generic_simple_pack_cuda_iov_non_cached_kernel( ddt_cuda_iov_dist_non_cached_t* cuda_iov_dist, int nb_blocks_used)
{
    uint32_t i, _copy_count;
    unsigned char *src, *dst;
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
        src = cuda_iov_dist[blockIdx.x + i * gridDim.x].src;
        dst = cuda_iov_dist[blockIdx.x + i * gridDim.x].dst;
        _copy_count = cuda_iov_dist[blockIdx.x + i * gridDim.x].nb_elements;
        alignment = cuda_iov_dist[blockIdx.x + i * gridDim.x].element_alignment;
        
        // if (threadIdx.x == 0) {
        //     printf("block %d, ali %d, nb_element %d\n", blockIdx.x, cuda_iov_dist[blockIdx.x].element_alignment[i], _copy_count);
        // }
        
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
#endif /* ! OPAL_DATATYPE_CUDA_DRY_RUN */
        }
    }
}

__global__ void opal_generic_simple_pack_cuda_iov_cached_kernel( ddt_cuda_iov_dist_cached_t* cuda_iov_dist, uint32_t cuda_iov_pos, uint32_t cuda_iov_count, uint32_t ddt_extent, uint32_t current_count, int nb_blocks_used, unsigned char* source_base, unsigned char* destination_base)
{
    uint32_t i, j;
    uint32_t _nb_bytes;    
    size_t src_offset, dst_offset;
    unsigned char *_source_tmp, *_destination_tmp;
    uint32_t current_cuda_iov_pos = cuda_iov_pos;
    size_t destination_disp = cuda_iov_dist[current_cuda_iov_pos].contig_disp;
    size_t contig_disp;
    uint32_t _my_cuda_iov_pos;
    uint32_t _my_cuda_iov_iteration;
    size_t ddt_size = cuda_iov_dist[cuda_iov_count].contig_disp;
    
    __shared__ uint32_t nb_tasks;
    uint32_t copy_count;
    uint8_t alignment;
    
    if (threadIdx.x == 0) {
        nb_tasks = nb_blocks_used / gridDim.x;
        if (blockIdx.x < (nb_blocks_used % gridDim.x)) {
            nb_tasks ++;
        }
    //    printf("cuda_iov_count %d, ddt_extent %d, current_count %d\n", cuda_iov_count, ddt_extent, current_count);
    //     printf("nb_tasks %d, griddim %d, nb_blocks_used %d, bloid %d \n", nb_tasks, gridDim.x, nb_blocks_used, blockIdx.x);
    }
    __syncthreads();
    
    for (i = 0; i < nb_tasks; i++) {
        /* these 3 variables are used multiple times, so put in in register */
        _my_cuda_iov_pos = (blockIdx.x + i * gridDim.x + current_cuda_iov_pos) % cuda_iov_count;
        _my_cuda_iov_iteration = (blockIdx.x + i * gridDim.x + current_cuda_iov_pos) / cuda_iov_count;
        contig_disp = cuda_iov_dist[_my_cuda_iov_pos].contig_disp;  
        
        src_offset = cuda_iov_dist[_my_cuda_iov_pos].ncontig_disp + (_my_cuda_iov_iteration + current_count) * ddt_extent;
        dst_offset = contig_disp + ddt_size * _my_cuda_iov_iteration - destination_disp;
        _nb_bytes = cuda_iov_dist[_my_cuda_iov_pos + 1].contig_disp - contig_disp;
        
        _source_tmp = source_base + src_offset;
        _destination_tmp = destination_base + dst_offset;
        /* block size is either multiple of ALIGNMENT_DOUBLE or residule */
        if ((uintptr_t)(_source_tmp) % ALIGNMENT_DOUBLE == 0 && (uintptr_t)(_destination_tmp) % ALIGNMENT_DOUBLE == 0 && _nb_bytes % ALIGNMENT_DOUBLE == 0) {
            alignment = ALIGNMENT_DOUBLE;
        } else if ((uintptr_t)(_source_tmp) % ALIGNMENT_FLOAT == 0 && (uintptr_t)(_destination_tmp) % ALIGNMENT_FLOAT == 0 && _nb_bytes % ALIGNMENT_FLOAT == 0) {
            alignment = ALIGNMENT_FLOAT;
        } else {
            alignment = ALIGNMENT_CHAR;
        }
        copy_count = _nb_bytes / alignment;
    /*    
        if (threadIdx.x == 0 && nb_tasks != 0) {
            printf("pack block %d, src_offset %ld, dst_offset %ld, count %d, nb_bytes %d, nb_tasks %d, i %d\n", blockIdx.x, src_offset, dst_offset, copy_count, _nb_bytes, nb_tasks, i);
        }
        __syncthreads();
      */  
        for (j = threadIdx.x; j < copy_count; j += blockDim.x) {
            if (j < copy_count) {
                _source_tmp = source_base + src_offset + j * alignment;
                _destination_tmp = destination_base + dst_offset + j * alignment;
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
}
