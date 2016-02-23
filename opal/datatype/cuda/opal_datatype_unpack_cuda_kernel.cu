#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#include "opal_datatype_cuda_internal.cuh"
#include <cuda.h>
#include <stdio.h> 


__global__ void opal_generic_simple_unpack_cuda_iov_non_cached_kernel( ddt_cuda_iov_dist_non_cached_t* cuda_iov_dist, int nb_blocks_used)
{
    uint32_t i, _copy_count;
    unsigned char *src, *dst;
    uint8_t alignment;
    unsigned char *_source_tmp, *_destination_tmp;
    
    __shared__ uint32_t nb_tasks;
    
    if (threadIdx.x == 0) {
        nb_tasks = nb_blocks_used / gridDim.x;
        if (blockIdx.x < nb_blocks_used % gridDim.x) {
            nb_tasks ++;
        }
    }
    __syncthreads();
    
    for (i = 0; i < nb_tasks; i++) {
        src = cuda_iov_dist[blockIdx.x + i * gridDim.x].src;
        dst = cuda_iov_dist[blockIdx.x + i * gridDim.x].dst;
        _copy_count = cuda_iov_dist[blockIdx.x + i * gridDim.x].nb_elements;
        alignment = cuda_iov_dist[blockIdx.x + i * gridDim.x].element_alignment;
        
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

#if 0
__global__ void opal_generic_simple_unpack_cuda_iov_cached_kernel( ddt_cuda_iov_dist_cached_t* cuda_iov_dist, uint32_t cuda_iov_pos, uint32_t cuda_iov_count, uint32_t ddt_extent, uint32_t current_count, int nb_blocks_used, unsigned char* destination_base, unsigned char* source_base, size_t cuda_iov_partial_length_start, size_t cuda_iov_partial_length_end)
{
    uint32_t i, j;
    size_t dst_offset, src_offset;
    unsigned char *_source_tmp, *_destination_tmp;
    uint32_t _nb_bytes;
    uint32_t current_cuda_iov_pos = cuda_iov_pos;
    size_t source_disp = cuda_iov_dist[current_cuda_iov_pos].contig_disp;
    size_t source_partial_disp = 0;
    size_t contig_disp; 
    uint32_t _my_cuda_iov_pos;
    uint32_t _my_cuda_iov_iteration;
    size_t ddt_size = cuda_iov_dist[cuda_iov_count].contig_disp;

    __shared__ uint32_t nb_tasks;
    uint32_t copy_count;
    uint8_t alignment;
    
    if (threadIdx.x == 0) {
        nb_tasks = nb_blocks_used / gridDim.x;
        if (blockIdx.x < nb_blocks_used % gridDim.x) {
            nb_tasks ++;
        }
     //   printf("cuda_iov_count %d, ddt_extent %d, current_count %d, ddt_size %d\n", cuda_iov_count, ddt_extent, current_count, ddt_size);
    }
    __syncthreads();
    
    if (cuda_iov_partial_length_start != 0) {
        source_partial_disp = (cuda_iov_dist[current_cuda_iov_pos+1].contig_disp - cuda_iov_dist[current_cuda_iov_pos].contig_disp) - cuda_iov_partial_length_start;
    }
    
    for (i = 0; i < nb_tasks; i++) {
        /* these 3 variables are used multiple times, so put in in register */
        _my_cuda_iov_pos = (blockIdx.x + i * gridDim.x + current_cuda_iov_pos) % cuda_iov_count;
        _my_cuda_iov_iteration = (blockIdx.x + i * gridDim.x + current_cuda_iov_pos) / cuda_iov_count;
        contig_disp = cuda_iov_dist[_my_cuda_iov_pos].contig_disp; 
        
        src_offset = contig_disp + ddt_size * _my_cuda_iov_iteration - source_disp - source_partial_disp;
        dst_offset = cuda_iov_dist[_my_cuda_iov_pos].ncontig_disp + (_my_cuda_iov_iteration + current_count) * ddt_extent;
        _nb_bytes = cuda_iov_dist[_my_cuda_iov_pos + 1].contig_disp - contig_disp;

        if (i == 0 && blockIdx.x == 0 && cuda_iov_partial_length_start != 0) {
            src_offset = contig_disp + ddt_size * _my_cuda_iov_iteration - source_disp;
            dst_offset = dst_offset + _nb_bytes - cuda_iov_partial_length_start;  
            _nb_bytes = cuda_iov_partial_length_start;
        } else if (i == nb_tasks-1 && (blockIdx.x == (nb_blocks_used-1) % gridDim.x) && cuda_iov_partial_length_end != 0) {
            _nb_bytes = cuda_iov_partial_length_end;
        }
        
        _destination_tmp = destination_base + dst_offset; 
        _source_tmp = source_base + src_offset;
        if ((uintptr_t)(_destination_tmp) % ALIGNMENT_DOUBLE == 0 && (uintptr_t)(_source_tmp) % ALIGNMENT_DOUBLE == 0 && _nb_bytes % ALIGNMENT_DOUBLE == 0) {
            alignment = ALIGNMENT_DOUBLE;
        } else if ((uintptr_t)(_destination_tmp) % ALIGNMENT_FLOAT == 0 && (uintptr_t)(_source_tmp) % ALIGNMENT_FLOAT == 0 && _nb_bytes % ALIGNMENT_FLOAT == 0) {
            alignment = ALIGNMENT_FLOAT;
        } else {
            alignment = ALIGNMENT_CHAR;
        }
        copy_count = _nb_bytes / alignment;
   /*     
        if (threadIdx.x == 0 && nb_tasks != 0) {
            printf("unpack block %d, src_offset %ld, dst_offset %ld, count %d, nb_bytes %d, nb_tasks %d, i %d\n", blockIdx.x, src_offset, dst_offset, copy_count, _nb_bytes, nb_tasks, i);
        }
        __syncthreads();
     */   
        for (j = threadIdx.x; j < copy_count; j += blockDim.x) {
/*            if (threadIdx.x == 0) {
                if (copy_count > blockDim.x) printf("copy_count %d, dim %d\n", copy_count, blockDim.x);
            }*/
            if (j < copy_count) {
                _source_tmp = source_base + src_offset + j * alignment;
                _destination_tmp = destination_base + dst_offset + j * alignment;
  /*              if (threadIdx.x == 0) {
                    printf("_src %p, dst %p, alignment %d, blk %d, j %d, count %d\n", _source_tmp, _destination_tmp, alignment, blockIdx.x, j, copy_count);
                }*/
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
}

#else 
__global__ void opal_generic_simple_unpack_cuda_iov_cached_kernel( ddt_cuda_iov_dist_cached_t* cuda_iov_dist, uint32_t cuda_iov_pos, uint32_t cuda_iov_count, uint32_t ddt_extent, uint32_t current_count, int nb_blocks_used, unsigned char* destination_base, unsigned char* source_base, size_t cuda_iov_partial_length_start, size_t cuda_iov_partial_length_end)
{
    uint32_t i, j;
    size_t dst_offset, src_offset;
    unsigned char *_source_tmp, *_destination_tmp;
    uint32_t _nb_bytes;
    uint32_t current_cuda_iov_pos = cuda_iov_pos;
    size_t source_disp = cuda_iov_dist[current_cuda_iov_pos].contig_disp;
    size_t source_partial_disp = 0;
    size_t contig_disp; 
    uint32_t _my_cuda_iov_pos;
    uint32_t _my_cuda_iov_iteration;
    size_t ddt_size = cuda_iov_dist[cuda_iov_count].contig_disp;

    __shared__ uint32_t nb_tasks_per_block;
    __shared__ uint32_t WARP_SIZE;
    __shared__ uint32_t nb_warp_per_block;
    uint32_t copy_count;
    uint8_t alignment;
    uint64_t tmp_var_64[KERNEL_UNROLL];
    uint32_t tmp_var_32[KERNEL_UNROLL];
    unsigned char tmp_var_8[KERNEL_UNROLL];
    uint32_t u, k;
    uint32_t copy_count_16, copy_count_8, copy_count_left;
    
    if (threadIdx.x == 0) {
        nb_tasks_per_block = nb_blocks_used / gridDim.x;
        if (blockIdx.x < nb_blocks_used % gridDim.x) {
            nb_tasks_per_block ++;
        }
        if (nb_tasks_per_block >= 4) {
            WARP_SIZE = 32;
        } else if (nb_tasks_per_block == 1) {
            WARP_SIZE = 32;//blockDim.x;
        } else {
            WARP_SIZE = 32;
        }
        nb_warp_per_block = blockDim.x / WARP_SIZE;
     //   printf("cuda_iov_count %d, ddt_extent %d, current_count %d, ddt_size %d\n", cuda_iov_count, ddt_extent, current_count, ddt_size);
    }
    __syncthreads();
    
    const uint32_t warp_id_per_block = threadIdx.x / WARP_SIZE;
    const uint32_t tid_per_warp = threadIdx.x & (WARP_SIZE - 1);
    
    if (cuda_iov_partial_length_start != 0) {
        source_partial_disp = (cuda_iov_dist[current_cuda_iov_pos+1].contig_disp - cuda_iov_dist[current_cuda_iov_pos].contig_disp) - cuda_iov_partial_length_start;
    }
    
    for (i = warp_id_per_block; i < nb_tasks_per_block; i+= nb_warp_per_block) {
        /* these 3 variables are used multiple times, so put in in register */
        _my_cuda_iov_pos = (blockIdx.x + i * gridDim.x + current_cuda_iov_pos) % cuda_iov_count;
        _my_cuda_iov_iteration = (blockIdx.x + i * gridDim.x + current_cuda_iov_pos) / cuda_iov_count;
        contig_disp = cuda_iov_dist[_my_cuda_iov_pos].contig_disp; 
        
        src_offset = contig_disp + ddt_size * _my_cuda_iov_iteration - source_disp - source_partial_disp;
        dst_offset = cuda_iov_dist[_my_cuda_iov_pos].ncontig_disp + (_my_cuda_iov_iteration + current_count) * ddt_extent;
        _nb_bytes = cuda_iov_dist[_my_cuda_iov_pos + 1].contig_disp - contig_disp;

        if (i == 0 && blockIdx.x == 0 && cuda_iov_partial_length_start != 0) {
            src_offset = contig_disp + ddt_size * _my_cuda_iov_iteration - source_disp;
            dst_offset = dst_offset + _nb_bytes - cuda_iov_partial_length_start;  
            _nb_bytes = cuda_iov_partial_length_start;
        } else if (i == nb_tasks_per_block-1 && (blockIdx.x == (nb_blocks_used-1) % gridDim.x) && cuda_iov_partial_length_end != 0) {
            _nb_bytes = cuda_iov_partial_length_end;
        }
        
        _destination_tmp = destination_base + dst_offset; 
        _source_tmp = source_base + src_offset;
        if ((uintptr_t)(_destination_tmp) % ALIGNMENT_DOUBLE == 0 && (uintptr_t)(_source_tmp) % ALIGNMENT_DOUBLE == 0 && _nb_bytes % ALIGNMENT_DOUBLE == 0) {
            alignment = ALIGNMENT_DOUBLE;
        } else if ((uintptr_t)(_destination_tmp) % ALIGNMENT_FLOAT == 0 && (uintptr_t)(_source_tmp) % ALIGNMENT_FLOAT == 0 && _nb_bytes % ALIGNMENT_FLOAT == 0) {
            alignment = ALIGNMENT_FLOAT;
        } else {
            alignment = ALIGNMENT_CHAR;
        }
        
        copy_count = _nb_bytes / alignment;
   /*     
        if (threadIdx.x == 0 && nb_tasks != 0) {
            printf("unpack block %d, src_offset %ld, dst_offset %ld, count %d, nb_bytes %d, nb_tasks %d, i %d\n", blockIdx.x, src_offset, dst_offset, copy_count, _nb_bytes, nb_tasks, i);
        }
        __syncthreads();
     */   
        if (alignment == ALIGNMENT_DOUBLE) {
            uint64_t *_source_base_64, *_destination_base_64; 
            copy_count_16 = copy_count  / (WARP_SIZE * UNROLL_16) * (WARP_SIZE * UNROLL_16);
            _source_base_64 = (uint64_t *)(source_base + src_offset);
            _destination_base_64 = (uint64_t *)(destination_base + dst_offset);
            if (copy_count_16 > 0) {
                for (k = 0; k < copy_count_16; k += UNROLL_16 * WARP_SIZE) {
                    #pragma unroll
                    for (u = 0; u < UNROLL_16; u++) {
                        j = tid_per_warp + u * WARP_SIZE + k;
                        tmp_var_64[u] = *(_source_base_64 + j);

                    }
                    #pragma unroll
                    for (u = 0; u < UNROLL_16; u++) {
                        j = tid_per_warp + u * WARP_SIZE + k;
                        *(_destination_base_64 + j) = tmp_var_64[u];

                    }
                }
            }
            _source_base_64 += copy_count_16;
            _destination_base_64 += copy_count_16;
            
            copy_count_8 = (copy_count - copy_count_16) / (WARP_SIZE * UNROLL_8) * (WARP_SIZE * UNROLL_8);
            if (copy_count_8 > 0) {
                for (k = 0; k < copy_count_8; k += UNROLL_8 * WARP_SIZE) {
                    #pragma unroll
                    for (u = 0; u < UNROLL_8; u++) {
                        j = tid_per_warp + u * WARP_SIZE + k;
                        tmp_var_64[u] = *(_source_base_64 + j);

                    }
                    #pragma unroll
                    for (u = 0; u < UNROLL_8; u++) {
                        j = tid_per_warp + u * WARP_SIZE + k;
                        *(_destination_base_64 + j) = tmp_var_64[u];

                    }
                }
            }
            _source_base_64 += copy_count_8;
            _destination_base_64 += copy_count_8;
        
            copy_count_left = copy_count - copy_count_16 - copy_count_8;
            if (copy_count_left > 0) {
                #pragma unroll
                for (u = 0; u < UNROLL_8; u++) {
                    j = tid_per_warp + u * WARP_SIZE;
                    if (j < copy_count_left) {
                        tmp_var_64[u] = *(_source_base_64 + j);
                    } else {
                        break;
                    }
                }
                #pragma unroll
                for (u = 0; u < UNROLL_8; u++) {
                    j = tid_per_warp + u * WARP_SIZE;
                    if (j < copy_count_left) {
                        *(_destination_base_64 + j) = tmp_var_64[u];
                    } else {
                        break;
                    }
                }
            }
        } else if (alignment == ALIGNMENT_FLOAT) {
            uint32_t *_source_base_32, *_destination_base_32;    
            copy_count_16 = copy_count  / (WARP_SIZE * UNROLL_16) * (WARP_SIZE * UNROLL_16);
            _source_base_32 = (uint32_t *)(source_base + src_offset);
            _destination_base_32 = (uint32_t *)(destination_base + dst_offset);
            if (copy_count_16 > 0) {
                for (k = 0; k < copy_count_16; k += UNROLL_16 * WARP_SIZE) {
                    #pragma unroll
                    for (u = 0; u < UNROLL_16; u++) {
                        j = tid_per_warp + u * WARP_SIZE + k;
                        tmp_var_32[u] = *(_source_base_32 + j);

                    }
                    #pragma unroll
                    for (u = 0; u < UNROLL_16; u++) {
                        j = tid_per_warp + u * WARP_SIZE + k;
                        *(_destination_base_32 + j) = tmp_var_32[u];

                    }
                }
            }
            _source_base_32 += copy_count_16;
            _destination_base_32 += copy_count_16;
        
            copy_count_8 = (copy_count - copy_count_16) / (WARP_SIZE * UNROLL_8) * (WARP_SIZE * UNROLL_8);
            if (copy_count_8 > 0) {
                for (k = 0; k < copy_count_8; k += UNROLL_8 * WARP_SIZE) {
                    #pragma unroll
                    for (u = 0; u < UNROLL_8; u++) {
                        j = tid_per_warp + u * WARP_SIZE + k;
                        tmp_var_32[u] = *(_source_base_32 + j);

                    }
                    #pragma unroll
                    for (u = 0; u < UNROLL_8; u++) {
                        j = tid_per_warp + u * WARP_SIZE + k;
                        *(_destination_base_32 + j) = tmp_var_32[u];

                    }
                }
            }
            _source_base_32 += copy_count_8;
            _destination_base_32 += copy_count_8;
        
            copy_count_left = copy_count - copy_count_16 - copy_count_8;
            if (copy_count_left > 0) {
                #pragma unroll
                for (u = 0; u < UNROLL_8; u++) {
                    j = tid_per_warp + u * WARP_SIZE;
                    if (j < copy_count_left) {
                        tmp_var_32[u] = *(_source_base_32 + j);
                    } else {
                        break;
                    }
                }
                #pragma unroll
                for (u = 0; u < UNROLL_8; u++) {
                    j = tid_per_warp + u * WARP_SIZE;
                    if (j < copy_count_left) {
                        *(_destination_base_32 + j) = tmp_var_32[u];
                    } else {
                        break;
                    }
                }
            }
        } else {
            unsigned char *_source_base_8, *_destination_base_8;
        
            copy_count_16 = copy_count  / (WARP_SIZE * UNROLL_16) * (WARP_SIZE * UNROLL_16);
            _source_base_8 = (unsigned char *)(source_base + src_offset);
            _destination_base_8 = (unsigned char *)(destination_base + dst_offset);
            if (copy_count_16 > 0) {
                for (k = 0; k < copy_count_16; k += UNROLL_16 * WARP_SIZE) {
                    #pragma unroll
                    for (u = 0; u < UNROLL_16; u++) {
                        j = tid_per_warp + u * WARP_SIZE + k;
                        tmp_var_8[u] = *(_source_base_8 + j);

                    }
                    #pragma unroll
                    for (u = 0; u < UNROLL_16; u++) {
                        j = tid_per_warp + u * WARP_SIZE + k;
                        *(_destination_base_8 + j) = tmp_var_8[u];

                    }
                }
            }
            _source_base_8 += copy_count_16;
            _destination_base_8 += copy_count_16;
        
            copy_count_8 = (copy_count - copy_count_16) / (WARP_SIZE * UNROLL_8) * (WARP_SIZE * UNROLL_8);
            if (copy_count_8 > 0) {
                for (k = 0; k < copy_count_8; k += UNROLL_8 * WARP_SIZE) {
                    #pragma unroll
                    for (u = 0; u < UNROLL_8; u++) {
                        j = tid_per_warp + u * WARP_SIZE + k;
                        tmp_var_8[u] = *(_source_base_8 + j);

                    }
                    #pragma unroll
                    for (u = 0; u < UNROLL_8; u++) {
                        j = tid_per_warp + u * WARP_SIZE + k;
                        *(_destination_base_8 + j) = tmp_var_8[u];

                    }
                }
            }
            _source_base_8 += copy_count_8;
            _destination_base_8 += copy_count_8;
        
            copy_count_left = copy_count - copy_count_16 - copy_count_8;
            if (copy_count_left > 0) {
                #pragma unroll
                for (u = 0; u < UNROLL_8; u++) {
                    j = tid_per_warp + u * WARP_SIZE;
                    if (j < copy_count_left) {
                        tmp_var_8[u] = *(_source_base_8 + j);
                    } else {
                        break;
                    }
                }
                #pragma unroll
                for (u = 0; u < UNROLL_8; u++) {
                    j = tid_per_warp + u * WARP_SIZE;
                    if (j < copy_count_left) {
                        *(_destination_base_8 + j) = tmp_var_8[u];
                    } else {
                        break;
                    }
                }
            }
        }
    }
}

#endif

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
