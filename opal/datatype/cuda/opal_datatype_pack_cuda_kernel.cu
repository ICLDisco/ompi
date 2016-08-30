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
    uint32_t i, u, tid, num_threads, warp_id, tid_per_warp, nb_warps, nb_warps_x, nb_warps_y, pos_x, pos_y, size_last_y, size_last_x;
    uint32_t size_nb, extent_nb;
    uint64_t *_source_tmp, *_destination_tmp, *source_64, *destination_64, *_source_left_tmp, *_destination_left_tmp;
    uint64_t val[UNROLL_16];
    
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    num_threads = gridDim.x * blockDim.x;
    warp_id = tid / CUDA_WARP_SIZE;
    tid_per_warp = threadIdx.x & (CUDA_WARP_SIZE-1);
    nb_warps = num_threads / CUDA_WARP_SIZE;
    
    extent_nb = extent / 8;
    size_nb = size / 8;
    source_64 = (uint64_t*)source;
    destination_64 = (uint64_t*)destination;
    
    nb_warps_x = size_nb / CUDA_WARP_SIZE;
    size_last_x = size_nb & (CUDA_WARP_SIZE-1);
    if ( size_last_x != 0) {
        nb_warps_x ++;
    } else {
        size_last_x = CUDA_WARP_SIZE;
    }
    nb_warps_y = copy_loops / UNROLL_16;
    size_last_y = copy_loops & (UNROLL_16-1);
    if ( size_last_y != 0) {
        nb_warps_y ++;
    } else {
        size_last_y = UNROLL_16;
    }
    // if (threadIdx.x == 0) {
    //     printf("warp_id %u, nb_warps_x %u, nb_warps_y %u, tid_per_warps %u, nb_warps %u\n", warp_id, nb_warps_x, nb_warps_y, tid_per_warp, nb_warps);
    // }
    
    const uint32_t extent_nb_times_UNROLL_16 =  extent_nb * UNROLL_16;
    const uint32_t size_nb_times_UNROLL_16 = size_nb * UNROLL_16;
    source_64 += tid_per_warp;
    destination_64 += tid_per_warp;
    
    for (i = warp_id; i < (nb_warps_x-1) * (nb_warps_y-1); i += nb_warps) {
        pos_x = i / (nb_warps_y-1);
        pos_y = i % (nb_warps_y-1);
        _source_tmp = source_64 + pos_y * extent_nb_times_UNROLL_16 + pos_x * CUDA_WARP_SIZE;
        _destination_tmp = destination_64 + pos_y * size_nb_times_UNROLL_16 + pos_x * CUDA_WARP_SIZE;
        #pragma unroll
        for (u = 0; u < UNROLL_16; u++) {
            val[u] = *(_source_tmp + u * extent_nb);
        }
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL_16; u++) {
            *(_destination_tmp + u * size_nb) = val[u];
        }
    }
    if (tid_per_warp < size_last_x) {
        pos_x = nb_warps_x - 1;
        _source_left_tmp = source_64 + pos_x * CUDA_WARP_SIZE;
        _destination_left_tmp = destination_64 + pos_x * CUDA_WARP_SIZE;
        for (i = warp_id; i < nb_warps_y-1; i += nb_warps) {
            _source_tmp = _source_left_tmp + i * extent_nb_times_UNROLL_16;
            _destination_tmp = _destination_left_tmp + i * size_nb_times_UNROLL_16;
            #pragma unroll
            for (u = 0; u < UNROLL_16; u++) {
                val[u] = *(_source_tmp + u * extent_nb);
            }
            #pragma unroll
            for (uint32_t u = 0; u < UNROLL_16; u++) {
                *(_destination_tmp + u * size_nb) = val[u];
            }
        }
    }
    
    pos_y = nb_warps_y - 1;
    _source_left_tmp = source_64 + pos_y * extent_nb_times_UNROLL_16;
    _destination_left_tmp = destination_64 + pos_y * size_nb_times_UNROLL_16;
    if (size_last_y == UNROLL_16) {
        for (i = warp_id; i < nb_warps_x-1; i += nb_warps) {
            _source_tmp = _source_left_tmp + i * CUDA_WARP_SIZE;
            _destination_tmp = _destination_left_tmp + i * CUDA_WARP_SIZE;
            #pragma unroll
            for (u = 0; u < UNROLL_16; u++) {
                val[u] = *(_source_tmp + u * extent_nb);
            }
            #pragma unroll
            for (uint32_t u = 0; u < UNROLL_16; u++) {
                *(_destination_tmp + u * size_nb) = val[u];
            }  
        } 
    } else {
        for (i = warp_id; i < nb_warps_x-1; i += nb_warps) {
            _source_tmp = _source_left_tmp + i * CUDA_WARP_SIZE;
            _destination_tmp = _destination_left_tmp + i * CUDA_WARP_SIZE;
            for (u = 0; u < size_last_y; u++) {
                *(_destination_tmp + u * size_nb) = *(_source_tmp + u * extent_nb);
            }
        }
    }
    
    if (warp_id == 0 && tid_per_warp < size_last_x) {
        _source_tmp = source_64 + (nb_warps_y-1) * extent_nb_times_UNROLL_16 + (nb_warps_x-1) * CUDA_WARP_SIZE;
        _destination_tmp = destination_64 + (nb_warps_y-1) * size_nb_times_UNROLL_16 + (nb_warps_x-1) * CUDA_WARP_SIZE;
        for (u = 0; u < size_last_y; u++) {
            *(_destination_tmp + u * size_nb) = *(_source_tmp + u * extent_nb);
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
        if (blockIdx.x < (nb_blocks_used % gridDim.x)) {
            nb_tasks_per_block ++;
        }
        WARP_SIZE = 32;
        nb_warp_per_block = blockDim.x / WARP_SIZE;
 //       nb_warp_per_block = 1;
     //   if (nb_tasks_per_block == )
    //    printf("cuda_iov_count %d, ddt_extent %d, current_count %d\n", cuda_iov_count, ddt_extent, current_count);
    //     printf("nb_tasks %d, griddim %d, nb_blocks_used %d, bloid %d \n", nb_tasks, gridDim.x, nb_blocks_used, blockIdx.x);
    }
    __syncthreads();
      
      const uint32_t warp_id_per_block = threadIdx.x / WARP_SIZE;
      const uint32_t tid_per_warp = threadIdx.x & (WARP_SIZE - 1);
 //     uint32_t warp_id_per_block = 0;
 //     uint32_t tid_per_warp = threadIdx.x;  
    
    for (i = warp_id_per_block; i < nb_tasks_per_block; i+= nb_warp_per_block) {
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
        
        //alignment = ALIGNMENT_DOUBLE;
        copy_count = _nb_bytes / alignment;
    /*    
        if (threadIdx.x == 0 && nb_tasks != 0) {
            printf("pack block %d, src_offset %ld, dst_offset %ld, count %d, nb_bytes %d, nb_tasks %d, i %d\n", blockIdx.x, src_offset, dst_offset, copy_count, _nb_bytes, nb_tasks, i);
        }
        __syncthreads();
      */
       /* if (threadIdx.x == 0){
            printf("bytes %d, copy count %d, alignment %d, task %d, nb_block_used %d\n", _nb_bytes, copy_count, alignment, i, nb_blocks_used);
        } */
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