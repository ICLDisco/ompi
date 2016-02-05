#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#include "opal_datatype_cuda_internal.cuh"
#include <stdio.h> 
#include <time.h>

#if 0
__global__ void pack_contiguous_loop_cuda_kernel_global( uint32_t copy_loops,
                                                         size_t size,
                                                         OPAL_PTRDIFF_TYPE extent,
                                                         unsigned char* source,
                                                         unsigned char* destination )
{
    uint32_t _i, tid, num_threads;
    uint32_t gap, nb_elements;
    uint64_t *_source_tmp, *_destination_tmp, *_src_disp_tmp;;
    
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    num_threads = gridDim.x * blockDim.x;
    
    gap = (extent - size) / 8;
    nb_elements = size / 8;
    _src_disp_tmp = (uint64_t*)source;
    _destination_tmp = (uint64_t*)destination;
    _destination_tmp += tid;
#if 0
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
#else
    for (_i = tid; _i < copy_loops*nb_elements; _i+=8*num_threads) {
        uint64_t val[16];
        uint32_t _j;
        uint32_t u;
        uint64_t *mysrc = _src_disp_tmp + tid;
        
        #pragma unroll      
        for (u = 0; u < 8; u++) {
            _j = _i + u * num_threads;
            val[u] = *(mysrc + _j/num_threads*num_threads + _j/nb_elements * gap);
        } 
        
        #pragma unroll
        for (u = 0; u < 8; u++) {
            *_destination_tmp = val[u];
            _destination_tmp += num_threads;
        } 
/*
        _j = _i;
        val[0] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

        _j += num_threads;
        val[1] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

        _j += num_threads;
        val[2] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

        _j += num_threads;
        val[3] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);
        
	_j += num_threads;
        val[4] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

	_j += num_threads;
        val[5] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

	_j += num_threads;
        val[6] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

	_j += num_threads;
        val[7] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

	_j += num_threads;
        val[8] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

	_j += num_threads;
        val[9] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

	_j += num_threads;
        val[10] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

        _j += num_threads;
        val[11] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

        _j += num_threads;
        val[12] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

        _j += num_threads;
        val[13] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

        _j += num_threads;
        val[14] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

        _j += num_threads;
        val[15] = *(_src_disp_tmp + tid + _j/num_threads*num_threads + _j/nb_elements * gap);

        *_destination_tmp = val[0];
        _destination_tmp += num_threads;
        *_destination_tmp = val[1];
        _destination_tmp += num_threads;
        *_destination_tmp = val[2];
        _destination_tmp += num_threads;
        *_destination_tmp = val[3];
        _destination_tmp += num_threads;
        *_destination_tmp = val[4];
        _destination_tmp += num_threads;
        *_destination_tmp = val[5];
        _destination_tmp += num_threads;
        *_destination_tmp = val[6];
        _destination_tmp += num_threads;
        *_destination_tmp = val[7];
        _destination_tmp += num_threads;
        *_destination_tmp = val[8];
        _destination_tmp += num_threads;
        *_destination_tmp = val[9];
        _destination_tmp += num_threads;
        *_destination_tmp = val[10];
        _destination_tmp += num_threads;
        *_destination_tmp = val[11];
        _destination_tmp += num_threads;
        *_destination_tmp = val[12];
        _destination_tmp += num_threads;
        *_destination_tmp = val[13];
        _destination_tmp += num_threads;
        *_destination_tmp = val[14];
        _destination_tmp += num_threads;
        *_destination_tmp = val[15];
        _destination_tmp += num_threads;
*/  
    }
#endif
}

#else

#define SEG_ADD(s) \
    l += s; \
    while (l >= lines) { \
	l -= lines; \
	c += width; \
    }

__global__ void pack_contiguous_loop_cuda_kernel_global( uint32_t lines,
                                                         size_t nb_size,
                                                         OPAL_PTRDIFF_TYPE nb_extent,
                                                         unsigned char * b_source,
                                                         unsigned char * b_destination )
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
  
    //size_t lines = (size_t)lines;
    size_t size = nb_size / 8;
    size_t extent = nb_extent / 8;
    uint64_t * source = (uint64_t *) b_source;
    uint64_t *destination = (uint64_t *) b_destination;
    uint64_t val[KERNEL_UNROLL];
    
    int col = 0;
    for (int width = 32; width > 0 && col < size; width >>= 1) {
    	while (size-col >= width) {
    	    const int warp_id = tid / width;
    	    const int warp_tid = tid & (width-1);
    	    const int warp_nb = num_threads / width;
    	    const int c = col + warp_tid;
            int l = warp_id * KERNEL_UNROLL;
    	    uint64_t *src = source + c;
    	    uint64_t *dst = destination + c;
    	    for (int b=0; b<lines/(KERNEL_UNROLL*warp_nb); b++) {
    		    #pragma unroll
    		    for (int u=0; u<KERNEL_UNROLL; u++) {
    		        val[u] = *(src+(l+u)*extent);
    		    }
    		    #pragma unroll
    		    for (int u=0; u<KERNEL_UNROLL; u++) {
    		        dst[(l+u)*size] = val[u];
    		    }
    		    l += warp_nb * KERNEL_UNROLL;
    	    }
    	    /* Finish non-unrollable case */
    	    for (int u=0; u<KERNEL_UNROLL && l<lines; u++) {
    		    dst[l*size] = *(src+l*extent);
    		    l++;
    	    }		
    	    col += width;
    	}
    }

    
}

/*
#define COLOFF_INC(jump, width, ext) \
     col += jump; \
     off += jump; \
     while (col >= width) { \
         col -= width; \
         off += ext - width; \
     }

#define ELEMSIZE 32

__global__ void pack_contiguous_loop_cuda_kernel_global( uint32_t
copy_loops,
size_t size,
OPAL_PTRDIFF_TYPE extent,
unsigned char * source,
unsigned char * destination )
{
     uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x ;
     uint32_t num_threads = gridDim.x * blockDim.x;

     int col = 0;
     int off = 0;

     COLOFF_INC(tid, size/ELEMSIZE, extent/ELEMSIZE);

     if (ELEMSIZE % 8 == 0) {
         volatile uint64_t * __restrict__ dst = (uint64_t*)destination +
tid * ELEMSIZE/8;
         for (int offset = tid; offset < copy_loops*size/ELEMSIZE;
offset+=num_threads) {
             const volatile uint64_t * __restrict__ src = (uint64_t*)source + off * ELEMSIZE/8;
#if 1
             uint64_t val[ELEMSIZE/8];
             #pragma unroll
             for (int i = 0; i < ELEMSIZE/8; i++) {
                 val[i] = src[i];
             }
             #pragma unroll
             for (int i = 0; i < ELEMSIZE/8; i++) {
                 dst[i] = val[i];
             }
#else
             #pragma unroll
             for (int i = 0; i < ELEMSIZE/8; i++) {
                 dst[i] = __ldg(src+i);
             }
#endif
             dst += num_threads*ELEMSIZE/8;
             COLOFF_INC(num_threads, size/ELEMSIZE, extent/ELEMSIZE);
         }
     }
}
*/
#endif


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

#if 0
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

#else
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
        if (nb_tasks_per_block >= 4) {
            WARP_SIZE = 32;
        } else if (nb_tasks_per_block == 1) {
            WARP_SIZE = blockDim.x;
        } else {
            WARP_SIZE = 64;
        }
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
        
     //   alignment = ALIGNMENT_DOUBLE;
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
#endif
