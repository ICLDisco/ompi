#ifndef OPAL_DATATYPE_CUDA_INTERNAL_H_HAS_BEEN_INCLUDED
#define OPAL_DATATYPE_CUDA_INTERNAL_H_HAS_BEEN_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include <sys/time.h>

#include "opal_datatype_orig_internal.h"


/* OPAL_CUDA */
// #define OPAL_DATATYPE_CUDA_DRY_RUN
#define OPAL_DATATYPE_CUDA_DEBUG
//#define OPAL_DATATYPE_CUDA_KERNEL_TIME
#define OPAL_DATATYPE_CUDA_DEBUG_LEVEL  0
#define OPAL_DATATYPE_CUDA_TIMING


#define IOV_ARRAY_SIZE          1
#define DT_CUDA_BUFFER_SIZE    1024*1024*200
#define DT_CUDA_FREE_LIST_SIZE  50

#define THREAD_PER_BLOCK    32
#define CUDA_WARP_SIZE      32
#define TASK_PER_THREAD     2
#define OPAL_GPU_INDEX      0
#define NB_STREAMS          4
#define CUDA_NB_IOV         4096
#define CUDA_IOV_LEN        1024*1204
#define CUDA_MAX_NB_BLOCKS  1024
#define CUDA_IOV_MAX_TASK_PER_BLOCK 200
#define ALIGNMENT_DOUBLE    8
#define ALIGNMENT_FLOAT     4
#define ALIGNMENT_CHAR      1

#define TIMER_DATA_TYPE struct timeval
#define GET_TIME(TV)   gettimeofday( &(TV), NULL )
#define ELAPSED_TIME(TSTART, TEND)  (((TEND).tv_sec - (TSTART).tv_sec) * 1000000 + ((TEND).tv_usec - (TSTART).tv_usec))



typedef struct {
    uint32_t description_index[200];     /* index of y direction */
    uint32_t description_local_index[200];   /* index of x direction */
    uint32_t dst_offset[200];
    uint32_t description_used;
} ddt_cuda_description_dist_t;

typedef struct {
    dt_stack_t pStack[DT_STATIC_STACK_SIZE];
    dt_elem_desc_t* description;
    struct iovec iov[IOV_ARRAY_SIZE];
    uint32_t stack_pos;
    uint32_t stack_size;
    unsigned char* pBaseBuf; /* const */
    OPAL_PTRDIFF_TYPE lb;  /* const */
    OPAL_PTRDIFF_TYPE ub;  /* const */
    size_t bConverted;
    size_t local_size; /* const */
    uint32_t out_size;
    size_t max_data;
    uint32_t description_count;
    uint32_t description_max_count;
    ddt_cuda_description_dist_t *description_dist;
} ddt_cuda_desc_t;

typedef struct {
    cudaStream_t opal_cuda_stream[NB_STREAMS];
    uint32_t current_stream_id;
} ddt_cuda_stream_t;

typedef struct {
    unsigned char* src[CUDA_IOV_MAX_TASK_PER_BLOCK];
    unsigned char* dst[CUDA_IOV_MAX_TASK_PER_BLOCK];
    uint32_t nb_elements[CUDA_IOV_MAX_TASK_PER_BLOCK];
    uint8_t element_alignment[CUDA_IOV_MAX_TASK_PER_BLOCK];
    uint32_t nb_tasks;
} ddt_cuda_iov_dist_t;

typedef struct ddt_cuda_buffer{
    unsigned char* gpu_addr;
    size_t size;
    struct ddt_cuda_buffer *next;
    struct ddt_cuda_buffer *prev;
} ddt_cuda_buffer_t;

typedef struct {
    ddt_cuda_buffer_t *head;
    ddt_cuda_buffer_t *tail;
    size_t nb_elements;
} ddt_cuda_list_t;

typedef struct {
    int device_id;
    unsigned char* gpu_buffer;
    ddt_cuda_list_t buffer_free;
    ddt_cuda_list_t buffer_used;
    size_t buffer_free_size;
    size_t buffer_used_size;
} ddt_cuda_device_t;

extern ddt_cuda_list_t *cuda_free_list;
extern ddt_cuda_device_t *cuda_device;
extern ddt_cuda_desc_t *cuda_desc_d, *cuda_desc_h;
extern unsigned char* pBaseBuf_GPU;
extern unsigned char *ddt_cuda_pack_buffer, *ddt_cuda_unpack_buffer;
extern size_t ddt_cuda_buffer_space;
extern ddt_cuda_stream_t* cuda_streams;
extern struct iovec cuda_iov[CUDA_NB_IOV];
extern uint32_t cuda_iov_count;
extern ddt_cuda_description_dist_t description_dist_h[CUDA_MAX_NB_BLOCKS];
extern ddt_cuda_description_dist_t* description_dist_d;
extern ddt_cuda_iov_dist_t cuda_iov_dist_h[NB_STREAMS][CUDA_MAX_NB_BLOCKS];
extern ddt_cuda_iov_dist_t* cuda_iov_dist_d[NB_STREAMS];
extern dt_elem_desc_t* description_d;
extern uint8_t opal_datatype_cuda_debug;

//extern uint8_t ALIGNMENT_DOUBLE, ALIGNMENT_FLOAT, ALIGNMENT_CHAR;

        
#if defined (OPAL_DATATYPE_CUDA_DEBUG) 
#define DBGPRINT(fmt, ...) printf(fmt, __VA_ARGS__) 
#else 
#define DBGPRINT(fmt, ...) 
#endif 

__device__ void pack_contiguous_loop_cuda_kernel( dt_elem_desc_t* ELEM,
                                                  uint32_t* COUNT,
                                                  unsigned char** SOURCE,
                                                  unsigned char** DESTINATION,
                                                  size_t* SPACE );
                                                            
__device__ void unpack_contiguous_loop_cuda_kernel( dt_elem_desc_t* ELEM,
                                                    uint32_t* COUNT,
                                                    unsigned char** SOURCE,
                                                    unsigned char** DESTINATION,
                                                    size_t* SPACE );
                                                  
__global__ void opal_generic_simple_pack_cuda_kernel(ddt_cuda_desc_t* cuda_desc);

__global__ void opal_generic_simple_pack_cuda_kernel_v2(ddt_cuda_desc_t* cuda_desc);

__global__ void opal_generic_simple_unpack_cuda_kernel(ddt_cuda_desc_t* cuda_desc);

__global__ void pack_contiguous_loop_cuda_kernel_global( uint32_t copy_loops,
                                                         size_t size,
                                                         OPAL_PTRDIFF_TYPE extent,
                                                         unsigned char* source,
                                                         unsigned char* destination );
                                                         
__global__ void unpack_contiguous_loop_cuda_kernel_global( uint32_t copy_loops,
                                                           size_t size,
                                                           OPAL_PTRDIFF_TYPE extent,
                                                           unsigned char* source,
                                                           unsigned char* destination );
                                                           
// __global__ void opal_generic_simple_pack_cuda_iov_kernel( ddt_cuda_description_dist_t* desc_dist_d, dt_elem_desc_t* desc_d, uint32_t required_blocks, struct iovec* iov, unsigned char* pBaseBuf);

__global__ void opal_generic_simple_pack_cuda_iov_kernel( ddt_cuda_iov_dist_t* cuda_iov_dist);

__global__ void opal_generic_simple_unpack_cuda_iov_kernel( ddt_cuda_iov_dist_t* cuda_iov_dist);

void opal_cuda_output(int output_id, const char *format, ...);

#if defined (OPAL_DATATYPE_CUDA_DEBUG)
#define DT_CUDA_DEBUG( INST ) if (opal_datatype_cuda_debug) { INST }
#else
#define DT_CUDA_DEBUG( INST )
#endif

extern "C"
{
int32_t opal_convertor_set_position_nocheck( opal_convertor_t* convertor, size_t* position );

int32_t opal_convertor_raw( opal_convertor_t* pConvertor, 
		                    struct iovec* iov, uint32_t* iov_count,
		                    size_t* length );
}

#endif  /* OPAL_DATATYPE_CUDA_INTERNAL_H_HAS_BEEN_INCLUDED */
