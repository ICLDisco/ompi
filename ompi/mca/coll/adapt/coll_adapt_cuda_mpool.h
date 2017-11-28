#ifndef COLL_ADAPT_CUDA_MPOOL_H
#define COLL_ADAPT_CUDA_MPOOL_H

#include "ompi_config.h"

#if OPAL_CUDA_SUPPORT

#include "opal/mca/event/event.h"
#include "opal/mca/mpool/mpool.h"

#define MPOOL_CPU 0x1
#define MPOOL_GPU 0x2
    
typedef struct coll_adapt_cuda_mpool_buffer {
    unsigned char* addr;
    size_t size;
    struct coll_adapt_cuda_mpool_buffer *next;
    struct coll_adapt_cuda_mpool_buffer *prev;
} coll_adapt_cuda_mpool_buffer_t;

typedef struct {
    coll_adapt_cuda_mpool_buffer_t *head;
    coll_adapt_cuda_mpool_buffer_t *tail;
    size_t nb_elements;
} coll_adapt_cuda_mpool_list_t;
    
typedef struct coll_adapt_cuda_mpool_module {
    mca_mpool_base_module_t super;
    unsigned char* base_ptr;
    coll_adapt_cuda_mpool_list_t buffer_free;
    coll_adapt_cuda_mpool_list_t buffer_used;
    coll_adapt_cuda_mpool_list_t free_list;
    size_t buffer_free_size;
    size_t buffer_used_size;
    size_t buffer_total_size;
} coll_adapt_cuda_mpool_module_t;

mca_mpool_base_module_t *coll_adapt_cuda_mpool_create (int mpool_type);


#endif

#endif
