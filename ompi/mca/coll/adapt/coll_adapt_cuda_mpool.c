#include "ompi_config.h"
#include "ompi/mca/coll/coll.h"
#include "coll_adapt_cuda.h"
#include "coll_adapt_cuda_mpool.h"
#include "opal/mca/common/cuda/common_cuda.h"

#define MPOOL_FREE_LIST_SIZE 100
#define MPOOL_SIZE 1024*1024*200
#define MPOOL_ALIGNMENT 4096

static inline coll_adapt_cuda_mpool_buffer_t* obj_coll_adapt_cuda_mpool_buffer_new(void)
{
    coll_adapt_cuda_mpool_buffer_t *p = (coll_adapt_cuda_mpool_buffer_t *)malloc(sizeof(coll_adapt_cuda_mpool_buffer_t));
    p->next = NULL;
    p->prev = NULL;
    p->size = 0;
    p->addr = NULL;
    return p; 
}

static inline void obj_coll_adapt_cuda_mpool_buffer_chop(coll_adapt_cuda_mpool_buffer_t *p)
{
    p->next = NULL;
    p->prev = NULL;
}

static inline void obj_coll_adapt_cuda_mpool_buffer_reset(coll_adapt_cuda_mpool_buffer_t *p)
{
    p->size = 0;
    p->addr = NULL;
}

static void init_coll_adapt_cuda_mpool_free_list(coll_adapt_cuda_mpool_list_t *flist)
{
    coll_adapt_cuda_mpool_list_t *list = NULL;
    coll_adapt_cuda_mpool_buffer_t *p, *prev;
    int i;
    list = flist;
    p = obj_coll_adapt_cuda_mpool_buffer_new();
    list->head = p;
    prev = p;
    for (i = 1; i < MPOOL_FREE_LIST_SIZE; i++) {
        p = obj_coll_adapt_cuda_mpool_buffer_new();
        prev->next = p;
        p->prev = prev;
        prev = p;
    }
    list->tail = p;
    list->nb_elements = MPOOL_FREE_LIST_SIZE;
}

static inline coll_adapt_cuda_mpool_buffer_t* coll_adapt_cuda_mpool_list_pop_tail(coll_adapt_cuda_mpool_list_t *list)
{
    coll_adapt_cuda_mpool_buffer_t *p = NULL;
    p = list->tail;
    if (p == NULL) {
        return p;
    } else {
        list->nb_elements --;
        if (list->head == p) {
            list->head = NULL;
            list->tail = NULL;
        } else {
            list->tail = p->prev;
            p->prev->next = NULL;
            obj_coll_adapt_cuda_mpool_buffer_chop(p);
        }
        return p;
    }
}

static inline void coll_adapt_cuda_mpool_list_push_head(coll_adapt_cuda_mpool_list_t *list, coll_adapt_cuda_mpool_buffer_t *item)
{
    coll_adapt_cuda_mpool_buffer_t * orig_head = list->head;
    assert(item->next == NULL && item->prev == NULL);
    list->head = item;
    item->next = orig_head;
    if (orig_head == NULL) {
        list->tail = item;
    } else {
        orig_head->prev = item;
    }
    list->nb_elements ++;
}

static inline void coll_adapt_cuda_mpool_list_push_tail(coll_adapt_cuda_mpool_list_t *list, coll_adapt_cuda_mpool_buffer_t *item)
{
    coll_adapt_cuda_mpool_buffer_t * orig_tail = list->tail;
    assert(item->next == NULL && item->prev == NULL);
    list->tail = item;
    item->prev = orig_tail;
    if (orig_tail == NULL) {
        list->head = item;
    } else {
        orig_tail->next = item;
    }
    list->nb_elements ++;
}

static inline void coll_adapt_cuda_mpool_list_delete(coll_adapt_cuda_mpool_list_t *list, coll_adapt_cuda_mpool_buffer_t *item)
{
    if (item->prev == NULL && item->next == NULL) {
        list->head = NULL;
        list->tail = NULL;
    }else if (item->prev == NULL && item->next != NULL) {
        list->head = item->next;
        item->next->prev = NULL;
    } else if (item->next == NULL && item->prev != NULL) {
        list->tail = item->prev;
        item->prev->next = NULL;
    } else {
        item->prev->next = item->next;
        item->next->prev = item->prev;
    }
    list->nb_elements --;
    obj_coll_adapt_cuda_mpool_buffer_chop(item);
}

static inline void coll_adapt_cuda_mpool_list_insert_before(coll_adapt_cuda_mpool_list_t *list, coll_adapt_cuda_mpool_buffer_t *item, coll_adapt_cuda_mpool_buffer_t *next)
{
    assert(item->next == NULL && item->prev == NULL);
    item->next = next;
    item->prev = next->prev;
    if (next->prev != NULL) {
        next->prev->next = item;
    }
    next->prev = item;
    if (list->head == next) {
        list->head = item;
    }
    list->nb_elements ++;
}

/**
 * Collapse the list of free buffers by mergining consecutive buffers. As the property of this list
 * is continously maintained, we only have to parse it up to the newest inserted elements.
 */
static inline void coll_adapt_cuda_mpool_list_item_merge_by_addr(coll_adapt_cuda_mpool_list_t *list, coll_adapt_cuda_mpool_buffer_t* last)
{
    coll_adapt_cuda_mpool_buffer_t *current = list->head;
    coll_adapt_cuda_mpool_buffer_t *next = NULL;
    void* stop_addr = last->addr;

    while(1) {  /* loop forever, the exit conditions are inside */
        if( NULL == (next = current->next) ) return;
        if ((current->addr + current->size) == next->addr) {
            current->size += next->size;
            coll_adapt_cuda_mpool_list_delete(list, next);
            free(next);  /* release the element, and try to continue merging */
            continue;
        }
        current = current->next;
        if( NULL == current ) return;
        if( current->addr > stop_addr ) return;
    }
}

/*
 *  Returns base address of shared memory mapping.
 */
static void *coll_adapt_cuda_mpool_base (mca_mpool_base_module_t *mpool)
{
    coll_adapt_cuda_mpool_module_t* mempool = (coll_adapt_cuda_mpool_module_t *)mpool;
    return mempool->base_ptr;
}

/**
  *  Allocate block of shared memory.
  */
static void *coll_adapt_cuda_mpool_alloc (mca_mpool_base_module_t *mpool,
                                          size_t size, size_t align,
                                          uint32_t flags)
{
    coll_adapt_cuda_mpool_module_t* mempool = (coll_adapt_cuda_mpool_module_t *)mpool;
    coll_adapt_cuda_mpool_buffer_t *ptr = mempool->buffer_free.head;
    while (ptr != NULL) {
        if (ptr->size < size) {  /* Not enough room in this buffer, check next */
            ptr = ptr->next;
            continue;
        }
        void *addr = ptr->addr;
        ptr->size -= size;
        if (ptr->size == 0) {
            coll_adapt_cuda_mpool_list_delete(&mempool->buffer_free, ptr);
            obj_coll_adapt_cuda_mpool_buffer_reset(ptr);
            /* hold on this ptr object, we will reuse it right away */
        } else {
            ptr->addr += size;
            ptr = coll_adapt_cuda_mpool_list_pop_tail(&mempool->free_list);
            if( NULL == ptr )
                ptr = obj_coll_adapt_cuda_mpool_buffer_new();
        }
        assert(NULL != ptr);
        ptr->size = size;
        ptr->addr = (unsigned char*)addr;
        coll_adapt_cuda_mpool_list_push_head(&mempool->buffer_used, ptr);
        mempool->buffer_used_size += size;
        mempool->buffer_free_size -= size;
    //    opal_output( 0, "Malloc buffer from mpool %p, size %lu\n", addr, size);
        return addr;
    }
    opal_output( 0, "no buffer for size %ld.\n", size);
    assert(0);
    return NULL;
}

/**
  * free function typedef
  */
static void coll_adapt_cuda_mpool_free(mca_mpool_base_module_t *mpool,
                                       void *addr)
{
    coll_adapt_cuda_mpool_module_t* mempool = (coll_adapt_cuda_mpool_module_t *)mpool;
    coll_adapt_cuda_mpool_buffer_t *ptr = mempool->buffer_used.head;

    /* Find the holder of this GPU allocation */
    for( ; (NULL != ptr) && (ptr->addr != addr); ptr = ptr->next );
    if (NULL == ptr) {  /* we could not find it. something went wrong */
        opal_output( 0, "addr %p is not managed.\n", addr);
        return;
    }
    coll_adapt_cuda_mpool_list_delete(&mempool->buffer_used, ptr);
    /* Insert the element in the list of free buffers ordered by the addr */
    coll_adapt_cuda_mpool_buffer_t *ptr_next = mempool->buffer_free.head;
    while (ptr_next != NULL) {
        if (ptr_next->addr > addr) {
            break;
        }
        ptr_next = ptr_next->next;
    }
    if (ptr_next == NULL) {  /* buffer_free is empty, or insert to last one */
        coll_adapt_cuda_mpool_list_push_tail(&mempool->buffer_free, ptr);
    } else {
        coll_adapt_cuda_mpool_list_insert_before(&mempool->buffer_free, ptr, ptr_next);
    }
    size_t size = ptr->size;
    coll_adapt_cuda_mpool_list_item_merge_by_addr(&mempool->buffer_free, ptr);
    mempool->buffer_free_size += size;
    mempool->buffer_used_size -= size;
  //  opal_output( 0, "Free buffer to mpool %p, size %lu\n", addr, size);
}


/*
 *  Initializes the mpool module.
 */
static void coll_adapt_cuda_mpool_module_init(coll_adapt_cuda_mpool_module_t * mpool, unsigned char* ptr, size_t size)
{
    mpool->super.mpool_base = coll_adapt_cuda_mpool_base;
    mpool->super.mpool_alloc = coll_adapt_cuda_mpool_alloc;
    mpool->super.mpool_free = coll_adapt_cuda_mpool_free;
  //  mpool->super.mpool_finalize = sm_module_finalize;
    mpool->super.flags = 0;

    init_coll_adapt_cuda_mpool_free_list(&(mpool->free_list));
    mpool->buffer_free_size = size;
    coll_adapt_cuda_mpool_buffer_t *p = obj_coll_adapt_cuda_mpool_buffer_new();
    p->size = size;
    p->addr = ptr;
    mpool->buffer_free.head = p;
    mpool->buffer_free.tail = mpool->buffer_free.head;
    mpool->buffer_free.nb_elements = 1;
    mpool->buffer_used.head = NULL;
    mpool->buffer_used.tail = NULL;
    mpool->buffer_used_size = 0;
    mpool->buffer_used.nb_elements = 0;
    mpool->buffer_total_size = size;
    
    mpool->base_ptr = ptr;
}

mca_mpool_base_module_t *coll_adapt_cuda_mpool_create (int mpool_type)
{
    coll_adapt_cuda_mpool_module_t *mpool_module = NULL;
    unsigned char *mpool_ptr = NULL;
    size_t mpool_size = MPOOL_SIZE;
    
    mpool_module = (coll_adapt_cuda_mpool_module_t *)malloc(sizeof(coll_adapt_cuda_mpool_module_t));
   // mpool_ptr = (unsigned char *)malloc(sizeof(char) * mpool_size);
    if (mpool_type == MPOOL_CPU) {
        posix_memalign((void **)&mpool_ptr, MPOOL_ALIGNMENT, mpool_size);
        mca_common_cuda_register(mpool_ptr, mpool_size, "adapt_cuda");
    } else if (mpool_type == MPOOL_GPU) {
        mca_common_cuda_alloc((void **)&mpool_ptr, mpool_size);
       // mpool_ptr = (unsigned char *)coll_adapt_cuda_malloc(mpool_size);
        printf("GPU MEMPOOL %p]\n", mpool_ptr);
    } else {
        opal_output(0, "Unsupported memory pool type %d\n", mpool_type);
    }
    
    coll_adapt_cuda_mpool_module_init(mpool_module, mpool_ptr, mpool_size);
    
    return (mca_mpool_base_module_t *)mpool_module;
}

