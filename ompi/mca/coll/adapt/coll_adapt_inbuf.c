#include "coll_adapt.h"
#include "coll_adapt_inbuf.h"

static void mca_coll_adapt_inbuf_constructor(mca_coll_adapt_inbuf_t *inbuf){
}

static void mca_coll_adapt_inbuf_destructor(mca_coll_adapt_inbuf_t *inbuf){
}

OBJ_CLASS_INSTANCE(mca_coll_adapt_inbuf_t, opal_free_list_item_t, mca_coll_adapt_inbuf_constructor, mca_coll_adapt_inbuf_destructor);

mca_coll_adapt_inbuf_t* coll_adapt_alloc_inbuf(opal_free_list_t *inbuf_list, size_t size, int buff_type)
{
    mca_coll_adapt_inbuf_t *inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(inbuf_list);
    //inbuf->buff = (char*)inbuf + sizeof(mca_coll_adapt_inbuf_t) + sizeof(char);
#if OPAL_CUDA_SUPPORT
    if (GPU_BUFFER == buff_type) {
        mca_mpool_base_module_t *mpool = mca_coll_adapt_component.pined_gpu_mpool;
        inbuf->buff = mpool->mpool_alloc(mpool, sizeof(char)*size, 0, 0);
    }
#endif
    //printf("inbuf %p, buff %p\n", inbuf, inbuf->buff);
    return inbuf;
}

int coll_adapt_free_inbuf(opal_free_list_t *inbuf_list, mca_coll_adapt_inbuf_t *inbuf, int buff_type)
{
#if OPAL_CUDA_SUPPORT
    if (GPU_BUFFER == buff_type) {
        mca_mpool_base_module_t *mpool = mca_coll_adapt_component.pined_gpu_mpool;
        mpool->mpool_free(mpool, inbuf->buff);
    }
#endif
    //inbuf->buff = NULL;
    opal_free_list_return(inbuf_list, (opal_free_list_item_t*)inbuf);
    
    return 0;
}
