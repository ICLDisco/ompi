#include "coll_adapt_item.h"

static void mca_coll_adapt_item_constructor(mca_coll_adapt_item_t *item){
}

static void mca_coll_adapt_item_destructor(mca_coll_adapt_item_t *item){
}

OBJ_CLASS_INSTANCE(mca_coll_adapt_item_t, opal_list_item_t, mca_coll_adapt_item_constructor, mca_coll_adapt_item_destructor);
