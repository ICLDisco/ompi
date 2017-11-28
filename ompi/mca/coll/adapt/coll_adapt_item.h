#include "opal/class/opal_list.h"
#include "coll_adapt_inbuf.h"

struct mca_coll_adapt_item_s {
    opal_list_item_t super;
    int id;     //fragment id
    int count;  //have received from how many children
    mca_coll_adapt_inbuf_t *inbuf_to_free[3];
    void *op_event;
};

typedef struct mca_coll_adapt_item_s mca_coll_adapt_item_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_item_t);