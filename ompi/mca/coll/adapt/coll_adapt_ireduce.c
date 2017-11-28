#include "ompi_config.h"
#include "ompi/communicator/communicator.h"
#include "coll_adapt.h"
#include "coll_adapt_algorithms.h"
#include "coll_adapt_context.h"
#include "coll_adapt_item.h"
#include "ompi/constants.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/mca/coll/base/coll_base_functions.h"     //COLL_BASE_COMPUTED_SEGCOUNT
#include "ompi/mca/coll/base/coll_base_topo.h"  //build tree

#if OPAL_CUDA_SUPPORT
#include "coll_adapt_cuda.h"
#include "coll_adapt_cuda_mpool.h"
#include "opal/mca/common/cuda/common_cuda.h"
#endif

#include "coll_adapt_op.h"

#define FREE_LIST_NUM_INBUF_LIST 10    //The start size of the context free list
#define FREE_LIST_MAX_INBUF_LIST 10000  //The max size of the context free list
#define FREE_LIST_INC_INBUF_LIST 10    //The incresment of the context free list
#define TEST printfno

/* Reduce algorithm variables */
static int coll_adapt_ireduce_algorithm = 0;
static size_t coll_adapt_ireduce_segment_size = 163740;
static int coll_adapt_ireduce_max_send_requests = 2;
static int coll_adapt_ireduce_max_recv_requests = 3;
static opal_free_list_t *coll_adapt_ireduce_context_free_list = NULL;
static int32_t coll_adapt_ireduce_context_free_list_enabled = 0;

typedef int (*mca_coll_adapt_ireduce_fn_t)(
    const void *sbuf,
    void *rbuf,
    int count, 
    struct ompi_datatype_t *datatype,
    struct ompi_op_t *op,
    int root, 
    struct ompi_communicator_t *comm, 
    ompi_request_t ** request, 
    mca_coll_base_module_t *module, 
    int ireduce_tag
);

static mca_coll_adapt_algorithm_index_t mca_coll_adapt_ireduce_algorithm_index[] = {
    {0, (uintptr_t)mca_coll_adapt_ireduce_tuned},
    {1, (uintptr_t)mca_coll_adapt_ireduce_binomial},
    {2, (uintptr_t)mca_coll_adapt_ireduce_in_order_binomial},
    {3, (uintptr_t)mca_coll_adapt_ireduce_binary},
    {4, (uintptr_t)mca_coll_adapt_ireduce_pipeline},
    {5, (uintptr_t)mca_coll_adapt_ireduce_chain},
    {6, (uintptr_t)mca_coll_adapt_ireduce_linear},
    {7, (uintptr_t)mca_coll_adapt_ireduce_topoaware_linear},
    {8, (uintptr_t)mca_coll_adapt_ireduce_topoaware_chain}
};

#if OPAL_CUDA_SUPPORT
static int reduce_async_op_free_inbuf(mca_coll_adapt_reduce_context_t *context, mca_coll_adapt_item_t * item);
static int reduce_send_context_async_op_callback(mca_coll_adapt_reduce_context_t *send_context);
#endif

int mca_coll_adapt_ireduce_init(void)
{
    mca_base_component_t *c = &mca_coll_adapt_component.super.collm_version;
    
    mca_base_component_var_register(c, "reduce_algorithm",
                                    "Algorithm of reduce",
                                    MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    &coll_adapt_ireduce_algorithm);
    
    mca_base_component_var_register(c, "reduce_segment_size",
                                    "Segment size in bytes used by default for reduce algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                    MCA_BASE_VAR_TYPE_SIZE_T, NULL, 0, 0,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    &coll_adapt_ireduce_segment_size);

    mca_base_component_var_register(c, "reduce_max_send_requests",
                                    "Maximum number of send requests",
                                    MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    &coll_adapt_ireduce_max_send_requests);
                                    
    mca_base_component_var_register(c, "reduce_max_recv_requests",
                                    "Maximum number of receive requests",
                                    MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    &coll_adapt_ireduce_max_recv_requests);
    return OMPI_SUCCESS;
}

int mca_coll_adapt_ireduce_fini(void)
{
    if ( NULL != coll_adapt_ireduce_context_free_list) {
        OBJ_RELEASE(coll_adapt_ireduce_context_free_list);
        coll_adapt_ireduce_context_free_list = NULL;
        coll_adapt_ireduce_context_free_list_enabled = 0;
        OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "reduce fini\n"));
    }
    return OMPI_SUCCESS;
}

//Can only work on commutative op
//TODO: fix lower bound stuff

static mca_coll_adapt_item_t * get_next_ready_item(opal_list_t* list, int num_children){
    mca_coll_adapt_item_t *item;
    if (opal_list_is_empty(list)) {
        return NULL;
    }
    for(item = (mca_coll_adapt_item_t *) opal_list_get_first(list);
        item != (mca_coll_adapt_item_t *) opal_list_get_end(list);
        item = (mca_coll_adapt_item_t *) ((opal_list_item_t *)item)->opal_list_next) {
        if (item->count == num_children) {
            opal_list_remove_item(list, (opal_list_item_t *)item);
            return item;
        }
    }
    return NULL;
}

static int add_to_list(opal_list_t* list, int id, mca_coll_adapt_inbuf_t *inbuf_to_free, mca_coll_adapt_item_t **op_item){
    mca_coll_adapt_item_t *item;
    int ret = 0;
    for(item = (mca_coll_adapt_item_t *) opal_list_get_first(list);
        item != (mca_coll_adapt_item_t *) opal_list_get_end(list);
        item = (mca_coll_adapt_item_t *) ((opal_list_item_t *)item)->opal_list_next) {
        if (item->id == id) {
            (item->count)++;
            item->inbuf_to_free[item->count-1] = inbuf_to_free;
            *op_item = item;
            ret = 1;
            break;
        }
    }
    if (ret == 0) {
        item = OBJ_NEW(mca_coll_adapt_item_t);
        item->id = id;
        item->count = 1;
        item->inbuf_to_free[item->count-1] = inbuf_to_free;
        item->op_event = NULL;
        *op_item = item;
        opal_list_append(list, (opal_list_item_t *)item);
        ret = 2;
    }
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "add_to_list_return %d\n", ret));
    return ret;
}

static mca_coll_adapt_inbuf_t * to_inbuf(char * buf, int distance){
    return (mca_coll_adapt_inbuf_t *)(buf - distance);
}

static int ireduce_request_fini(mca_coll_adapt_reduce_context_t *context)
{
    int i;
    #if OPAL_CUDA_SUPPORT
    mca_common_cuda_sync_op_stream(0);
    #endif
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Singal\n", ompi_comm_rank(context->con->comm)));
    ompi_request_t *temp_req = context->con->request;
    if (context->con->accumbuf != NULL) {
        if (context->con->rank != context->con->root) {
            for (i=0; i<context->con->num_segs; i++) {
                //wei opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)to_inbuf(context->con->accumbuf[i], context->con->distance));
                if (CPU_BUFFER == context->con->buff_type) {
                    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Return accumbuf %d %p\n", ompi_comm_rank(context->con->comm), i, to_inbuf(context->con->accumbuf[i], context->con->distance)));
                    coll_adapt_free_inbuf(context->con->inbuf_list, to_inbuf(context->con->accumbuf[i], context->con->distance), context->con->buff_type);
                }
                #if OPAL_CUDA_SUPPORT
                else {
                    coll_adapt_free_inbuf(context->con->inbuf_list, context->con->accumbuf_to_inbuf[i], context->con->buff_type);
                    context->con->accumbuf_to_inbuf[i] = NULL;
                }
                #endif
            }
        }
        free(context->con->accumbuf);
    }
    OBJ_RELEASE(context->con->recv_list);
    for (i=0; i<context->con->num_segs; i++) {
        OBJ_RELEASE(context->con->mutex_op_list[i]);
    }
    free(context->con->mutex_op_list);
    OBJ_RELEASE(context->con->mutex_num_recv_segs);
    OBJ_RELEASE(context->con->mutex_recv_list);
    OBJ_RELEASE(context->con->mutex_num_sent);
    fflush(stdout);
    if (context->con->tree->tree_nextsize > 0) {
        OBJ_RELEASE(context->con->inbuf_list);
        free(context->con->next_recv_segs);
    }
    OBJ_RELEASE(context->con);
    OBJ_RELEASE(context->con);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "return context_list\n"));
    opal_free_list_return(coll_adapt_ireduce_context_free_list, (opal_free_list_item_t*)context);
    ompi_request_complete(temp_req, 1);
    return OMPI_SUCCESS;
}

static int send_cb(ompi_request_t *req){
    mca_coll_adapt_reduce_context_t *context = (mca_coll_adapt_reduce_context_t *) req->req_complete_cb_data;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: ireduce_send_cb, peer %d, seg_id %d\n", context->con->rank, context->peer, context->frag_id));
    int err;
    
    opal_atomic_sub_32(&(context->con->ongoing_send), 1);
    
    //send a new segment
    //list is not empty
    OPAL_THREAD_LOCK (context->con->mutex_recv_list);
    mca_coll_adapt_item_t *item = get_next_ready_item(context->con->recv_list, context->con->tree->tree_nextsize);
    OPAL_THREAD_UNLOCK (context->con->mutex_recv_list);
    
    if (item != NULL) {
        //get new context item from free list
        mca_coll_adapt_reduce_context_t * send_context = (mca_coll_adapt_reduce_context_t *) opal_free_list_wait(coll_adapt_ireduce_context_free_list);
        if (context->con->tree->tree_nextsize > 0) {
            send_context->buff = context->con->accumbuf[item->id];
            
        }
        else{
            send_context->buff = context->buff + (item->id - context->frag_id) * context->con->segment_increment;
        }
        send_context->frag_id = item->id;
        send_context->peer = context->peer;
        send_context->con = context->con;
        OBJ_RETAIN(context->con);
        
        opal_atomic_add_32(&(context->con->ongoing_send), 1);
        
        int send_count = send_context->con->seg_count;
        if (item->id == (send_context->con->num_segs - 1)) {
            send_count = send_context->con->count - item->id * send_context->con->seg_count;
        }
        
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: In send_cb, create isend to seg %d, peer %d, tag %d\n", send_context->con->rank, send_context->frag_id, send_context->peer, (send_context->con->ireduce_tag << 16) + send_context->frag_id));
        
        ompi_request_t *send_req;
        err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, (context->con->ireduce_tag << 16) + send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
        
        //release the item
        OBJ_RELEASE(item);
        
        //invoke send call back
        ompi_request_set_callback(send_req, send_cb, send_context);
    }
    
    OPAL_THREAD_LOCK(context->con->mutex_num_sent);
    int32_t num_sent = ++(context->con->num_sent_segs);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output,"[%d]: In send_cb, root = %d, num_sent = %d, num_segs = %d\n", context->con->rank, context->con->tree->tree_root, num_sent, context->con->num_segs));
    //check whether signal the condition, non root and sent all the segments
    if (context->con->tree->tree_root != context->con->rank && num_sent == context->con->num_segs) {
        OPAL_THREAD_UNLOCK(context->con->mutex_num_sent);
        ireduce_request_fini(context);
    }
    else{
        OPAL_THREAD_UNLOCK(context->con->mutex_num_sent);
        OBJ_RELEASE(context->con);
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output,"return context_list\n"));
        opal_free_list_return(coll_adapt_ireduce_context_free_list, (opal_free_list_item_t*)context);
    }
    OPAL_THREAD_UNLOCK(req->req_lock);
    req->req_free(&req);
    return 1;
}

static int recv_cb(ompi_request_t *req){
    mca_coll_adapt_reduce_context_t *context = (mca_coll_adapt_reduce_context_t *) req->req_complete_cb_data;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: ireduce_recv_cb, peer %d, seg_id %d\n", context->con->rank, context->peer, context->frag_id));
    
    mca_coll_adapt_inbuf_t *inbuf_to_free = NULL;
    int op_async_not_free = 0;
    
    int err;
    //atomic
    int32_t new_id = opal_atomic_add_32(&(context->con->next_recv_segs[context->child_id]), 1);
    
    //receive new segment
    if (new_id < context->con->num_segs) {
        char * temp_recv_buf = NULL;
        mca_coll_adapt_inbuf_t * inbuf = NULL;
        //set inbuf, if it it first child, recv on rbuf, else recv on inbuf
        if (context->child_id == 0 && context->con->sbuf != MPI_IN_PLACE && context->con->root == context->con->rank) {
            temp_recv_buf = (char *)context->con->rbuf + (ptrdiff_t)new_id * (ptrdiff_t)context->con->segment_increment;
        }
        else {
            //wei inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(context->con->inbuf_list);
            OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: In recv_cb, alloc inbuf\n", context->con->rank));
            inbuf = (mca_coll_adapt_inbuf_t *) coll_adapt_alloc_inbuf(context->con->inbuf_list, context->con->real_seg_size, context->con->buff_type);
            temp_recv_buf = inbuf->buff - context->con->lower_bound;
        }
        //get new context item from free list
        mca_coll_adapt_reduce_context_t * recv_context = (mca_coll_adapt_reduce_context_t *) opal_free_list_wait(coll_adapt_ireduce_context_free_list);
        recv_context->buff = temp_recv_buf;
        recv_context->frag_id = new_id;
        recv_context->child_id = context->child_id;
        recv_context->peer = context->peer;
        recv_context->con = context->con;
        OBJ_RETAIN(context->con);
        recv_context->inbuf = inbuf;
        int recv_count = recv_context->con->seg_count;
        if (new_id == (recv_context->con->num_segs - 1)) {
            recv_count = recv_context->con->count - new_id * recv_context->con->seg_count;
        }
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: In recv_cb, create irecv for seg %d, peer %d, inbuf %p, tag %d\n", context->con->rank, recv_context->frag_id, recv_context->peer, (void *)inbuf, (recv_context->con->ireduce_tag << 16) + recv_context->frag_id));
        ompi_request_t *recv_req;
        MCA_PML_CALL(irecv(temp_recv_buf, recv_count, recv_context->con->datatype, recv_context->peer, (recv_context->con->ireduce_tag << 16) + recv_context->frag_id, recv_context->con->comm, &recv_req));
        //invoke recvive call back
        ompi_request_set_callback(recv_req, recv_cb, recv_context);
    }
    
    //do the op
    int op_count = context->con->seg_count;
    if (context->frag_id == (context->con->num_segs - 1)) {
        op_count = context->con->count - context->frag_id * context->con->seg_count;
    }
    
    int keep_inbuf = 0;
    OPAL_THREAD_LOCK(context->con->mutex_op_list[context->frag_id]);
    if (context->con->accumbuf[context->frag_id] == NULL) {
        if (context->inbuf == NULL) {
            OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: set accumbuf to rbuf\n", context->con->rank));
            context->con->accumbuf[context->frag_id] = context->buff;
        }
        else {
            keep_inbuf = 1;
            OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: set accumbuf to inbuf\n", context->con->rank));
            context->con->accumbuf[context->frag_id] = context->inbuf->buff - context->con->lower_bound;
            if (GPU_BUFFER == context->con->buff_type) {
                context->con->accumbuf_to_inbuf[context->frag_id] = context->inbuf;
                op_async_not_free = 1;
            }
        }
        //op sbuf and accmbuf to accumbuf
        coll_adapt_op_reduce(context->con->op, context->con->sbuf + (ptrdiff_t)context->frag_id * (ptrdiff_t)context->con->segment_increment, context->con->accumbuf[context->frag_id], op_count, context->con->datatype, context->con->buff_type);
        
    }
    else {
        if (context->inbuf == NULL) {
            //op rbuf and accumbuf to rbuf
            OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: op rbuf and accumbuf to rbuf\n", context->con->rank));
            coll_adapt_op_reduce(context->con->op, context->con->accumbuf[context->frag_id], context->buff, op_count, context->con->datatype, context->con->buff_type);
            //free old accumbuf
            //opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)to_inbuf(context->con->accumbuf[context->frag_id], context->con->distance));
            if (CPU_BUFFER == context->con->buff_type) {
                OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: free old accumbuf %p\n", context->con->rank, to_inbuf(context->con->accumbuf[context->frag_id], context->con->distance)));
                coll_adapt_free_inbuf(context->con->inbuf_list, to_inbuf(context->con->accumbuf[context->frag_id], context->con->distance), context->con->buff_type);
            }
#if OPAL_CUDA_SUPPORT  
            else {
                //coll_adapt_free_inbuf(context->con->inbuf_list, context->con->accumbuf_to_inbuf[context->frag_id], context->con->buff_type);
                //context->con->accumbuf_to_inbuf[context->frag_id] = NULL;
                op_async_not_free = 1;
                inbuf_to_free = context->con->accumbuf_to_inbuf[context->frag_id];
                
            }
#endif
            //set accumbut to rbuf
            context->con->accumbuf[context->frag_id] = context->buff;
        }
        else {
            //op inbuf and accmbuf to accumbuf
            OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: op inbuf and accmbuf to accumbuf\n", context->con->rank));
            coll_adapt_op_reduce(context->con->op, context->inbuf->buff - context->con->lower_bound, context->con->accumbuf[context->frag_id], op_count, context->con->datatype, context->con->buff_type);
#if OPAL_CUDA_SUPPORT 
            if (GPU_BUFFER == context->con->buff_type) {
                inbuf_to_free = context->inbuf;
                op_async_not_free = 1;
            }
#endif
        }
    }
    
    OPAL_THREAD_UNLOCK(context->con->mutex_op_list[context->frag_id]);
    
    //set recv list
    OPAL_THREAD_LOCK (context->con->mutex_recv_list);
    mca_coll_adapt_item_t *op_item = NULL;
    add_to_list(context->con->recv_list, context->frag_id, inbuf_to_free, &op_item);
#if OPAL_CUDA_SUPPORT
    /* now fragment from all children has been received, we record the cuda event for the op */ 
    if (GPU_BUFFER == context->con->buff_type) 
        if (op_item->count == context->con->tree->tree_nextsize) {
            void *op_cuda_stream = mca_common_cuda_get_op_stream(0);
            op_item->op_event = mca_common_cuda_get_op_event_item();
            mca_common_cuda_record_op_event_item(op_item->op_event, op_cuda_stream);
    }
#endif
    OPAL_THREAD_UNLOCK (context->con->mutex_recv_list);
    
    //send to parent
    if (context->con->rank != context->con->tree->tree_root && context->con->ongoing_send < coll_adapt_ireduce_max_send_requests) {
        //atomic
        OPAL_THREAD_LOCK (context->con->mutex_recv_list);
        mca_coll_adapt_item_t *item = get_next_ready_item(context->con->recv_list, context->con->tree->tree_nextsize);
        OPAL_THREAD_UNLOCK (context->con->mutex_recv_list);
        
        if (item != NULL) {
            //get new context item from free list
            mca_coll_adapt_reduce_context_t * send_context = (mca_coll_adapt_reduce_context_t *) opal_free_list_wait(coll_adapt_ireduce_context_free_list);
            send_context->buff = context->con->accumbuf[context->frag_id];
            send_context->frag_id = item->id;
            send_context->peer = context->con->tree->tree_prev;
            send_context->con = context->con;
            OBJ_RETAIN(context->con);
            //atomic
            opal_atomic_add_32(&(context->con->ongoing_send), 1);
            
            int send_count = send_context->con->seg_count;
            if (item->id == (send_context->con->num_segs - 1)) {
                send_count = send_context->con->count - item->id * send_context->con->seg_count;
            }
#if OPAL_CUDA_SUPPORT
            if (GPU_BUFFER == context->con->buff_type) {
                /* op is not done, we save it and send in callback */
                if (0 == mca_common_cuda_query_op_event_item(item->op_event)) {
                    opal_output(0, "op not done before send, record op %p\n", item->op_event);
                    send_context->flags = COLL_ADAPT_CONTEXT_FLAGS_CUDA_REDUCE;
                    send_context->cuda_callback = reduce_send_context_async_op_callback;
                    send_context->buff_to_free_item = item;
                    mca_common_cuda_save_op_event("op in coll_adapt_cuda_reduce", item->op_event, (void *)send_context);
                    goto RECV_CB_SKIP_SEND;
                } else {
                    opal_output(0, "op done before send, return op %p\n", item->op_event);
                    mca_common_cuda_return_op_event_item(item->op_event);
                    reduce_async_op_free_inbuf(context, item);
                }
            }
#endif
            OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: In recv_cb, create isend to seg %d, peer %d, tag %d\n", send_context->con->rank, send_context->frag_id, send_context->peer, (send_context->con->ireduce_tag << 16) + send_context->frag_id));
            
            ompi_request_t *send_req;
            err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, (send_context->con->ireduce_tag << 16) + send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
            
            //release the item
            OBJ_RELEASE(item);
            
            //invoke send call back
            ompi_request_set_callback(send_req, send_cb, send_context);
        }
    }
    
RECV_CB_SKIP_SEND: ;  
    OPAL_THREAD_LOCK (context->con->mutex_num_recv_segs);
    int num_recv_segs_t = ++(context->con->num_recv_segs);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: In recv_cb, tree = %p, root = %d, num_recv = %d, num_segs = %d, num_child = %d\n", context->con->rank, (void *)context->con->tree, context->con->tree->tree_root, num_recv_segs_t, context->con->num_segs, context->con->tree->tree_nextsize));
    //if this is root and has received all the segments
    if (context->con->tree->tree_root == context->con->rank && num_recv_segs_t == context->con->num_segs * context->con->tree->tree_nextsize) {
        OPAL_THREAD_UNLOCK (context->con->mutex_num_recv_segs);
        if (!keep_inbuf && context->inbuf != NULL) {
            // wei opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
            OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: root free context inbuf %p", context->con->rank, context->inbuf));
            coll_adapt_free_inbuf(context->con->inbuf_list, context->inbuf, context->con->buff_type);
        }
        ireduce_request_fini(context);
    }
    else{
        OPAL_THREAD_UNLOCK (context->con->mutex_num_recv_segs);
        if (!keep_inbuf && context->inbuf != NULL && !op_async_not_free) {
            //wei opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
            OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: free context inbuf %p", context->con->rank, context->inbuf));
            coll_adapt_free_inbuf(context->con->inbuf_list, context->inbuf, context->con->buff_type);
        }
        OBJ_RELEASE(context->con);
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: return context_list", context->con->rank));
        opal_free_list_return(coll_adapt_ireduce_context_free_list, (opal_free_list_item_t*)context);
    }
    OPAL_THREAD_UNLOCK(req->req_lock);
    req->req_free(&req);
    return 1;
}

int mca_coll_adapt_ireduce(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    if (count == 0) {
        return MPI_SUCCESS;
    }
    else {
        int rank = ompi_comm_rank(comm);
        if (rank == root) {
            OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "ireduce root %d, algorithm %d, coll_adapt_ireduce_segment_size %zu, coll_adapt_ireduce_max_send_requests %d, coll_adapt_ireduce_max_recv_requests %d\n", 
            root, coll_adapt_ireduce_algorithm, coll_adapt_ireduce_segment_size, coll_adapt_ireduce_max_send_requests, coll_adapt_ireduce_max_recv_requests));
        }
        //get ireduce tag
        int ireduce_tag = opal_atomic_add_32(&(comm->c_ireduce_tag), 1);
        ireduce_tag = (ireduce_tag % 4096) + 4096;
#if OPAL_CUDA_SUPPORT
        if (mca_common_is_cuda_buffer(rbuf) && mca_common_is_cuda_buffer(sbuf)) {
            return mca_coll_adapt_ireduce_cuda(sbuf, rbuf, count, dtype, op, root, comm, request, module, ireduce_tag);
        }
#endif
        mca_coll_adapt_ireduce_fn_t reduce_func = (mca_coll_adapt_ireduce_fn_t)mca_coll_adapt_ireduce_algorithm_index[coll_adapt_ireduce_algorithm].algorithm_fn_ptr;
        return reduce_func(sbuf, rbuf, count, dtype, op, root, comm, request, module, ireduce_tag);
    }
}

int mca_coll_adapt_ireduce_tuned(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag){
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_chain(1, comm, root);
    size_t seg_size = coll_adapt_ireduce_segment_size;
    int err= mca_coll_adapt_ireduce_generic(sbuf, rbuf, count, dtype, op, root, comm, request, module, tree, seg_size, ireduce_tag, CPU_BUFFER);
    //ompi_coll_base_topo_destroy_tree(&tree);
    return err;
}

int mca_coll_adapt_ireduce_binomial(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_bmtree) && (coll_comm->cached_bmtree_root == root) ) ) {
        if( coll_comm->cached_bmtree ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_bmtree) );
        }
        coll_comm->cached_bmtree = ompi_coll_base_topo_build_bmtree(comm, root);
        coll_comm->cached_bmtree_root = root;
    }
    size_t seg_size = coll_adapt_ireduce_segment_size;
    return mca_coll_adapt_ireduce_generic(sbuf, rbuf, count, dtype, op, root, comm, request, module, coll_comm->cached_bmtree, seg_size, ireduce_tag, CPU_BUFFER);
}

int mca_coll_adapt_ireduce_in_order_binomial(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_in_order_bmtree) && (coll_comm->cached_in_order_bmtree_root == root) ) ) {
        if( coll_comm->cached_in_order_bmtree ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_in_order_bmtree) );
        }
        coll_comm->cached_in_order_bmtree = ompi_coll_base_topo_build_in_order_bmtree(comm, root);
        coll_comm->cached_in_order_bmtree_root = root;
    }
    size_t seg_size = coll_adapt_ireduce_segment_size;
    return mca_coll_adapt_ireduce_generic(sbuf, rbuf, count, dtype, op, root, comm, request, module, coll_comm->cached_in_order_bmtree, seg_size, ireduce_tag, CPU_BUFFER);
}

int mca_coll_adapt_ireduce_binary(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_bintree) && (coll_comm->cached_bintree_root == root) ) ) {
        if( coll_comm->cached_bintree ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_bintree) );
        }
        coll_comm->cached_bintree = ompi_coll_base_topo_build_tree(2, comm, root);
        coll_comm->cached_bintree_root = root;
    }
    size_t seg_size = coll_adapt_ireduce_segment_size;
    return mca_coll_adapt_ireduce_generic(sbuf, rbuf, count, dtype, op, root, comm, request, module, coll_comm->cached_bintree, seg_size, ireduce_tag, CPU_BUFFER);
}


//int mca_coll_adapt_ireduce_pipeline(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag){
//    mca_coll_base_comm_t *coll_comm = module->base_data;
//    if( !( (coll_comm->cached_pipeline) && (coll_comm->cached_pipeline_root == root) ) ) {
//        if( coll_comm->cached_pipeline ) { /* destroy previous binomial if defined */
//            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_pipeline) );
//        }
//        coll_comm->cached_pipeline = ompi_coll_base_topo_build_chain(1, comm, root);
//        coll_comm->cached_pipeline_root = root;
//    }
//    return mca_coll_adapt_ireduce_generic(sbuf, rbuf, count, dtype, op, root, comm, request, module, coll_comm->cached_pipeline, ireduce_tag);
//}

int mca_coll_adapt_ireduce_pipeline(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag){
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_chain(1, comm, root);
    size_t seg_size = coll_adapt_ireduce_segment_size;
    int err= mca_coll_adapt_ireduce_generic(sbuf, rbuf, count, dtype, op, root, comm, request, module, tree, seg_size, ireduce_tag, CPU_BUFFER);
    //ompi_coll_base_topo_destroy_tree(&tree);
    return err;
}


int mca_coll_adapt_ireduce_chain(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_chain) && (coll_comm->cached_chain_root == root) ) ) {
        if( coll_comm->cached_chain ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_chain) );
        }
        coll_comm->cached_chain = ompi_coll_base_topo_build_chain(4, comm, root);
        coll_comm->cached_chain_root = root;
    }
    size_t seg_size = coll_adapt_ireduce_segment_size;
    return mca_coll_adapt_ireduce_generic(sbuf, rbuf, count, dtype, op, root, comm, request, module, coll_comm->cached_chain, seg_size, ireduce_tag, CPU_BUFFER);
}

int mca_coll_adapt_ireduce_linear(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_linear) && (coll_comm->cached_linear_root == root) ) ) {
        if( coll_comm->cached_linear ) { /* destroy previous tree if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_linear) );
        }
        int fanout = ompi_comm_size(comm) - 1;
        ompi_coll_tree_t * tree;
        if (fanout > 1) {
            tree = ompi_coll_base_topo_build_tree(ompi_comm_size(comm) - 1, comm, root);
        }
        else{
            tree = ompi_coll_base_topo_build_chain(1, comm, root);
        }
        coll_comm->cached_linear = tree;
        coll_comm->cached_linear_root = root;
    }
    size_t seg_size = coll_adapt_ireduce_segment_size;
    return mca_coll_adapt_ireduce_generic(sbuf, rbuf, count, dtype, op, root, comm, request, module, coll_comm->cached_linear, seg_size, ireduce_tag, CPU_BUFFER);
}

int mca_coll_adapt_ireduce_topoaware_linear(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_topolinear) && (coll_comm->cached_topolinear_root == root) ) ) {
        if( coll_comm->cached_topolinear ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_topolinear) );
        }
        coll_comm->cached_topolinear = ompi_coll_base_topo_build_topoaware_linear(comm, root, module, 3, 0, NULL);
        coll_comm->cached_topolinear_root = root;
    }
    size_t seg_size = coll_adapt_ireduce_segment_size;
    return mca_coll_adapt_ireduce_generic(sbuf, rbuf, count, dtype, op, root, comm, request, module, coll_comm->cached_topolinear, seg_size, ireduce_tag, CPU_BUFFER);
}

//int mca_coll_adapt_ireduce_topoaware_chain(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag){
//    mca_coll_base_comm_t *coll_comm = module->base_data;
//    if( !( (coll_comm->cached_topochain) && (coll_comm->cached_topochain_root == root) ) ) {
//        if( coll_comm->cached_topochain ) { /* destroy previous binomial if defined */
//            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_topochain) );
//        }
//        coll_comm->cached_topochain = ompi_coll_base_topo_build_topoaware_chain(comm, root, module, 0, NULL);
//        coll_comm->cached_topochain_root = root;
//    }
//    return mca_coll_adapt_ireduce_generic(sbuf, rbuf, count, dtype, op, root, comm, request, module, coll_comm->cached_topochain, ireduce_tag);
//}

int mca_coll_adapt_ireduce_topoaware_chain(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag){
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_topoaware_chain(comm, root, module, 3, 0, NULL);
    //print_tree(tree, ompi_comm_rank(comm));
    size_t seg_size = coll_adapt_ireduce_segment_size;
    int err =  mca_coll_adapt_ireduce_generic(sbuf, rbuf, count, dtype, op, root, comm, request, module, tree, seg_size, ireduce_tag, CPU_BUFFER);
    //ompi_coll_base_topo_destroy_tree(&tree);
    return err;
}


int mca_coll_adapt_ireduce_generic(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree, size_t seg_size, int ireduce_tag, int buff_type){
    
    ptrdiff_t extent, lower_bound, segment_increment;
    ptrdiff_t true_lower_bound, true_extent, real_seg_size;
    size_t typelng;
    int seg_count = count, num_segs, rank, recv_count, send_count, i, j, err, min, distance = 0;
    int32_t seg_index;
    int * next_recv_segs = NULL;
    char **accumbuf = NULL;      //used to store the accumuate result, pointer to every segment
    opal_free_list_t * inbuf_list; //a free list contain all recv data
    opal_mutex_t * mutex_recv_list;
    opal_mutex_t * mutex_num_recv_segs;
    opal_mutex_t * mutex_num_sent;
    opal_mutex_t ** mutex_op_list;
    opal_list_t * recv_list;     //a list to store the segments need to be sent
    
    /* set up gpu mpool */
#if OPAL_CUDA_SUPPORT
    if (GPU_BUFFER == buff_type) {
        mca_mpool_base_module_t *mpool = NULL;
        if (mca_coll_adapt_component.pined_gpu_mpool == NULL) {
            mca_coll_adapt_component.pined_gpu_mpool = coll_adapt_cuda_mpool_create(MPOOL_GPU);
        }
        mpool = mca_coll_adapt_component.pined_gpu_mpool;
        assert(mpool != NULL);
    }
#endif    
    
    // Determine number of segments and number of elements sent per operation
    rank = ompi_comm_rank(comm);
    ompi_datatype_get_extent( dtype, &lower_bound, &extent );
    ompi_datatype_type_size( dtype, &typelng );
    COLL_BASE_COMPUTED_SEGCOUNT( seg_size, typelng, seg_count );
    num_segs = (count + seg_count - 1) / seg_count;
    segment_increment = (ptrdiff_t)seg_count * extent;
    ompi_datatype_get_true_extent(dtype, &true_lower_bound, &true_extent);
    real_seg_size = true_extent + (ptrdiff_t)(seg_count - 1) * extent;
    
    //set up free list
    if (0 == coll_adapt_ireduce_context_free_list_enabled) {
        int32_t context_free_list_enabled = opal_atomic_add_32(&(coll_adapt_ireduce_context_free_list_enabled), 1);
        if (1 == context_free_list_enabled) {
            coll_adapt_ireduce_context_free_list = OBJ_NEW(opal_free_list_t);
            opal_free_list_init(coll_adapt_ireduce_context_free_list,
                                sizeof(mca_coll_adapt_reduce_context_t),
                                opal_cache_line_size,
                                OBJ_CLASS(mca_coll_adapt_reduce_context_t),
                                0,opal_cache_line_size,
                                mca_coll_adapt_component.adapt_context_free_list_min,
                                mca_coll_adapt_component.adapt_context_free_list_max,
                                mca_coll_adapt_component.adapt_context_free_list_inc,
                                NULL, 0, NULL, NULL, NULL);
        }
    }
    
    //not leaf
    if (tree->tree_nextsize > 0) {
        inbuf_list = OBJ_NEW(opal_free_list_t);
        size_t inbuf_extend_size = 0;
        if (CPU_BUFFER == buff_type) { /* not GPU buff, inbuf = inbuf_t + seg_size */
            inbuf_extend_size = real_seg_size;
        }
        opal_free_list_init(inbuf_list,
                            sizeof(mca_coll_adapt_inbuf_t) + inbuf_extend_size,
                            opal_cache_line_size,
                            OBJ_CLASS(mca_coll_adapt_inbuf_t),
                            0,opal_cache_line_size,
                            FREE_LIST_NUM_INBUF_LIST,
                            FREE_LIST_MAX_INBUF_LIST,
                            FREE_LIST_INC_INBUF_LIST,
                            NULL, 0, NULL, NULL, NULL);
        //set up next_recv_segs
        next_recv_segs = (int32_t *)malloc(sizeof(int32_t) * tree->tree_nextsize);
        //wei mca_coll_adapt_inbuf_t * temp_inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(inbuf_list);
        mca_coll_adapt_inbuf_t * temp_inbuf = coll_adapt_alloc_inbuf(inbuf_list, real_seg_size, buff_type);
        distance = (char *)temp_inbuf->buff - lower_bound - (char *)temp_inbuf; //address of inbuf->buff to address of inbuf
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: distance %d, inbuf %p, inbuf->buff %p, inbuf->buff-lb %p, to_inbuf %p, inbuf_list %p\n", rank, distance, temp_inbuf, (char *)temp_inbuf->buff, (char *)temp_inbuf->buff - lower_bound, to_inbuf((char *)temp_inbuf->buff - lower_bound, distance), inbuf_list));
        //wei opal_free_list_return(inbuf_list, (opal_free_list_item_t*)temp_inbuf);
        coll_adapt_free_inbuf(inbuf_list, temp_inbuf, buff_type);
    }
    else {
        inbuf_list = NULL;
        next_recv_segs = NULL;
    }
    
    ompi_request_t * temp_request = NULL;
    //set up request
    temp_request = OBJ_NEW(ompi_request_t);
    OMPI_REQUEST_INIT(temp_request, false);
    temp_request->req_state = OMPI_REQUEST_ACTIVE;
    temp_request->req_type = 0;
    temp_request->req_free = adapt_request_free;
    temp_request->req_status.MPI_SOURCE = 0;
    temp_request->req_status.MPI_TAG = 0;
    temp_request->req_status.MPI_ERROR = 0;
    temp_request->req_status._cancelled = 0;
    temp_request->req_status._ucount = 0;
    *request = temp_request;

    //set up mutex
    mutex_recv_list = OBJ_NEW(opal_mutex_t);
    mutex_num_recv_segs = OBJ_NEW(opal_mutex_t);
    mutex_op_list = (opal_mutex_t **)malloc(sizeof(opal_mutex_t *) * num_segs);
    for (i=0; i<num_segs; i++) {
        mutex_op_list[i] = OBJ_NEW(opal_mutex_t);
    }
    mutex_num_sent = OBJ_NEW(opal_mutex_t);
    //create recv_list
    recv_list = OBJ_NEW(opal_list_t);
    
    //Set constant context for send and recv call back
    mca_coll_adapt_constant_reduce_context_t *con = OBJ_NEW(mca_coll_adapt_constant_reduce_context_t);
    con->count = count;
    con->seg_count = seg_count;
    con->datatype = dtype;
    con->comm = comm;
    con->segment_increment = segment_increment;
    con->num_segs = num_segs;
    con->request = temp_request;
    con->rank = rank;
    con->num_recv_segs = 0;
    con->num_sent_segs = 0;
    con->next_recv_segs = next_recv_segs;
    con->mutex_recv_list = mutex_recv_list;
    con->mutex_num_recv_segs = mutex_num_recv_segs;
    con->mutex_num_sent = mutex_num_sent;
    con->mutex_op_list = mutex_op_list;
    con->op = op;
    con->tree = tree;
    con->inbuf_list = inbuf_list;
    con->recv_list = recv_list;
    con->lower_bound = lower_bound;
    con->ongoing_send = 0;
    con->sbuf = (char *)sbuf;
    con->rbuf = (char *)rbuf;
    con->root = root;
    con->distance = distance;
    con->ireduce_tag = ireduce_tag;
    con->buff_type = buff_type;
    con->real_seg_size = real_seg_size;
    
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: start ireduce root %d tag %d\n", rank, tree->tree_root, ireduce_tag));

    // non leaf nodes
    if (tree->tree_nextsize > 0) {
        //set accumbuf
        accumbuf = (char **) malloc (sizeof(char*) * num_segs);
        if (root == rank && sbuf == MPI_IN_PLACE) {
            for (i=0; i<num_segs; i++) {
                accumbuf[i] = (char *)rbuf + (ptrdiff_t)i * (ptrdiff_t)segment_increment;
            }
        }
        else{
            for (i=0; i<num_segs; i++) {
                accumbuf[i] = NULL;
            }
        }
        
        con->accumbuf = accumbuf;

        /* set inbuf coresponding to accumbuf */
        if (GPU_BUFFER == buff_type) {
            con->accumbuf_to_inbuf = (mca_coll_adapt_inbuf_t **) malloc (sizeof(mca_coll_adapt_inbuf_t*) * num_segs);
            for (i=0; i<num_segs; i++) {
                con->accumbuf_to_inbuf[i] = NULL;
            }
        } else {
            con->accumbuf_to_inbuf = NULL;
        }
        
        
        //for the first batch of segments
        if (num_segs <= coll_adapt_ireduce_max_recv_requests) {
            min = num_segs;
        }
        else{
            min = coll_adapt_ireduce_max_recv_requests;
        }
        for (i=0; i<tree->tree_nextsize; i++) {
            next_recv_segs[i] = min - 1;
        }
        
        for( j = 0; j < min; j++ ) {
            //for each child
            for( i = 0; i < tree->tree_nextsize; i++ ) {
                seg_index = j;
                if (seg_index < num_segs) {
                    recv_count = seg_count;
                    if( seg_index == (num_segs-1) ){
                        recv_count = count - (ptrdiff_t)seg_count * (ptrdiff_t)seg_index;
                    }
                    char * temp_recv_buf = NULL;
                    mca_coll_adapt_inbuf_t * inbuf = NULL;
                    //set inbuf, if it it first child, recv on rbuf, else recv on inbuf
                    if (i==0 && sbuf != MPI_IN_PLACE && root == rank) {
                        temp_recv_buf = (char *)rbuf + (ptrdiff_t)j * (ptrdiff_t)segment_increment;
                    }
                    else {
                       //wei inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(inbuf_list);
                        inbuf = coll_adapt_alloc_inbuf(inbuf_list, real_seg_size, buff_type);
                        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: In ireduce, alloc inbuf %p\n", rank, inbuf));
                        temp_recv_buf = inbuf->buff - lower_bound;
                    }
                    //get context
                    mca_coll_adapt_reduce_context_t * context = (mca_coll_adapt_reduce_context_t *) opal_free_list_wait(coll_adapt_ireduce_context_free_list);
                    context->buff = temp_recv_buf;
                    context->frag_id = seg_index;
                    context->child_id = i;              //the id of peer in in the tree
                    context->peer = tree->tree_next[i];   //the actural rank of the peer
                    context->con = con;
                    OBJ_RETAIN(con);
                    context->inbuf = inbuf;
                    
                    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: In ireduce, create irecv for seg %d, peer %d, recv_count %d, inbuf %p tag %d\n", context->con->rank, context->frag_id, context->peer, recv_count, (void *)inbuf, (ireduce_tag << 16) + seg_index));
                    
                    //create a recv request
                    ompi_request_t *recv_req;
                    err = MCA_PML_CALL(irecv(temp_recv_buf, recv_count, dtype, tree->tree_next[i], (ireduce_tag << 16) + seg_index, comm, &recv_req));
                    if (MPI_SUCCESS != err) {
                        return err;
                    }
                    //invoke recv call back
                    ompi_request_set_callback(recv_req, recv_cb, context);
                }
            }
        }
    }
    
    //leaf nodes
    else{
        mca_coll_adapt_item_t *item;
        //set up recv_list
        for(seg_index = 0; seg_index < num_segs; seg_index++) {
            item = OBJ_NEW(mca_coll_adapt_item_t);
            item->id = seg_index;
            item->count = tree->tree_nextsize;
            opal_list_append(recv_list, (opal_list_item_t *)item);
        }
        if (num_segs <= coll_adapt_ireduce_max_send_requests) {
            min = num_segs;
        }
        else{
            min = coll_adapt_ireduce_max_send_requests;
        }
        con->accumbuf = accumbuf;
        con->accumbuf_to_inbuf = NULL;
        for(i = 0; i < min; i++) {
            OPAL_THREAD_LOCK (mutex_recv_list);
            item = get_next_ready_item(recv_list, tree->tree_nextsize);
            OPAL_THREAD_UNLOCK (mutex_recv_list);
            if (item != NULL) {
                send_count = seg_count;
                if(item->id == (num_segs-1)){
                    send_count = count - (ptrdiff_t)seg_count * (ptrdiff_t)item->id;
                }
                mca_coll_adapt_reduce_context_t * context = (mca_coll_adapt_reduce_context_t *) opal_free_list_wait(coll_adapt_ireduce_context_free_list);
                context->buff = (char*)sbuf + (ptrdiff_t)item->id * (ptrdiff_t)segment_increment;
                context->frag_id = item->id;
                context->peer = tree->tree_prev;   //the actural rank of the peer
                context->con = con;
                OBJ_RETAIN(con);
                context->inbuf = NULL;
                
                //atomic
                opal_atomic_add_32(&(context->con->ongoing_send), 1);
                OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: In ireduce, create isend to seg %d, peer %d, send_count %d tag %d\n", context->con->rank, context->frag_id, context->peer, send_count, (ireduce_tag << 16) + context->frag_id));
                
                //create send request
                ompi_request_t *send_req;
                err = MCA_PML_CALL( isend(context->buff, send_count, dtype, tree->tree_prev, (ireduce_tag << 16) + context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req) );
                if (MPI_SUCCESS != err) {
                    return err;
                }
                //release the item
                OBJ_RELEASE(item);
                
                //invoke send call back
                ompi_request_set_callback(send_req, send_cb, context);
            }
        }
        
    }
    
    return MPI_SUCCESS;
}

#if OPAL_CUDA_SUPPORT
int mca_coll_adapt_ireduce_cuda(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag)
{
    coll_adapt_ireduce_segment_size = 524288;
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if (0 == mca_coll_adapt_component.coll_adapt_cuda_enabled) {
        coll_adapt_cuda_init();
    }
    if( !( (coll_comm->cached_topochain) && (coll_comm->cached_topochain_root == root) ) ) {
        if( coll_comm->cached_topochain ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_topochain) );
        }
        ompi_coll_topo_gpu_t *gpu_topo = (ompi_coll_topo_gpu_t *)malloc(sizeof(ompi_coll_topo_gpu_t));
        coll_adapt_cuda_get_gpu_topo(gpu_topo);
        coll_comm->cached_topochain = ompi_coll_base_topo_build_topoaware_chain(comm, root, module, 4, 1, (void*)gpu_topo);
        //coll_comm->cached_topochain = ompi_coll_base_topo_build_topoaware_chain(comm, root, module, 3, 0, NULL);
        coll_comm->cached_topochain_root = root;
        coll_adapt_cuda_free_gpu_topo(gpu_topo);
        free(gpu_topo);
    }
    else {
    }
    //print_tree(coll_comm->cached_topochain, ompi_comm_rank(comm));
    return mca_coll_adapt_ireduce_generic(sbuf, rbuf, count, dtype, op, root, comm, request, module, coll_comm->cached_topochain, coll_adapt_ireduce_segment_size, ireduce_tag, GPU_BUFFER);
}

static int reduce_async_op_free_inbuf(mca_coll_adapt_reduce_context_t *context, mca_coll_adapt_item_t * item)
{
    int i;
    for (i = 0; i < item->count; i++) {
        if (item->inbuf_to_free[i] != NULL) {
            coll_adapt_free_inbuf(context->con->inbuf_list, item->inbuf_to_free[i], context->con->buff_type);
            item->inbuf_to_free[i] = NULL;
        }
    }
    return 1;
}

static int reduce_send_context_async_op_callback(mca_coll_adapt_reduce_context_t *send_context)
{
    opal_output(0, "progress reduce cb\n");
    ompi_request_t *send_req;
    mca_coll_adapt_item_t *item = send_context->buff_to_free_item; 
    reduce_async_op_free_inbuf(send_context, item);
    
    assert(item->op_event != NULL);
    mca_common_cuda_return_op_event_item(item->op_event);
    
    int send_count = send_context->con->seg_count;
    if (item->id == (send_context->con->num_segs - 1)) {
        send_count = send_context->con->count - item->id * send_context->con->seg_count;
    }

    int err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, (send_context->con->ireduce_tag << 16) + send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
    
    //release the item
    OBJ_RELEASE(item);
    ompi_request_set_callback(send_req, send_cb, send_context);
    return OMPI_SUCCESS;
}

#endif


