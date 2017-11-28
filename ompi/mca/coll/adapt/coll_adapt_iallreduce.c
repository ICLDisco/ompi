#include "ompi_config.h"
#include "ompi/communicator/communicator.h"
#include "coll_adapt_algorithms.h"
#include "coll_adapt_context.h"
#include "coll_adapt_item.h"
#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "opal/util/bit_ops.h"      //opal_next_poweroftwo

#define FREE_LIST_NUM_CONTEXT_LIST 10    //The start size of the context free list
#define FREE_LIST_MAX_CONTEXT_LIST 10000  //The max size of the context free list
#define FREE_LIST_INC_CONTEXT_LIST 10    //The incresment of the context free list
#define FREE_LIST_NUM_INBUF_LIST 2    //The start size of the context free list
#define FREE_LIST_MAX_INBUF_LIST 10000  //The max size of the context free list
#define FREE_LIST_INC_INBUF_LIST 2    //The incresment of the context free list

#define TEST printfno
#define COUNT_TIME 0

double starttime_0, endtime_0;
double totaltime = 0;

static int send_rd_cb(ompi_request_t *req);
static int recv_rd_cb(ompi_request_t *req);

static int send_rd_cb(ompi_request_t *req){
    mca_coll_adapt_allreduce_context_t *context = (mca_coll_adapt_allreduce_context_t *) req->req_complete_cb_data;
    
    TEST("[%d]: send_rd_cb, peer = %d, distance = %d, inbuf_ready = %d, sendbuf_ready = %d, inbuf = %p\n", ompi_comm_rank(context->con->comm), context->peer, context->distance, context->con->inbuf_ready, context->con->sendbuf_ready, (void *)context->inbuf);
    int err;
    int rank = ompi_comm_rank(context->con->comm);
    //set new distance
    int new_distance = 0;
    if (context->distance == 0) {
        new_distance = 1;
        //recv from rank+1 at last round, since this node just recv once at last,
        //so there is no need to use inbuf_list, set distance to adjsize+1
        mca_coll_adapt_allreduce_context_t * recv_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context->con->context_list);
        recv_context->inbuf = NULL;
        recv_context->newrank = context->newrank;
        recv_context->distance = context->con->adjsize+1;
        recv_context->peer = rank+1;
        recv_context->con = context->con;
        OBJ_RETAIN(recv_context->con);
        //create a recv request
        ompi_request_t *recv_req;
        TEST("[%d]: Recv(start in send_rd_cb): distance %d from %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
        err = MCA_PML_CALL(irecv(recv_context->con->recvbuf, recv_context->con->count, recv_context->con->datatype, recv_context->peer, recv_context->distance, recv_context->con->comm, &recv_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke recv call back
        ompi_request_set_callback(recv_req, recv_rd_cb, recv_context);
        
    }
    else{
        new_distance = context->distance << 1;
    }
    OPAL_THREAD_LOCK(context->con->mutex_buf);
    context->con->sendbuf_ready++;
    int ready = context->con->sendbuf_ready && context->con->inbuf_ready;
    OPAL_THREAD_UNLOCK(context->con->mutex_buf);
    if (ready) {
        opal_atomic_add_32(&(context->con->sendbuf_ready), -1);
        opal_atomic_add_32(&(context->con->inbuf_ready), -1);
        
        int newremote = 0;
        int remote = 0;
        mca_coll_adapt_allreduce_context_t * recv_context = NULL;
        //recv from new distance
        if (new_distance < context->con->adjsize && context->newrank >= 0) {
            mca_coll_adapt_inbuf_t * inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(context->con->inbuf_list);
            recv_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context->con->context_list);
            recv_context->inbuf = inbuf;
            recv_context->newrank = context->newrank;
            recv_context->distance = new_distance;
            newremote = context->newrank ^ new_distance;
            remote = (newremote < context->con->extra_ranks)?(newremote * 2 + 1):(newremote + context->con->extra_ranks);
            recv_context->peer = remote;
            recv_context->con = context->con;
            OBJ_RETAIN(recv_context->con);
            //create a recv request
            ompi_request_t *recv_req;
            TEST("[%d]: Recv(start in send cb): distance %d from %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
            err = MCA_PML_CALL(irecv(inbuf->buff-context->con->lower_bound, recv_context->con->count, recv_context->con->datatype, recv_context->peer, recv_context->distance, recv_context->con->comm, &recv_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke recv call back
            ompi_request_set_callback(recv_req, recv_rd_cb, recv_context);
        }
        //do the operation, commutative
        char* recvbuf;
        if (context->inbuf != NULL) {
            recvbuf = context->inbuf->buff-context->con->lower_bound;
            //sendbuf = recvbuf + sendbuf
            TEST("[%d]: recv_rd_cb, distance = %d, recvbuf[1] = %d,sendbuf[1] = %d\n", rank, context->distance, ((int *)recvbuf)[1], ((int *)context->con->sendbuf)[1]);
            ompi_op_reduce(context->con->op, recvbuf, context->con->sendbuf, context->con->count, context->con->datatype);
            TEST("[%d]: recv_rd_cb, distance = %d, sendbuf[1] = %d\n", rank, context->distance, ((int *)context->con->sendbuf)[1]);
            
        }
        else{
            recvbuf = context->con->recvbuf;
            //recvbuf = recvbuf + sendbuf
            TEST("[%d]: recv_rd_cb, distance = %d, recvbuf[1] = %d,sendbuf[1] = %d\n", rank, context->distance, ((int *)recvbuf)[1], ((int *)context->con->sendbuf)[1]);
            ompi_op_reduce(context->con->op, context->con->sendbuf, recvbuf, context->con->count, context->con->datatype);
            TEST("[%d]: recv_rd_cb, distance = %d, recvbuf[1] = %d\n", rank, context->distance, ((int *)recvbuf)[1]);
            context->con->sendbuf = recvbuf;
            
        }
        
        //send to new distance
        if (new_distance < context->con->adjsize && context->newrank >= 0) {
            mca_coll_adapt_allreduce_context_t * send_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context->con->context_list);
            send_context->inbuf = recv_context->inbuf;
            send_context->newrank = context->newrank;
            send_context->distance = new_distance;
            send_context->peer = remote;
            send_context->con = context->con;
            OBJ_RETAIN(send_context->con);
            
            //create a send request
            ompi_request_t *send_req;
            TEST("[%d]: Send(start in send cb): distance %d to %d, ongoing send %d\n", ompi_comm_rank(send_context->con->comm), send_context->distance, send_context->peer, send_context->con->total_send);
            err = MCA_PML_CALL(isend(send_context->con->sendbuf, send_context->con->count, send_context->con->datatype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke send call back
            ompi_request_set_callback(send_req, send_rd_cb, send_context);
        }
        
        //this is the last send
        if (new_distance >= context->con->adjsize){
            if (context->newrank >=0) {
                //at last, send to rank - 1
                if (rank < (2 * context->con->extra_ranks) && rank % 2 == 1) {
                    mca_coll_adapt_allreduce_context_t * send_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context->con->context_list);
                    send_context->inbuf = NULL;
                    send_context->newrank = context->newrank;
                    send_context->distance = context->con->adjsize+1;
                    send_context->peer = rank-1;
                    send_context->con = context->con;
                    OBJ_RETAIN(send_context->con);
                    //set new_distance, so in this turn would not enter the complete part
                    new_distance = context->distance;
                    //create a send request
                    ompi_request_t *send_req;
                    TEST("[%d]: Send(start in send cb, Last): distance %d to %d, ongoing send %d\n", ompi_comm_rank(send_context->con->comm), send_context->distance, send_context->peer, send_context->con->total_send);
                    err = MCA_PML_CALL(isend(send_context->con->sendbuf, send_context->con->count, send_context->con->datatype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
                    if (MPI_SUCCESS != err) {
                        return err;
                    }
                    //invoke send call back
                    ompi_request_set_callback(send_req, send_rd_cb, send_context);
                }
                //copy to recvbuf
                //ompi_datatype_copy_content_same_ddt(context->con->datatype, context->con->count, context->con->recvbuf, context->con->sendbuf);
            }
        }
    }
    
    opal_mutex_t * mutex_temp = context->con->mutex_total_send;
    OPAL_THREAD_LOCK(mutex_temp);
    TEST("[%d]: adjsize %d, new_distance %d, new_rank %d, total_send %d\n", rank, context->con->adjsize, new_distance, context->newrank, context->con->total_send);
    //this is the last send the node with newrank < 0 only do one send
    if (context->con->total_send == 1) {
        OPAL_THREAD_UNLOCK(mutex_temp);
        int complete;
        complete = opal_atomic_add_32(&(context->con->complete), 1);
        
        TEST("[%d]: last send, complete = %d, total_send = %d\n", ompi_comm_rank(context->con->comm), complete, context->con->total_send);
        if (complete == 2) {
            //signal
            TEST("[%d]: last send, signal\n", ompi_comm_rank(context->con->comm));
            ompi_request_t *temp_req = context->con->request;
            if (ready && context->inbuf != NULL) {
                TEST("[%d]: send_rd_cb return inbuf item\n", rank);
                opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
            }
            opal_free_list_t * temp = context->con->context_list;
            OBJ_RELEASE(context->con->inbuf_list);
            OBJ_RELEASE(context->con->mutex_buf);
            OBJ_RELEASE(context->con->mutex_total_send);
            OBJ_RELEASE(context->con->mutex_total_recv);
            OBJ_RELEASE(context->con);
            OBJ_RELEASE(context->con);
            opal_free_list_return(temp, (opal_free_list_item_t*)context);
            OBJ_RELEASE(temp);
            ompi_request_complete(temp_req, 1);
            if (COUNT_TIME) {
                endtime_0 = MPI_Wtime();
                totaltime += (endtime_0 - starttime_0);
                printf("[%d]: Total Time in Iallreduce: %lf, start %lf, end %lf\n", rank, totaltime, starttime_0, endtime_0);
            }
        }
    }
    else{
        context->con->total_send--;
        if (ready && context->inbuf != NULL) {
            TEST("[%d]: send_rd_cb return inbuf item\n", rank);
            opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
        }
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        OPAL_THREAD_UNLOCK(mutex_temp);
    }
    OPAL_THREAD_UNLOCK (req->req_lock);
    req->req_free(&req);
    TEST("[%d]: send_rd_cb finish\n", rank);
    return 1;
}

static int recv_rd_cb(ompi_request_t *req){
    
    mca_coll_adapt_allreduce_context_t *context = (mca_coll_adapt_allreduce_context_t *) req->req_complete_cb_data;
    
    TEST("[%d]: recv_rd_cb, peer = %d, distance = %d, inbuf_ready = %d, sendbuf_ready = %d, inbuf = %p\n", ompi_comm_rank(context->con->comm), context->peer, context->distance, context->con->inbuf_ready, context->con->sendbuf_ready, (void *)context->inbuf);
    int err;
    int rank = ompi_comm_rank(context->con->comm);
    //set new distance
    int new_distance = 0;
    if (context->distance == 0) {
        new_distance = 1;
    }
    else{
        new_distance = context->distance << 1;
    }
    
    OPAL_THREAD_LOCK(context->con->mutex_buf);
    context->con->inbuf_ready++;
    int ready = context->con->sendbuf_ready && context->con->inbuf_ready;
    OPAL_THREAD_UNLOCK(context->con->mutex_buf);
    if (ready) {
        opal_atomic_add_32(&(context->con->sendbuf_ready), -1);
        opal_atomic_add_32(&(context->con->inbuf_ready), -1);
        
        int newremote = 0;
        int remote = 0;
        mca_coll_adapt_allreduce_context_t * recv_context = NULL;
        //recv from new distance
        if (new_distance < context->con->adjsize && context->newrank >= 0) {
            mca_coll_adapt_inbuf_t * inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(context->con->inbuf_list);
            recv_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context->con->context_list);
            recv_context->inbuf = inbuf;
            recv_context->newrank = context->newrank;
            recv_context->distance = new_distance;
            newremote = context->newrank ^ new_distance;
            remote = (newremote < context->con->extra_ranks)?(newremote * 2 + 1):(newremote + context->con->extra_ranks);
            recv_context->peer = remote;
            recv_context->con = context->con;
            OBJ_RETAIN(recv_context->con);
            //create a recv request
            ompi_request_t *recv_req;
            TEST("[%d]: Recv(start in recv cb): distance %d from %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
            err = MCA_PML_CALL(irecv(inbuf->buff-context->con->lower_bound, recv_context->con->count, recv_context->con->datatype, recv_context->peer, recv_context->distance, recv_context->con->comm, &recv_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke recv call back
            ompi_request_set_callback(recv_req, recv_rd_cb, recv_context);
        }
        //do the operation, commutative
        char* recvbuf;
        if (context->inbuf != NULL) {
            recvbuf = context->inbuf->buff-context->con->lower_bound;
            //sendbuf = recvbuf + sendbuf
            TEST("[%d]: recv_rd_cb, distance = %d, recvbuf[1] = %d,sendbuf[1] = %d\n", rank, context->distance, ((int *)recvbuf)[1], ((int *)context->con->sendbuf)[1]);
            ompi_op_reduce(context->con->op, recvbuf, context->con->sendbuf, context->con->count, context->con->datatype);
            TEST("[%d]: recv_rd_cb, distance = %d, sendbuf[1] = %d\n", rank, context->distance, ((int *)context->con->sendbuf)[1]);

        }
        else{
            recvbuf = context->con->recvbuf;
            //recvbuf = recvbuf + sendbuf
            TEST("[%d]: recv_rd_cb, distance = %d, recvbuf[1] = %d,sendbuf[1] = %d\n", rank, context->distance, ((int *)recvbuf)[1], ((int *)context->con->sendbuf)[1]);
            ompi_op_reduce(context->con->op, context->con->sendbuf, recvbuf, context->con->count, context->con->datatype);
            TEST("[%d]: recv_rd_cb, distance = %d, recvbuf[1] = %d\n", rank, context->distance, ((int *)recvbuf)[1]);
            context->con->sendbuf = recvbuf;

        }
        
        //send to new distance
        if (new_distance < context->con->adjsize && context->newrank >= 0) {
            mca_coll_adapt_allreduce_context_t * send_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context->con->context_list);
            send_context->inbuf = recv_context->inbuf;
            send_context->newrank = context->newrank;
            send_context->distance = new_distance;
            send_context->peer = remote;
            send_context->con = context->con;
            OBJ_RETAIN(send_context->con);
            
            //create a send request
            ompi_request_t *send_req;
            TEST("[%d]: Send(start in recv cb): distance %d to %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
            err = MCA_PML_CALL(isend(send_context->con->sendbuf, send_context->con->count, send_context->con->datatype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke send call back
            ompi_request_set_callback(send_req, send_rd_cb, send_context);
        }
        
        //this is the last recv
        if (new_distance >= context->con->adjsize){
            if (context->newrank >=0) {
                //at last, send to rank - 1
                if (rank < (2 * context->con->extra_ranks) && rank % 2 == 1) {
                    mca_coll_adapt_allreduce_context_t * send_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context->con->context_list);
                    send_context->inbuf = NULL;
                    send_context->newrank = context->newrank;
                    send_context->distance = context->con->adjsize+1;
                    send_context->peer = rank-1;
                    send_context->con = context->con;
                    OBJ_RETAIN(send_context->con);
                    
                    //create a send request
                    ompi_request_t *send_req;
                    TEST("[%d]: Send(start in recv cb, Last): distance %d to %d\n", ompi_comm_rank(send_context->con->comm), send_context->distance, send_context->peer);
                    err = MCA_PML_CALL(isend(send_context->con->sendbuf, send_context->con->count, send_context->con->datatype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
                    if (MPI_SUCCESS != err) {
                        return err;
                    }
                    //invoke send call back
                    ompi_request_set_callback(send_req, send_rd_cb, send_context);
                }
                //copy to recvbuf
                //ompi_datatype_copy_content_same_ddt(context->con->datatype, context->con->count, context->con->recvbuf, context->con->sendbuf);
            }
        }
    }
    
    opal_mutex_t * mutex_temp = context->con->mutex_total_recv;
    OPAL_THREAD_LOCK(mutex_temp);
    //this is the last recv, the node with newrank < 0 only do one recv
    if (context->con->total_recv == 1){
        OPAL_THREAD_UNLOCK(mutex_temp);
        int complete = opal_atomic_add_32(&(context->con->complete), 1);
        TEST("[%d]: last recv, complete = %d\n", ompi_comm_rank(context->con->comm), complete);
        if (complete == 2) {
            //signal
            TEST("[%d]: last recv, signal\n", ompi_comm_rank(context->con->comm));
            ompi_request_t *temp_req = context->con->request;
            if (ready && context->inbuf != NULL) {
                TEST("[%d]: recv_rd_cb return inbuf item\n", rank);
                opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
            }
            opal_free_list_t * temp = context->con->context_list;
            OBJ_RELEASE(context->con->inbuf_list);
            OBJ_RELEASE(context->con->mutex_buf);
            OBJ_RELEASE(context->con->mutex_total_send);
            OBJ_RELEASE(context->con->mutex_total_recv);
            OBJ_RELEASE(context->con);
            OBJ_RELEASE(context->con);
            opal_free_list_return(temp, (opal_free_list_item_t*)context);
            OBJ_RELEASE(temp);
            ompi_request_complete(temp_req, 1);
            if (COUNT_TIME) {
                endtime_0 = MPI_Wtime();
                totaltime += (endtime_0 - starttime_0);
                printf("[%d]: Total Time in Iallreduce: %lf, start %lf, end %lf\n", rank, totaltime, starttime_0, endtime_0);
            }
        }
    }
    else{
        context->con->total_recv--;
        if (ready && context->inbuf != NULL) {
            TEST("[%d]: recv_rd_cb return inbuf item\n", rank);
            opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
        }
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        OPAL_THREAD_UNLOCK(mutex_temp);
    }
    OPAL_THREAD_UNLOCK (req->req_lock);
    req->req_free(&req);
    TEST("[%d]: recv_rd_cb finish\n", rank);
    return 1;
}

int mca_coll_adapt_iallreduce_intra_recursivedoubling(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    
    if (COUNT_TIME) {
        starttime_0 = MPI_Wtime();
    }
    
    ptrdiff_t extent, lower_bound, true_lower_bound, true_extent;
    int size, rank, adjsize, extra_ranks;
    char *accumbuf = NULL;
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    int err;
    //set up request
    ompi_request_t * temp_request = NULL;
    temp_request = OBJ_NEW(ompi_request_t);
    OMPI_REQUEST_INIT(temp_request, false);
    temp_request->req_type = 0;
    temp_request->req_free = adapt_request_free;
    temp_request->req_status.MPI_SOURCE = 0;
    temp_request->req_status.MPI_TAG = 0;
    temp_request->req_status.MPI_ERROR = 0;
    temp_request->req_status._cancelled = 0;
    temp_request->req_status._ucount = 0;
    *request = temp_request;
    
    /* Special case for size == 1 */
    if (1 == size) {
        if (MPI_IN_PLACE != sbuf) {
            ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, (char*)sbuf);
        }
        ompi_request_complete(temp_request, 1);
        return MPI_SUCCESS;
    }
    
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    ompi_datatype_get_true_extent(dtype, &true_lower_bound, &true_extent);
    
    /* Allocate and initialize temporary send buffer */
    accumbuf = (char*) malloc(sizeof(char *));
    if (MPI_IN_PLACE == sbuf) {
        accumbuf = rbuf;
    }
    else{
        accumbuf = sbuf;
    }
    
    /* Determine nearest power of two less than or equal to size */
    /* size = 10, adjsize =16 */
    adjsize = opal_next_poweroftwo (size);
    adjsize >>= 1;
    extra_ranks = size - adjsize;
    
    //set up free list
    opal_free_list_t * context_list = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(context_list,
                        sizeof(mca_coll_adapt_allreduce_context_t),
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_allreduce_context_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM_CONTEXT_LIST,
                        FREE_LIST_MAX_CONTEXT_LIST,
                        FREE_LIST_INC_CONTEXT_LIST,
                        NULL, 0, NULL, NULL, NULL);
    
    opal_free_list_t * inbuf_list = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(inbuf_list,
                        sizeof(mca_coll_adapt_inbuf_t) + true_extent + (ptrdiff_t)(count - 1) * extent,
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_inbuf_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM_INBUF_LIST,
                        FREE_LIST_MAX_INBUF_LIST,
                        FREE_LIST_INC_INBUF_LIST,
                        NULL, 0, NULL, NULL, NULL);
    
    //set up mutex
    opal_mutex_t * mutex_buf = OBJ_NEW(opal_mutex_t);
    opal_mutex_t * mutex_total_send = OBJ_NEW(opal_mutex_t);
    opal_mutex_t * mutex_total_recv = OBJ_NEW(opal_mutex_t);
    
    
    //Set constant context for send and recv call back
    mca_coll_adapt_constant_allreduce_context_t *con = OBJ_NEW(mca_coll_adapt_constant_allreduce_context_t);
    con->sendbuf = accumbuf;
    con->recvbuf = rbuf;
    con->count = count;
    con->datatype = dtype;
    con->comm = comm;
    con->request = temp_request;
    con->context_list = context_list;
    con->op = op;
    con->lower_bound = lower_bound;
    con->extra_ranks = extra_ranks;
    con->inbuf_list = inbuf_list;
    con->complete = 0;
    con->adjsize = adjsize;
    con->sendbuf_ready = 0;     //use to decide if sendbuf is ready for reuse
    con->inbuf_ready = 0;     //use to decide if inbuf has the data already
    con->total_send = 0;         //to tell how many sends are needed in total
    con->total_recv = 0;        //to tell how many recvs are needed in total
    con->mutex_buf = mutex_buf;
    con->mutex_total_send = mutex_total_send;
    con->mutex_total_recv = mutex_total_recv;
    
    /* Handle non-power-of-two case:
     - Even ranks less than 2 * extra_ranks send their data to (rank + 1), and
     sets new rank to -1.
     - Odd ranks less than 2 * extra_ranks receive data from (rank - 1),
     apply appropriate operation, and set new rank to rank/2
     - Everyone else sets rank to rank - extra_ranks
     Turn non-power-of-two case into power of two case to improve performance.
     Suppose size = 2^n + extra_ranks. By combining every pair of 2 nodes among 2 * extra_ranks of nodes into one node, 2*extra_ranks nodes become extra_ranks node.
     So the size become 2^n. The goal is to remove the extra_ranks number of nodes.
     */
    if (rank <  (2 * extra_ranks)) {
        if (0 == (rank % 2)) {
            TEST("[%d]: Case 1\n", rank);
            con->total_send = 1;
            con->total_recv = 1;
            int newrank = -1;
            //send to rank+1
            mca_coll_adapt_allreduce_context_t * send_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context_list);
            send_context->inbuf = NULL;
            send_context->newrank = newrank;
            send_context->distance = 0;
            send_context->peer = rank+1;
            send_context->con = con;
            OBJ_RETAIN(con);
            
            //create a send request
            ompi_request_t *send_req;
            TEST("[%d]: Send(start in main): distance %d to %d, ongoing send %d\n", ompi_comm_rank(send_context->con->comm), send_context->distance, send_context->peer, send_context->con->total_send);
            
            err = MCA_PML_CALL(isend(con->sendbuf, count, dtype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke send call back
            ompi_request_set_callback(send_req, send_rd_cb, send_context);
            
        }
        else {
            TEST("[%d]: Case 2\n", rank);
            con->total_send = log2_int(adjsize)+1;
            con->total_recv = con->total_send;
            int newrank = rank>>1;
            //recv from rank-1
            mca_coll_adapt_inbuf_t * inbuf = NULL;
            if (MPI_IN_PLACE == sbuf) {
                inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(inbuf_list);
            }
            mca_coll_adapt_allreduce_context_t * recv_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context_list);
            recv_context->inbuf = inbuf;
            recv_context->newrank = newrank;
            recv_context->distance = 0;
            recv_context->peer = rank-1;
            recv_context->con = con;
            OBJ_RETAIN(con);
            //there is no send going, sendbuf is ready, sendbuf is accumbuf
            opal_atomic_add_32(&(recv_context->con->sendbuf_ready), 1);
            //create a recv request
            ompi_request_t *recv_req;
            TEST("[%d]: Recv(start in main): distance %d from %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
            if (inbuf != NULL) {
                err = MCA_PML_CALL(irecv(inbuf->buff-lower_bound, count, dtype, recv_context->peer, recv_context->distance, comm, &recv_req));
            }
            else {
                err = MCA_PML_CALL(irecv(con->recvbuf, count, dtype, recv_context->peer, recv_context->distance, comm, &recv_req));
            }
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke recv call back
            ompi_request_set_callback(recv_req, recv_rd_cb, recv_context);
        }
    }
    else {
        TEST("[%d]: Case 3\n", rank);
        con->total_send = log2_int(adjsize);
        con->total_recv = con->total_send;
        int newrank = rank-extra_ranks;
        int newremote;
        int remote;
        //recv from distance = 1
        mca_coll_adapt_inbuf_t * inbuf = NULL;
        if (MPI_IN_PLACE == sbuf) {
            inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(inbuf_list);
        }
        mca_coll_adapt_allreduce_context_t * recv_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context_list);
        recv_context->inbuf = inbuf;
        recv_context->newrank = newrank;
        recv_context->distance = 1;
        /* Determine remote node */
        newremote = recv_context->newrank ^ recv_context->distance;
        remote = (newremote < extra_ranks)?(newremote * 2 + 1):(newremote + extra_ranks);
        recv_context->peer = remote;
        recv_context->con = con;
        OBJ_RETAIN(con);
        //create a recv request
        ompi_request_t *recv_req;
        TEST("[%d]: Recv(start in main): distance %d from %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
        if (inbuf != NULL) {
            err = MCA_PML_CALL(irecv(inbuf->buff-lower_bound, count, dtype, recv_context->peer, recv_context->distance, comm, &recv_req));
        }
        else {
            err = MCA_PML_CALL(irecv(con->recvbuf, count, dtype, recv_context->peer, recv_context->distance, comm, &recv_req));
        }
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke recv call back
        ompi_request_set_callback(recv_req, recv_rd_cb, recv_context);
        
        //send to distance = 1
        mca_coll_adapt_allreduce_context_t * send_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context_list);
        send_context->inbuf = recv_context->inbuf;
        send_context->newrank = newrank;
        send_context->distance = 1;
        send_context->peer = remote;
        send_context->con = con;
        OBJ_RETAIN(con);
        
        //create a send request
        ompi_request_t *send_req;
        TEST("[%d]: Send(start in main): distance %d to %d, ongoing send %d\n", ompi_comm_rank(send_context->con->comm), send_context->distance, send_context->peer, send_context->con->total_send);
        
        err = MCA_PML_CALL(isend(con->sendbuf, count, dtype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke send call back
        ompi_request_set_callback(send_req, send_rd_cb, send_context);
    }
    
    return MPI_SUCCESS;
}

static int recv_ring_cb(ompi_request_t *req);
static int send_ring_cb(ompi_request_t *req);
static int recv_ring_allgather_cb(ompi_request_t *req);
static int send_ring_allgather_cb(ompi_request_t *req);


static int send_ring_cb(ompi_request_t *req){
    mca_coll_adapt_allreduce_ring_context_t *context = (mca_coll_adapt_allreduce_ring_context_t *) req->req_complete_cb_data;
    
    TEST("[%d]: send_ring_cb, peer = %d, seg = %d\n", ompi_comm_rank(context->con->comm), context->peer, context->block*context->con->num_phases+context->phase);
    int rank = ompi_comm_rank(context->con->comm);
    int err;
    
    //recv from rank-1 for seg block*num_phases+phase
    mca_coll_adapt_allreduce_ring_context_t * recv_context_t = (mca_coll_adapt_allreduce_ring_context_t *) opal_free_list_wait(context->con->context_list);
    //no matter MPI_IN_PLACE or not, always recv on rbuf
    recv_context_t->buff = ((char*)context->con->rbuf) + (ptrdiff_t)(context->block_offset + context->phase_offset) * context->con->extent;
    recv_context_t->peer = context->con->tree->tree_prev; //(rank + size - 1) % size;
    recv_context_t->block = context->block;
    recv_context_t->block_offset = context->block_offset;
    recv_context_t->phase = context->phase;
    recv_context_t->phase_offset = context->phase_offset;
    recv_context_t->phase_count = context->phase_count;
    recv_context_t->inbuf = NULL;
    recv_context_t->con = context->con;
    OBJ_RETAIN(recv_context_t->con);
    //create a recv request
    ompi_request_t *recv_req_t;
    TEST("[%d]: Recv(start in send_ring_cb, whole): seg %d from %d count %d\n", rank, recv_context_t->block * recv_context_t->con->num_phases + recv_context_t->phase, recv_context_t->peer, recv_context_t->phase_count);
    err = MCA_PML_CALL(irecv(recv_context_t->buff, recv_context_t->phase_count, recv_context_t->con->dtype, recv_context_t->peer, recv_context_t->block * recv_context_t->con->num_phases + recv_context_t->phase, recv_context_t->con->comm, &recv_req_t));
    if (MPI_SUCCESS != err) {
        return err;
    }
    //invoke recv call back
    ompi_request_set_callback(recv_req_t, recv_ring_allgather_cb, recv_context_t);
    
    opal_free_list_t * temp = context->con->context_list;
    OBJ_RELEASE(context->con);
    opal_free_list_return(temp, (opal_free_list_item_t*)context);
    
    OPAL_THREAD_UNLOCK (req->req_lock);
    req->req_free(&req);
    TEST("[%d]: send_ring_cb finish\n", rank);
    return 1;
}

static int recv_ring_cb(ompi_request_t *req){
    mca_coll_adapt_allreduce_ring_context_t *context = (mca_coll_adapt_allreduce_ring_context_t *) req->req_complete_cb_data;
    
    TEST("[%d]: recv_ring_cb, peer = %d, seg = %d\n", ompi_comm_rank(context->con->comm), context->peer, context->block*context->con->num_phases+context->phase);
    int size = ompi_comm_size(context->con->comm);
    int rank = ompi_comm_rank(context->con->comm);
    int vrank = context->con->tree->tree_root;
    int err;
    //recv from rank-1 for seg (block-1)*num_phases+phase
    int block = (context->block + size - 1) % size;
    int block_count, phase_count;
    ptrdiff_t phase_offset, block_offset;
    int early_segcount;
    int late_segcount;
    int split_phase;
    char *sbuf = context->con->sbuf;
    if (context->block != ((vrank+1+size) % size)) {
        block_count = ((block < context->con->split_block)? context->con->early_blockcount : context->con->late_blockcount);
        block_offset = ((block < context->con->split_block) ? ((ptrdiff_t)block * (ptrdiff_t)context->con->early_blockcount) : ((ptrdiff_t)block * (ptrdiff_t)context->con->late_blockcount + context->con->split_block));
        COLL_BASE_COMPUTE_BLOCKCOUNT(block_count, context->con->num_phases, split_phase, early_segcount, late_segcount);
        phase_count = ((context->phase < split_phase) ? early_segcount : late_segcount);
        phase_offset = ((context->phase < split_phase) ? ((ptrdiff_t)context->phase * (ptrdiff_t)early_segcount) : ((ptrdiff_t)context->phase * (ptrdiff_t)late_segcount + split_phase));
        mca_coll_adapt_allreduce_ring_context_t * recv_context = (mca_coll_adapt_allreduce_ring_context_t *) opal_free_list_wait(context->con->context_list);
        if (sbuf == MPI_IN_PLACE) {
            mca_coll_adapt_inbuf_t * inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(context->con->inbuf_list);
            recv_context->buff = inbuf->buff - context->con->lower_bound;
            recv_context->inbuf = inbuf;
        }
        else {
            recv_context->buff = ((char*)context->con->rbuf) + (ptrdiff_t)(block_offset + phase_offset) * context->con->extent;
            recv_context->inbuf = NULL;
        }
        recv_context->peer = context->peer;
        recv_context->block = block;
        recv_context->block_offset = block_offset;
        recv_context->phase = context->phase;
        recv_context->phase_offset = phase_offset;
        recv_context->phase_count = phase_count;
        recv_context->con = context->con;
        OBJ_RETAIN(recv_context->con);
        //create a recv request
        ompi_request_t *recv_req;
        TEST("[%d]: Recv(start in recv_cb): seg %d from %d count %d\n", rank, recv_context->block * recv_context->con->num_phases + recv_context->phase, recv_context->peer, recv_context->phase_count);
        err = MCA_PML_CALL(irecv(recv_context->buff, recv_context->phase_count, recv_context->con->dtype, recv_context->peer, recv_context->block * recv_context->con->num_phases + recv_context->phase, recv_context->con->comm, &recv_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke recv call back
        ompi_request_set_callback(recv_req, recv_ring_cb, recv_context);
    }
    
    block = context->block;
    char * tmprecv = ((char*)context->con->rbuf) + (ptrdiff_t)(context->block_offset + context->phase_offset) * context->con->extent;
    if (sbuf == MPI_IN_PLACE) {
        //op inbuf and rbuf to rbuf
        TEST("[%d]: Op(in recv_cb), seg %d, rbuf %f, inbuf %f\n", rank, block * context->con->num_phases + context->phase, ((float *)tmprecv)[0], ((float *)context->buff)[0]);
        ompi_op_reduce(context->con->op, context->buff, tmprecv, context->phase_count, context->con->dtype);
        TEST("[%d]: Op(in recv_cb), seg %d, rbuf %f, inbuf %f\n", rank, block * context->con->num_phases + context->phase, ((float *)tmprecv)[0], ((float *)context->buff)[0]);
    }
    else {
        //op sbuf and rbuf to rbuf
        char * tmpsend = ((char*)context->con->sbuf) + (ptrdiff_t)(context->block_offset + context->phase_offset) * context->con->extent;
        TEST("[%d]: Op(in recv_cb), seg %d, rbuf %f, sbuf %f\n", rank, block * context->con->num_phases + context->phase, ((float *)tmprecv)[0], ((float *)tmpsend)[0]);
        ompi_op_reduce(context->con->op, tmpsend, tmprecv, context->phase_count, context->con->dtype);
        TEST("[%d]: Op(in recv_cb), seg %d, rbuf %f, sbuf %f\n", rank, block * context->con->num_phases + context->phase, ((float *)tmprecv)[0], ((float *)tmpsend)[0]);
        
    }
    
    if (context->block != ((vrank+1+size) % size)) {
        //send rbuf to rank+1
        mca_coll_adapt_allreduce_ring_context_t * send_context = (mca_coll_adapt_allreduce_ring_context_t *) opal_free_list_wait(context->con->context_list);
        send_context->buff = tmprecv;
        send_context->peer = context->con->tree->tree_next[0]; //(rank+1) % size;
        send_context->block = block;
        send_context->block_offset = context->block_offset;
        send_context->phase = context->phase;
        send_context->phase_offset = context->phase_offset;
        send_context->phase_count = context->phase_count;
        send_context->inbuf = NULL;
        send_context->con = context->con;
        OBJ_RETAIN(send_context->con);
        //send op result to rank+1
        ompi_request_t *send_req;
        TEST("[%d]: Send(start in recv_cb): seg %d to %d count %d\n", rank, block * context->con->num_phases + context->phase, send_context->peer, context->phase_count);
        err = MCA_PML_CALL(isend(tmprecv, context->phase_count, context->con->dtype, send_context->peer, block * context->con->num_phases + context->phase, MCA_PML_BASE_SEND_SYNCHRONOUS, context->con->comm, &send_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke send call back
        ompi_request_set_callback(send_req, send_ring_cb, send_context);
        
    }
    else {
        //send final op result to rank+1
        mca_coll_adapt_allreduce_ring_context_t * send_context = (mca_coll_adapt_allreduce_ring_context_t *) opal_free_list_wait(context->con->context_list);
        send_context->buff = tmprecv;
        send_context->peer = context->con->tree->tree_next[0];
        send_context->block = block;
        send_context->block_offset = context->block_offset;
        send_context->phase = context->phase;
        send_context->phase_offset = context->phase_offset;
        send_context->phase_count = context->phase_count;
        send_context->inbuf = NULL;
        send_context->con = context->con;
        OBJ_RETAIN(send_context->con);
        ompi_request_t *send_req;
        TEST("[%d]: Send(start in recv_cb, whole): seg %d to %d count %d\n", rank, context->block * context->con->num_phases + context->phase, send_context->peer, context->phase_count);
        err = MCA_PML_CALL(isend(tmprecv, context->phase_count, context->con->dtype, send_context->peer, context->block * context->con->num_phases + context->phase, MCA_PML_BASE_SEND_SYNCHRONOUS, context->con->comm, &send_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke send call back
        ompi_request_set_callback(send_req, send_ring_allgather_cb, send_context);
        
    }
    
    if (context->inbuf !=NULL) {
        opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
    }
    opal_free_list_t * temp = context->con->context_list;
    OBJ_RELEASE(context->con);
    opal_free_list_return(temp, (opal_free_list_item_t*)context);
    
    OPAL_THREAD_UNLOCK (req->req_lock);
    req->req_free(&req);
    TEST("[%d]: recv_ring_cb finish\n", rank);
    return 1;
    
}

static int send_ring_allgather_cb(ompi_request_t *req){
    mca_coll_adapt_allreduce_ring_context_t *context = (mca_coll_adapt_allreduce_ring_context_t *) req->req_complete_cb_data;
    
    TEST("[%d]: send_ring_allgather_cb, peer = %d, seg = %d\n", ompi_comm_rank(context->con->comm), context->peer, context->block*context->con->num_phases+context->phase);
    int size = ompi_comm_size(context->con->comm);
    int rank = ompi_comm_rank(context->con->comm);
    
    OPAL_THREAD_LOCK(context->con->mutex_complete);
    int complete = ++context->con->complete;
    if (complete == context->con->num_phases*size) {
        //signal
        OPAL_THREAD_UNLOCK(context->con->mutex_complete);
        TEST("[%d]: send_ring_allgather_cb signal, complete %d\n", rank, context->con->complete);
        ompi_request_t *temp_req = context->con->request;
        if (context->inbuf != NULL) {
            opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
        }
        OBJ_RELEASE(context->con->inbuf_list);
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con->mutex_complete);
        OBJ_RELEASE(context->con);
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        OBJ_RELEASE(temp);
        ompi_request_complete(temp_req, 1);
        
        OPAL_THREAD_UNLOCK (req->req_lock);
        req->req_free(&req);
        TEST("[%d]: send_ring_allgather_cb signal finish\n", rank);
        return 1;
        
    }
    else {
        OPAL_THREAD_UNLOCK(context->con->mutex_complete);
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        
        OPAL_THREAD_UNLOCK (req->req_lock);
        req->req_free(&req);
        TEST("[%d]: send_ring_allgather_cb finish complete %d\n", rank, complete);
        return 1;
        
    }
}

static int recv_ring_allgather_cb(ompi_request_t *req){
    mca_coll_adapt_allreduce_ring_context_t *context = (mca_coll_adapt_allreduce_ring_context_t *) req->req_complete_cb_data;
    
    TEST("[%d]: recv_ring_allgather_cb, peer = %d, seg = %d\n", ompi_comm_rank(context->con->comm), context->peer, context->block*context->con->num_phases+context->phase);
    int size = ompi_comm_size(context->con->comm);
    int rank = ompi_comm_rank(context->con->comm);
    int vrank = context->con->tree->tree_root;
    int err;
    
    //TODO:rank+2 notice
    if (context->block != ((vrank+2) % size)) {
        mca_coll_adapt_allreduce_ring_context_t * send_context = (mca_coll_adapt_allreduce_ring_context_t *) opal_free_list_wait(context->con->context_list);
        send_context->buff = context->buff;
        send_context->peer = context->con->tree->tree_next[0];  //(rank+1) % size;
        send_context->block = context->block;
        send_context->block_offset = context->block_offset;
        send_context->phase = context->phase;
        send_context->phase_offset = context->phase_offset;
        send_context->phase_count = context->phase_count;
        send_context->inbuf = NULL;
        send_context->con = context->con;
        OBJ_RETAIN(send_context->con);
        ompi_request_t *send_req;
        TEST("[%d]: Send(start in recv_allgather_cb): seg %d to %d count %d\n", rank, context->block * context->con->num_phases + context->phase, send_context->peer, context->phase_count);
        err = MCA_PML_CALL(isend(send_context->buff, context->phase_count, context->con->dtype, send_context->peer, context->block * context->con->num_phases + context->phase, MCA_PML_BASE_SEND_SYNCHRONOUS, context->con->comm, &send_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke send call back
        ompi_request_set_callback(send_req, send_ring_allgather_cb, send_context);
    }
    else {
        OPAL_THREAD_LOCK(context->con->mutex_complete);
        int complete = ++context->con->complete;
        if (complete == context->con->num_phases*size) {
            //signal
            OPAL_THREAD_UNLOCK(context->con->mutex_complete);
            TEST("[%d]: send_ring_allgather_cb signal, complete %d\n", rank, context->con->complete);
            ompi_request_t *temp_req = context->con->request;
            if (context->inbuf != NULL) {
                opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
            }
            OBJ_RELEASE(context->con->inbuf_list);
            opal_free_list_t * temp = context->con->context_list;
            OBJ_RELEASE(context->con->mutex_complete);
            OBJ_RELEASE(context->con);
            OBJ_RELEASE(context->con);
            opal_free_list_return(temp, (opal_free_list_item_t*)context);
            OBJ_RELEASE(temp);
            ompi_request_complete(temp_req, 1);
            
            OPAL_THREAD_UNLOCK (req->req_lock);
            req->req_free(&req);
            TEST("[%d]: send_ring_allgather_cb signal finish\n", rank);
            return 1;
            
        }
        else {
            OPAL_THREAD_UNLOCK(context->con->mutex_complete);
        }
        
    }
    if (context->inbuf != NULL) {
        opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
    }
    opal_free_list_t * temp = context->con->context_list;
    OBJ_RELEASE(context->con);
    opal_free_list_return(temp, (opal_free_list_item_t*)context);
    
    OPAL_THREAD_UNLOCK (req->req_lock);
    req->req_free(&req);
    TEST("[%d]: recv_ring_gather_cb finish\n", rank);
    return 1;
    
    
    
}
int mca_coll_adapt_iallreduce_intra_ring_segmented(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree, uint32_t segsize){
    
    size_t typelng;
    int err;
    int size = ompi_comm_size(comm);
    int rank = ompi_comm_rank(comm);
    int vrank = tree->tree_root;
    int segcount = count;
    int split_block;
    int split_phase;
    int early_blockcount;
    int late_blockcount;
    int early_segcount;
    int late_segcount;
    int num_phases;
    ptrdiff_t max_real_segsize;
    ptrdiff_t lower_bound, extent, gap;
    
    //set up request
    ompi_request_t * temp_request = NULL;
    temp_request = OBJ_NEW(ompi_request_t);
    OMPI_REQUEST_INIT(temp_request, false);
    temp_request->req_type = 0;
    temp_request->req_free = adapt_request_free;
    temp_request->req_status.MPI_SOURCE = 0;
    temp_request->req_status.MPI_TAG = 0;
    temp_request->req_status.MPI_ERROR = 0;
    temp_request->req_status._cancelled = 0;
    temp_request->req_status._ucount = 0;
    *request = temp_request;
    
    /* Special case for size == 1 */
    if (1 == size) {
        if (MPI_IN_PLACE != sbuf) {
            ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, (char*)sbuf);
        }
        ompi_request_complete(temp_request, 1);
        
        return MPI_SUCCESS;
    }
    
    /* Determine segment count based on the suggested segment size */
    ompi_datatype_type_size(dtype, &typelng);
    COLL_BASE_COMPUTED_SEGCOUNT(segsize, typelng, segcount);
    TEST("segsize %d, typelng %d, segcount %d, count %d\n", segsize, typelng, segcount, count);
    
    /* Special case for count less than size * segcount - use regular ring */
    if (count < (size * segcount)) {
        TEST("Segsize is too big\n");
        if (count < size) {
            TEST("Message is too small\n");
            return mca_coll_adapt_iallreduce_intra_recursivedoubling(sbuf, rbuf, count, dtype, op, comm, request, module);
        }
        else {
            TEST("Set num phases = 1\n");
            COLL_BASE_COMPUTE_BLOCKCOUNT(count, size, split_block, early_segcount, late_segcount );
            num_phases = 1;
        }
    }
    else {
        /* Determine the number of phases of the algorithm */
        num_phases = count / (size * segcount);
        if ((count % (size * segcount) >= size) &&
            (count % (size * segcount) > ((size * segcount) / 2))) {
            num_phases++;
        }
    }
    
    /* Determine the number of elements per block and corresponding
     block sizes.
     The blocks are divided into "early" and "late" ones:
     blocks 0 .. (split_block - 1) are "early" and
     blocks (split_block) .. (size - 1) are "late".
     Early blocks are at most 1 element larger than the late ones.
     Note, these blocks will be split into num_phases segments,
     out of the largest one will have early_segcount elements.
     */
    
    COLL_BASE_COMPUTE_BLOCKCOUNT(count, size, split_block, early_blockcount, late_blockcount );
    COLL_BASE_COMPUTE_BLOCKCOUNT(early_blockcount, num_phases, split_phase, early_segcount, late_segcount);
    
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    max_real_segsize = opal_datatype_span(&dtype->super, early_segcount, &gap);
    
    TEST("num_phases = %d, max_real_segsize = %d, lower_bound = %d\n", num_phases, max_real_segsize, lower_bound);
    
    //set up free list
    opal_free_list_t * context_list = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(context_list,
                        sizeof(mca_coll_adapt_allreduce_ring_context_t),
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_allreduce_ring_context_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM_CONTEXT_LIST,
                        FREE_LIST_MAX_CONTEXT_LIST,
                        FREE_LIST_INC_CONTEXT_LIST,
                        NULL, 0, NULL, NULL, NULL);
    
    opal_free_list_t * inbuf_list = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(inbuf_list,
                        sizeof(mca_coll_adapt_inbuf_t) + max_real_segsize,
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_inbuf_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM_INBUF_LIST,
                        FREE_LIST_MAX_INBUF_LIST,
                        FREE_LIST_INC_INBUF_LIST,
                        NULL, 0, NULL, NULL, NULL);
    
    //set up mutex
    opal_mutex_t * mutex_complete = OBJ_NEW(opal_mutex_t);
    
    //    /* Handle MPI_IN_PLACE */
    //    if (MPI_IN_PLACE != sbuf) {
    //        err = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, (char*)sbuf);
    //        if (MPI_SUCCESS != err) {
    //            return err;
    //        }
    //    }
    
    //Set constant context for send and recv call back
    mca_coll_adapt_constant_allreduce_ring_context_t *con = OBJ_NEW(mca_coll_adapt_constant_allreduce_ring_context_t);
    con->rbuf = rbuf;
    con->sbuf = sbuf;
    con->dtype = dtype;
    con->comm = comm;
    con->count = count;
    con->mutex_complete = mutex_complete;
    con->complete = 0;
    con->split_block = split_block;
    con->inbuf_list = inbuf_list;
    con->context_list = context_list;
    con->num_phases = num_phases;
    con->early_blockcount = early_blockcount;
    con->late_blockcount = late_blockcount;
    con->lower_bound = lower_bound;
    con->extent = extent;
    con->op = op;
    con->request = temp_request;
    con->tree = tree;
    
    int phase;
    int block;
    int block_count, phase_count;
    ptrdiff_t phase_offset, block_offset;
    //for the first block
    for (phase=0; phase<num_phases; phase++) {
        //recv from rank-1
        block = (vrank + size - 1) % size;
        block_count = ((block < split_block)? early_blockcount : late_blockcount);
        block_offset = ((block < split_block) ? ((ptrdiff_t)block * (ptrdiff_t)early_blockcount) : ((ptrdiff_t)block * (ptrdiff_t)late_blockcount + split_block));
        COLL_BASE_COMPUTE_BLOCKCOUNT(block_count, num_phases, split_phase, early_segcount, late_segcount);
        phase_count = ((phase < split_phase) ? early_segcount : late_segcount);
        phase_offset = ((phase < split_phase) ? ((ptrdiff_t)phase * (ptrdiff_t)early_segcount) : ((ptrdiff_t)phase * (ptrdiff_t)late_segcount + split_phase));
        mca_coll_adapt_allreduce_ring_context_t * recv_context = (mca_coll_adapt_allreduce_ring_context_t *) opal_free_list_wait(context_list);
        if (sbuf == MPI_IN_PLACE) {
            mca_coll_adapt_inbuf_t * inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(inbuf_list);
            recv_context->buff = inbuf->buff-lower_bound;
            recv_context->inbuf = inbuf;
        }
        else {
            recv_context->buff = ((char*)rbuf) + (ptrdiff_t)(block_offset + phase_offset) * extent;
            recv_context->inbuf = NULL;
        }
        recv_context->peer = tree->tree_prev; //(rank + size - 1) % size;
        recv_context->block = block;
        recv_context->block_offset = block_offset;
        recv_context->phase = phase;
        recv_context->phase_offset = phase_offset;
        recv_context->phase_count = phase_count;
        recv_context->con = con;
        OBJ_RETAIN(con);
        //create a recv request
        ompi_request_t *recv_req;
        TEST("[%d]: Recv(start in main): seg %d from %d count %d inbuf %p\n", rank, block*num_phases+phase, recv_context->peer, phase_count, (void *)recv_context->inbuf);
        err = MCA_PML_CALL(irecv(recv_context->buff, phase_count, dtype, recv_context->peer, block*num_phases+phase, comm, &recv_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke recv call back
        ompi_request_set_callback(recv_req, recv_ring_cb, recv_context);
        
        //send to rank+1
        block = vrank;
        block_count = ((block < split_block)? early_blockcount : late_blockcount);
        block_offset = ((block < split_block) ? ((ptrdiff_t)block * (ptrdiff_t)early_blockcount) : ((ptrdiff_t)block * (ptrdiff_t)late_blockcount + split_block));
        COLL_BASE_COMPUTE_BLOCKCOUNT(block_count, num_phases, split_phase, early_segcount, late_segcount);
        phase_count = ((phase < split_phase) ? early_segcount : late_segcount);
        phase_offset = ((phase < split_phase) ? ((ptrdiff_t)phase * (ptrdiff_t)early_segcount) : ((ptrdiff_t)phase * (ptrdiff_t)late_segcount + split_phase));
        mca_coll_adapt_allreduce_ring_context_t * send_context = (mca_coll_adapt_allreduce_ring_context_t *) opal_free_list_wait(context_list);
        if (sbuf == MPI_IN_PLACE) {
            send_context->buff = ((char*)rbuf) + (ptrdiff_t)(block_offset + phase_offset) * extent;
        }
        else {
            send_context->buff = ((char*)sbuf) + (ptrdiff_t)(block_offset + phase_offset) * extent;
        }
        send_context->peer = tree->tree_next[0]; //(rank + 1) % size;
        send_context->block = block;
        send_context->block_offset = block_offset;
        send_context->phase = phase;
        send_context->phase_offset = phase_offset;
        send_context->phase_count = phase_count;
        send_context->inbuf = NULL;
        send_context->con = con;
        OBJ_RETAIN(con);
        //create a send request
        ompi_request_t *send_req;
        TEST("[%d]: Send(start in main): seg %d to %d count %d\n", ompi_comm_rank(comm), block*num_phases+phase, send_context->peer, phase_count);
        err = MCA_PML_CALL(isend(send_context->buff, phase_count, dtype, send_context->peer, block*num_phases+phase, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke send call back
        ompi_request_set_callback(send_req, send_ring_cb, send_context);
    }
    
    return MPI_SUCCESS;
    
}

int temp_count = 0;

int mca_coll_adapt_iallreduce(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    //printf("ADAPT\n");
    TEST("[%" PRIx64 "] Adapt iallreduce %d, count %d\n", gettid(), temp_count++, count);

    //get iallreduce tag
    int size = ompi_comm_size(comm);
    int iallreduce_tag = opal_atomic_add_32(&(comm->c_iallreduce_tag), size);
    iallreduce_tag = (iallreduce_tag % 4096) + 4097;

    //    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_topoaware_ring(comm, module);
    //    //print_tree(tree, ompi_comm_rank(comm));
    //eturn mca_coll_adapt_iallreduce_intra_ring_segmented(sbuf, rbuf, count, dtype, op, comm, request, module, tree, 1024000);
    //int r = mca_coll_adapt_iallreduce_intra_generic(sbuf, rbuf, count, dtype, op, comm, request, module, NULL, 1024000, iallreduce_tag);
    int r = mca_coll_adapt_iallreduce_intra_recursivedoubling(sbuf, rbuf, count, dtype, op, comm, request, module);
    return r;
}

