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

#define SEND_NUM 2    //send how many neighbor at first
#define RECV_NUM 3    //recv how many neighbor at first
#define FREE_LIST_NUM_CONTEXT_LIST 10    //The start size of the context free list
#define FREE_LIST_MAX_CONTEXT_LIST 10000  //The max size of the context free list
#define FREE_LIST_INC_CONTEXT_LIST 10    //The incresment of the context free list
#define TEST printfno

//return value indicate the whole operation is finished or not
static int send_cb(ompi_request_t *req){
    mca_coll_adapt_alltoallv_context_t *context = (mca_coll_adapt_alltoallv_context_t *) req->req_complete_cb_data;
    
    opal_atomic_add_32(&(context->con->finished_send), 1);

    int rank, size, err;
    size = ompi_comm_size(context->con->comm);
    rank = ompi_comm_rank(context->con->comm);
    
    TEST("[%d]: send_cb, peer = %d, distance = %d\n", ompi_comm_rank(context->con->comm), context->peer, context->distance);
    
    int new_distance = context->con->next_send_distance;
    opal_atomic_add_32(&(context->con->next_send_distance), 1);
    if (new_distance < size) {
        mca_coll_adapt_alltoallv_context_t * send_context = (mca_coll_adapt_alltoallv_context_t *) opal_free_list_wait(context->con->context_list);
        send_context->distance = new_distance;
        send_context->peer = (rank+new_distance)%size;
        send_context->start = ((char *)context->con->sbuf) + (ptrdiff_t)context->con->sdisps[send_context->peer] * context->con->sext;
        send_context->con = context->con;
        OBJ_RETAIN(send_context->con);
        
        //create a send request
        ompi_request_t *send_req;
        TEST("[%d]: Send(start in send_cb): distance %d to %d\n", ompi_comm_rank(send_context->con->comm), send_context->distance, send_context->peer);
        opal_atomic_add_32(&(send_context->con->ongoing_send), 1);
        err = MCA_PML_CALL(isend(send_context->start, send_context->con->scounts[send_context->peer], send_context->con->sdtype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke send call back
        if(!ompi_request_set_callback(send_req, send_cb, send_context)) {
            send_cb(send_req);
        }
    }
    
    opal_free_list_t * temp = context->con->context_list;
    //check whether complete the request
    if (context->con->finished_send >= size && context->con->ongoing_send == 1) {
        TEST("[%d]: Last send\n", ompi_comm_rank(context->con->comm));
        opal_atomic_add_32(&(context->con->complete), 1);
        //all finished
        if (context->con->complete == 2) {
            TEST("[%d]: Signal in send\n", ompi_comm_rank(context->con->comm));
            ompi_request_t *temp_req = context->con->request;
            if (MPI_IN_PLACE == context->con->origin_sbuf) {
                free(context->con->sbuf);
            }
            OBJ_RELEASE(context->con);
            OBJ_RELEASE(context->con);
            opal_free_list_return(temp, (opal_free_list_item_t*)context);
            OBJ_RELEASE(temp);
            ompi_request_complete(temp_req, 1);
            return 1;
        }
    }
    opal_atomic_add_32(&(context->con->ongoing_send), -1);
    OBJ_RELEASE(context->con);
    opal_free_list_return(temp, (opal_free_list_item_t*)context);
    
    return 0;
}

static int recv_cb(ompi_request_t *req){
    mca_coll_adapt_alltoallv_context_t *context = (mca_coll_adapt_alltoallv_context_t *) req->req_complete_cb_data;
    
    opal_atomic_add_32(&(context->con->finished_recv), 1);

    int rank, size, err;
    size = ompi_comm_size(context->con->comm);
    rank = ompi_comm_rank(context->con->comm);
    
    TEST("[%d]: recv_cb, peer = %d, distance = %d\n", ompi_comm_rank(context->con->comm), context->peer, context->distance);
    
    int new_distance = context->con->next_recv_distance;
    opal_atomic_add_32(&(context->con->next_recv_distance), 1);
    if (new_distance < size) {
        mca_coll_adapt_alltoallv_context_t * recv_context = (mca_coll_adapt_alltoallv_context_t *) opal_free_list_wait(context->con->context_list);
        recv_context->distance = new_distance;
        recv_context->peer = (rank-new_distance+size)%size;
        recv_context->start = ((char *)context->con->rbuf) + (ptrdiff_t)context->con->rdisps[recv_context->peer] * context->con->rext;
        recv_context->con = context->con;
        OBJ_RETAIN(recv_context->con);
        
        //create a recv request
        ompi_request_t *recv_req;
        TEST("[%d]: Recv(start in recv_cb): distance %d from %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
        opal_atomic_add_32(&(recv_context->con->ongoing_recv), 1);
        err = MCA_PML_CALL(irecv(recv_context->start, recv_context->con->rcounts[recv_context->peer], recv_context->con->rdtype, recv_context->peer, recv_context->distance, recv_context->con->comm, &recv_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke recv call back
        if(!ompi_request_set_callback(recv_req, recv_cb, recv_context)) {
            if (recv_cb(recv_req)) {
                return 1;
            }
        }
    }
    
    opal_free_list_t * temp = context->con->context_list;
    //check whether complete the request
    if (context->con->finished_recv >= size  && context->con->ongoing_recv == 1) {
        TEST("[%d]: Last recv\n", ompi_comm_rank(context->con->comm));
        opal_atomic_add_32(&(context->con->complete), 1);
        if (context->con->complete == 2) {
            TEST("[%d]: Singal in recv\n", ompi_comm_rank(context->con->comm));
            ompi_request_t *temp_req = context->con->request;
            if (MPI_IN_PLACE == context->con->origin_sbuf) {
                free(context->con->sbuf);
            }
            OBJ_RELEASE(context->con);
            OBJ_RELEASE(context->con);
            opal_free_list_return(temp, (opal_free_list_item_t*)context);
            OBJ_RELEASE(temp);
            ompi_request_complete(temp_req, 1);
            return 1;
        }
    }
    opal_atomic_add_32(&(context->con->ongoing_recv), -1);
    OBJ_RELEASE(context->con);
    opal_free_list_return(temp, (opal_free_list_item_t*)context);
    
    return 0;
}


int mca_coll_adapt_alltoallv(const void *sbuf, const int *scounts, const int *sdisps, struct ompi_datatype_t *sdtype, void* rbuf, const int *rcounts, const int *rdisps, struct ompi_datatype_t *rdtype, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    TEST("In adapt alltoallv\n");
    
    int rank, size, i;
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    
    ptrdiff_t sext, rext;
    
    char* temp_sbuf;
    //handle MPI_IN_PLACE, set up temp_sbuf
    if (MPI_IN_PLACE == sbuf) {
        ptrdiff_t lb, true_lb, true_sext;
        ompi_datatype_get_extent(sdtype, &lb, &sext);
        ompi_datatype_get_true_extent(sdtype, &true_lb, &true_sext);
        /* Allocate and initialize temporary send buffer */
        int total_scounts = 0;
        for (i=0; i<size; i++) {
            total_scounts+=scounts[i];
        }
        temp_sbuf = (char*) malloc(true_sext + (ptrdiff_t)(total_scounts - 1) * sext);
        ompi_datatype_copy_content_same_ddt(sdtype, total_scounts, temp_sbuf, (char*)rbuf);
    }
    else{
        temp_sbuf = (char*)sbuf;
    }
    ompi_datatype_type_extent(sdtype, &sext);
    ompi_datatype_type_extent(rdtype, &rext);

    ompi_request_t * temp_request = NULL;
    //set up request
    temp_request = OBJ_NEW(ompi_request_t);
    OMPI_REQUEST_INIT(temp_request, false);
    temp_request->req_type = 0;
    temp_request->req_free = adapt_request_free;
    temp_request->req_status.MPI_SOURCE = 0;
    temp_request->req_status.MPI_TAG = 0;
    temp_request->req_status.MPI_ERROR = 0;
    temp_request->req_status._cancelled = 0;
    temp_request->req_status._ucount = 0;
    
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

    //Set constant context for send and recv call back
    mca_coll_adapt_constant_alltoallv_context_t *con = OBJ_NEW(mca_coll_adapt_constant_alltoallv_context_t);
    con->sbuf = temp_sbuf;
    con->scounts = scounts;
    con->sdisps = sdisps;
    con->sdtype = sdtype;
    con->rbuf = rbuf;
    con->rcounts = rcounts;
    con->rdisps = rdisps;
    con->rdtype = rdtype;
    con->comm = comm;
    con->request = temp_request;
    con->context_list = context_list;
    con->sext = sext;
    con->rext = rext;
    con->origin_sbuf = sbuf;
    con->finished_send = 0;
    con->finished_recv = 0;
    con->ongoing_send = 0;
    con->ongoing_recv = 0;
    con->complete = 0;
    con->next_send_distance = 0;
    con->next_recv_distance = 0;

    int min, err;
    //recv from first a few neighbors
    min = (RECV_NUM < size) ? RECV_NUM : size;
    for (i=0; i<min; i++) {
        if (con->next_recv_distance < size) {
            mca_coll_adapt_alltoallv_context_t * recv_context = (mca_coll_adapt_alltoallv_context_t *) opal_free_list_wait(context_list);
            recv_context->distance = con->next_recv_distance;
            opal_atomic_add_32(&(con->next_recv_distance), 1);
            recv_context->peer = (rank-recv_context->distance+size)%size;
            recv_context->start = ((char *)rbuf) + (ptrdiff_t)rdisps[recv_context->peer] * rext;
            recv_context->con = con;
            OBJ_RETAIN(con);
            
            //create a recv request
            ompi_request_t *recv_req;
            TEST("[%d]: Recv(start in main): distance %d from %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
            opal_atomic_add_32(&(recv_context->con->ongoing_recv), 1);
            err = MCA_PML_CALL(irecv(recv_context->start, rcounts[recv_context->peer], rdtype, recv_context->peer, recv_context->distance, comm, &recv_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke recv call back
            if(!ompi_request_set_callback(recv_req, recv_cb, recv_context)) {
                recv_cb(recv_req);
            }
        }
    }
    
    //send to first a few neighbors
    min = (SEND_NUM < size) ? SEND_NUM : size;
    for (i=0; i<min; i++) {
        if (con->next_send_distance < size) {
            mca_coll_adapt_alltoallv_context_t * send_context = (mca_coll_adapt_alltoallv_context_t *) opal_free_list_wait(context_list);
            send_context->distance = con->next_send_distance;
            opal_atomic_add_32(&(con->next_send_distance), 1);
            send_context->peer = (rank+send_context->distance)%size;
            send_context->start = ((char *)sbuf) + (ptrdiff_t)sdisps[send_context->peer] * sext;
            send_context->con = con;
            OBJ_RETAIN(con);
            
            //create a send request
            ompi_request_t *send_req;
            TEST("[%d]: Send(start in main): distance %d to %d\n", ompi_comm_rank(send_context->con->comm), send_context->distance, send_context->peer);
            opal_atomic_add_32(&(send_context->con->ongoing_send), 1);
            err = MCA_PML_CALL(isend(send_context->start, scounts[send_context->peer], sdtype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke send call back
            if(!ompi_request_set_callback(send_req, send_cb, send_context)) {
                send_cb(send_req);
            }
        }
    }
    
    ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);
    
    return MPI_SUCCESS;
}
