#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_base_topo.h"  //ompi_coll_tree_t
#include "ompi/mca/coll/base/coll_base_functions.h" //COLL_BASE_COMPUTE_BLOCKCOUNT
#include <math.h>	

typedef struct mca_coll_adapt_algorithm_index_s {
    int algorithm_index;
    uintptr_t algorithm_fn_ptr;
}mca_coll_adapt_algorithm_index_t;	

/* Bcast */
int mca_coll_adapt_ibcast_init(void);
int mca_coll_adapt_ibcast_fini(void);

int mca_coll_adapt_bcast(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_adapt_ibcast(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);
int mca_coll_adapt_ibcast_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree, size_t seg_size, int ibcast_tag, int gpu_use_cpu_buff);
int mca_coll_adapt_ibcast_tuned(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_in_order_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_pipeline(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_topoaware_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_topoaware_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_cuda(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_two_trees_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t** trees, int ibcast_tag);
int mca_coll_adapt_ibcast_two_trees_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_two_trees_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_two_chains(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);


/* Reduce */
int mca_coll_adapt_ireduce_init(void);
int mca_coll_adapt_ireduce_fini(void);

int mca_coll_adapt_reduce(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_ireduce(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ireduce_generic(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree, size_t seg_size, int ireduce_tag, int is_gpu_buff);
int mca_coll_adapt_ireduce_tuned(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag);
int mca_coll_adapt_ireduce_binomial(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag);
int mca_coll_adapt_ireduce_in_order_binomial(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag);
int mca_coll_adapt_ireduce_binary(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag);
int mca_coll_adapt_ireduce_pipeline(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag);
int mca_coll_adapt_ireduce_chain(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag);
int mca_coll_adapt_ireduce_linear(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag);
int mca_coll_adapt_ireduce_topoaware_linear(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag);
int mca_coll_adapt_ireduce_topoaware_chain(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag);


/* Allreduce */
int mca_coll_adapt_allreduce_intra_nonoverlapping(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_allreduce_intra_recursivedoubling(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_allreduce_intra_generic(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, int iallreduce_tag);

int mca_coll_adapt_allreduce(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_iallreduce_intra_recursivedoubling(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_iallreduce_intra_ring_segmented(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree, uint32_t segsize);

int mca_coll_adapt_iallreduce_intra_generic(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree, uint32_t segsize, int iallreduce_tag);

int mca_coll_adapt_iallreduce(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_alltoallv(const void *sbuf, const int *scounts, const int *sdisps, struct ompi_datatype_t *sdtype, void* rbuf, const int *rcounts, const int *rdisps, struct ompi_datatype_t *rdtype, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_ialltoallv(const void *sbuf, const int *scounts, const int *sdisps, struct ompi_datatype_t *sdtype, void* rbuf, const int *rcounts, const int *rdisps, struct ompi_datatype_t *rdtype, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

/* CUDA */
#if OPAL_CUDA_SUPPORT
int mca_coll_adapt_ibcast_cuda(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ireduce_cuda(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ireduce_tag);
#endif
//get thread id for test
//static inline uint64_t gettid(void) {
//    pthread_t ptid = pthread_self();
//    uint64_t threadId = 0;
//    int min;
//    if (sizeof(threadId) < sizeof(ptid)) {
//        min = sizeof(threadId);
//    }
//    else
//        min = sizeof(ptid);
//    memcpy(&threadId, &ptid, min);
//    return threadId;
//}

static void printfno(){
    
}

static int log2_int(int n)
{
    int i=0;
    while (n>>=1) ++i;
    return i;
}

//print tree for test
static inline void print_tree(ompi_coll_tree_t* tree, int rank) {
    int i;
    printf("[%d, prev = %d, next_size = %d, root =%d]:", rank, tree->tree_prev, tree->tree_nextsize, tree->tree_root);
    for( i = 0; i < tree->tree_nextsize; i++ ){
        printf(" %d", tree->tree_next[i]);
    }
    if (rank == tree->tree_root) {
        printf(" root = %d", tree->tree_root);
    }
    printf("\n");
}

static inline int adapt_request_free(ompi_request_t** request)
{
    OPAL_THREAD_LOCK ((*request)->req_lock);
    (*request)->req_state = OMPI_REQUEST_INVALID;
    OPAL_THREAD_UNLOCK ((*request)->req_lock);
    OBJ_RELEASE(*request);
    *request = MPI_REQUEST_NULL;
    return OMPI_SUCCESS;
}
