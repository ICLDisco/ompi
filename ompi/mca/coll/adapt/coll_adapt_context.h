#include "ompi/mca/coll/coll.h"
#include "opal/class/opal_free_list.h"
#include "opal/class/opal_list.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/op/op.h"
#include "ompi/mca/coll/base/coll_base_topo.h"
#include "coll_adapt_inbuf.h"

#if OPAL_CUDA_SUPPORT  
#define CPU_BUFFER_MEMCPY_DONE  1
#define CPU_BUFFER_MEMCPY_NOT_DONE  0
#define CPU_BUFFER_MEMCPY_PENDING  2

#define COLL_ADAPT_CONTEXT_FLAGS_CUDA_BCAST     0x2
#define COLL_ADAPT_CONTEXT_FLAGS_CUDA_REDUCE     0x4
#endif

/* bcast constant context in bcast context */
struct mca_coll_adapt_constant_bcast_context_s {
    opal_object_t  super;
    int root;
    size_t count;
    size_t seg_count;
    ompi_datatype_t * datatype;
    ompi_communicator_t * comm;
    int real_seg_size;
    int num_segs;
    ompi_request_t * request;
    opal_mutex_t * mutex;
    int* recv_array;
    int* send_array;
    int num_recv_segs; /* store the length of the fragment array, how many fragments are recevied */
    int num_recv_fini;  /* store how many segs is finish recving */
    int num_sent_segs;  /* number of sent segments */
    ompi_coll_tree_t * tree;
    int ibcast_tag;
    int gpu_use_cpu_buff;
#if OPAL_CUDA_SUPPORT    
    char **cpu_buff_list;
    int *cpu_buff_memcpy_flags;
    int *cpu_buff_list_ref_count;
#endif
};

typedef struct mca_coll_adapt_constant_bcast_context_s mca_coll_adapt_constant_bcast_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_constant_bcast_context_t);


/* bcast context of each segment*/
typedef struct mca_coll_adapt_bcast_context_s mca_coll_adapt_bcast_context_t;

typedef int (*mca_coll_adapt_bcast_cuda_callback_fn_t)(mca_coll_adapt_bcast_context_t *context);

struct mca_coll_adapt_bcast_context_s {
    opal_free_list_item_t super;
#if OPAL_CUDA_SUPPORT
    int flags;
#endif
    char *buff;
    int frag_id;
    int child_id;
    int peer;
    mca_coll_adapt_constant_bcast_context_t * con;
#if OPAL_CUDA_SUPPORT
    int debug_flag; 
    size_t send_count;
    mca_coll_adapt_bcast_cuda_callback_fn_t cuda_callback; 
#endif
};

OBJ_CLASS_DECLARATION(mca_coll_adapt_bcast_context_t);

/* reduce constant context in reduce context */
struct mca_coll_adapt_constant_reduce_context_s {
    opal_object_t  super;
    size_t count;
    size_t seg_count;
    ompi_datatype_t * datatype;
    ompi_communicator_t * comm;
    size_t real_seg_size;
    int segment_increment;      /* increment of each segment */
    int num_segs;
    ompi_request_t * request;
    int rank;
    int32_t num_recv_segs; /* store the length of the fragment array, how many fragments are recevied */
    int32_t num_sent_segs;  /* number of sent segments */
    int32_t* next_recv_segs;  /* next seg need to be received for every children */
    opal_mutex_t * mutex_recv_list;     /* use to lock recv list */
    opal_mutex_t * mutex_num_recv_segs;     /* use to lock num_recv_segs */
    opal_mutex_t * mutex_num_sent;     /* use to lock num_sent */
    opal_mutex_t ** mutex_op_list;   /* use to lock each segment when do the reduce op */
    ompi_op_t * op;  /* reduce operation */
    ompi_coll_tree_t * tree;
    char ** accumbuf;   /* accumulate buff, used in reduce */
    mca_coll_adapt_inbuf_t ** accumbuf_to_inbuf;  /* inbuf list address of accumbuf */
    opal_free_list_t *inbuf_list;
    opal_list_t *recv_list;    /* a list to store the segments which are received and not yet be sent */
    ptrdiff_t lower_bound;
    int32_t ongoing_send;   /* how many send is posted but not finished */
    char * sbuf;    /* inputed sbuf */
    char * rbuf;    /* inputed rbuf */
    int root;
    int distance;   /* address of inbuf->buff to address of inbuf */
    int ireduce_tag;
    int buff_type; /* memory type, CPU or GPU */
};

typedef struct mca_coll_adapt_constant_reduce_context_s mca_coll_adapt_constant_reduce_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_constant_reduce_context_t);

/* reduce context of each segment */
typedef struct mca_coll_adapt_reduce_context_s mca_coll_adapt_reduce_context_t;

typedef int (*mca_coll_adapt_reduce_cuda_callback_fn_t)(mca_coll_adapt_reduce_context_t *context);

struct mca_coll_adapt_reduce_context_s {
    opal_free_list_item_t super;
#if OPAL_CUDA_SUPPORT
    int flags;
#endif
    char *buff;
    int frag_id;
    int child_id;
    int peer;
    mca_coll_adapt_constant_reduce_context_t * con;
    mca_coll_adapt_inbuf_t *inbuf;  /* only used in reduce, store the incoming segment */
#if OPAL_CUDA_SUPPORT
    void *buff_to_free_item;
    mca_coll_adapt_reduce_cuda_callback_fn_t cuda_callback; 
#endif
};

OBJ_CLASS_DECLARATION(mca_coll_adapt_reduce_context_t);
