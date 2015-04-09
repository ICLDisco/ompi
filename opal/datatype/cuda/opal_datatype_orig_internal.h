#ifndef OPAL_DATATYPE_ORIG_INTERNAL_H_HAS_BEEN_INCLUDED
#define OPAL_DATATYPE_ORIG_INTERNAL_H_HAS_BEEN_INCLUDED

#include <stdbool.h>

#include "opal_config.h"

/* original OMPI */
#define OPAL_DECLSPEC

#define OPAL_PTRDIFF_TYPE ptrdiff_t
#define DT_STATIC_STACK_SIZE    5                /**< This should be sufficient for most applications */

#if OPAL_ENABLE_DEBUG
/* Any kind of unique ID should do the job */
#define OPAL_OBJ_MAGIC_ID ((0xdeafbeedULL << 32) + 0xdeafbeedULL)
#endif

/* keep the last 16 bits free for data flags */
#define CONVERTOR_DATATYPE_MASK    0x0000FFFF
#define CONVERTOR_SEND_CONVERSION  0x00010000
#define CONVERTOR_RECV             0x00020000
#define CONVERTOR_SEND             0x00040000
#define CONVERTOR_HOMOGENEOUS      0x00080000
#define CONVERTOR_NO_OP            0x00100000
#define CONVERTOR_WITH_CHECKSUM    0x00200000
#define CONVERTOR_CUDA             0x00400000
#define CONVERTOR_CUDA_ASYNC       0x00800000
#define CONVERTOR_TYPE_MASK        0x00FF0000
#define CONVERTOR_STATE_START      0x01000000
#define CONVERTOR_STATE_COMPLETE   0x02000000
#define CONVERTOR_STATE_ALLOC      0x04000000
#define CONVERTOR_COMPLETED        0x08000000

#define OPAL_DATATYPE_LOOP           0
#define OPAL_DATATYPE_END_LOOP       1
#define OPAL_DATATYPE_LB             2
#define OPAL_DATATYPE_UB             3
#define OPAL_DATATYPE_FIRST_TYPE     4 /* Number of first real type */
#define OPAL_DATATYPE_INT1           4
#define OPAL_DATATYPE_INT2           5
#define OPAL_DATATYPE_INT4           6
#define OPAL_DATATYPE_INT8           7
#define OPAL_DATATYPE_INT16          8
#define OPAL_DATATYPE_UINT1          9
#define OPAL_DATATYPE_UINT2          10
#define OPAL_DATATYPE_UINT4          11
#define OPAL_DATATYPE_UINT8          12
#define OPAL_DATATYPE_UINT16         13
#define OPAL_DATATYPE_FLOAT2         14
#define OPAL_DATATYPE_FLOAT4         15
#define OPAL_DATATYPE_FLOAT8         16
#define OPAL_DATATYPE_FLOAT12        17
#define OPAL_DATATYPE_FLOAT16        18
#define OPAL_DATATYPE_FLOAT_COMPLEX  19
#define OPAL_DATATYPE_DOUBLE_COMPLEX 20
#define OPAL_DATATYPE_LONG_DOUBLE_COMPLEX 21
#define OPAL_DATATYPE_BOOL           22
#define OPAL_DATATYPE_WCHAR          23
#define OPAL_DATATYPE_UNAVAILABLE    24

/* flags for the datatypes. */
#define OPAL_DATATYPE_FLAG_UNAVAILABLE   0x0001  /**< datatypes unavailable on the build (OS or compiler dependant) */
#define OPAL_DATATYPE_FLAG_PREDEFINED    0x0002  /**< cannot be removed: initial and predefined datatypes */
#define OPAL_DATATYPE_FLAG_COMMITED      0x0004  /**< ready to be used for a send/recv operation */
#define OPAL_DATATYPE_FLAG_OVERLAP       0x0008  /**< datatype is unpropper for a recv operation */
#define OPAL_DATATYPE_FLAG_CONTIGUOUS    0x0010  /**< contiguous datatype */
#define OPAL_DATATYPE_FLAG_NO_GAPS       0x0020  /**< no gaps around the datatype, aka OPAL_DATATYPE_FLAG_CONTIGUOUS and extent == size */
#define OPAL_DATATYPE_FLAG_USER_LB       0x0040  /**< has a user defined LB */
#define OPAL_DATATYPE_FLAG_USER_UB       0x0080  /**< has a user defined UB */
#define OPAL_DATATYPE_FLAG_DATA          0x0100  /**< data or control structure */
/*
 * We should make the difference here between the predefined contiguous and non contiguous
 * datatypes. The OPAL_DATATYPE_FLAG_BASIC is held by all predefined contiguous datatypes.
 */
#define OPAL_DATATYPE_FLAG_BASIC         (OPAL_DATATYPE_FLAG_PREDEFINED | \
                                          OPAL_DATATYPE_FLAG_CONTIGUOUS | \
                                          OPAL_DATATYPE_FLAG_NO_GAPS |    \
                                          OPAL_DATATYPE_FLAG_DATA |       \
                                          OPAL_DATATYPE_FLAG_COMMITED)
 
/* typedefs ***********************************************************/

typedef struct opal_object_t opal_object_t;
typedef struct opal_class_t opal_class_t;
typedef void (*opal_construct_t) (opal_object_t *);
typedef void (*opal_destruct_t) (opal_object_t *);


/* types **************************************************************/

/**
* Class descriptor.
*
* There should be a single instance of this descriptor for each class
* definition.
*/
struct opal_class_t {
  const char *cls_name;           /**< symbolic name for class */
  opal_class_t *cls_parent;       /**< parent class descriptor */
  opal_construct_t cls_construct; /**< class constructor */
  opal_destruct_t cls_destruct;   /**< class destructor */
  int cls_initialized;            /**< is class initialized */
  int cls_depth;                  /**< depth of class hierarchy tree */
  opal_construct_t *cls_construct_array;
                                  /**< array of parent class constructors */
  opal_destruct_t *cls_destruct_array;
                                  /**< array of parent class destructors */
  size_t cls_sizeof;              /**< size of an object instance */
};

/**
 * Base object.
 *
 * This is special and does not follow the pattern for other classes.
 */
struct opal_object_t {
#if OPAL_ENABLE_DEBUG
    /** Magic ID -- want this to be the very first item in the
        struct's memory */
    uint64_t obj_magic_id;
#endif
    opal_class_t *obj_class;            /**< class descriptor */
    volatile int32_t obj_reference_count;   /**< reference count */
#if OPAL_ENABLE_DEBUG
   const char* cls_init_file_name;        /**< In debug mode store the file where the object get contructed */
   int   cls_init_lineno;           /**< In debug mode store the line number where the object get contructed */
#endif  /* OPAL_ENABLE_DEBUG */
};

/**
 * Declaration for class descriptor
 *
 * @param NAME          Name of class
 *
 * Put this in NAME.h
 */
#define OBJ_CLASS_DECLARATION(NAME)             \
    extern opal_class_t NAME ## _class

/**
 * Return a pointer to the class descriptor associated with a
 * class type.
 *
 * @param NAME          Name of class
 * @return              Pointer to class descriptor
 */
#define OBJ_CLASS(NAME)     (&(NAME ## _class))

/**
 * For static initializations of OBJects.
 *
 * @param NAME   Name of the class to initialize
 */
#if OPAL_ENABLE_DEBUG
#define OPAL_OBJ_STATIC_INIT(BASE_CLASS) { OPAL_OBJ_MAGIC_ID, OBJ_CLASS(BASE_CLASS), 1, __FILE__, __LINE__ }
#else
#define OPAL_OBJ_STATIC_INIT(BASE_CLASS) { OBJ_CLASS(BASE_CLASS), 1 }
#endif



struct ddt_elem_id_description {
    uint16_t   flags;  /**< flags for the record */
    uint16_t   type;   /**< the basic data type id */
};
typedef struct ddt_elem_id_description ddt_elem_id_description;

/* the basic element. A data description is composed
 * by a set of basic elements.
 */
struct ddt_elem_desc {
    ddt_elem_id_description common;           /**< basic data description and flags */
    uint32_t                count;            /**< number of blocks */
    uint32_t                blocklen;         /**< number of elements on each block */
    OPAL_PTRDIFF_TYPE       extent;           /**< extent of each block (in bytes) */
    OPAL_PTRDIFF_TYPE       disp;             /**< displacement of the first block */
};
typedef struct ddt_elem_desc ddt_elem_desc_t;

struct ddt_loop_desc {
    ddt_elem_id_description common;           /**< basic data description and flags */
    uint32_t                loops;            /**< number of elements */
    uint32_t                items;            /**< number of items in the loop */
    size_t                  unused;           /**< not used right now */
    OPAL_PTRDIFF_TYPE       extent;           /**< extent of the whole loop */
};
typedef struct ddt_loop_desc ddt_loop_desc_t;

struct ddt_endloop_desc {
    ddt_elem_id_description common;           /**< basic data description and flags */
    uint32_t                items;            /**< number of elements */
    uint32_t                unused;           /**< not used right now */
    size_t                  size;             /**< real size of the data in the loop */
    OPAL_PTRDIFF_TYPE       first_elem_disp;  /**< the displacement of the first block in the loop */
};
typedef struct ddt_endloop_desc ddt_endloop_desc_t;

union dt_elem_desc {
    ddt_elem_desc_t    elem;
    ddt_loop_desc_t    loop;
    ddt_endloop_desc_t end_loop;
};
typedef union dt_elem_desc dt_elem_desc_t;

/* dt_type_description */
typedef uint32_t opal_datatype_count_t;

struct dt_type_desc_t {
    opal_datatype_count_t  length;  /**< the maximum number of elements in the description array */
    opal_datatype_count_t  used;    /**< the number of used elements in the description array */
    dt_elem_desc_t*        desc;
};
typedef struct dt_type_desc_t dt_type_desc_t;

/*
 * The datatype description.
 */
#define OPAL_DATATYPE_MAX_PREDEFINED 25
#define OPAL_DATATYPE_MAX_SUPPORTED  47
#define OPAL_MAX_OBJECT_NAME         64

struct opal_datatype_t {
    opal_object_t      super;    /**< basic superclass */
    uint16_t           flags;    /**< the flags */
    uint16_t           id;       /**< data id, normally the index in the data array. */
    uint32_t           bdt_used; /**< bitset of which basic datatypes are used in the data description */
    size_t             size;     /**< total size in bytes of the memory used by the data if
                                      the data is put on a contiguous buffer */
    OPAL_PTRDIFF_TYPE  true_lb;  /**< the true lb of the data without user defined lb and ub */
    OPAL_PTRDIFF_TYPE  true_ub;  /**< the true ub of the data without user defined lb and ub */
    OPAL_PTRDIFF_TYPE  lb;       /**< lower bound in memory */
    OPAL_PTRDIFF_TYPE  ub;       /**< upper bound in memory */
    /* --- cacheline 1 boundary (64 bytes) --- */
    size_t             nbElems;  /**< total number of elements inside the datatype */
    uint32_t           align;    /**< data should be aligned to */

    /* Attribute fields */
    char               name[OPAL_MAX_OBJECT_NAME];  /**< name of the datatype */
    /* --- cacheline 2 boundary (128 bytes) was 8-12 bytes ago --- */
    dt_type_desc_t     desc;     /**< the data description */
    dt_type_desc_t     opt_desc; /**< short description of the data used when conversion is useless
                                      or in the send case (without conversion) */

    uint32_t           btypes[OPAL_DATATYPE_MAX_SUPPORTED];
                                 /**< basic elements count used to compute the size of the
                                      datatype for remote nodes. The length of the array is dependent on
                                      the maximum number of datatypes of all top layers.
                                      Reason being is that Fortran is not at the OPAL layer. */
    /* --- cacheline 5 boundary (320 bytes) was 32-36 bytes ago --- */

    /* size: 352, cachelines: 6, members: 15 */
    /* last cacheline: 28-32 bytes */
};

typedef struct opal_datatype_t opal_datatype_t;

OPAL_DECLSPEC OBJ_CLASS_DECLARATION( opal_datatype_t );

/* convertor and stack */
typedef struct opal_convertor_t opal_convertor_t;

typedef int32_t (*convertor_advance_fct_t)( opal_convertor_t* pConvertor,
                                            struct iovec* iov,
                                            uint32_t* out_size,
                                            size_t* max_data );
typedef void*(*memalloc_fct_t)( size_t* pLength, void* userdata );
typedef void*(*memcpy_fct_t)( void* dest, const void* src, size_t n, opal_convertor_t* pConvertor );

/* The master convertor struct (defined in convertor_internal.h) */
struct opal_convertor_master_t;

struct dt_stack_t {
    int32_t           index;    /**< index in the element description */
    int16_t           type;     /**< the type used for the last pack/unpack (original or OPAL_DATATYPE_UINT1) */
    size_t            count;    /**< number of times we still have to do it */
    OPAL_PTRDIFF_TYPE disp;     /**< actual displacement depending on the count field */
};
typedef struct dt_stack_t dt_stack_t;

typedef int32_t (*conversion_fct_t)( opal_convertor_t* pConvertor, uint32_t count,
                                     const void* from, size_t from_len, OPAL_PTRDIFF_TYPE from_extent,
                                     void* to, size_t to_length, OPAL_PTRDIFF_TYPE to_extent,
                                     OPAL_PTRDIFF_TYPE *advance );

typedef struct opal_convertor_master_t {
    struct opal_convertor_master_t* next;
    uint32_t                        remote_arch;
    uint32_t                        flags;
    uint32_t                        hetero_mask;
    const size_t                    remote_sizes[OPAL_DATATYPE_MAX_PREDEFINED];
    conversion_fct_t*               pFunctions;   /**< the convertor functions pointer */
} opal_convertor_master_t;

#define MAX_IPC_EVENT_HANDLE   10

struct opal_convertor_t {
    opal_object_t                 super;          /**< basic superclass */
    uint32_t                      remoteArch;     /**< the remote architecture */
    uint32_t                      flags;          /**< the properties of this convertor */
    size_t                        local_size;     /**< overall length data on local machine, compared to bConverted */
    size_t                        remote_size;    /**< overall length data on remote machine, compared to bConverted */
    const opal_datatype_t*        pDesc;          /**< the datatype description associated with the convertor */
    const dt_type_desc_t*         use_desc;       /**< the version used by the convertor (normal or optimized) */
    opal_datatype_count_t         count;          /**< the total number of full datatype elements */
    uint32_t                      stack_size;     /**< size of the allocated stack */
    /* --- cacheline 1 boundary (64 bytes) --- */
    unsigned char*                pBaseBuf;       /**< initial buffer as supplied by the user */
    dt_stack_t*                   pStack;         /**< the local stack for the actual conversion */
    convertor_advance_fct_t       fAdvance;       /**< pointer to the pack/unpack functions */
    struct opal_convertor_master_t* master;       /**< the master convertor */

    /* All others fields get modified for every call to pack/unpack functions */
    uint32_t                      stack_pos;      /**< the actual position on the stack */
    uint32_t                      partial_length; /**< amount of data left over from the last unpack */
    size_t                        bConverted;     /**< # of bytes already converted */
    uint32_t                      checksum;       /**< checksum computed by pack/unpack operation */
    uint32_t                      csum_ui1;       /**< partial checksum computed by pack/unpack operation */
    size_t                        csum_ui2;       /**< partial checksum computed by pack/unpack operation */
     /* --- cacheline 2 boundary (128 bytes) --- */
    dt_stack_t                    static_stack[DT_STATIC_STACK_SIZE];  /**< local stack for small datatypes */
    /* --- cacheline 3 boundary (192 bytes) was 56 bytes ago --- */

#if OPAL_CUDA_SUPPORT
    memcpy_fct_t                  cbmemcpy;       /**< memcpy or cuMemcpy */
    void *                        stream;         /**< CUstream for async copy */

    unsigned char *               gpu_buffer_ptr; /**< GPU buffer used for pack/unpack */
    unsigned char *               gpu_buffer_ptr_source; /**< source address of GPU buffer start to pack, update in packing function */
    uint64_t *                    pipeline_event[MAX_IPC_EVENT_HANDLE]; /**< cuda event for pipeline */
#endif
    /* size: 248, cachelines: 4, members: 20 */
    /* last cacheline: 56 bytes */
};

struct iovec {  
    void *iov_base; /* Starting address */  
    size_t iov_len; /* Length in bytes */  
};


OPAL_DECLSPEC extern union dt_elem_desc opal_datatype_predefined_elem_desc[2 * OPAL_DATATYPE_MAX_PREDEFINED];

#define OPAL_DATATYPE_INIT_BTYPES_ARRAY_UNAVAILABLE { 0 }
#define OPAL_DATATYPE_INIT_BTYPES_ARRAY(NAME) { [OPAL_DATATYPE_ ## NAME] = 1 }

#define OPAL_DATATYPE_INIT_NAME(NAME) "OPAL_" #NAME

/*
 * Macro to initialize the main description for basic types, setting the pointer
 * into the array opal_datatype_predefined_type_desc, which is initialized at
 * runtime in opal_datatype_init(). Each basic type has two desc-elements....
 */
#define OPAL_DATATYPE_INIT_DESC_PREDEFINED(NAME)                                     \
    {                                                                                \
        .length = 1, .used = 1,                                                      \
        .desc = &(opal_datatype_predefined_elem_desc[2 * OPAL_DATATYPE_ ## NAME])    \
    }
#define OPAL_DATATYPE_INIT_DESC_NULL  {.length = 0, .used = 0, .desc = NULL}

#define OPAL_DATATYPE_INITIALIZER_UNAVAILABLE_NAMED( NAME, FLAGS )                   \
    {                                                                                \
        .super = OPAL_OBJ_STATIC_INIT(opal_datatype_t),                              \
        .flags = OPAL_DATATYPE_FLAG_UNAVAILABLE | OPAL_DATATYPE_FLAG_PREDEFINED | (FLAGS), \
        .id = OPAL_DATATYPE_ ## NAME,                                                \
        .bdt_used = 0,                                                               \
        .size = 0,                                                                   \
        .true_lb = 0, .true_ub = 0, .lb = 0, .ub = 0,                                \
        .align = 0,                                                                  \
        .nbElems = 1,                                                                \
        .name = OPAL_DATATYPE_INIT_NAME(NAME),                                       \
        .desc = OPAL_DATATYPE_INIT_DESC_PREDEFINED(UNAVAILABLE),                     \
        .opt_desc = OPAL_DATATYPE_INIT_DESC_PREDEFINED(UNAVAILABLE),                 \
        .btypes = OPAL_DATATYPE_INIT_BTYPES_ARRAY_UNAVAILABLE                        \
    }

#define OPAL_DATATYPE_INITIALIZER_EMPTY( FLAGS )                        \
    {                                                                   \
        .super = OPAL_OBJ_STATIC_INIT(opal_datatype_t),                 \
        .flags = OPAL_DATATYPE_FLAG_PREDEFINED | (FLAGS),               \
        .id = 0,                                                        \
        .bdt_used = 0,                                                  \
        .size = 0,                                                      \
        .true_lb = 0, .true_ub = 0, .lb = 0, .ub = 0,                   \
        .align = 0,                                                     \
        .nbElems = 1,                                                   \
        .name = OPAL_DATATYPE_INIT_NAME(EMPTY),                         \
        .desc = OPAL_DATATYPE_INIT_DESC_NULL,                           \
        .opt_desc = OPAL_DATATYPE_INIT_DESC_NULL,                       \
        .btypes = OPAL_DATATYPE_INIT_BTYPES_ARRAY_UNAVAILABLE           \
    }

#define OPAL_DATATYPE_INIT_BASIC_TYPE( TYPE, NAME, FLAGS )              \
    {                                                                   \
        .super = OPAL_OBJ_STATIC_INIT(opal_datatype_t),                 \
        .flags = OPAL_DATATYPE_FLAG_PREDEFINED | (FLAGS),               \
        .id = TYPE,                                                     \
        .bdt_used = (((uint32_t)1)<<(TYPE)),                            \
        .size = 0,                                                      \
        .true_lb = 0, .true_ub = 0, .lb = 0, .ub = 0,                   \
        .align = 0,                                                     \
        .nbElems = 1,                                                   \
        .name = OPAL_DATATYPE_INIT_NAME(NAME),                          \
        .desc = OPAL_DATATYPE_INIT_DESC_NULL,                           \
        .opt_desc = OPAL_DATATYPE_INIT_DESC_NULL,                       \
        .btypes = OPAL_DATATYPE_INIT_BTYPES_ARRAY(NAME)                 \
    }
    
#define OPAL_DATATYPE_INIT_BASIC_DATATYPE( TYPE, ALIGN, NAME, FLAGS )                \
    {                                                                                \
        .super = OPAL_OBJ_STATIC_INIT(opal_datatype_t),                              \
        .flags = OPAL_DATATYPE_FLAG_BASIC | (FLAGS),                                 \
        .id = OPAL_DATATYPE_ ## NAME,                                                \
        .bdt_used = (((uint32_t)1)<<(OPAL_DATATYPE_ ## NAME)),                       \
        .size = sizeof(TYPE),                                                        \
        .true_lb = 0, .true_ub = sizeof(TYPE), .lb = 0, .ub = sizeof(TYPE),          \
        .align = (ALIGN),                                                            \
        .nbElems = 1,                                                                \
        .name = OPAL_DATATYPE_INIT_NAME(NAME),                                       \
        .desc = OPAL_DATATYPE_INIT_DESC_PREDEFINED(NAME),                            \
        .opt_desc = OPAL_DATATYPE_INIT_DESC_PREDEFINED(NAME),                        \
        .btypes = OPAL_DATATYPE_INIT_BTYPES_ARRAY(NAME)                              \
    }

#define OPAL_DATATYPE_INITIALIZER_LOOP(FLAGS)       OPAL_DATATYPE_INIT_BASIC_TYPE( OPAL_DATATYPE_LOOP, LOOP, FLAGS )
#define OPAL_DATATYPE_INITIALIZER_END_LOOP(FLAGS)   OPAL_DATATYPE_INIT_BASIC_TYPE( OPAL_DATATYPE_END_LOOP, END_LOOP, FLAGS )
#define OPAL_DATATYPE_INITIALIZER_LB(FLAGS)         OPAL_DATATYPE_INIT_BASIC_TYPE( OPAL_DATATYPE_LB, LB, FLAGS )
#define OPAL_DATATYPE_INITIALIZER_UB(FLAGS)         OPAL_DATATYPE_INIT_BASIC_TYPE( OPAL_DATATYPE_UB, UB, FLAGS ) 
#define OPAL_DATATYPE_INITIALIZER_INT1(FLAGS)       OPAL_DATATYPE_INIT_BASIC_DATATYPE( int8_t, OPAL_ALIGNMENT_INT8, INT1, FLAGS )
#define OPAL_DATATYPE_INITIALIZER_INT2(FLAGS)       OPAL_DATATYPE_INIT_BASIC_DATATYPE( int16_t, OPAL_ALIGNMENT_INT16, INT2, FLAGS )
#define OPAL_DATATYPE_INITIALIZER_INT4(FLAGS)       OPAL_DATATYPE_INIT_BASIC_DATATYPE( int32_t, OPAL_ALIGNMENT_INT32, INT4, FLAGS )
#define OPAL_DATATYPE_INITIALIZER_INT8(FLAGS)       OPAL_DATATYPE_INIT_BASIC_DATATYPE( int64_t, OPAL_ALIGNMENT_INT64, INT8, FLAGS )
#ifdef HAVE_INT128_T
#define OPAL_DATATYPE_INITIALIZER_INT16(FLAGS)      OPAL_DATATYPE_INIT_BASIC_DATATYPE( int128_t, OPAL_ALIGNMENT_INT128, INT16, FLAGS )
#else
#define OPAL_DATATYPE_INITIALIZER_INT16(FLAGS)      OPAL_DATATYPE_INITIALIZER_UNAVAILABLE_NAMED( INT16, FLAGS )
#endif
#define OPAL_DATATYPE_INITIALIZER_UINT1(FLAGS)      OPAL_DATATYPE_INIT_BASIC_DATATYPE( uint8_t, OPAL_ALIGNMENT_INT8, UINT1, FLAGS )
#define OPAL_DATATYPE_INITIALIZER_UINT2(FLAGS)      OPAL_DATATYPE_INIT_BASIC_DATATYPE( uint16_t, OPAL_ALIGNMENT_INT16, UINT2, FLAGS )
#define OPAL_DATATYPE_INITIALIZER_UINT4(FLAGS)      OPAL_DATATYPE_INIT_BASIC_DATATYPE( uint32_t, OPAL_ALIGNMENT_INT32, UINT4, FLAGS )
#define OPAL_DATATYPE_INITIALIZER_UINT8(FLAGS)      OPAL_DATATYPE_INIT_BASIC_DATATYPE( uint64_t, OPAL_ALIGNMENT_INT64, UINT8, FLAGS )
#ifdef HAVE_UINT128_T
#define OPAL_DATATYPE_INITIALIZER_UINT16(FLAGS)     OPAL_DATATYPE_INIT_BASIC_DATATYPE( uint128_t, OPAL_ALIGNMENT_INT128, UINT16, FLAGS )
#else
#define OPAL_DATATYPE_INITIALIZER_UINT16(FLAGS)     OPAL_DATATYPE_INITIALIZER_UNAVAILABLE_NAMED( INT16, FLAGS )
#endif

#if SIZEOF_FLOAT == 2
#define OPAL_DATATYPE_INITIALIZER_FLOAT2(FLAGS)     OPAL_DATATYPE_INIT_BASIC_DATATYPE( float, OPAL_ALIGNMENT_FLOAT, FLOAT2, FLAGS )
#elif SIZEOF_DOUBLE == 2
#define OPAL_DATATYPE_INITIALIZER_FLOAT2(FLAGS)     OPAL_DATATYPE_INIT_BASIC_DATATYPE( double, OPAL_ALIGNMENT_DOUBLE, FLOAT2, FLAGS )
#elif SIZEOF_LONG_DOUBLE == 2
#define OPAL_DATATYPE_INITIALIZER_FLOAT2(FLAGS)     OPAL_DATATYPE_INIT_BASIC_DATATYPE( long double, OPAL_ALIGNMENT_LONG_DOUBLE, FLOAT2, FLAGS )
#else
#define OPAL_DATATYPE_INITIALIZER_FLOAT2(FLAGS)     OPAL_DATATYPE_INITIALIZER_UNAVAILABLE_NAMED( FLOAT2, FLAGS )
#endif

#if SIZEOF_FLOAT == 4
#define OPAL_DATATYPE_INITIALIZER_FLOAT4(FLAGS)     OPAL_DATATYPE_INIT_BASIC_DATATYPE( float, OPAL_ALIGNMENT_FLOAT, FLOAT4, FLAGS )
#elif SIZEOF_DOUBLE == 4
#define OPAL_DATATYPE_INITIALIZER_FLOAT4(FLAGS)     OPAL_DATATYPE_INIT_BASIC_DATATYPE( double, OPAL_ALIGNMENT_DOUBLE, FLOAT4, FLAGS )
#elif SIZEOF_LONG_DOUBLE == 4
#define OPAL_DATATYPE_INITIALIZER_FLOAT4(FLAGS)     OPAL_DATATYPE_INIT_BASIC_DATATYPE( long double, OPAL_ALIGNMENT_LONG_DOUBLE, FLOAT4, FLAGS )
#else
#define OPAL_DATATYPE_INITIALIZER_FLOAT4(FLAGS)     OPAL_DATATYPE_INITIALIZER_UNAVAILABLE_NAMED( FLOAT4, FLAGS )
#endif

#if SIZEOF_FLOAT == 8
#define OPAL_DATATYPE_INITIALIZER_FLOAT8(FLAGS)     OPAL_DATATYPE_INIT_BASIC_DATATYPE( float, OPAL_ALIGNMENT_FLOAT, FLOAT8, FLAGS )
#elif SIZEOF_DOUBLE == 8
#define OPAL_DATATYPE_INITIALIZER_FLOAT8(FLAGS)     OPAL_DATATYPE_INIT_BASIC_DATATYPE( double, OPAL_ALIGNMENT_DOUBLE, FLOAT8, FLAGS )
#elif SIZEOF_LONG_DOUBLE == 8
#define OPAL_DATATYPE_INITIALIZER_FLOAT8(FLAGS)     OPAL_DATATYPE_INIT_BASIC_DATATYPE( long double, OPAL_ALIGNMENT_LONG_DOUBLE, FLOAT8, FLAGS )
#else
#define OPAL_DATATYPE_INITIALIZER_FLOAT8(FLAGS)     OPAL_DATATYPE_INITIALIZER_UNAVAILABLE_NAMED( FLOAT8, FLAGS )
#endif

#if SIZEOF_FLOAT == 12
#define OPAL_DATATYPE_INITIALIZER_FLOAT12(FLAGS)    OPAL_DATATYPE_INIT_BASIC_DATATYPE( float, OPAL_ALIGNMENT_FLOAT, FLOAT12, FLAGS )
#elif SIZEOF_DOUBLE == 12
#define OPAL_DATATYPE_INITIALIZER_FLOAT12(FLAGS)    OPAL_DATATYPE_INIT_BASIC_DATATYPE( double, OPAL_ALIGNMENT_DOUBLE, FLOAT12, FLAGS )
#elif SIZEOF_LONG_DOUBLE == 12
#define OPAL_DATATYPE_INITIALIZER_FLOAT12(FLAGS)    OPAL_DATATYPE_INIT_BASIC_DATATYPE( long double, OPAL_ALIGNMENT_LONG_DOUBLE, FLOAT12, FLAGS )
#else
#define OPAL_DATATYPE_INITIALIZER_FLOAT12(FLAGS)    OPAL_DATATYPE_INITIALIZER_UNAVAILABLE_NAMED( FLOAT12, FLAGS )
#endif

#if SIZEOF_FLOAT == 16
#define OPAL_DATATYPE_INITIALIZER_FLOAT16(FLAGS)    OPAL_DATATYPE_INIT_BASIC_DATATYPE( float, OPAL_ALIGNMENT_FLOAT, FLOAT16, FLAGS )
#elif SIZEOF_DOUBLE == 16
#define OPAL_DATATYPE_INITIALIZER_FLOAT16(FLAGS)    OPAL_DATATYPE_INIT_BASIC_DATATYPE( double, OPAL_ALIGNMENT_DOUBLE, FLOAT16, FLAGS )
#elif SIZEOF_LONG_DOUBLE == 16
#define OPAL_DATATYPE_INITIALIZER_FLOAT16(FLAGS)    OPAL_DATATYPE_INIT_BASIC_DATATYPE( long double, OPAL_ALIGNMENT_LONG_DOUBLE, FLOAT16, FLAGS )
#else
#define OPAL_DATATYPE_INITIALIZER_FLOAT16(FLAGS)    OPAL_DATATYPE_INITIALIZER_UNAVAILABLE_NAMED( FLOAT16, FLAGS )
#endif

#if HAVE_FLOAT__COMPLEX
#define OPAL_DATATYPE_INITIALIZER_FLOAT_COMPLEX(FLAGS) OPAL_DATATYPE_INIT_BASIC_DATATYPE( float _Complex, OPAL_ALIGNMENT_FLOAT_COMPLEX, FLOAT_COMPLEX, FLAGS )
#else
#define OPAL_DATATYPE_INITIALIZER_FLOAT_COMPLEX(FLAGS) OPAL_DATATYPE_INITIALIZER_UNAVAILABLE_NAMED( FLOAT_COMPLEX, FLAGS)
#endif

#if HAVE_DOUBLE__COMPLEX
#define OPAL_DATATYPE_INITIALIZER_DOUBLE_COMPLEX(FLAGS) OPAL_DATATYPE_INIT_BASIC_DATATYPE( double _Complex, OPAL_ALIGNMENT_DOUBLE_COMPLEX, DOUBLE_COMPLEX, FLAGS )
#else
#define OPAL_DATATYPE_INITIALIZER_DOUBLE_COMPLEX(FLAGS) OPAL_DATATYPE_INITIALIZER_UNAVAILABLE_NAMED( DOUBLE_COMPLEX, FLAGS)
#endif

#if HAVE_LONG_DOUBLE__COMPLEX
#define OPAL_DATATYPE_INITIALIZER_LONG_DOUBLE_COMPLEX(FLAGS) OPAL_DATATYPE_INIT_BASIC_DATATYPE( long double _Complex, OPAL_ALIGNMENT_LONG_DOUBLE_COMPLEX, LONG_DOUBLE_COMPLEX, FLAGS )
#else
#define OPAL_DATATYPE_INITIALIZER_LONG_DOUBLE_COMPLEX(FLAGS) OPAL_DATATYPE_INITIALIZER_UNAVAILABLE_NAMED( LONG_DOUBLE_COMPLEX, FLAGS)
#endif

#define OPAL_DATATYPE_INITIALIZER_BOOL(FLAGS)       OPAL_DATATYPE_INIT_BASIC_DATATYPE( _Bool, OPAL_ALIGNMENT_BOOL, BOOL, FLAGS )

#if OPAL_ALIGNMENT_WCHAR != 0
#define OPAL_DATATYPE_INITIALIZER_WCHAR(FLAGS)      OPAL_DATATYPE_INIT_BASIC_DATATYPE( wchar_t, OPAL_ALIGNMENT_WCHAR, WCHAR, FLAGS )
#else
#define OPAL_DATATYPE_INITIALIZER_WCHAR(FLAGS)      OPAL_DATATYPE_INITIALIZER_UNAVAILABLE_NAMED( WCHAR, FLAGS )
#endif
    
#define SAVE_STACK( PSTACK, INDEX, TYPE, COUNT, DISP) \
do { \
   (PSTACK)->index    = (INDEX); \
   (PSTACK)->type     = (TYPE); \
   (PSTACK)->count    = (COUNT); \
   (PSTACK)->disp     = (DISP); \
} while(0)

#define PUSH_STACK( PSTACK, STACK_POS, INDEX, TYPE, COUNT, DISP) \
do { \
   dt_stack_t* pTempStack = (PSTACK) + 1; \
   if (threadIdx.x == 0) {  \
       SAVE_STACK( pTempStack, (INDEX), (TYPE), (COUNT), (DISP) );  \
   }    \
   __syncthreads(); \
   (STACK_POS)++; \
   (PSTACK) = pTempStack; \
} while(0)

#define UPDATE_INTERNAL_COUNTERS( DESCRIPTION, POSITION, ELEMENT, COUNTER ) \
    do {                                                                \
        (ELEMENT) = &((DESCRIPTION)[(POSITION)]);                       \
        (COUNTER) = (ELEMENT)->elem.count;                              \
    } while (0)   

OPAL_DECLSPEC extern const size_t opal_datatype_basicDatatypesSize[OPAL_DATATYPE_MAX_PREDEFINED];

#define     OPAL_DATATYPE_LOOP_SIZE         0
#define     OPAL_DATATYPE_END_LOOP_SIZE     0
#define     OPAL_DATATYPE_LB_SIZE           0
#define     OPAL_DATATYPE_UB_SIZE           0
#define     OPAL_DATATYPE_INT1_SIZE         sizeof(int8_t)
#define     OPAL_DATATYPE_INT2_SIZE         sizeof(int16_t)
#define     OPAL_DATATYPE_INT4_SIZE         sizeof(int32_t)
#define     OPAL_DATATYPE_INT8_SIZE         sizeof(int64_t)
#ifdef HAVE_INT128_T
#   define  OPAL_DATATYPE_INT16_SIZE        sizeof(int128_t)       /* Yes, double-machine word integers are available */
#else
#   define  OPAL_DATATYPE_INT16_SIZE        0
#endif

#define     OPAL_DATATYPE_UINT1_SIZE        sizeof(uint8_t)
#define     OPAL_DATATYPE_UINT2_SIZE        sizeof(uint16_t)
#define     OPAL_DATATYPE_UINT4_SIZE        sizeof(uint32_t)
#define     OPAL_DATATYPE_UINT8_SIZE        sizeof(uint64_t)
#ifdef HAVE_UINT128_T
#   define  OPAL_DATATYPE_UINT16_SIZE       sizeof(uint128_t)      /* Yes, double-machine word integers are available */
#else
#   define  OPAL_DATATYPE_UINT16_SIZE       0
#endif

#if SIZEOF_FLOAT == 2
#   define  OPAL_DATATYPE_FLOAT2_SIZE       sizeof(float)
#elif SIZEOF_DOUBLE == 2
#   define  OPAL_DATATYPE_FLOAT2_SIZE       sizeof(double)
#elif SIZEOF_LONG_DOUBLE == 2
#   define  OPAL_DATATYPE_FLOAT2_SIZE       sizeof(long double)
#else
#   define  OPAL_DATATYPE_FLOAT2_SIZE       0
#endif

#if SIZEOF_FLOAT == 4
#   define  OPAL_DATATYPE_FLOAT4_SIZE       sizeof(float)
#elif SIZEOF_DOUBLE == 4
#   define  OPAL_DATATYPE_FLOAT4_SIZE       sizeof(double)
#elif SIZEOF_LONG_DOUBLE == 4
#   define  OPAL_DATATYPE_FLOAT4_SIZE       sizeof(long double)
#else
#   define  OPAL_DATATYPE_FLOAT4_SIZE       0
#endif

#if SIZEOF_FLOAT == 8
#   define  OPAL_DATATYPE_FLOAT8_SIZE       sizeof(float)
#elif SIZEOF_DOUBLE == 8
#   define  OPAL_DATATYPE_FLOAT8_SIZE       sizeof(double)
#elif SIZEOF_LONG_DOUBLE == 8
#   define  OPAL_DATATYPE_FLOAT8_SIZE       sizeof(long double)
#else
#   define  OPAL_DATATYPE_FLOAT8_SIZE       0
#endif

#if SIZEOF_FLOAT == 12
#   define  OPAL_DATATYPE_FLOAT12_SIZE      sizeof(float)
#elif SIZEOF_DOUBLE == 12
#   define  OPAL_DATATYPE_FLOAT12_SIZE      sizeof(double)
#elif SIZEOF_LONG_DOUBLE == 12
#   define  OPAL_DATATYPE_FLOAT12_SIZE      sizeof(long double)
#else
#   define  OPAL_DATATYPE_FLOAT12_SIZE      0
#endif

#if SIZEOF_FLOAT == 16
#   define  OPAL_DATATYPE_FLOAT16_SIZE      sizeof(float)
#elif SIZEOF_DOUBLE == 16
#   define  OPAL_DATATYPE_FLOAT16_SIZE      sizeof(double)
#elif SIZEOF_LONG_DOUBLE == 16
#   define  OPAL_DATATYPE_FLOAT16_SIZE      sizeof(long double)
#else
#   define  OPAL_DATATYPE_FLOAT16_SIZE      0
#endif
        
#if HAVE_FLOAT__COMPLEX
#   define  OPAL_DATATYPE_FLOAT_COMPLEX_SIZE    sizeof(float _Complex)
#else
#   define  OPAL_DATATYPE_FLOAT_COMPLEX_SIZE    0
#endif

#if HAVE_DOUBLE__COMPLEX
#   define  OPAL_DATATYPE_DOUBLE_COMPLEX_SIZE    sizeof(float _Complex)
#else
#   define  OPAL_DATATYPE_DOUBLE_COMPLEX_SIZE    0
#endif
    
#if HAVE_LONG_DOUBLE__COMPLEX
#   define  OPAL_DATATYPE_LONG_DOUBLE_COMPLEX_SIZE    sizeof(float _Complex)
#else
#   define  OPAL_DATATYPE_LONG_DOUBLE_COMPLEX_SIZE    0
#endif

#define     OPAL_DATATYPE_BOOL_SIZE         sizeof(_Bool)
#if OPAL_ALIGNMENT_WCHAR != 0
#   define  OPAL_DATATYPE_WCHAR_SIZE        sizeof(wchar_t)
#else 
#   define  OPAL_DATATYPE_WCHAR_SIZE        0
#endif

#define     OPAL_DATATYPE_UNAVAILABLE_SIZE  0

#endif  /* OPAL_DATATYPE_ORIG_INTERNAL_H_HAS_BEEN_INCLUDED */
