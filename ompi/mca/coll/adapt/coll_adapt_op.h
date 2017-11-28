#ifndef COLL_ADAPT_OP_H
#define COLL_ADAPT_OP_H

#include "coll_adapt.h"
static inline int coll_adapt_op_reduce(ompi_op_t * op, void *source, void *target, int count, ompi_datatype_t * dtype, int buff_type) 
{
    if (buff_type == CPU_BUFFER) {
        ompi_op_reduce(op, source, target, count, dtype);
    }
#if OPAL_CUDA_SUPPORT    
    else if (buff_type == GPU_BUFFER) {
        coll_adapt_cuda_op_reduce(op, source, target, count, dtype);
    } 
#endif
    else {
        opal_output(0, "Unsupported buffer type %d\n", buff_type);
    }
}

#endif /* COLL_ADAPT_OP_H */
