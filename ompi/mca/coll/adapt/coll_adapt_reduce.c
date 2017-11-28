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

int mca_coll_adapt_reduce(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    if (count == 0) {
        return MPI_SUCCESS;
    }
    else {
        ompi_request_t * request;
        int retval = mca_coll_adapt_ireduce(sbuf, rbuf, count, dtype, op, root, comm, &request, module);
        ompi_request_wait(&request, MPI_STATUS_IGNORE);
        return retval;
    }
}