//TODO: change tag to 3 bits
//TODO: add sent part in root use sent array to acheive always send the next one
//TODO: move receve before send
#include "ompi_config.h"
#include "ompi/mca/pml/pml.h"
#include "coll_adapt.h"
#include "coll_adapt_algorithms.h"
#include "coll_adapt_context.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/coll/base/coll_base_functions.h"     //COLL_BASE_COMPUTED_SEGCOUNT
#include "opal/util/bit_ops.h"
#include "opal/sys/atomic.h"                //atomic
#include "ompi/mca/pml/ob1/pml_ob1.h"       //dump

int mca_coll_adapt_bcast(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    if (count == 0) {
        return MPI_SUCCESS;
    }
    else {
        ompi_request_t * request;
        int retval = mca_coll_adapt_ibcast(buff, count, datatype, root, comm, &request, module);
        ompi_request_wait(&request, MPI_STATUS_IGNORE);
        return retval;
    }
}