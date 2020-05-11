/**
 * Copyright (c) 2019      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "coll_solo.h"

int mca_coll_solo_bcast_intra(void *buff, int count,
                              struct ompi_datatype_t *dtype, 
                              int root,
                              struct ompi_communicator_t *comm, 
                              mca_coll_base_module_t * module)
{
    return mca_coll_solo_bcast_linear_intra_memcpy(buff, count, dtype, root, comm, module);
}

/* linear bcast with memcpy */
int mca_coll_solo_bcast_linear_intra_memcpy(void *buff, int count,
                                            struct ompi_datatype_t *dtype, 
                                            int root,
                                            struct ompi_communicator_t *comm, 
                                            mca_coll_base_module_t * module)
{
    mca_coll_solo_module_t *solo_module = (mca_coll_solo_module_t *) module;

    int rank = ompi_comm_rank(comm);
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    /* Enable solo module if necessary */
    if (!solo_module->enabled) {
        mca_coll_solo_lazy_enable(module, comm);
    }
    /* Init the data_buf - shared among all the processes */
    int id;
    char *data_buf;
    if ((size_t) count * extent <= mca_coll_solo_component.static_block_size) {
        data_buf = solo_module->data_bufs[root];
    } else if ((size_t) count * extent <= mca_coll_solo_component.mpool_small_block_size) {
        if (rank == root) {
            id = mca_coll_solo_mpool_request(mca_coll_solo_component.solo_mpool, count * extent);
        }
        mca_coll_solo_bcast_linear_intra_memcpy(&id, 1, MPI_INT, root, comm, module);
        data_buf = mca_coll_solo_mpool_calculate(mca_coll_solo_component.solo_mpool, id, 
                                                 count * extent);
    } else {
        return mca_coll_solo_bcast_pipeline_intra_memcpy(buff, count, dtype, root, comm, module, 
                                                         mca_coll_solo_component.mpool_small_block_size);
    }

    /* Root copy data to the shared memory block */
    if (rank == root) {
        mca_coll_solo_pack_to_shared(buff, (void *) data_buf, dtype, count, extent);
    }
    mac_coll_solo_barrier_intra(comm, module);
    /* Other processes copy data from the shared memory block */
    if (rank != root) {
        mca_coll_solo_unpack_from_shared(buff, (void *) data_buf, dtype, count, extent);
    }
    mac_coll_solo_barrier_intra(comm, module);
    if ((size_t) count * extent > mca_coll_solo_component.static_block_size &&
        (size_t) count * extent <= mca_coll_solo_component.mpool_large_block_size) {
        if (rank == root) {
            mca_coll_solo_mpool_return(mca_coll_solo_component.solo_mpool, id, count * extent);
        }
    }
    return OMPI_SUCCESS;
}

int mca_coll_solo_bcast_pipeline_intra_memcpy(void *buff, int count,
                                              struct ompi_datatype_t *dtype, 
                                              int root,
                                              struct ompi_communicator_t *comm, 
                                              mca_coll_base_module_t * module,
                                              size_t seg_size)
{
    mca_coll_solo_module_t *solo_module = (mca_coll_solo_module_t *) module;

    int rank = ompi_comm_rank(comm);
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    /* Enable solo module if necessary */
    if (!solo_module->enabled) {
        mca_coll_solo_lazy_enable(module, comm);
    }
    /* Init the data_bufs - shared among all the processes, needs two for the pipelining */
    int ids[2];
    char *data_bufs[2];
    int i;
    for (i = 0; i < 2; i++) {
        if (rank == root) {
            ids[i] = mca_coll_solo_mpool_request(mca_coll_solo_component.solo_mpool, seg_size);
        }
    }
    mca_coll_solo_bcast_linear_intra_memcpy(ids, 2, MPI_INT, root, comm, module);
    for (i = 0; i < 2; i++) {
        data_bufs[i] = mca_coll_solo_mpool_calculate(mca_coll_solo_component.solo_mpool, ids[i], 
                                                     seg_size);
    }
    
    int seg_count = seg_size / extent;
    int num_segments = (count + seg_count - 1) / seg_count;
    int last_count = count - seg_count * (num_segments - 1);

    for (i = 0; i <= num_segments; i++) {
        int cur = i & 1;
        int pre = !cur;
        if (i == 0) {
            /* In the first iteration, root copies data to the current shared memory block */
            if (rank == root) {
                mca_coll_solo_pack_to_shared(buff, (void *) data_bufs[cur], dtype, seg_count, extent);
            }
        }
        else if ( i == num_segments) {
            /* In the last iteration, other processes copy data from the previous shared memory block */
            mca_coll_solo_unpack_from_shared(((char *) buff) + seg_count * extent * (i - 1), (void *) data_bufs[pre], dtype, last_count, extent);
        }
        else {
            /** 
             * For other iterations, root copies data to the current shared memory block and 
             * other proceeses copy data from the previous shared memory block.
             */
            if (rank == root) {
                int temp_count = seg_count;
                if ( i == num_segments - 1) {
                    temp_count = last_count;
                }
                mca_coll_solo_pack_to_shared(((char *) buff) + seg_count * extent * i, data_bufs[cur], dtype, temp_count, extent);
            }
            else {
                mca_coll_solo_unpack_from_shared(((char *) buff) + seg_count * extent * (i - 1), (void *) data_bufs[pre], dtype, seg_count, extent);
            }
        }
        mac_coll_solo_barrier_intra(comm, module);
    }

    /* Return the data_bufs */
    for (i = 0; i < 2; i++) {
        if (rank == root) {
            mca_coll_solo_mpool_return(mca_coll_solo_component.solo_mpool, ids[i], seg_size);
        }
    }
    
    return OMPI_SUCCESS;
}
