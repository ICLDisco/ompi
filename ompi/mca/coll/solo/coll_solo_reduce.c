/*
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

int mca_coll_solo_reduce_intra(const void *sbuf, void *rbuf, int count,
                                 struct ompi_datatype_t *dtype,
                                 struct ompi_op_t *op,
                                 int root,
                                 struct ompi_communicator_t *comm, 
                                 mca_coll_base_module_t * module)
{
    if (ompi_op_is_commute(op)) {
        return mca_coll_solo_reduce_ring_intra_memcpy(sbuf, rbuf, count, dtype, op, root, comm, module);
    }
    else {
        return ompi_coll_base_reduce_intra_basic_linear(sbuf, rbuf, count, dtype, op, root, comm, module);
    }
}

/**
 * Each process operates a part of the shared data buffer in turn.
 * Suppose the number of processes is 4.
 * Step 1:
 * |  P0  |  P1  |  P2  |  P3  |
 * Step 2:
 * |  P1  |  P2  |  P3  |  P0  |
 * Step 3:
 * |  P2  |  P3  |  P0  |  P1  |
 * Step 4:
 * |  P3  |  P0  |  P1  |  P2  |
 * At last, root copies data back from the shared data buffer.
 */
int mca_coll_solo_reduce_ring_intra_memcpy(const void *sbuf, void *rbuf,
                                           int count,
                                           struct ompi_datatype_t *dtype,
                                           struct ompi_op_t *op,
                                           int root, 
                                           struct ompi_communicator_t *comm, 
                                           mca_coll_base_module_t * module)
{
    mca_coll_solo_module_t *solo_module = (mca_coll_solo_module_t *) module;
    int size = ompi_comm_size(comm);
    int rank = ompi_comm_rank(comm);
    int i;

    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);

    /* Set up segment count */
    int seg_count, l_seg_count;
    seg_count = count / size;
    l_seg_count = seg_count;
    if (rank == size - 1) {
        seg_count = count - rank * l_seg_count;
    }

    /* Enable solo module if necessary */
    if (!solo_module->enabled) {
        mca_coll_solo_lazy_enable(module, comm);
    }

    char **data_bufs = NULL;
    int *ids = NULL;
    if ((size_t) l_seg_count * extent <= mca_coll_solo_component.static_block_size) {
        data_bufs = solo_module->data_bufs;
    } else if ((size_t) l_seg_count * extent <= mca_coll_solo_component.mpool_large_block_size) {
        data_bufs = (char **) malloc(sizeof(char *) * size);
        ids = (int *) malloc(sizeof(int) * size);
        ids[rank] =
            mca_coll_solo_mpool_request(mca_coll_solo_component.solo_mpool, l_seg_count * extent);

        ompi_coll_base_allgather_intra_recursivedoubling(MPI_IN_PLACE, 0,
                                                         MPI_DATATYPE_NULL,
                                                         ids,
                                                         1, MPI_INT, comm,
                                                         (mca_coll_base_module_t *)
                                                         solo_module);
        for (i = 0; i < size; i++) {
            data_bufs[i] =
                mca_coll_solo_mpool_calculate(mca_coll_solo_component.solo_mpool, ids[i],
                                                l_seg_count * extent);
        }
    } else {
        /* For the messages which are greater than mpool_large_block_size*np, invoke this reduce multiple times */
        int seg_count = mca_coll_solo_component.mpool_large_block_size / extent;
        int num_segments = (count + seg_count - 1) / seg_count;
        int last_count = count - seg_count * (num_segments - 1);
        for (int i = 0; i < num_segments; i++) {
            char *temp_sbuf;
            if (sbuf == MPI_IN_PLACE)
                temp_sbuf = MPI_IN_PLACE;
            else
                temp_sbuf = (char *)sbuf + seg_count * extent * i;
            char *temp_rbuf = (char *)rbuf + seg_count * extent * i;
            int temp_count = seg_count;
            if (i == num_segments - 1) {
                temp_count = last_count;
            }
            mca_coll_solo_reduce_ring_intra_memcpy(temp_sbuf, temp_rbuf, temp_count, dtype, op, 
                                                   root, comm, module);
        }
        return MPI_SUCCESS;
    }

    char *sbuf_temp = (char *)sbuf;
    if( sbuf == MPI_IN_PLACE ) {
        sbuf_temp = (char *)rbuf;
    }

    *(int *) (solo_module->ctrl_bufs[rank]) = rank;
    mac_coll_solo_barrier_intra(comm, module);

    int cur = rank;
    for (i = 0; i < size; i++) {
        if (cur != size - 1) {
            seg_count = l_seg_count;
        } else {
            seg_count = count - cur * l_seg_count;
        }
        /* At first iteration, copy local data to the shared data buffer */
        if (cur == rank) {
            mca_coll_solo_copy((void *) ((char *) sbuf_temp + cur * l_seg_count * extent), 
                               (void *) data_bufs[cur], dtype, seg_count, extent);
            mac_coll_solo_barrier_intra(comm, module);
        }
        /* For other iterations, do operations on the shared data buffer */
        else {
            ompi_op_reduce(op, (char *) sbuf_temp + cur * l_seg_count * extent,
                           data_bufs[cur], seg_count, dtype);
            mac_coll_solo_barrier_intra(comm, module);
        }
        cur = (cur - 1 + size) % size;
        *(int *) (solo_module->ctrl_bufs[rank]) =
            (*(int *) (solo_module->ctrl_bufs[rank]) + 1) % size;
        mac_coll_solo_barrier_intra(comm, module);

    }
    /* At last, root copies data from the solo data buffer */
    if (rank == root) {
        char *c;
        c = rbuf;
        for (i = 0; i < size; i++) {
            if (i != size - 1) {
                seg_count = l_seg_count;
            } else {
                seg_count = count - i * l_seg_count;
            }
            mca_coll_solo_copy((void *) data_bufs[i], (void *) c, dtype, seg_count, extent);
            c = c + seg_count * extent;
        }
    }
    mac_coll_solo_barrier_intra(comm, module);
    if ((size_t) l_seg_count * extent > mca_coll_solo_component.static_block_size && 
        (size_t) l_seg_count * extent <= mca_coll_solo_component.mpool_large_block_size) {
        mca_coll_solo_mpool_return(mca_coll_solo_component.solo_mpool, ids[rank],
                                   l_seg_count * extent);
        if (ids != NULL) {
            free(ids);
            ids = NULL;
        }

        if (data_bufs != NULL) {
            free(data_bufs);
            data_bufs = NULL;
        }
    }

    return OMPI_SUCCESS;
}
