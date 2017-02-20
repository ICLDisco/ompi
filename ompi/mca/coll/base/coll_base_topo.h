/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_BASE_TOPO_H_HAS_BEEN_INCLUDED
#define MCA_COLL_BASE_TOPO_H_HAS_BEEN_INCLUDED

#include "ompi_config.h"

#define MAXTREEFANOUT 32

BEGIN_C_DECLS

typedef struct ompi_coll_tree_t {
    int32_t tree_root;
    int32_t tree_fanout;
    int32_t tree_bmtree;
    int32_t tree_prev;
    int32_t tree_next[MAXTREEFANOUT];
    int32_t tree_nextsize;
} ompi_coll_tree_t;

ompi_coll_tree_t*
ompi_coll_base_topo_build_tree( int fanout,
                               struct ompi_communicator_t* com,
                               int root );
ompi_coll_tree_t*
ompi_coll_base_topo_build_in_order_bintree( struct ompi_communicator_t* comm );

ompi_coll_tree_t*
ompi_coll_base_topo_build_bmtree( struct ompi_communicator_t* comm,
                                 int root );
ompi_coll_tree_t*
ompi_coll_base_topo_build_in_order_bmtree( struct ompi_communicator_t* comm,
                                          int root );
ompi_coll_tree_t*
ompi_coll_base_topo_build_chain( int fanout,
                                struct ompi_communicator_t* com,
                                int root );

int ompi_coll_base_topo_destroy_tree( ompi_coll_tree_t** tree );

/* debugging stuff, will be removed later */
int ompi_coll_base_topo_dump_tree (ompi_coll_tree_t* tree, int rank);

ompi_coll_tree_t*
ompi_coll_base_topo_build_topoaware_linear(struct ompi_communicator_t* comm,
                                        int root, mca_coll_base_module_t *module );
ompi_coll_tree_t*
ompi_coll_base_topo_build_topoaware_chain(struct ompi_communicator_t* comm,
                                          int root, mca_coll_base_module_t *module );

void get_topo(int *topo, struct ompi_communicator_t* comm);

typedef struct ompi_coll_topo_helper_t {
    int num_group;  /*for a level, how many groups are there*/
    int* start_loc; /*the starting point of each group*/
} ompi_coll_topo_helper_t;

void sort_topo(int *topo, int start, int end, int size, int *ranks_a, int level);
void set_helper(ompi_coll_topo_helper_t *helper, int *ranks_a, int *ranks_s, int *topo, int root, int size);
void free_helper(ompi_coll_topo_helper_t *helper);
void print_helper(ompi_coll_topo_helper_t *helper);

int to_vrank(int rank, int *ranks, int size);
int to_rank(int vrank, int *ranks, int size);
void move_group_forward(int *ranks, int size, int start, int end);
void build_topoaware_chain(int count, int *start_loc, int rank, int rank_loc, int size, ompi_coll_tree_t *tree, int *ranks_a, int *ranks_s);
void build_topoaware_linear(int count, int *start_loc, int rank, int rank_loc, int size, ompi_coll_tree_t *tree, int *ranks_a, int *ranks_s);

END_C_DECLS

#endif  /* MCA_COLL_BASE_TOPO_H_HAS_BEEN_INCLUDED */
