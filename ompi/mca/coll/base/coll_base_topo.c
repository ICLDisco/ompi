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
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpi.h"
#include "opal/util/bit_ops.h"
#include "ompi/constants.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "coll_base_topo.h"
#include "ompi/mca/pml/pml.h"   //for probe
#if OPAL_CUDA_SUPPORT
#include "opal/mca/common/cuda/common_cuda.h"
#endif
#include <math.h>
#include <limits.h>
/*
 * Some static helpers.
 */
static int pown( int fanout, int num )
{
    int j, p = 1;
    if( num < 0 ) return 0;
    if (1==num) return fanout;
    if (2==fanout) {
        return p<<num;
    }
    else {
        for( j = 0; j < num; j++ ) { p*= fanout; }
    }
    return p;
}

static int calculate_level( int fanout, int rank )
{
    int level, num;
    if( rank < 0 ) return -1;
    for( level = 0, num = 0; num <= rank; level++ ) {
        num += pown(fanout, level);
    }
    return level-1;
}

static int calculate_num_nodes_up_to_level( int fanout, int level )
{
    /* just use geometric progression formula for sum:
     a^0+a^1+...a^(n-1) = (a^n-1)/(a-1) */
    return ((pown(fanout,level) - 1)/(fanout - 1));
}

/*
 * And now the building functions.
 *
 * An example for fanout = 2, comm_size = 7
 *
 *              0           <-- delta = 1 (fanout^0)
 *            /   \
 *           1     2        <-- delta = 2 (fanout^1)
 *          / \   / \
 *         3   5 4   6      <-- delta = 4 (fanout^2)
 */

ompi_coll_tree_t*
ompi_coll_base_topo_build_tree( int fanout,
                               struct ompi_communicator_t* comm,
                               int root )
{
    int rank, size, schild, sparent, shiftedrank, i;
    int level; /* location of my rank in the tree structure of size */
    int delta; /* number of nodes on my level */
    int slimit; /* total number of nodes on levels above me */
    ompi_coll_tree_t* tree;
    
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "coll:base:topo_build_tree Building fo %d rt %d", fanout, root));
    
    if (fanout<1) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "coll:base:topo_build_tree invalid fanout %d", fanout));
        return NULL;
    }
    if (fanout>MAXTREEFANOUT) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo_build_tree invalid fanout %d bigger than max %d", fanout, MAXTREEFANOUT));
        return NULL;
    }
    
    /*
     * Get size and rank of the process in this communicator
     */
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    
    tree = (ompi_coll_tree_t*)malloc(sizeof(ompi_coll_tree_t));
    if (!tree) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo_build_tree PANIC::out of memory"));
        return NULL;
    }
    
    tree->tree_root     = MPI_UNDEFINED;
    tree->tree_nextsize = MPI_UNDEFINED;
    
    /*
     * Set root
     */
    tree->tree_root = root;
    
    /*
     * Initialize tree
     */
    tree->tree_fanout   = fanout;
    tree->tree_bmtree   = 0;
    tree->tree_root     = root;
    tree->tree_prev     = -1;
    tree->tree_nextsize = 0;
    for( i = 0; i < fanout; i++ ) {
        tree->tree_next[i] = -1;
    }
    
    /* return if we have less than 2 processes */
    if( size < 2 ) {
        return tree;
    }
    
    /*
     * Shift all ranks by root, so that the algorithm can be
     * designed as if root would be always 0
     * shiftedrank should be used in calculating distances
     * and position in tree
     */
    shiftedrank = rank - root;
    if( shiftedrank < 0 ) {
        shiftedrank += size;
    }
    
    /* calculate my level */
    level = calculate_level( fanout, shiftedrank );
    delta = pown( fanout, level );
    
    /* find my children */
    for( i = 0; i < fanout; i++ ) {
        schild = shiftedrank + delta * (i+1);
        if( schild < size ) {
            tree->tree_next[i] = (schild+root)%size;
            tree->tree_nextsize = tree->tree_nextsize + 1;
        } else {
            break;
        }
    }
    
    /* find my parent */
    slimit = calculate_num_nodes_up_to_level( fanout, level );
    sparent = shiftedrank;
    if( sparent < fanout ) {
        sparent = 0;
    } else {
        while( sparent >= slimit ) {
            sparent -= delta/fanout;
        }
    }
    tree->tree_prev = (sparent+root)%size;
    
    return tree;
}

/*
 * Constructs in-order binary tree which can be used for non-commutative reduce
 * operations.
 * Root of this tree is always rank (size-1) and fanout is 2.
 * Here are some of the examples of this tree:
 * size == 2     size == 3     size == 4                size == 9
 *      1             2             3                        8
 *     /             / \          /   \                    /   \
 *    0             1  0         2     1                  7     3
 *                                    /                 /  \   / \
 *                                   0                 6    5 2   1
 *                                                         /     /
 *                                                        4     0
 */
ompi_coll_tree_t*
ompi_coll_base_topo_build_in_order_bintree( struct ompi_communicator_t* comm )
{
    int rank, size, myrank, rightsize, delta, parent, lchild, rchild;
    ompi_coll_tree_t* tree;
    
    /*
     * Get size and rank of the process in this communicator
     */
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    
    tree = (ompi_coll_tree_t*)malloc(sizeof(ompi_coll_tree_t));
    if (!tree) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                     "coll:base:topo_build_tree PANIC::out of memory"));
        return NULL;
    }
    
    tree->tree_root     = MPI_UNDEFINED;
    tree->tree_nextsize = MPI_UNDEFINED;
    
    /*
     * Initialize tree
     */
    tree->tree_fanout   = 2;
    tree->tree_bmtree   = 0;
    tree->tree_root     = size - 1;
    tree->tree_prev     = -1;
    tree->tree_nextsize = 0;
    tree->tree_next[0]  = -1;
    tree->tree_next[1]  = -1;
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "coll:base:topo_build_in_order_tree Building fo %d rt %d",
                 tree->tree_fanout, tree->tree_root));
    
    /*
     * Build the tree
     */
    myrank = rank;
    parent = size - 1;
    delta = 0;
    
    while ( 1 ) {
        /* Compute the size of the right subtree */
        rightsize = size >> 1;
        
        /* Determine the left and right child of this parent  */
        lchild = -1;
        rchild = -1;
        if (size - 1 > 0) {
            lchild = parent - 1;
            if (lchild > 0) {
                rchild = rightsize - 1;
            }
        }
        
        /* The following cases are possible: myrank can be
         - a parent,
         - belong to the left subtree, or
         - belong to the right subtee
         Each of the cases need to be handled differently.
         */
        
        if (myrank == parent) {
            /* I am the parent:
             - compute real ranks of my children, and exit the loop. */
            if (lchild >= 0) tree->tree_next[0] = lchild + delta;
            if (rchild >= 0) tree->tree_next[1] = rchild + delta;
            break;
        }
        if (myrank > rchild) {
            /* I belong to the left subtree:
             - If I am the left child, compute real rank of my parent
             - Iterate down through tree:
             compute new size, shift ranks down, and update delta.
             */
            if (myrank == lchild) {
                tree->tree_prev = parent + delta;
            }
            size = size - rightsize - 1;
            delta = delta + rightsize;
            myrank = myrank - rightsize;
            parent = size - 1;
            
        } else {
            /* I belong to the right subtree:
             - If I am the right child, compute real rank of my parent
             - Iterate down through tree:
             compute new size and parent,
             but the delta and rank do not need to change.
             */
            if (myrank == rchild) {
                tree->tree_prev = parent + delta;
            }
            size = rightsize;
            parent = rchild;
        }
    }
    
    if (tree->tree_next[0] >= 0) { tree->tree_nextsize = 1; }
    if (tree->tree_next[1] >= 0) { tree->tree_nextsize += 1; }
    
    return tree;
}

int ompi_coll_base_topo_destroy_tree( ompi_coll_tree_t** tree )
{
    ompi_coll_tree_t *ptr;
    
    if ((!tree)||(!*tree)) {
        return OMPI_SUCCESS;
    }
    
    ptr = *tree;
    
    free (ptr);
    *tree = NULL;   /* mark tree as gone */
    
    return OMPI_SUCCESS;
}

int ompi_coll_base_topo_destroy_two_trees( ompi_coll_tree_t** trees )
{
    free(trees[0]);
    free(trees[1]);
    free(trees);
    *trees = NULL;   /* mark tree as gone */
    
    return OMPI_SUCCESS;
}

/*
 *
 * Here are some of the examples of this tree:
 * size == 2                   size = 4                 size = 8
 *      0                           0                        0
 *     /                            | \                    / | \
 *    1                             2  1                  4  2  1
 *                                     |                     |  |\
 *                                     3                     6  5 3
 *                                                                |
 *                                                                7
 */
ompi_coll_tree_t*
ompi_coll_base_topo_build_bmtree( struct ompi_communicator_t* comm,
                                 int root )
{
    int childs = 0, rank, size, mask = 1, index, remote, i;
    ompi_coll_tree_t *bmtree;
    
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo:build_bmtree rt %d", root));
    
    /*
     * Get size and rank of the process in this communicator
     */
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    
    index = rank -root;
    
    bmtree = (ompi_coll_tree_t*)malloc(sizeof(ompi_coll_tree_t));
    if (!bmtree) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo:build_bmtree PANIC out of memory"));
        return NULL;
    }
    
    bmtree->tree_bmtree   = 1;
    
    bmtree->tree_root     = MPI_UNDEFINED;
    bmtree->tree_nextsize = MPI_UNDEFINED;
    for( i = 0;i < MAXTREEFANOUT; i++ ) {
        bmtree->tree_next[i] = -1;
    }
    
    if( index < 0 ) index += size;
    
    mask = opal_next_poweroftwo(index);
    
    /* Now I can compute my father rank */
    if( root == rank ) {
        bmtree->tree_prev = root;
    } else {
        remote = (index ^ (mask >> 1)) + root;
        if( remote >= size ) remote -= size;
        bmtree->tree_prev = remote;
    }
    /* And now let's fill my childs */
    while( mask < size ) {
        remote = (index ^ mask);
        if( remote >= size ) break;
        remote += root;
        if( remote >= size ) remote -= size;
        if (childs==MAXTREEFANOUT) {
            OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo:build_bmtree max fanout incorrect %d needed %d", MAXTREEFANOUT, childs));
            free(bmtree);
            return NULL;
        }
        bmtree->tree_next[childs] = remote;
        mask <<= 1;
        childs++;
    }
    bmtree->tree_nextsize = childs;
    bmtree->tree_root     = root;
    return bmtree;
}

/*
 * Constructs in-order binomial tree which can be used for gather/scatter
 * operations.
 *
 * Here are some of the examples of this tree:
 * size == 2                   size = 4                 size = 8
 *      0                           0                        0
 *     /                          / |                      / | \
 *    1                          1  2                     1  2  4
 *                                  |                        |  | \
 *                                  3                        3  5  6
 *                                                                 |
 *                                                                 7
 */
ompi_coll_tree_t*
ompi_coll_base_topo_build_in_order_bmtree( struct ompi_communicator_t* comm,
                                          int root )
{
    int childs = 0, rank, vrank, size, mask = 1, remote, i;
    ompi_coll_tree_t *bmtree;
    
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo:build_in_order_bmtree rt %d", root));
    
    /*
     * Get size and rank of the process in this communicator
     */
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    
    vrank = (rank - root + size) % size;
    
    bmtree = (ompi_coll_tree_t*)malloc(sizeof(ompi_coll_tree_t));
    if (!bmtree) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo:build_bmtree PANIC out of memory"));
        return NULL;
    }
    
    bmtree->tree_bmtree   = 1;
    bmtree->tree_root     = MPI_UNDEFINED;
    bmtree->tree_nextsize = MPI_UNDEFINED;
    for(i=0;i<MAXTREEFANOUT;i++) {
        bmtree->tree_next[i] = -1;
    }
    
    if (root == rank) {
        bmtree->tree_prev = root;
    }
    
    while (mask < size) {
        remote = vrank ^ mask;
        if (remote < vrank) {
            bmtree->tree_prev = (remote + root) % size;
            break;
        } else if (remote < size) {
            bmtree->tree_next[childs] = (remote + root) % size;
            childs++;
            if (childs==MAXTREEFANOUT) {
                OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                             "coll:base:topo:build_bmtree max fanout incorrect %d needed %d",
                             MAXTREEFANOUT, childs));
                free(bmtree);
                return NULL;
            }
        }
        mask <<= 1;
    }
    bmtree->tree_nextsize = childs;
    bmtree->tree_root     = root;
    
    return bmtree;
}


ompi_coll_tree_t*
ompi_coll_base_topo_build_chain( int fanout,
                                struct ompi_communicator_t* comm,
                                int root )
{
    int i, maxchainlen, mark, head, len, rank, size, srank /* shifted rank */;
    ompi_coll_tree_t *chain;
    
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo:build_chain fo %d rt %d", fanout, root));
    
    /*
     * Get size and rank of the process in this communicator
     */
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    
    if( fanout < 1 ) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo:build_chain WARNING invalid fanout of ZERO, forcing to 1 (pipeline)!"));
        fanout = 1;
    }
    if (fanout>MAXTREEFANOUT) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo:build_chain WARNING invalid fanout %d bigger than max %d, forcing to max!", fanout, MAXTREEFANOUT));
        fanout = MAXTREEFANOUT;
    }
    
    /*
     * Allocate space for topology arrays if needed
     */
    chain = (ompi_coll_tree_t*)malloc( sizeof(ompi_coll_tree_t) );
    if (!chain) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo:build_chain PANIC out of memory"));
        fflush(stdout);
        return NULL;
    }
    chain->tree_root     = MPI_UNDEFINED;
    chain->tree_nextsize = -1;
    for(i=0;i<fanout;i++) chain->tree_next[i] = -1;
    
    /*
     * Set root & numchain
     */
    chain->tree_root = root;
    if( (size - 1) < fanout ) {
        chain->tree_nextsize = size-1;
        fanout = size-1;
    } else {
        chain->tree_nextsize = fanout;
    }
    
    /*
     * Shift ranks
     */
    srank = rank - root;
    if (srank < 0) srank += size;
    
    /*
     * Special case - fanout == 1
     */
    if( fanout == 1 ) {
        if( srank == 0 ) chain->tree_prev = -1;
        else chain->tree_prev = (srank-1+root)%size;
        
        if( (srank + 1) >= size) {
            chain->tree_next[0] = -1;
            chain->tree_nextsize = 0;
        } else {
            chain->tree_next[0] = (srank+1+root)%size;
            chain->tree_nextsize = 1;
        }
        return chain;
    }
    
    /* Let's handle the case where there is just one node in the communicator */
    if( size == 1 ) {
        chain->tree_next[0] = -1;
        chain->tree_nextsize = 0;
        chain->tree_prev = -1;
        return chain;
    }
    /*
     * Calculate maximum chain length
     */
    maxchainlen = (size-1) / fanout;
    if( (size-1) % fanout != 0 ) {
        maxchainlen++;
        mark = (size-1)%fanout;
    } else {
        mark = fanout+1;
    }
    
    /*
     * Find your own place in the list of shifted ranks
     */
    if( srank != 0 ) {
        int column;
        if( srank-1 < (mark * maxchainlen) ) {
            column = (srank-1)/maxchainlen;
            head = 1+column*maxchainlen;
            len = maxchainlen;
        } else {
            column = mark + (srank-1-mark*maxchainlen)/(maxchainlen-1);
            head = mark*maxchainlen+1+(column-mark)*(maxchainlen-1);
            len = maxchainlen-1;
        }
        
        if( srank == head ) {
            chain->tree_prev = 0; /*root*/
        } else {
            chain->tree_prev = srank-1; /* rank -1 */
        }
        if( srank == (head + len - 1) ) {
            chain->tree_next[0] = -1;
            chain->tree_nextsize = 0;
        } else {
            if( (srank + 1) < size ) {
                chain->tree_next[0] = srank+1;
                chain->tree_nextsize = 1;
            } else {
                chain->tree_next[0] = -1;
                chain->tree_nextsize = 0;
            }
        }
        chain->tree_prev = (chain->tree_prev+root)%size;
        if( chain->tree_next[0] != -1 ) {
            chain->tree_next[0] = (chain->tree_next[0]+root)%size;
        }
    } else {
        /*
         * Unshift values
         */
        chain->tree_prev = -1;
        chain->tree_next[0] = (root+1)%size;
        for( i = 1; i < fanout; i++ ) {
            chain->tree_next[i] = chain->tree_next[i-1] + maxchainlen;
            if( i > mark ) {
                chain->tree_next[i]--;
            }
            chain->tree_next[i] %= size;
        }
        chain->tree_nextsize = fanout;
    }
    
    return chain;
}

//segment number must >=2 and the number of nodes must >=3
ompi_coll_tree_t**
ompi_coll_base_topo_build_two_trees_binary(struct ompi_communicator_t* comm,
                                           int root ){
    int i, j, rank, size, vrank;
    ompi_coll_tree_t** two_trees;
    //build the two tree array
    two_trees = (ompi_coll_tree_t **)malloc(2 * sizeof(ompi_coll_tree_t *));
    two_trees[0] = (ompi_coll_tree_t*)malloc(sizeof(ompi_coll_tree_t));
    two_trees[1] = (ompi_coll_tree_t*)malloc(sizeof(ompi_coll_tree_t));
    if (!two_trees || !two_trees[0] || !two_trees[1]) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                     "coll:base:topo:build_two_tree PANIC out of memory"));
        return NULL;
    }
    
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    vrank = (rank-root+size-1)%size;
    //printf("rank = %d, vrank = %d\n", rank, vrank);
    //int recovered_rank = (vrank+1+root)%size;      //only for test
    //printf("rank = %d, recovered_rank = %d\n", rank, recovered_rank);   //for test
    int p = size-1;       //number of node in the two tree, exclude the root
    int h = ceil(log(p+1)/log(2));
    //printf("p = %d, h = %d\n", p, h);
    
    //init the two trees
    two_trees[0]->tree_bmtree   = 0;
    two_trees[0]->tree_root     = root;
    two_trees[0]->tree_nextsize = 0;
    for( i = 0;i < MAXTREEFANOUT; i++ ) {
        two_trees[0]->tree_next[i] = -1;
    }
    
    two_trees[1]->tree_bmtree   = 0;
    two_trees[1]->tree_root     = root;
    two_trees[1]->tree_nextsize = 0;
    for( i = 0;i < MAXTREEFANOUT; i++ ) {
        two_trees[1]->tree_next[i] = -1;
    }
    if (p <= 1) {
        return two_trees;
    }
    
    //build tree 0
    if (vrank != p){
        for (j=0; j<h; j++) {
            //to decide which level the node will be in
            if (vrank % ((int)pow(2,j+1)) == (int)pow(2,j)-1){
                if (j != 0) {
                    //find children for all the non leaf nodes
                    //find left child, left child always in the boundary
                    int child_vrank = vrank-(int)pow(2,j-1);
                    two_trees[0]->tree_next[0] = (child_vrank+1+root)%size;
                    two_trees[0]->tree_nextsize += 1;
                    
                    //find right child, right child is not always in the boundary
                    child_vrank = vrank+(int)pow(2,j-1);
                    if (child_vrank < p && child_vrank >= 0) {
                        two_trees[0]->tree_next[1] = (child_vrank+1+root)%size;
                        two_trees[0]->tree_nextsize += 1;
                    }
                    //if the right children is not in the boundary
                    if (two_trees[0]->tree_nextsize < 2) {
                        //it is not the last node
                        if (vrank != p-1) {
                            int right_p = p-vrank-1;
                            int right_h = ceil(log(right_p+1)/log(2));
                            two_trees[0]->tree_next[1] = (vrank+(int)pow(2,right_h-1)+1+root)%size;
                            two_trees[0]->tree_nextsize += 1;
                        }
                    }
                }
                if (j != h-1) {
                    //find parent for all the non root nodes
                    int parent_vrank = (vrank-(int)pow(2,j)+size)%size;
                    if (parent_vrank % ((int)pow(2,j+2)) != (int)pow(2,j+1)-1) {
                        parent_vrank = (vrank+(int)pow(2,j)+size)%size;
                    }
                    two_trees[0]->tree_prev = (parent_vrank+1+root)%size;
                    //printf("rank = %d, parent = %d\n", rank, two_trees[0]->tree_prev);   //for test
                }
                else{
                    two_trees[0]->tree_prev = root;
                }
            }
        }
    }
    else{
        two_trees[0]->tree_next[0] = ((int)pow(2,h-1)+root)%size;
        two_trees[0]->tree_nextsize += 1;
        two_trees[0]->tree_prev = -1;
        
    }
    
    //build tree 1
    if (vrank != p){
        vrank = (vrank+1)%(size-1);
        //printf("rank = %d, vrank = %d\n", rank, vrank);
        //printf("rank = %d, recovered_rank = %d\n", rank, ((vrank+size-2)%(size-1)+1+root)%size);
        for (j=0; j<h; j++) {
            //to decide which level the node will be in
            if (vrank % ((int)pow(2,j+1)) == (int)pow(2,j)-1){
                if (j != 0) {
                    //find children for all the non leaf nodes
                    //find left child, left child always in the boundary
                    int child_vrank = vrank-(int)pow(2,j-1);
                    two_trees[1]->tree_next[0] = ((child_vrank+size-2)%(size-1)+1+root)%size;
                    two_trees[1]->tree_nextsize += 1;
                    
                    //find right child, right child is not always in the boundary
                    child_vrank = vrank+(int)pow(2,j-1);
                    if (child_vrank < p && child_vrank >= 0) {
                        two_trees[1]->tree_next[1] = ((child_vrank+size-2)%(size-1)+1+root)%size;
                        two_trees[1]->tree_nextsize += 1;
                    }
                    //if the right children is not in the boundary
                    if (two_trees[1]->tree_nextsize < 2) {
                        //it is not the last node
                        if (vrank != p-1) {
                            int right_p = p-vrank-1;
                            int right_h = ceil(log(right_p+1)/log(2));
                            two_trees[1]->tree_next[1] = ((vrank+(int)pow(2,right_h-1)+size-2)%(size-1)+1+root)%size;
                            two_trees[1]->tree_nextsize += 1;
                        }
                    }
                }
                if (j != h-1) {
                    //find parent for all the non root nodes
                    int parent_vrank = (vrank-(int)pow(2,j)+size)%size;
                    if (parent_vrank % ((int)pow(2,j+2)) != (int)pow(2,j+1)-1) {
                        parent_vrank = (vrank+(int)pow(2,j)+size)%size;
                    }
                    two_trees[1]->tree_prev = ((parent_vrank+size-2)%(size-1)+1+root)%size;
                    //printf("rank = %d, parent = %d\n", rank, two_trees[1]->tree_prev);   //for test
                }
                else{
                    two_trees[1]->tree_prev = root;
                }
            }
        }
    }
    else{
        two_trees[1]->tree_next[0] = (((int)pow(2,h-1)-1+size-2)%(size-1)+1+root)%size;
        two_trees[1]->tree_nextsize += 1;
        two_trees[1]->tree_prev = -1;
    }
    return two_trees;
}

static void add_edge(int** tree, int length, int parent, int child){
    int i;
    for (i=0; i<length; i++) {
        if (tree[parent][i] == -1) {
            break;
        }
    }
    tree[parent][i] = child;
}

static void divide_group(int** tree, int length, int start, int end, int tree_id){
    //2 elements in a group
    if (end - start == 1) {
        if (tree_id == 0){
            add_edge(tree, length, start, end);
        }
        else {
            add_edge(tree, length, end, start);
        }
    }
    //3 elements in a group
    else if(end - start == 2){
        if (tree_id == 0) {
            add_edge(tree, length, start, start+1);
            add_edge(tree, length, start+1, end);
        }
        else {
            add_edge(tree, length, end, start);
            add_edge(tree, length, start, start+1);
        }
    }
    else if(end - start > 2){
        int mid = (end-start)/2+start;
        if (tree_id == 0) {
            add_edge(tree, length, start, mid+1);
        }
        else {
            add_edge(tree, length, end, mid);
        }
        divide_group(tree, length, start, mid, tree_id);
        divide_group(tree, length, mid+1, end, tree_id);
    }
}

//segment number must >=2 and the number of nodes must >=3
ompi_coll_tree_t**
ompi_coll_base_topo_build_two_trees_binomial(struct ompi_communicator_t* comm,
                                             int root ){
    int i, j, rank, size, vrank;
    ompi_coll_tree_t** two_trees;
    //build the two tree array
    two_trees = (ompi_coll_tree_t **)malloc(2 * sizeof(ompi_coll_tree_t *));
    two_trees[0] = (ompi_coll_tree_t*)malloc(sizeof(ompi_coll_tree_t));
    two_trees[1] = (ompi_coll_tree_t*)malloc(sizeof(ompi_coll_tree_t));
    if (!two_trees || !two_trees[0] || !two_trees[1]) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                     "coll:base:topo:build_two_tree PANIC out of memory"));
        return NULL;
    }
    
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    vrank = (rank-root+size-1)%size;    //root has the highest vrank
    //printf("rank = %d, vrank = %d\n", rank, vrank);
    //int recovered_rank = (vrank+1+root)%size;      //only for test
    //printf("rank = %d, recovered_rank = %d\n", rank, recovered_rank);   //for test
    int p = size-1;       //number of node in the two tree, exclude the root
    int max_num_child = ceil(log(p)/log(2));
    
    //calculate the topology
    //build a 2D array for tree0 and init with -1
    int** tree0 = (int **)malloc(p * sizeof(int *));
    for (i=0; i<p; i++) {
        tree0[i] = (int *)malloc(max_num_child * sizeof(int));
        for (j=0; j<max_num_child; j++) {
            tree0[i][j] = -1;
        }
    }
    divide_group(tree0, max_num_child, 0, p-1, 0);
    
    int** tree1 = (int **)malloc(p * sizeof(int *));
    for (i=0; i<p; i++) {
        tree1[i] = (int *)malloc(max_num_child * sizeof(int));
        for (j=0; j<max_num_child; j++) {
            tree1[i][j] = -1;
        }
    }
    divide_group(tree1, max_num_child, 0, p-1, 1);
    
    //init the two trees
    two_trees[0]->tree_bmtree   = 1;
    two_trees[0]->tree_root     = root;
    two_trees[0]->tree_nextsize = 0;
    for( i = 0;i < MAXTREEFANOUT; i++ ) {
        two_trees[0]->tree_next[i] = -1;
    }
    
    two_trees[1]->tree_bmtree   = 1;
    two_trees[1]->tree_root     = root;
    two_trees[1]->tree_nextsize = 0;
    for( i = 0;i < MAXTREEFANOUT; i++ ) {
        two_trees[1]->tree_next[i] = -1;
    }
    if (p <= 1) {
        return two_trees;
    }
    
    //build tree 0
    if (vrank != p) {
        //find children for all the nodes except root
        int num_child = 0;
        for (i=0; i<max_num_child; i++) {
            int child_vrank = tree0[vrank][i];
            if (child_vrank != -1) {
                two_trees[0]->tree_next[num_child] = (child_vrank+1+root)%size;
                two_trees[0]->tree_nextsize += 1;
                num_child++;
            }
        }
        //find parent for all the nodes
        if (vrank == 0) {
            two_trees[0]->tree_prev = root;
        }
        else{
            for (i=0; i<p; i++) {
                for (j=0; j<max_num_child; j++) {
                    if (vrank == tree0[i][j]) {
                        two_trees[0]->tree_prev = (i+1+root)%size;
                    }
                }
            }
        }
    }
    else{
        two_trees[0]->tree_next[0] = (0+1+root)%size;
        two_trees[0]->tree_nextsize += 1;
        two_trees[0]->tree_prev = -1;
    }
    
    //build tree 1
    if (vrank != p) {
        //find children for all the nodes except root
        int num_child = 0;
        for (i=0; i<max_num_child; i++) {
            int child_vrank = tree1[vrank][i];
            if (child_vrank != -1) {
                two_trees[1]->tree_next[num_child] = (child_vrank+1+root)%size;
                two_trees[1]->tree_nextsize += 1;
                num_child++;
            }
        }
        //find parent for all the nodes
        if (vrank == p-1) {
            two_trees[1]->tree_prev = root;
        }
        else{
            for (i=0; i<p; i++) {
                for (j=0; j<max_num_child; j++) {
                    if (vrank == tree1[i][j]) {
                        two_trees[1]->tree_prev = (i+1+root)%size;
                    }
                }
            }
        }
    }
    else{
        two_trees[1]->tree_next[0] = (p-1+1+root)%size;
        two_trees[1]->tree_nextsize += 1;
        two_trees[1]->tree_prev = -1;
    }
    
    return two_trees;
}

ompi_coll_tree_t**
ompi_coll_base_topo_build_two_chains(struct ompi_communicator_t* comm,
                                     int root ){
    int i, rank, size, vrank;
    ompi_coll_tree_t** two_trees;
    //build the two tree array
    two_trees = (ompi_coll_tree_t **)malloc(2 * sizeof(ompi_coll_tree_t *));
    two_trees[0] = (ompi_coll_tree_t*)malloc(sizeof(ompi_coll_tree_t));
    two_trees[1] = (ompi_coll_tree_t*)malloc(sizeof(ompi_coll_tree_t));
    if (!two_trees || !two_trees[0] || !two_trees[1]) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                     "coll:base:topo:build_two_tree PANIC out of memory"));
        return NULL;
    }
    
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    vrank = (rank-root+size-1)%size;    //root has the highest vrank
    //printf("rank = %d, vrank = %d\n", rank, vrank);
    //int recovered_rank = (vrank+1+root)%size;      //only for test
    //printf("rank = %d, recovered_rank = %d\n", rank, recovered_rank);   //for test
    int p = size-1;       //number of node in the two tree, exclude the root
    
    //init the two trees
    two_trees[0]->tree_bmtree   = 1;
    two_trees[0]->tree_root     = root;
    two_trees[0]->tree_nextsize = 0;
    for( i = 0;i < MAXTREEFANOUT; i++ ) {
        two_trees[0]->tree_next[i] = -1;
    }
    
    two_trees[1]->tree_bmtree   = 1;
    two_trees[1]->tree_root     = root;
    two_trees[1]->tree_nextsize = 0;
    for( i = 0;i < MAXTREEFANOUT; i++ ) {
        two_trees[1]->tree_next[i] = -1;
    }
    if (p <= 1) {
        return two_trees;
    }
    
    
    //build the topology
    //build chain 0
    if (vrank != p) {
        if (vrank != p-1) {
            two_trees[0]->tree_next[0] = (vrank+1+1+root)%size;
            two_trees[0]->tree_nextsize += 1;
        }
        two_trees[0]->tree_prev = (vrank-1+1+root)%size;
    }
    else {
        two_trees[0]->tree_next[0] = (0+1+root)%size;
        two_trees[0]->tree_nextsize += 1;
        two_trees[0]->tree_prev = -1;
    }
    
    //build chain 1
    if (vrank != p) {
        if (vrank != 0) {
            two_trees[1]->tree_next[0] = (vrank-1+1+root)%size;
            two_trees[1]->tree_nextsize += 1;
        }
        two_trees[1]->tree_prev = (vrank+1+1+root)%size;
    }
    else {
        two_trees[1]->tree_next[0] = (p-1+1+root)%size;
        two_trees[1]->tree_nextsize += 1;
        two_trees[1]->tree_prev = -1;
    }
    return two_trees;
}

int ompi_coll_base_topo_dump_tree (ompi_coll_tree_t* tree, int rank)
{
    int i;
    
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "coll:base:topo:topo_dump_tree %1d tree root %d"
                 " fanout %d BM %1d nextsize %d prev %d",
                 rank, tree->tree_root, tree->tree_bmtree, tree->tree_fanout,
                 tree->tree_nextsize, tree->tree_prev));
    if( tree->tree_nextsize ) {
        for( i = 0; i < tree->tree_nextsize; i++ )
            OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"[%1d] %d", i, tree->tree_next[i]));
    }
    return (0);
}

//TODO: Need to change int to bits
int pow10_int(int pow_value) {
    int i, result = 1;
    for (i=0; i<pow_value; i++) {
        result *= 10;
    }
    return result;
}

int hostname_to_number(char* hostname, int size) {
    int i=0, j=0;
    char * number_array = (char *)malloc(sizeof(char)*size);
    while (hostname[i] != '\0'){
        if(hostname[i] >= '0' && hostname[i] <= '9'){
            number_array[j++] = hostname[i];
        }
        i++;
    }
    int number = 0;
    for (i=0; i<j; i++){
        number += (number_array[i]-'0') * pow10_int(j-1-i);
    }
    free(number_array);
    return number;
}

void get_topo(int *topo, struct ompi_communicator_t* comm, int nb_topo_level){
    int r_rank, i;
    int size = ompi_comm_size(comm);
    ompi_proc_t* proc;
    int * self_topo = (int *)malloc(sizeof(int) * nb_topo_level);
    int * same_numa = (int *)malloc(sizeof(int) * size);
    for (i=0; i<size; i++) {
        same_numa[i] = INT_MAX;
    }
    int same_numa_count = 0;
    //set daemon vpid
    //self_topo[0] = OMPI_RTE_MY_NODEID;
    char hostname[1024];
    //printf("[%d]: %s\n", ompi_comm_rank(comm), hostname);
    gethostname(hostname, 1024);
    self_topo[0] = hostname_to_number(hostname, 1024);
    //set numa id
    for (r_rank=0; r_rank < size; r_rank++) {
        proc = ompi_group_peer_lookup(comm->c_local_group, r_rank);
        if (OPAL_PROC_ON_LOCAL_NUMA(proc->super.proc_flags)) {
            same_numa[same_numa_count] = r_rank;
            same_numa_count++;
        }
    }
    int min = size;
    for (i=0; i<same_numa_count; i++) {
        if (same_numa[i] < min) {
            min = same_numa[i];
        }
    }
    self_topo[1] = min;

    //set core id
    self_topo[2] = ompi_comm_rank(comm);
    
    printf("[topo %d]: %d %d %d\n", ompi_comm_rank(comm), self_topo[0], self_topo[1], self_topo[2]);
    fflush(stdout);
    //do allgather
    comm->c_coll->coll_allgather(self_topo, nb_topo_level, MPI_INT, topo, nb_topo_level, MPI_INT, comm, comm->c_coll->coll_allgather_module);
    printf("[topo %d]: after allgather\n", ompi_comm_rank(comm));
    fflush(stdout);
    free(same_numa);
    free(self_topo);
    

//    
//    topo[0*nb_topo_level+0] = 0;
//    topo[0*nb_topo_level+1] = 0;
//    topo[0*nb_topo_level+2] = 0;
//    
//    topo[1*nb_topo_level+0] = 1;
//    topo[1*nb_topo_level+1] = 1;
//    topo[1*nb_topo_level+2] = 1;
//    
//    topo[2*nb_topo_level+0] = 0;
//    topo[2*nb_topo_level+1] = 2;
//    topo[2*nb_topo_level+2] = 2;
//    
//    topo[3*nb_topo_level+0] = 1;
//    topo[3*nb_topo_level+1] = 3;
//    topo[3*nb_topo_level+2] = 3;
//
//    topo[4*nb_topo_level+0] = 0;
//    topo[4*nb_topo_level+1] = 0;
//    topo[4*nb_topo_level+2] = 4;
//    
//    topo[5*nb_topo_level+0] = 1;
//    topo[5*nb_topo_level+1] = 1;
//    topo[5*nb_topo_level+2] = 5;
//    
//    topo[6*nb_topo_level+0] = 0;
//    topo[6*nb_topo_level+1] = 2;
//    topo[6*nb_topo_level+2] = 6;
//    
//    topo[7*nb_topo_level+0] = 1;
//    topo[7*nb_topo_level+1] = 3;
//    topo[7*nb_topo_level+2] = 7;
//
//    topo[8*nb_topo_level+0] = 0;
//    topo[8*nb_topo_level+1] = 0;
//    topo[8*nb_topo_level+2] = 8;
//    
//    topo[9*nb_topo_level+0] = 1;
//    topo[9*nb_topo_level+1] = 1;
//    topo[9*nb_topo_level+2] = 9;
//    
//    topo[10*nb_topo_level+0] = 0;
//    topo[10*nb_topo_level+1] = 2;
//    topo[10*nb_topo_level+2] = 10;
//    
//    topo[11*nb_topo_level+0] = 1;
//    topo[11*nb_topo_level+1] = 3;
//    topo[11*nb_topo_level+2] = 11;
//    
//    topo[12*nb_topo_level+0] = 0;
//    topo[12*nb_topo_level+1] = 0;
//    topo[12*nb_topo_level+2] = 12;
//    
//    topo[13*nb_topo_level+0] = 1;
//    topo[13*nb_topo_level+1] = 1;
//    topo[13*nb_topo_level+2] = 13;
//    
//    topo[14*nb_topo_level+0] = 0;
//    topo[14*nb_topo_level+1] = 2;
//    topo[14*nb_topo_level+2] = 14;
//    
//    topo[15*nb_topo_level+0] = 1;
//    topo[15*nb_topo_level+1] = 3;
//    topo[15*nb_topo_level+2] = 15;

}

/* check if r_rank is in the same gpu socket of my_rank */
static int rank_on_current_gpu_numa(int r_rank, int my_rank, int *topo, int size, int topo_level) {
    int i;
    int rank_level = topo_level - 1;
    int numa_level = 1;
    int my_rank_numa;
    int r_rank_numa;
    for (i = 0; i < size; i++) {
        if (topo[i*topo_level + rank_level] == my_rank) {
            my_rank_numa = topo[i*topo_level + numa_level];
        }
        if (topo[i*topo_level + rank_level] == r_rank) {
            r_rank_numa = topo[i*topo_level + numa_level];
        }
    }
    if (my_rank_numa == r_rank_numa) {
        return 1;
    } else {
        return 0;
    }
}


void get_topo_gpu(int *topo, struct ompi_communicator_t* comm, int nb_topo_level, ompi_coll_topo_gpu_t* gpu_topo){
    int r_rank, i, j, my_rank;
    int size = ompi_comm_size(comm);
    int * self_topo = (int *)malloc(sizeof(int) * nb_topo_level);
    int * same_numa = (int *)malloc(sizeof(int) * size);
    for (i=0; i<size; i++) {
        same_numa[i] = size;
    }
    
    int same_numa_count = 0;
    //set daemon vpid
    self_topo[0] = OMPI_RTE_MY_NODEID;

    my_rank = ompi_comm_rank(comm);
    
    //set gpu socket id
    self_topo[1] = self_topo[0] * 10 + gpu_topo->gpu_numa[gpu_topo->nb_gpus + gpu_topo->gpu_id];
    //set gpu id
    self_topo[2] = gpu_topo->gpu_id + self_topo[0] * 10;
  //  self_topo[2] = my_rank;
    //set rank
    self_topo[3] = my_rank;

    //do allgather
    comm->c_coll->coll_allgather(self_topo, nb_topo_level, MPI_INT, topo, nb_topo_level, MPI_INT, comm, comm->c_coll->coll_allgather_module);
    
    for (r_rank=0; r_rank < size; r_rank++) {
        if (rank_on_current_gpu_numa(r_rank, my_rank, topo, size, nb_topo_level)) {
            same_numa[same_numa_count] = r_rank;
            same_numa_count++;
        }
    }
    int min = size;
    for (i=0; i<same_numa_count; i++) {
        if (same_numa[i] < min) {
            min = same_numa[i];
        }
    }
    
    self_topo[1] = min;
    
//    printf("[topo %d]: %d %d %d\n", ompi_comm_rank(comm), self_topo[0], self_topo[1], self_topo[2], self_topo[3]);
    
    comm->c_coll->coll_allgather(self_topo, nb_topo_level, MPI_INT, topo, nb_topo_level, MPI_INT, comm, comm->c_coll->coll_allgather_module);
    
    int *topo_bak = (int *)malloc(sizeof(int)*size*nb_topo_level);
    memcpy(topo_bak, topo, sizeof(int)*size*nb_topo_level);
    int max, max_index;
    for (j = 0; j < size; j++) {
        max_index = 0;
        max = topo_bak[2];
        for (i = 0; i < size; i++) {
            if (topo_bak[i*nb_topo_level + 2] != -1 && topo_bak[i*nb_topo_level + 2] > max) {
                max = topo_bak[i*nb_topo_level + 2];
                max_index = i;
            }
        }
        topo[max_index * nb_topo_level + 2] = size-j-1;
        topo_bak[max_index * nb_topo_level + 2] = -1;
    }
    free(topo_bak);
    
    printf("[topo %d]: %d %d %d %d, real device %d\n", ompi_comm_rank(comm), topo[my_rank * nb_topo_level], topo[my_rank * nb_topo_level + 1], topo[my_rank * nb_topo_level + 2], topo[my_rank * nb_topo_level + 3], gpu_topo->gpu_id);
    
    free(same_numa);
    free(self_topo);
}

//convert rank to shift rank, shift rank to viritual rank
int to_vrank(int rank, int *ranks, int size){
    int i;
    for (i=0; i<size; i++) {
        if (ranks[i] == rank) {
            return i;
        }
    }
    return -1;
    
}
//convert shift rank to rank, viritual rank to shift rank
int to_rank(int vrank, int *ranks, int size){
    return ranks[vrank];
    
}

//In ranks array, find shift rank start to end and move them forward
void move_group_forward(int *ranks, int size, int start, int end){
    int length = end - start+1;
    int i, start_loc;
    for (i=0; i<size; i++) {
        if(ranks[i] == start) {
            start_loc = i;
            break;
        }
    }
    for (i=start_loc-1; i>=0; i--) {
        ranks[i+length] = ranks[i];
    }
    for (i=0; i<length; i++) {
        ranks[i] = start+i;
    }
}

void sort_topo(int *topo, int start, int end, int size, int *ranks_a, int level, int nb_topo_level){
    if (level > nb_topo_level-1 || start >= end) {
        return;
    }
    int i, j;
//    printf("Sort [start %d, end %d, level %d]\n", start, end, level);
//    printf("Before sort: ");
//    for (i=start; i<=end; i++) {
//        printf("%d ", topo[i*nb_topo_level+level]);
//    }
//    printf("\n");
    int min = INT_MAX;
    int min_loc = -1;
    for (i=start; i<=end; i++) {
        //find min
        for (j=i; j<=end; j++) {
            if (topo[j*nb_topo_level+level] < min) {
                min = topo[j*nb_topo_level+level];
                min_loc = j;
                //printf("i %d j %d min %d, min_loc %d\n", i, j, min, min_loc);
                
            }
        }
        //swap i and min_loc
        //printf("min %d, min_loc %d\n", min, min_loc);
        int temp;
        for (j=0; j<nb_topo_level; j++) {
            temp = topo[i*nb_topo_level+j];
            topo[i*nb_topo_level+j] = topo[min_loc*nb_topo_level+j];
            topo[min_loc*nb_topo_level+j] = temp;
        }
        temp = ranks_a[i];
        ranks_a[i] = ranks_a[min_loc];
        ranks_a[min_loc] = temp;
        min = INT_MAX;
        min_loc = -1;
    }
//    printf("After sort: ");
//    for (i=start; i<=end; i++) {
//        printf("%d ", topo[i*nb_topo_level+level]);
//    }
//    printf("\n");
    int last;
    int new_start;
    int new_end;
    for (i=start; i<=end; i++) {
        if (i == start) {
            last = topo[i*nb_topo_level+level];
            new_start = start;
        }
        else if (i == end) {
            new_end = end;
            sort_topo(topo, new_start, new_end, size, ranks_a, level+1, nb_topo_level);
        }
        else if (last != topo[i*nb_topo_level+level]) {
            new_end = i-1;
            sort_topo(topo, new_start, new_end, size, ranks_a, level+1, nb_topo_level);
            new_start = i;
            last = topo[i*nb_topo_level+level];
        }
    }
}

//get the starting point of each gourp on every level
void set_helper(ompi_coll_topo_helper_t *helper, int32_t *rank_topo_array, int *ranks_a, int *ranks_s, int *topo, int root, int size, int nb_topo_level){
    int i, j;
//
//        for (i=0; i<size; i++) {
//            printf("[%d]: ", i);
//            int j;
//            for (j=0; j<nb_topo_level; j++) {
//                printf("%d ", topo[i*nb_topo_level+j]);
//            }
//            printf("\n");
//        }
//    printf("ranks_a: ");
//    for (i=0; i<size; i++) {
//        printf("%d ", ranks_a[i]);
//    }
//    printf("\n");
    
    for (i = 0; i < size; i ++) {
        rank_topo_array[i] = nb_topo_level-1;
    }

    int count = 0;
    int rank;
    int *temp = (int *)malloc(sizeof(int)*size);
    for (i=0; i<nb_topo_level; i++) {
        count = 0;
        int this_group = -1;
        for (j=0; j<size; j++) {    //j is shifted rank
            if (this_group != topo[j*nb_topo_level+i]) {
                //printf("this_group %d, count %d, j %d\n", this_group, count, j);
                this_group = topo[j*nb_topo_level+i];
                temp[count] = j;
                count++;
            }
        }
        helper[i].num_group = count;
        if (count != 0) {
            helper[i].start_loc = (int *)malloc(sizeof(int)*count);
        }
        for (j=0; j<count; j++) {
            helper[i].start_loc[j] = temp[j];
            rank = to_rank(helper[i].start_loc[j], ranks_s, size);
            if (rank_topo_array[rank] > i) {
                rank_topo_array[rank] = i;
            }
        }
        //if there are more than one group in this level
        if (count > 1) {
            for (j=0; j<count; j++) {
                if (to_vrank(root, ranks_a, size) >= temp[j]) {
                    int end;
                    if (j == count-1) {
                        end = size-1;
                    }
                    else{
                        end = temp[j+1]-1;
                    }
                    //find the group with root in this level
                    if (to_vrank(root, ranks_a, size) <= end) {
                        //move that group forward
                        //printf("Move group forward: start %d, end %d\n", temp[j], end);
                        move_group_forward(ranks_s, size, temp[j], end);
                    }
                }
            }
        }
    }
    
//    print_helper(helper);
//    printf("ranks_s: ");
//    for (i=0; i<size; i++) {
//        printf("%d ", ranks_s[i]);
//    }
//    printf("\n");

    //set helper with vranks
    for (i=0; i<nb_topo_level; i++) {
        count = 0;
        int this_group = -1;
        for (j=0; j<size; j++) {    //j is virtal rank
            if (this_group != topo[to_rank(j, ranks_s, size)*nb_topo_level+i]) {
                this_group = topo[to_rank(j, ranks_s, size)*nb_topo_level+i];
                temp[count] = j;
                count++;
            }
        }
        helper[i].num_group = count;
        for (j=0; j<count; j++) {
            helper[i].start_loc[j] = temp[j];
        }
    }
    
    free(temp);
}

void free_helper(ompi_coll_topo_helper_t *helper, int nb_topo_level){
    int i;
    for (i=0; i<nb_topo_level; i++) {
        if (helper[i].num_group != 0) {
            free(helper[i].start_loc);
        }
    }
    free(helper);
}

void print_helper(ompi_coll_topo_helper_t *helper, int nb_topo_level){
    int i, j;
    printf("print helper, topo level %d\n", nb_topo_level);
    for (i=0; i<nb_topo_level; i++) {
        printf("[Topo Level %d]: ", i);
        for (j=0; j<helper[i].num_group; j++) {
            printf("%d ", helper[i].start_loc[j]);
        }
        printf("\n");
    }
}


ompi_coll_tree_t*
ompi_coll_base_topo_build_topoaware_linear(struct ompi_communicator_t* comm, int root, mca_coll_base_module_t *module, int nb_topo_level, int device_type, void *topo_info){
    
    int i, j;
    ompi_coll_tree_t *tree = (ompi_coll_tree_t*)malloc(sizeof(ompi_coll_tree_t));
    if (!tree) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo_build_tree PANIC::out of memory"));
        return NULL;
    }
    
    //Set root
    tree->tree_root = root;
    
    //Initialize tree
    tree->tree_fanout   = 0;
    tree->tree_bmtree   = 0;
    tree->tree_root     = root;
    tree->tree_prev     = -1;
    tree->tree_nextsize = 0;
    for( i = 0; i < MAXTREEFANOUT; i++ ) {
        tree->tree_next[i] = -1;
    }
    
    int size, rank;
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    
    int *ranks_a;
    int *ranks_s = (int *)malloc(sizeof(int)*size);   //ranks[0] store which shift rank has virtual rank 0
    
    for (i=0; i<size; i++) {
        ranks_s[i] = i;
    }
    
    int *topo;
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_topo) && (coll_comm->cached_old_comm == comm) ) ) {
        if( coll_comm->cached_topo ) {
            free(coll_comm->cached_topo);
            free(coll_comm->cached_ranks_a);
        }
        topo = (int *)malloc(sizeof(int)*size*nb_topo_level);
        if (device_type == 1) {
            get_topo_gpu(topo, comm, nb_topo_level, (ompi_coll_topo_gpu_t *)topo_info);
        } else {
            get_topo(topo, comm, nb_topo_level);
        }
        ranks_a = (int *)malloc(sizeof(int)*size);   //ranks[0] store which actual rank has shift rank 0
        for (i=0; i<size; i++) {
            ranks_a[i] = i;
        }
        //sort the topo such that each group is contiguous
        sort_topo(topo, 0, size-1, size, ranks_a, 0, nb_topo_level);
        coll_comm->cached_topo = topo;
        coll_comm->cached_old_comm = comm;
        coll_comm->cached_ranks_a = ranks_a;
    }
    else{
        topo = coll_comm->cached_topo;
        ranks_a = coll_comm->cached_ranks_a;
    }
    
    ompi_coll_topo_helper_t *helper = (ompi_coll_topo_helper_t *) malloc(sizeof(ompi_coll_topo_helper_t)*nb_topo_level);
    int32_t *rank_topo_array = (int32_t *)malloc(sizeof(int32_t) * size);
    set_helper(helper, rank_topo_array, ranks_a, ranks_s, topo, root, size, 3);
    //print_helper(helper);
    
    int vrank = to_vrank(to_vrank(rank, ranks_a, size), ranks_s, size);
    
    int head = 0;
    int tail = size-1;
    int new_head = -1;
    int new_tail = -1;
    int rank_loc = -1;
    
    for (i=0; i<nb_topo_level; i++) {
        //count how many groups on this level between head and tail
        int count = 0;
        int exist = 0;  //to judge if rank is one of the group heads
        int end = 0;
        int *temp_start_loc = (int *)malloc(sizeof(int)*helper[i].num_group);
        //printf("head = %d, tail = %d, num_group %d\n", head, tail, helper[i].num_group);
        for (j=0; j<helper[i].num_group; j++) {
            if (helper[i].start_loc[j] >= head) {
                if (helper[i].start_loc[j] <= tail) {
                    temp_start_loc[count] = helper[i].start_loc[j];
                    end = tail;
                    if (j != helper[i].num_group-1) {
                        end = helper[i].start_loc[j+1]-1;
                    }
                    //printf("i %d, j %d, end %d\n", i, j ,end);
                    if (vrank >= helper[i].start_loc[j] &&  vrank <= end) {
                        //printf("vrank = %d, j = %d, start_loc[j] = %d\n", vrank, j, helper[i].start_loc[j]);
                        if (vrank == helper[i].start_loc[j]) {
                            //printf("set exist\n");
                            exist = 1;
                            rank_loc = count;
                            new_head = vrank;
                        }
                        else {
                            new_head = helper[i].start_loc[j];
                        }
                        new_tail = tail;
                        if (j != helper[i].num_group-1) {
                            new_tail = helper[i].start_loc[j+1]-1;
                        }
                    }
                    count++;
                }
                else {
                    break;
                }
            }
        }
        
        head = new_head;
        tail = new_tail;
        //printf("exist %d, count = %d\n", exist, count);
        //if rank is one of the group heads
        if (exist) {
            build_topoaware_linear(count, temp_start_loc, rank, rank_loc, size, tree, ranks_a, ranks_s);
        }
        
        free(temp_start_loc);
    }
    
    free_helper(helper, 3);
    free(ranks_s);
    free(rank_topo_array);
    
    return tree;
}

ompi_coll_tree_t*
ompi_coll_base_topo_build_topoaware_chain(struct ompi_communicator_t* comm, int root, mca_coll_base_module_t *module, int nb_topo_level, int device_type, void *topo_info ){
    int i, j;
    ompi_coll_tree_t *tree = (ompi_coll_tree_t*)malloc(sizeof(ompi_coll_tree_t));
    if (!tree) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo_build_tree PANIC::out of memory"));
        return NULL;
    }
    
    //Set root
    tree->tree_root = root;
    
    //Initialize tree
    tree->tree_fanout   = 0;
    tree->tree_bmtree   = 0;
    tree->tree_root     = root;
    tree->tree_prev     = -1;
    tree->tree_nextsize = 0;
    for( i = 0; i < MAXTREEFANOUT; i++ ) {
        tree->tree_next[i] = -1;
    }
    
    int size, rank;
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    
    int *ranks_a;
    int *ranks_s = (int *)malloc(sizeof(int)*size);   //ranks[0] store which shift rank has virtual rank 0
    
    for (i=0; i<size; i++) {
        ranks_s[i] = i;
    }

    int *topo;
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_topo) && (coll_comm->cached_old_comm == comm) ) ) {
        if( coll_comm->cached_topo ) {
            free(coll_comm->cached_topo);
            free(coll_comm->cached_ranks_a);
        }
        topo = (int *)malloc(sizeof(int)*size*nb_topo_level);
        if (device_type == 1) {
            get_topo_gpu(topo, comm, nb_topo_level, (ompi_coll_topo_gpu_t *)topo_info);
        } else {
            get_topo(topo, comm, nb_topo_level);
        }
        ranks_a = (int *)malloc(sizeof(int)*size);   //ranks[0] store which actual rank has shift rank 0
        for (i=0; i<size; i++) {
            ranks_a[i] = i;
        }
        //sort the topo such that each group is contiguous
        sort_topo(topo, 0, size-1, size, ranks_a, 0, nb_topo_level);
        coll_comm->cached_topo = topo;
        coll_comm->cached_old_comm = comm;
        coll_comm->cached_ranks_a = ranks_a;
    }
    else{
        topo = coll_comm->cached_topo;
        ranks_a = coll_comm->cached_ranks_a;
    }
    
    ompi_coll_topo_helper_t *helper = (ompi_coll_topo_helper_t *) malloc(sizeof(ompi_coll_topo_helper_t)*nb_topo_level);
    int32_t *rank_topo_array = (int32_t *)malloc(sizeof(int32_t) * size);
    set_helper(helper, rank_topo_array, ranks_a, ranks_s, topo, root, size, nb_topo_level);
    //print_helper(helper);
    
    int vrank = to_vrank(to_vrank(rank, ranks_a, size), ranks_s, size);
    
    int head = 0;
    int tail = size-1;
    int new_head = -1;
    int new_tail = -1;
    int rank_loc = -1;
    
    tree->topo_flags = rank_topo_array[vrank];
    
    for (i=0; i<nb_topo_level; i++) {
        //count how many groups on this level between head and tail
        int count = 0;
        int exist = 0;  //to judge if rank is one of the group heads
        int end = 0;
        int *temp_start_loc = (int *)malloc(sizeof(int)*helper[i].num_group);
        //printf("head = %d, tail = %d, num_group %d\n", head, tail, helper[i].num_group);
        if (helper[i].num_group <= 0) {
            continue;
        }
        for (j=0; j<helper[i].num_group; j++) {
            if (helper[i].start_loc[j] >= head) {
                if (helper[i].start_loc[j] <= tail) {
                    temp_start_loc[count] = helper[i].start_loc[j];
                    end = tail;
                    if (j != helper[i].num_group-1) {
                        end = helper[i].start_loc[j+1]-1;
                    }
                    //printf("i %d, j %d, end %d\n", i, j ,end);
                    if (vrank >= helper[i].start_loc[j] &&  vrank <= end) {
                        //printf("vrank = %d, j = %d, start_loc[j] = %d\n", vrank, j, helper[i].start_loc[j]);
                        if (vrank == helper[i].start_loc[j]) {
                            //printf("set exist\n");
                            exist = 1;
                            rank_loc = count;
                            new_head = vrank;
                        }
                        else {
                            new_head = helper[i].start_loc[j];
                        }
                        new_tail = tail;
                        if (j != helper[i].num_group-1) {
                            new_tail = helper[i].start_loc[j+1]-1;
                        }
                    }
                    count++;
                }
                else {
                    break;
                }
            }
        }
        
        head = new_head;
        tail = new_tail;
        //printf("exist %d, count = %d\n", exist, count);
        //if rank is one of the group heads
        if (exist) {
         //   printf("level %d, rank %d\n", i, rank);
            build_topoaware_chain(count, temp_start_loc, rank, rank_loc, size, tree, ranks_a, ranks_s, rank_topo_array);
        }
        
        free(temp_start_loc);
    }
    
    free_helper(helper, nb_topo_level);
    free(ranks_s);
    free(rank_topo_array);
    
    return tree;
}




void build_topoaware_chain(int count, int *start_loc, int rank, int rank_loc, int size, ompi_coll_tree_t *tree, int *ranks_a, int *ranks_s, int32_t *rank_topo_array){
    //printf("build_topo_chain, count = %d, rank = %d, rank_loc=%d\n", count, rank, rank_loc);
    if (count == 1) {
        return;
    }
    else {
        int vrank;
        if (rank_loc == 0) {
            tree->tree_next[tree->tree_nextsize] = to_rank(to_rank(start_loc[rank_loc+1], ranks_s, size), ranks_a, size);
            vrank = to_vrank(to_vrank(tree->tree_next[tree->tree_nextsize], ranks_a, size), ranks_s, size);
            tree->tree_next_topo_flags[tree->tree_nextsize] = rank_topo_array[vrank];
            tree->tree_nextsize+=1;
            //printf("[rank %d]: set next %d\n", rank, tree->tree_next[tree->tree_nextsize-1]);
        }
        else if (rank_loc == count - 1) {
            tree->tree_prev = to_rank(to_rank(start_loc[rank_loc-1], ranks_s, size), ranks_a, size);
            vrank = to_vrank(to_vrank(tree->tree_prev, ranks_a, size), ranks_s, size);
            tree->tree_prev_topo_flags = rank_topo_array[vrank];
            //printf("[rank %d]: set prev %d\n", rank, tree->tree_prev);
        }
        else {
            tree->tree_next[tree->tree_nextsize] = to_rank(to_rank(start_loc[rank_loc+1], ranks_s, size), ranks_a, size);
            vrank = to_vrank(to_vrank(tree->tree_next[tree->tree_nextsize], ranks_a, size), ranks_s, size);
            tree->tree_next_topo_flags[tree->tree_nextsize] = rank_topo_array[vrank];
            tree->tree_nextsize+=1;
            //printf("[rank %d]: set next %d\n", rank, tree->tree_next[tree->tree_nextsize-1]);
            tree->tree_prev = to_rank(to_rank(start_loc[rank_loc-1], ranks_s, size), ranks_a, size);
            vrank = to_vrank(to_vrank(tree->tree_prev, ranks_a, size), ranks_s, size);
            tree->tree_prev_topo_flags = rank_topo_array[vrank];
            //printf("[rank %d]: set prev %d\n", rank, tree->tree_prev);
        }
    }
}

void build_topoaware_linear(int count, int *start_loc, int rank, int rank_loc, int size, ompi_coll_tree_t *tree, int *ranks_a, int *ranks_s){
    //printf("build_topo_linear, count = %d, rank = %d, rank_loc=%d\n", count, rank, rank_loc);
    if (count == 1) {
        return;
    }
    else {
        if (rank_loc == 0) {
            int i;
            for (i=1; i<count; i++) {
                tree->tree_next[tree->tree_nextsize] = to_rank(to_rank(start_loc[i], ranks_s, size), ranks_a, size);
                tree->tree_nextsize+=1;
                //printf("[rank %d]: set next %d\n", rank, tree->tree_next[tree->tree_nextsize-1]);
            }
            
        }
        else {
            tree->tree_prev = to_rank(to_rank(start_loc[0], ranks_s, size), ranks_a, size);
            //printf("[rank %d]: set prev %d\n", rank, tree->tree_prev);
        }
    }
}

ompi_coll_tree_t*
ompi_coll_base_topo_build_topoaware_ring(struct ompi_communicator_t* comm,
                                         mca_coll_base_module_t *module , int nb_topo_level){
    int i;
    ompi_coll_tree_t *tree = (ompi_coll_tree_t*)malloc(sizeof(ompi_coll_tree_t));
    if (!tree) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:topo_build_tree PANIC::out of memory"));
        return NULL;
    }
    
    //Initialize tree
    tree->tree_fanout   = 0;
    tree->tree_bmtree   = 0;
    tree->tree_root     = -1;       //use to store vrank here
    tree->tree_prev     = -1;
    tree->tree_nextsize = 0;
    for( i = 0; i < MAXTREEFANOUT; i++ ) {
        tree->tree_next[i] = -1;
    }
    
    int size, rank;
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    
    int *topo;
    int *ranks_a;
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_topo) && (coll_comm->cached_old_comm == comm) ) ) {
        if( coll_comm->cached_topo ) {
            free(coll_comm->cached_topo);
            free(coll_comm->cached_ranks_a);
        }
        topo = (int *)malloc(sizeof(int)*size*nb_topo_level);
        get_topo(topo, comm, nb_topo_level);
        coll_comm->cached_topo = topo;
        coll_comm->cached_old_comm = comm;
        ranks_a = (int *)malloc(sizeof(int)*size);   //ranks[0] store which actual rank has shift rank 0
        for (i=0; i<size; i++) {
            ranks_a[i] = i;
        }
        sort_topo(topo, 0, size-1, size, ranks_a, 0, nb_topo_level);
    }
    else{
        topo = coll_comm->cached_topo;
        ranks_a = coll_comm->cached_ranks_a;
    }

    int vrank = to_vrank(rank, ranks_a, size);
    tree->tree_root = vrank;
    tree->tree_prev = to_rank((vrank-1+size)%size, ranks_a, size);
    tree->tree_nextsize = 1;
    tree->tree_next[0] = to_rank((vrank+1)%size, ranks_a, size);
    
    return tree;
}

