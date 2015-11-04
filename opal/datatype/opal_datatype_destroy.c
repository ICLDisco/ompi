/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2009 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"
#include "opal/constants.h"
#include "opal/datatype/opal_datatype.h"
#include "opal/datatype/opal_datatype_internal.h"
#if OPAL_CUDA_SUPPORT
#include "opal/datatype/opal_convertor.h"
#include "opal/datatype/opal_datatype_cuda.h"
#endif /* OPAL_CUDA_SUPPORT */   

int32_t opal_datatype_destroy( opal_datatype_t** dt )
{
    opal_datatype_t* pData = *dt;
    
#if OPAL_CUDA_SUPPORT   
    /* free cuda iov */
    if (opal_datatype_cuda_kernel_support== 1 && pData->cuda_iov_dist != NULL && pData->cuda_iov_dist != (void*)0xDEADBEEF) {
        opal_cuda_iov_dist_fini(pData->cuda_iov_dist);
        pData->cuda_iov_dist = NULL;
        pData->cuda_iov_count = 0;
    }
#endif /* OPAL_CUDA_SUPPORT */

    if( (pData->flags & OPAL_DATATYPE_FLAG_PREDEFINED) &&
        (pData->super.obj_reference_count <= 1) )
        return OPAL_ERROR;

    OBJ_RELEASE( pData );
    *dt = NULL;
    return OPAL_SUCCESS;
}
