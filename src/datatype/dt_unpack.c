/* -*- Mode: C; c-basic-offset:4 ; -*- */

#include "lam_config.h"

#include "datatype.h"
#include "datatype_internal.h"

#ifdef HAVE_ALLOCA_H
#include <alloca.h>
#endif
#include <stdlib.h>

static int convertor_unpack_homogeneous( lam_convertor_t* pConv, struct iovec* iov, unsigned int out_size );
static int convertor_unpack_general( lam_convertor_t* pConvertor,
                                     struct iovec* pInputv, unsigned int inputCount );

void dump_stack( dt_stack_t* pStack, int stack_pos, dt_elem_desc_t* pDesc, char* name )
{
   printf( "\nStack %p stack_pos %d name %s\n", (void*)pStack, stack_pos, name );
   for( ;stack_pos >= 0; stack_pos-- ) {
      printf( "%d: pos %d count %d disp %ld end_loop %d ", stack_pos, pStack[stack_pos].index,
              pStack[stack_pos].count, pStack[stack_pos].disp, pStack[stack_pos].end_loop );
      if( pStack[stack_pos].index != -1 )
         printf( "[desc count %d disp %ld extent %d]\n",
                 pDesc[pStack[stack_pos].index].count,
                 pDesc[pStack[stack_pos].index].disp,
                 pDesc[pStack[stack_pos].index].extent );
      else
         printf( "\n" );
   }
   printf( "\n" );
}

/*
 *  Remember that the first item in the stack (ie. position 0) is the number
 * of times the datatype is involved in the operation (ie. the count argument
 * in the MPI_ call).
 */
/* Convert data from multiple input buffers (as received from the network layer)
 * to a contiguous output buffer with a predefined size.
 * Return 0 if everything went OK and if there is still room before the complete
 *          conversion of the data (need additional call with others input buffers )
 *        1 if everything went fine and the data was completly converted
 *       -1 something wrong occurs.
 */
static int convertor_unpack_general( lam_convertor_t* pConvertor,
                                     struct iovec* pInputv,
                                     unsigned int inputCount )
{
   dt_stack_t* pStack;   /* pointer to the position on the stack */
   int pos_desc;         /* actual position in the description of the derived datatype */
   int count_desc;       /* the number of items already done in the actual pos_desc */
   int end_loop;         /* last element in the actual loop */
   int type;             /* type at current position */
   unsigned int advance; /* number of bytes that we should advance the buffer */
   int rc;
   long disp_desc = 0;   /* compute displacement for truncated data */
   long disp;            /* displacement at the beging of the last loop */
   dt_desc_t *pData = pConvertor->pDesc;
   dt_elem_desc_t* pElems;
   char* pOutput = pConvertor->pBaseBuf;
   int oCount = (pData->ub - pData->lb) * pConvertor->count;
   char* pInput = pInputv[0].iov_base;
   int iCount = pInputv[0].iov_len;

   if( pData->opt_desc.desc != NULL ) pElems = pData->opt_desc.desc;
   else                               pElems = pData->desc.desc;

   DUMP( "convertor_decode( %p, {%p, %d}, %d )\n", pConvertor,
         pInputv[0].iov_base, pInputv[0].iov_len, inputCount );
   pStack = pConvertor->pStack + pConvertor->stack_pos;
   pos_desc  = pStack->index;
   disp = 0;
   if( pos_desc == -1 ) {
      pos_desc = 0;
      count_desc = pElems[0].count;
      disp_desc = pElems[0].disp;
   } else {
      count_desc = pStack->count;
      if( pElems[pos_desc].type != DT_LOOP ) {
         pConvertor->stack_pos--;
         pStack--;
         disp = pStack->disp;
         disp_desc = ( pElems[pos_desc].disp +
                       (pElems[pos_desc].count - count_desc) * pElems[pos_desc].extent);
      }
   }
   DUMP_STACK( pConvertor->pStack, pConvertor->stack_pos, pElems, "starting" );
   DUMP( "remember position on stack %d last_elem at %d\n", pConvertor->stack_pos, pos_desc );
   DUMP( "top stack info {index = %d, count = %d}\n", 
         pStack->index, pStack->count );

  next_loop:
   end_loop = pStack->end_loop;
   while( pConvertor->stack_pos >= 0 ) {
      if( pos_desc == end_loop ) { /* end of the current loop */
         while( --(pStack->count) == 0 ) { /* end of loop */
            pConvertor->stack_pos--;
            pStack--;
            if( pConvertor->stack_pos == -1 )
               return 1;  /* completed */
         }
         pos_desc = pStack->index;
         if( pos_desc == -1 )
            pStack->disp += (pData->ub - pData->lb);
         else
            pStack->disp += pElems[pos_desc].extent;
         pos_desc++;
         disp = pStack->disp;
         count_desc = pElems[pos_desc].count;
         disp_desc = pElems[pos_desc].disp;
         goto next_loop;
      }
      if( pElems[pos_desc].type == DT_LOOP ) {
         do {
            PUSH_STACK( pStack, pConvertor->stack_pos,
                        pos_desc, pElems[pos_desc].count,
                        disp, pos_desc + pElems[pos_desc].disp + 1 );
            pos_desc++;
         } while( pElems[pos_desc].type == DT_LOOP ); /* let's start another loop */
         DUMP_STACK( pConvertor->pStack, pConvertor->stack_pos, pElems, "advance loops" );
         /* update the current state */
         count_desc = pElems[pos_desc].count;
         disp_desc = pElems[pos_desc].disp;
         goto next_loop;
      }
      /* now here we have a basic datatype */
      type = pElems[pos_desc].type;
      rc = pConvertor->pFunctions[type]( count_desc,
                                         pInput, iCount, pElems[pos_desc].extent,
                                         pOutput + disp + disp_desc, oCount, pElems[pos_desc].extent,
                                         &advance );
      if( rc <= 0 ) {
         printf( "trash in the input buffer\n" );
         return -1;
      }
      iCount -= advance;      /* decrease the available space in the buffer */
      pInput += advance;      /* increase the pointer to the buffer */
      pConvertor->bConverted += advance;
      if( rc != count_desc ) {
         /* not all data has been converted. Keep the state */
         PUSH_STACK( pStack, pConvertor->stack_pos,
                     pos_desc, count_desc - rc,
                     disp + rc * pElems[pos_desc].extent, pos_desc );
         if( iCount != 0 )
            printf( "there is still room in the input buffer %d bytes\n", iCount );
         return 0;
      }
      pConvertor->converted += rc;  /* number of elementd converted so far */
      pos_desc++;  /* advance to the next data */
      count_desc = pElems[pos_desc].count;
      disp_desc = pElems[pos_desc].disp;
      if( iCount == 0 ) break;  /* break if there is no more data in the buffer */
   }

   /* out of the loop: we have complete the data conversion or no more space
    * in the buffer.
    */
   if( pConvertor->pStack[0].count < 0 ) return 1;  /* data succesfully converted */

   /* I complete an element, next step I should go to the next one */
   PUSH_STACK( pStack, pConvertor->stack_pos, pos_desc,
               pElems[pos_desc].count, disp, pos_desc );

   return 0;
}

static int convertor_unpack_homogeneous( lam_convertor_t* pConv, struct iovec* iov, unsigned int out_size )
{
   dt_stack_t* pStack;   /* pointer to the position on the stack */
   int pos_desc;         /* actual position in the description of the derived datatype */
   int type;             /* type at current position */
   int i;                /* counter for basic datatype with extent */
   int stack_pos = 0;    /* position on the stack */
   long lastDisp = 0, lastLength = 0;
   char* pSrcBuf;
   dt_desc_t* pData = pConv->pDesc;
   dt_elem_desc_t* pElems;

   pSrcBuf = iov[0].iov_base;

   if( pData->flags & DT_FLAG_CONTIGUOUS ) {
      long extent = pData->ub - pData->lb;
      char* pDstBuf = pConv->pBaseBuf + pData->true_lb + pConv->bConverted;

      if( pData->size == extent ) {
         long length = pConv->count * pData->size;

	 if( length > iov[0].iov_len )
             length = iov[0].iov_len;
         /* contiguous data or basic datatype with count */
         MEMCPY( pDstBuf, pSrcBuf, length );
         pConv->bConverted += length;
      } else {
         type = iov[0].iov_len;
         for( pos_desc = 0; pos_desc < pConv->count; pos_desc++ ) {
            MEMCPY( pDstBuf, pSrcBuf, pData->size );
            pSrcBuf += pData->size;
            pDstBuf += extent;
            type -= pData->size;
         }
         pConv->bConverted += type;
      }
      return (pConv->bConverted == (pData->size * pConv->count));
   }

   pStack = pConv->pStack;
   pStack->count = pConv->count;
   pStack->index = -1;
   pStack->disp = 0;
   pos_desc  = 0;

   if( pData->opt_desc.desc != NULL ) {
      pElems = pData->opt_desc.desc;
      pStack->end_loop = pData->opt_desc.used;
   } else {
      pElems = pData->desc.desc;
      pStack->end_loop = pData->desc.used;
   }

   DUMP_STACK( pStack, stack_pos, pElems, "starting" );
   DUMP( "remember position on stack %d last_elem at %d\n", stack_pos, pos_desc );
   DUMP( "top stack info {index = %d, count = %d}\n", 
         pStack->index, pStack->count );
  next_loop:
   while( pos_desc <= pStack->end_loop ) {
      if( pos_desc == pStack->end_loop ) { /* end of the current loop */
         if( --(pStack->count) == 0 ) { /* end of loop */
            pStack--;
            if( --stack_pos == -1 ) break;
         } else {
            pos_desc = pStack->index;
            if( pos_desc == -1 )
               pStack->disp += (pData->ub - pData->lb);
            else
               pStack->disp += pElems[pos_desc].extent;
         }
         pos_desc++;
         goto next_loop;
      }
      if( pElems[pos_desc].type == DT_LOOP ) {
         if( pElems[pos_desc].flags & DT_FLAG_CONTIGUOUS ) {
            dt_elem_desc_t* pLast = &( pElems[pos_desc + pElems[pos_desc].disp]);
            if( lastLength == 0 ) {
               MEMCPY( pConv->pBaseBuf + lastDisp, pSrcBuf, lastLength );
               pSrcBuf += lastLength;
            }
            lastLength = pLast->extent;
            for( i = 0; i < (pElems[pos_desc].count - 1); i++ ) {
               MEMCPY( pConv->pBaseBuf + lastDisp, pSrcBuf, lastLength );
               pSrcBuf += pLast->extent;
               lastDisp += pElems[pos_desc].extent;
            }
            pos_desc += pElems[pos_desc].disp + 1;
            goto next_loop;
         } else {
            do {
               PUSH_STACK( pStack, stack_pos, pos_desc, pElems[pos_desc].count,
                           pStack->disp, pos_desc + pElems[pos_desc].disp );
               pos_desc++;
            } while( pElems[pos_desc].type == DT_LOOP ); /* let's start another loop */
         }
      }
      /* now here we have a basic datatype */
      type = pElems[pos_desc].type;
      if( (lastDisp + lastLength) == (pStack->disp + pElems[pos_desc].disp) ) {
         lastLength += pElems[pos_desc].count * basicDatatypes[type].size;
      } else {
         MEMCPY( pConv->pBaseBuf + lastDisp, pSrcBuf, lastLength );
         pSrcBuf += lastLength;
		 printf( "increase by %ld bytes\n", lastLength );
         pConv->bConverted += lastLength;
         lastDisp = pStack->disp + pElems[pos_desc].disp;
         lastLength = pElems[pos_desc].count * basicDatatypes[type].size;
      }
      pos_desc++;  /* advance to the next data */
   }

   MEMCPY( pConv->pBaseBuf + lastDisp, pSrcBuf, lastLength );
   pConv->bConverted += lastLength;

   /* cleanup the stack */
   return 0;
}

int lam_convertor_unpack( lam_convertor_t* pConvertor,
                          struct iovec* pInputv,
                          unsigned int inputCount )
{
   dt_desc_t *pData = pConvertor->pDesc;
   char* pOutput = pConvertor->pBaseBuf;
   char* pInput = pInputv[0].iov_base;
   int rc;

   if( pConvertor->count == 0 ) return 1;  /* nothing to do */

   if( pConvertor->flags & DT_FLAG_CONTIGUOUS ) {
      if( pInputv[0].iov_base == NULL ) {
         rc = pConvertor->count * pData->size;
         if( pInputv[0].iov_len == 0 ) {  /* give me the whole buffer */
            pInputv[0].iov_base = pConvertor->pBaseBuf + pData->true_lb;
            pInputv[0].iov_len = rc;
            return 1;
         } else {  /* what about the next chunk ? */
            pInputv[0].iov_base = pConvertor->pBaseBuf + pData->true_lb + pConvertor->bConverted;
            if( pInputv[0].iov_len > (rc - pConvertor->bConverted) )
               pInputv[0].iov_len = rc - pConvertor->bConverted;
            pConvertor->bConverted += pInputv[0].iov_len;
            return (pConvertor->bConverted == rc);
         }
      }
   }
   if( (pInput >= pOutput) && (pInput < (pOutput + pConvertor->count * (pData->ub - pData->lb))) ) {
      return 1;
   }
   return lam_convertor_progress( pConvertor, pInputv, inputCount );
}

/* Return value:
 *     0 : nothing has been done
 * positive value: number of item converted.
 * negative value: -1 * number of items converted, less data provided than expected
 *                and there are less data than the size on the remote host of the
 *                basic datatype.
 */
#define COPY_TYPE( TYPENAME, TYPE ) \
static int copy_##TYPENAME( unsigned int count, \
                            char* from, unsigned int from_len, long from_extent, \
                            char* to, unsigned int to_len, long to_extent, \
                            int* used )                                 \
{ \
   int i, res = 1; \
   unsigned int remote_TYPE_size = sizeof(TYPE); /* TODO */ \
\
   if( (remote_TYPE_size * count) > from_len ) { \
      count = from_len / remote_TYPE_size; \
      if( (count * remote_TYPE_size) != from_len ) { \
         DUMP( "oops should I keep this data somewhere (excedent %d bytes)?\n", \
               from_len - (count * remote_TYPE_size) ); \
         res = -1; \
      } \
      DUMP( "correct: copy %s count %d from buffer %p with length %d to %p space %d\n", \
            #TYPE, count, from, from_len, to, to_len ); \
   } else \
      DUMP( "         copy %s count %d from buffer %p with length %d to %p space %d\n", \
            #TYPE, count, from, from_len, to, to_len ); \
\
   if( (from_extent == sizeof(TYPE)) && (to_extent == sizeof(TYPE)) ) { \
      MEMCPY( to, from, count * sizeof(TYPE) ); \
   } else { \
      for( i = 0; i < count; i++ ) { \
         MEMCPY( to, from, sizeof(TYPE) ); \
         to += to_extent; \
         from += from_extent; \
      } \
   } \
   *used = count * sizeof(TYPE) ; \
   return res * count; \
}

COPY_TYPE( char, char )
COPY_TYPE( short, short )
COPY_TYPE( int, int )
COPY_TYPE( float, float )
COPY_TYPE( long, long )
/*COPY_TYPE( double, double );*/
COPY_TYPE( long_long, long long )
COPY_TYPE( long_double, long double )
COPY_TYPE( complex_float, complex_float_t )
COPY_TYPE( complex_double, complex_double_t )

static int copy_double( unsigned int count,
                        char* from, unsigned int from_len, long from_extent,
                        char* to, unsigned int to_len, long to_extent,
                        int* used )
{
   int i, res = 1;
   unsigned int remote_double_size = sizeof(double); /* TODO */

   if( (remote_double_size * count) > from_len ) {
      count = from_len / remote_double_size;
      if( (count * remote_double_size) != from_len ) {
         DUMP( "oops should I keep this data somewhere (excedent %d bytes)?\n",
               from_len - (count * remote_double_size) );
         res = -1;
      }
      DUMP( "correct: copy %s count %d from buffer %p with length %d to %p space %d\n",
            "double", count, from, from_len, to, to_len );
   } else
      DUMP( "         copy %s count %d from buffer %p with length %d to %p space %d\n",
            "double", count, from, from_len, to, to_len );

   
   if( (from_extent == sizeof(double)) && (to_extent == sizeof(double)) ) {
      MEMCPY( to, from, count * sizeof(double) );
   } else {
      for( i = 0; i < count; i++ ) {      
         MEMCPY( to, from, sizeof(double) );     
         to += to_extent;
         from += from_extent;
      }
   }
   *used = count * sizeof(double) ;
   return res * count;
}

conversion_fct_t copy_functions[DT_MAX_PREDEFINED] = {
   (conversion_fct_t)NULL,                 /*    DT_LOOP           */ 
   (conversion_fct_t)NULL,                 /*    DT_LB             */ 
   (conversion_fct_t)NULL,                 /*    DT_UB             */ 
   (conversion_fct_t)NULL,                 /*    DT_SPACE          */ 
   (conversion_fct_t)copy_char,            /*    DT_CHAR           */ 
   (conversion_fct_t)copy_char,            /*    DT_BYTE           */ 
   (conversion_fct_t)copy_short,           /*    DT_SHORT          */ 
   (conversion_fct_t)copy_int,             /*    DT_INT            */ 
   (conversion_fct_t)copy_float,           /*    DT_FLOAT          */ 
   (conversion_fct_t)copy_long,            /*    DT_LONG           */ 
   (conversion_fct_t)copy_double,          /*    DT_DOUBLE         */ 
   (conversion_fct_t)copy_long_long,       /*    DT_LONG_LONG      */ 
   (conversion_fct_t)copy_long_double,     /*    DT_LONG_DOUBLE    */ 
   (conversion_fct_t)copy_complex_float,   /*    DT_COMPLEX_FLOAT  */ 
   (conversion_fct_t)copy_complex_double,  /*    DT_COMPLEX_DOUBLE */ 
};

/* Should we supply buffers to the convertor or can we use directly
 * the user buffer ?
 */
int lam_convertor_need_buffers( lam_convertor_t* pConvertor )
{
   if( pConvertor->flags & DT_FLAG_CONTIGUOUS ) return 0;
   return 1;
}

extern int local_sizes[DT_MAX_PREDEFINED];
int lam_convertor_init_for_recv( lam_convertor_t* pConv, unsigned int flags,
                                 dt_desc_t* pData, int count,
                                 void* pUserBuf, int starting_point )
{
   OBJ_RETAIN( pData );
   pConv->pDesc = pData;
   pConv->flags = CONVERTOR_RECV;
   if( pConv->pStack != NULL ) free( pConv->pStack );
   pConv->pStack = (dt_stack_t*)malloc(sizeof(dt_stack_t) * (pData->btypes[DT_LOOP] + 2) );
   if( starting_point == 0 ) {
      pConv->stack_pos = 0;
      pConv->pStack[0].index = -1;         /* fake entry for the first step */
      pConv->pStack[0].count = count;      /* fake entry for the first step */
      pConv->pStack[0].disp  = 0;
      /* first we should decide which data representation will be used TODO */
      pConv->pStack[0].end_loop = pData->desc.used;
   } else {
   }
   pConv->pBaseBuf = pUserBuf;
   pConv->available_space = count * (pData->ub - pData->lb);
   pConv->count = count;
   pConv->pFunctions = copy_functions;
   pConv->converted = 0;
   pConv->bConverted = 0;
   if( (pData->flags & DT_FLAG_CONTIGUOUS) && (pData->size == (pData->ub - pData->lb)) )
      pConv->flags |= DT_FLAG_CONTIGUOUS;
   pConv->fAdvance = convertor_unpack_homogeneous;
   return 0;
}

/* Get the number of elements from the data associated with this convertor that can be
 * retrieved from a recevied buffer with the size iSize.
 * To spped-up this function you should use it with a iSize == to the modulo
 * of the original size and the size of the data.
 * This function should be called with a initialized clean convertor.
 * Return value:
 *   positive = number of basic elements inside
 *   negative = some error occurs
 */
int lam_ddt_get_element_count( dt_desc_t* pData, int iSize )
{
   dt_stack_t* pStack;   /* pointer to the position on the stack */
   int pos_desc;         /* actual position in the description of the derived datatype */
   int end_loop;         /* last element in the actual loop */
   int type;             /* type at current position */
   int rc, nbElems = 0;
   int stack_pos = 0;

   DUMP( "dt_count_elements( %p, %d )\n", pData, iSize );
   pStack = alloca( sizeof(pStack) * (pData->btypes[DT_LOOP] + 2) );
   pStack->count = 1;
   pStack->index = -1;
   pStack->end_loop = pData->desc.used;
   pStack->disp = 0;
   pos_desc  = 0;

   DUMP_STACK( pStack, stack_pos, pElems, "starting" );
   DUMP( "remember position on stack %d last_elem at %d\n", stack_pos, pos_desc );
   DUMP( "top stack info {index = %d, count = %d}\n", 
         pStack->index, pStack->count );

  next_loop:
   end_loop = pStack->end_loop;
   while( stack_pos >= 0 ) {
      if( pos_desc == end_loop ) { /* end of the current loop */
         while( --(pStack->count) == 0 ) { /* end of loop */
            stack_pos--;
            pStack--;
            if( stack_pos == -1 )
               return nbElems;  /* completed */
         }
         pos_desc = pStack->index;
         if( pos_desc == -1 )
            pStack->disp += (pData->ub - pData->lb);
         else
            pStack->disp += pData->desc.desc[pos_desc].extent;
         pos_desc++;
         goto next_loop;
      }
      if( pData->desc.desc[pos_desc].type == DT_LOOP ) {
         do {
            PUSH_STACK( pStack, stack_pos, pos_desc, pData->desc.desc[pos_desc].count,
                        0, pos_desc + pData->desc.desc[pos_desc].disp );
            pos_desc++;
         } while( pData->desc.desc[pos_desc].type == DT_LOOP ); /* let's start another loop */
         DUMP_STACK( pStack, stack_pos, pData->desc, "advance loops" );
         goto next_loop;
      }
      /* now here we have a basic datatype */
      type = pData->desc.desc[pos_desc].type;
      rc = pData->desc.desc[pos_desc].count * basicDatatypes[type].size;
      if( rc >= iSize ) {
         nbElems += iSize / basicDatatypes[type].size;
         break;
      }
      nbElems += pData->desc.desc[pos_desc].count;
      iSize -= rc;

      pos_desc++;  /* advance to the next data */
   }

   /* cleanup the stack */
   return nbElems;
}

int lam_ddt_copy_content_same_ddt( dt_desc_t* pData, int count,
                                   char* pDestBuf, char* pSrcBuf )
{
   dt_stack_t* pStack;   /* pointer to the position on the stack */
   int pos_desc;         /* actual position in the description of the derived datatype */
   int type;             /* type at current position */
   int stack_pos = 0;
   long lastDisp = 0, lastLength = 0;
   dt_elem_desc_t* pElems;

   if( (pData->flags & DT_FLAG_BASIC) == DT_FLAG_BASIC ) {
      /* basic datatype with count */
      MEMCPY( pDestBuf, pSrcBuf, pData->size * count );
      return 0;
   }

   pStack = alloca( sizeof(pStack) * (pData->btypes[DT_LOOP]+1) );
   pStack->count = count;
   pStack->index = -1;
   pStack->disp = 0;
   pos_desc  = 0;

   if( pData->opt_desc.desc != NULL ) {
      pElems = pData->opt_desc.desc;
      pStack->end_loop = pData->opt_desc.used;
   } else {
      pElems = pData->desc.desc;
      pStack->end_loop = pData->desc.used;
   }

   DUMP_STACK( pStack, stack_pos, pElems, "starting" );
   DUMP( "remember position on stack %d last_elem at %d\n", stack_pos, pos_desc );
   DUMP( "top stack info {index = %d, count = %d}\n", 
         pStack->index, pStack->count );

  next_loop:
   while( pos_desc <= pStack->end_loop ) {
      if( pos_desc == pStack->end_loop ) { /* end of the current loop */
         if( --(pStack->count) == 0 ) { /* end of loop */
            pStack--;
            if( --stack_pos == -1 ) break;
         } else
            pos_desc = pStack->index;
         if( pos_desc == -1 )
            pStack->disp += (pData->ub - pData->lb);
         else
            pStack->disp += pElems[pos_desc].extent;
         pos_desc++;
         goto next_loop;
      }
      if( pElems[pos_desc].type == DT_LOOP ) {
         do {
            PUSH_STACK( pStack, stack_pos, pos_desc, pElems[pos_desc].count,
                        pStack->disp, pos_desc + pElems[pos_desc].disp );
            pos_desc++;
         } while( pElems[pos_desc].type == DT_LOOP ); /* let's start another loop */
         DUMP_STACK( pStack, stack_pos, pElems, "advance loops" );
         goto next_loop;
      }
      /* now here we have a basic datatype */
      type = pElems[pos_desc].type;
      if( (lastDisp + lastLength) == (pStack->disp + pElems[pos_desc].disp) ) {
         lastLength += pElems[pos_desc].count * basicDatatypes[type].size;
      } else {
         MEMCPY( pDestBuf + lastDisp, pSrcBuf + lastDisp, lastLength );
         lastDisp = pStack->disp + pElems[pos_desc].disp;
         lastLength = pElems[pos_desc].count * basicDatatypes[type].size;
      }
      pos_desc++;  /* advance to the next data */
   }

   MEMCPY( pDestBuf + lastDisp, pSrcBuf + lastDisp, lastLength );
   /* cleanup the stack */
   return 0;
}
