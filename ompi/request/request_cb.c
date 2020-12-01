/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2020      High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */


#include "ompi_config.h"
#include "request.h"
#include "request_cb.h"
#include "opal/class/opal_fifo.h"
#include "opal/sys/atomic.h"
#include "opal/threads/thread_usage.h"


static opal_free_list_t request_callback_freelist;

/**
 * FIFO of completed requests that need the user-defined completion callback
 * invoked. Use atomic push to avoid race conditions if multiple threads
 * complete requests.
 */
#ifdef USE_FIFO
opal_fifo_t *request_cont_fifo = NULL;
#else
opal_list_t *request_cont_list = NULL;
#endif
static opal_mutex_t request_cont_lock;

/**
 * Flag indicating whether the progress callback has been registered.
 */
static int32_t progress_callback_registered;


static void ompi_request_cont_construct(ompi_request_cont_t* cont)
{
    cont->cont_req   = NULL;
    cont->cont_cb    = NULL;
    cont->cont_data  = NULL;
    cont->num_active = 0;
}

static void ompi_request_cont_destruct(ompi_request_cont_t* cont)
{
    assert(cont->cont_req   == NULL);
    assert(cont->cont_cb    == NULL);
    assert(cont->cont_data  == NULL);
    assert(cont->num_active == 0);
}


OBJ_CLASS_INSTANCE(
    ompi_request_cont_t,
    opal_free_list_item_t,
    ompi_request_cont_construct,
    ompi_request_cont_destruct);

static inline
void ompi_request_cont_destroy(ompi_request_cont_t *cont, ompi_request_t *cont_req)
{
    if (opal_using_threads()) {
        opal_atomic_lock(&cont_req->cont_lock);
    }
    int num_active = --cont_req->cont_num_active;
    assert(num_active >= 0);
    if (0 == num_active) {
        assert(!REQUEST_COMPLETE(cont_req));
        opal_atomic_wmb();
        /* signal that all continuations were found complete */
        ompi_request_complete(cont_req, true);
    }
    if (opal_using_threads()) {
        opal_atomic_unlock(&cont_req->cont_lock);
    }
    OBJ_RELEASE(cont_req);

#ifdef OPAL_ENABLE_DEBUG
    cont->cont_cb   = NULL;
    cont->cont_data = NULL;
    cont->cont_req  = NULL;
#endif // OPAL_ENABLE_DEBUG
    opal_free_list_return(&request_callback_freelist, &cont->super);
}

/**
 * Process a callback. Returns the callback object to the freelist.
 */
static inline
void ompi_request_cont_invoke(ompi_request_cont_t *cont)
{
    ompi_request_t *cont_req = cont->cont_req;
    assert(NULL != cont_req);
    assert(OMPI_REQUEST_CONT == cont_req->req_type);

    MPIX_Continue_cb_function *fn = cont->cont_cb;
    void *cont_data = cont->cont_data;
    MPI_Status *statuses = cont->cont_status;
    fn(statuses, cont_data);
    ompi_request_cont_destroy(cont, cont_req);
}

/*
  * Allow multiple threads to progress callbacks concurrently
  * but protect from recursive progressing
  */
static opal_thread_local int in_progress = 0;

static
int ompi_request_cont_progress_some(const uint32_t max)
{

#ifdef USE_FIFO
    if (in_progress || opal_fifo_is_empty(request_cont_fifo)) return 0;
#else
    if (in_progress || opal_list_is_empty(request_cont_list)) return 0;
#endif
    int completed = 0;
    in_progress = 1;

    do {
        ompi_request_cont_t *cb;
        OPAL_THREAD_LOCK(&request_cont_lock);
#ifdef USE_FIFO
        cb = (ompi_request_cont_t*)opal_fifo_pop_st(request_cont_fifo);
#else
        cb = (ompi_request_cont_t*)opal_list_remove_first(request_cont_list);
#endif
        OPAL_THREAD_UNLOCK(&request_cont_lock);
        if (NULL == cb) break;
        ompi_request_cont_invoke(cb);
        completed++;
    } while (max > completed);

    in_progress = 0;

    return completed;
}

int ompi_request_cont_progress_callback()
{
    return ompi_request_cont_progress_some(1);
}

int ompi_request_cont_progress_request(ompi_request_t *cont_req)
{
    if (in_progress) return 0;
    if (NULL == cont_req->cont_complete_list) {
    /* progress as many as possible */
    return ompi_request_cont_progress_some(UINT32_MAX);
    }
    if (opal_list_is_empty(cont_req->cont_complete_list)) {
        return 0;
    }

    in_progress = 1;

    int completed = 0;
    while (!opal_list_is_empty(cont_req->cont_complete_list)) {
        ompi_request_cont_t *cb;
        if (opal_using_threads()) {
            opal_atomic_lock(&cont_req->cont_lock);
            cb = (ompi_request_cont_t *) opal_list_remove_first(cont_req->cont_complete_list);
            opal_atomic_unlock(&cont_req->cont_lock);
        } else {
            cb = (ompi_request_cont_t *) opal_list_remove_first(cont_req->cont_complete_list);
        }
        if (NULL == cb) break;

        ompi_request_cont_invoke(cb);
        completed++;
    }

    in_progress = 0;

    return completed;
}


int ompi_request_cont_progress_register_request(ompi_request_t *cont_req)
{
    if (NULL == cont_req->cont_complete_list) return OMPI_SUCCESS;

    if (opal_using_threads()) {
        OPAL_THREAD_LOCK(&request_cont_lock);
        opal_atomic_lock(&cont_req->cont_lock);
    }

    /* signal that from now on all continuations should go into the global queue */
    cont_req->cont_global_progress = true;

    /* move all complete local continuations into the global queue */
#ifdef USE_FIFO
    ompi_request_cont_t *cb;
    while (NULL != (cb = (ompi_request_cont_t *)opal_list_remove_first(cont_req->cont_complete_list))) {
        opal_fifo_push_st(request_cont_fifo, &cb->super.super);
    }
#else
    opal_list_join(request_cont_list, opal_list_get_begin(request_cont_list), cont_req->cont_complete_list);
#endif

    if (opal_using_threads()) {
        opal_atomic_unlock(&cont_req->cont_lock);
        OPAL_THREAD_UNLOCK(&request_cont_lock);
    }

    return OMPI_SUCCESS;
}

int ompi_request_cont_progress_deregister_request(ompi_request_t *cont_req)
{
    if (opal_using_threads()) {
        opal_atomic_lock(&cont_req->cont_lock);
        cont_req->cont_global_progress = false;
        opal_atomic_unlock(&cont_req->cont_lock);
    } else {
        cont_req->cont_global_progress = false;
    }

    return OMPI_SUCCESS;
}

int ompi_request_cont_init(void)
{
    progress_callback_registered = 0;

    OBJ_CONSTRUCT(&request_cont_lock, opal_mutex_t);
#ifdef USE_FIFO
    request_cont_fifo = malloc(sizeof(*request_cont_fifo));
    OBJ_CONSTRUCT(request_cont_fifo, opal_fifo_t);
#else
    request_cont_list = malloc(sizeof(*request_cont_list));
    OBJ_CONSTRUCT(request_cont_list, opal_list_t);
#endif

    OBJ_CONSTRUCT(&request_callback_freelist, opal_free_list_t);
    opal_free_list_init(&request_callback_freelist,
                        sizeof(ompi_request_cont_t),
                        opal_cache_line_size,
                        OBJ_CLASS(ompi_request_cont_t),
                        0, opal_cache_line_size,
                        0, -1 , 8, NULL, 0, NULL, NULL, NULL);

    return OMPI_SUCCESS;
}

int ompi_request_cont_finalize(void)
{
    if (progress_callback_registered) {
        opal_progress_unregister(&ompi_request_cont_progress_callback);
    }
#ifdef USE_FIFO
    OBJ_DESTRUCT(request_cont_fifo);
    free(request_cont_fifo);
#else
    OBJ_DESTRUCT(request_cont_list);
    free(request_cont_list);
#endif
    OBJ_DESTRUCT(&request_cont_lock);
    OBJ_DESTRUCT(&request_callback_freelist);

    return OMPI_SUCCESS;
}

/**
 * Enqueue the continuation for later invocation.
 */
void
ompi_request_cont_enqueue_complete(ompi_request_cont_t *cont)
{
    int retry;
    do {
        retry = 0;
        if (NULL != cont->cont_req->cont_complete_list && !cont->cont_req->cont_global_progress) {
            opal_atomic_lock(&cont->cont_req->cont_lock);
            if (OPAL_UNLIKELY(cont->cont_req->cont_global_progress)) {
                opal_atomic_unlock(&cont->cont_req->cont_lock);
                /* try again, this time target the global list */
                retry = 1;
                continue;
            }
            opal_list_append(cont->cont_req->cont_complete_list, &cont->super.super);
            opal_atomic_unlock(&cont->cont_req->cont_lock);
        } else {
            OPAL_THREAD_LOCK(&request_cont_lock);
#ifdef USE_FIFO
            opal_fifo_push_st(request_cont_fifo, &cont->super.super);
#else
            opal_list_append(request_cont_list, &cont->super.super);
#endif
            if (OPAL_UNLIKELY(!progress_callback_registered)) {
                opal_progress_register_post(&ompi_request_cont_progress_callback);
                progress_callback_registered = true;
            }
            OPAL_THREAD_UNLOCK(&request_cont_lock);
        }
    } while (retry);
}

/**
 * Create and initialize a continuation object.
 */
static inline
ompi_request_cont_t *ompi_request_cont_create(
  int                         count,
  ompi_request_t             *cont_req,
  MPIX_Continue_cb_function  *cont_cb,
  void                       *cont_data,
  MPI_Status                 *cont_status)
{
    ompi_request_cont_t *cont;
    cont = (ompi_request_cont_t *)opal_free_list_get(&request_callback_freelist);
    cont->cont_req  = cont_req;
    cont->cont_cb   = cont_cb;
    cont->cont_data = cont_data;
    cont->num_active = count;
    cont->cont_status = cont_status;

    /* signal that the continuation request has a new continuation */
    OBJ_RETAIN(cont_req);
    if (opal_using_threads()) {
        opal_atomic_lock(&cont_req->cont_lock);
    }
    int32_t num_active = cont_req->cont_num_active++;
    if (num_active == 0) {
        /* (re)activate the continuation request upon first registration */
        assert(REQUEST_COMPLETE(cont_req));
        cont_req->cont_obj     = REQUEST_CONT_NONE;
        cont_req->req_complete = REQUEST_PENDING;
    }
    if (opal_using_threads()) {
        opal_atomic_unlock(&cont_req->cont_lock);
    }

    return cont;
}

int ompi_request_cont_register(
  ompi_request_t             *cont_req,
  const int                   count,
  ompi_request_t             *requests[],
  MPIX_Continue_cb_function  *cont_cb,
  void                       *cont_data,
  bool                       *all_complete,
  ompi_status_public_t        statuses[])
{
    assert(OMPI_REQUEST_CONT == cont_req->req_type);

    /* Set status objects if required */
    if (MPI_STATUSES_IGNORE != statuses) {
        for (int i = 0; i < count; ++i) {
            ompi_request_t *request = requests[i];
            if (MPI_REQUEST_NULL != request) {
                request->cont_status = &statuses[i];
            }
        }
    }
    *all_complete = false;

    ompi_request_cont_t *cont = ompi_request_cont_create(count, cont_req, cont_cb, cont_data, statuses);

    opal_atomic_wmb();

    int32_t num_registered = 0;
    for (int i = 0; i < count; ++i) {
        if (MPI_REQUEST_NULL != requests[i]) {
            void *cont_compare = REQUEST_CONT_NONE;
            if (REQUEST_CONT_NONE == requests[i]->cont_obj &&
                OPAL_ATOMIC_COMPARE_EXCHANGE_STRONG_PTR(&requests[i]->cont_obj,
                                                        &cont_compare, cont)) {
                ++num_registered;
            } else {
                ompi_request_t *request = requests[i];
                assert(REQUEST_COMPLETE(requests[i]) || REQUEST_CONT_COMPLETED == cont_compare);
                /* set the status, if necessary */
                if (NULL != request->cont_status) {
                    *request->cont_status = request->req_status;
                }
                /* inactivate / free the request */
                if (request->req_persistent) {
                    if (!request->req_transient) {
                        request->req_state = OMPI_REQUEST_INACTIVE;
                    }
                } else {
                    /* the request is complete, release the request object */
                    ompi_request_free(&request);
                }
            }
            /* take ownership of any non-persistent request */
            if (!requests[i]->req_persistent)
            {
                requests[i] = MPI_REQUEST_NULL;
            }
        }
    }

    int num_complete = count - num_registered;
    int32_t last_num_active = OPAL_THREAD_ADD_FETCH32(&cont->num_active,
                                                      -num_complete);
    if (0 == last_num_active && 0 < num_complete) {
        /**
         * set flag and return the continuation to the free-list
         */
        *all_complete = true;
        ompi_request_cont_destroy(cont, cont_req);
    }

    return OMPI_SUCCESS;
}

/**
 * Continuation request management
 */

static int ompi_request_cont_req_free(ompi_request_t** cont_req)
{
    OMPI_REQUEST_FINI(*cont_req);
    (*cont_req)->req_state = OMPI_REQUEST_INVALID;
    OBJ_RELEASE(*cont_req);
    *cont_req = &ompi_request_null.request;
    return OMPI_SUCCESS;
}

int ompi_request_cont_allocate_cont_req(ompi_request_t **cont_req, ompi_info_t *info)
{
    ompi_request_t *res = OBJ_NEW(ompi_request_t);

    if (OPAL_LIKELY(NULL != cont_req)) {
        res->req_type = OMPI_REQUEST_CONT;
        res->req_complete = REQUEST_COMPLETED;
        res->req_state = OMPI_REQUEST_ACTIVE;
        res->req_persistent = true;
        res->req_transient  = true;
        res->req_free = ompi_request_cont_req_free;
        res->req_status = ompi_status_empty; /* always returns MPI_SUCCESS */
        opal_atomic_lock_init(&res->cont_lock, 0);

        int flag;
        bool test_poll = false;
        ompi_info_get_bool(info, "continue_test_poll", &test_poll, &flag);

        if (flag && test_poll) {
            res->cont_complete_list = OBJ_NEW(opal_list_t);
        } else {
            res->cont_complete_list = NULL;
        }

        *cont_req = res;

        return MPI_SUCCESS;
    }

    return OMPI_ERR_OUT_OF_RESOURCE;
}
