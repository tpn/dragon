/*
 * Dragon DDict glue for miniAMR. See miniamr_ddict.h for how to wire it in.
 *
 * All Dragon C-API use is isolated here. Grounded in src/include/dragon/ddict.h:
 *   dragon_ddict_attach, dragon_ddict_local_manager, dragon_ddict_manager,
 *   dragon_ddict_create_request, dragon_ddict_write_bytes, dragon_ddict_put,
 *   dragon_ddict_finalize_request, dragon_ddict_detach.
 *
 * The put sequence, per the header's contract: write the key bytes, call put
 * (which ends the key and begins the value), write the value bytes, finalize.
 */

#include "miniamr_ddict.h"

#include <stdio.h>
#include <string.h>
#include <dragon/ddict.h>
#include <dragon/return_codes.h>

/* Two handles held for the life of the rank: the attached dictionary, and a
 * copy pinned to this node's local manager. Puts go through the pinned copy so
 * the data stays on this node. */
static dragonDDictDescr_t g_dd;
static dragonDDictDescr_t g_local;
static int g_open = 0;


int miniamr_ddict_open(const char *serial)
{
    if (serial == NULL)
        return 1;

    /* NULL timeout: block rather than fail, matching the Python default. */
    if (dragon_ddict_attach(serial, &g_dd, NULL) != DRAGON_SUCCESS)
        return 2;

    uint64_t local_id;
    if (dragon_ddict_local_manager(&g_dd, &local_id) != DRAGON_SUCCESS)
        return 3;

    /* A client copy that only talks to the node-local manager -- the C
     * equivalent of Python's store.manager(id). */
    if (dragon_ddict_manager(&g_dd, &g_local, local_id) != DRAGON_SUCCESS)
        return 4;

    g_open = 1;
    return 0;
}


int miniamr_ddict_put(const char *key, const uint8_t *buf, size_t nbytes)
{
    if (!g_open)
        return 1;

    dragonDDictRequestDescr_t req;
    if (dragon_ddict_create_request(&g_local, &req) != DRAGON_SUCCESS)
        return 2;

    /* Key first. */
    if (dragon_ddict_write_bytes(&req, strlen(key), (uint8_t *) key)
            != DRAGON_SUCCESS)
        return 3;

    /* put() ends the key and begins the value. */
    if (dragon_ddict_put(&req) != DRAGON_SUCCESS)
        return 4;

    /* Then the value bytes. */
    if (nbytes > 0 && dragon_ddict_write_bytes(&req, nbytes, (uint8_t *) buf)
            != DRAGON_SUCCESS)
        return 5;

    if (dragon_ddict_finalize_request(&req) != DRAGON_SUCCESS)
        return 6;

    return 0;
}


int miniamr_ddict_close(void)
{
    if (!g_open)
        return 0;
    dragon_ddict_detach(&g_local);
    dragon_ddict_detach(&g_dd);
    g_open = 0;
    return 0;
}
