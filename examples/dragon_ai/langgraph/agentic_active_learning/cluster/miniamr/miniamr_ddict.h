/*
 * Write miniAMR's mesh into the Dragon DDict from C.
 *
 * This lives in the hpc-pe-dragon-dragon repo (not in miniAMR) so it ships to
 * the cluster with the demo. On the allocation you clone miniAMR, copy this
 * file and miniamr_ddict.c into miniAMR/ref, add the hooks below, and build.
 *
 * It keeps every rank's field on its OWN node by writing to the node-local
 * manager, so the large data never leaves the node that computed it.
 *
 * Three calls, nothing more:
 *   miniamr_ddict_open(serial)          attach, once per rank at startup
 *   miniamr_ddict_put(key, buf, nbytes) write one block's bytes, node-local
 *   miniamr_ddict_close()               detach at shutdown
 *
 * WIRING (done by the integrator in miniAMR/ref):
 *   1. Python side hands the serialized DDict descriptor to the job as an env
 *      var (campaign.py sets env={"MINIAMR_DDICT": mesh.serialize(), ...}).
 *   2. In main.c, after MPI_Init:
 *        miniamr_ddict_open(getenv("MINIAMR_DDICT"));
 *      and read the run id: const char *run = getenv("MINIAMR_RUN");
 *   3. After the timestep loop, pack each active block and put it.
 *      blocks[].array is double[var][x][y][z]; active when blocks[n].number>=0.
 *      Pack into a contiguous double buffer using miniAMR's own block dims
 *      (num_vars, x_block_size, y_block_size, z_block_size). Write blocks under
 *      a contiguous index and a per-rank count, so the reader fetches them by
 *      exact key instead of listing the node. Sketch:
 *
 *        char key[64];
 *        int nb = 0;
 *        for (int n = 0; n < max_active_block; n++) {
 *           block *bp = &blocks[n];
 *           if (bp->number < 0) continue;
 *           // copy bp->array[v][i][j][k] into packed[] (v,i,j,k C order)
 *           snprintf(key, sizeof(key), "mesh/%s/r%d/b%d", run, my_pe, nb);
 *           miniamr_ddict_put(key, (uint8_t*) packed, packed_bytes);
 *           nb++;
 *        }
 *        int32_t nb32 = (int32_t) nb;   // how many blocks this rank wrote
 *        snprintf(key, sizeof(key), "mesh/%s/r%d/n", run, my_pe);
 *        miniamr_ddict_put(key, (uint8_t*) &nb32, sizeof(nb32));
 *   4. miniamr_ddict_close() before MPI_Finalize.
 *   5. Makefile: add Dragon's include and lib dirs and link libdragon:
 *        CPPFLAGS += -I$(DRAGON_BASE_DIR)/include
 *        LDFLAGS  += -L$(DRAGON_BASE_DIR)/lib
 *        LDLIBS   += -ldragon
 *        OBJS     += miniamr_ddict.o
 *
 * See the cluster README's "Build miniAMR with the DDict hook" for the exact,
 * copy-pasteable version of these edits.
 *
 * The Python reader (cluster/mesh_v2.py) gets these bytes back with
 * store.manager(m)[key] and views them as float64 [num_vars, x, y, z], C order.
 * That layout is a contract -- keep it in sync with physics.MESH_DIMS.
 */

#ifndef MINIAMR_DDICT_H
#define MINIAMR_DDICT_H

#include <stddef.h>
#include <stdint.h>

/* Attach to the DDict and pin this rank to its node-local manager.
 * serial is the string from Python's store.serialize(). Returns 0 on success. */
int miniamr_ddict_open(const char *serial);

/* Write one value (a block's packed doubles) under key, to the local manager.
 * Returns 0 on success. */
int miniamr_ddict_put(const char *key, const uint8_t *buf, size_t nbytes);

/* Detach. Returns 0 on success. */
int miniamr_ddict_close(void);

#endif /* MINIAMR_DDICT_H */
