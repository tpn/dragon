"""
The expensive part: miniAMR, a real MPI simulation used as a stand-in for a
production solver.

Unlike a Python function, this is a compiled MPI program, so campaign.py starts
it as a Dragon ``batch.job`` tied to a chosen machine. Built with the hook in
miniamr/miniamr_ddict.h, it writes its result straight into the shared store from
C, on the machine it ran on, so there are no output files to parse.

This file only builds the command line and records the memory layout that the C
writer and the Python reader (mesh_v2.py) must agree on. It imports nothing from
Dragon; starting the job and choosing its machine happen in campaign.py.

The two names used throughout:
  * dials    -- the settings of one run, [object_speed, object_size]. A sphere is
                pushed through the grid; how fast and how big it is decides how
                much extra detail the grid has to add to follow it.
  * quantity -- the single number we keep from a run, worked out in mesh_v2.py.
"""

from __future__ import annotations


# Binary produced by `make` in miniAMR/ref (built WITH miniamr_ddict.c so it
# writes its mesh into the DDict). Point this at the built executable.
MINIAMR_BIN = "miniAMR.x"

# Fixed run settings, demo-tuned so each run is seconds not minutes. miniAMR is
# communication-bound, so the levers that matter are num_vars, num_tsteps and
# stages_per_ts -- all cut well below the miniAMR defaults (40, 100, 20). The
# refinement-footprint signal the surrogate learns is unaffected.
NX = NY = NZ = 8             # cells per block, per direction
NUM_VARS = 8                # variables per cell (default 40; drives comm cost)
NUM_REFINE = 5              # levels of refinement the mesh may reach; level 6
                            # overflows MAX_BLOCKS (miniAMR aborts with "Need
                            # more blocks"), so 5 is the safe ceiling here
MAX_BLOCKS = 6000           # per-rank block cap
NUM_TSTEPS = 80             # longer runs widen the wall-time gap between heavy
                            # and light spheres, so the heavy ones straddle the
                            # next wave more often (default 100)
STAGES_PER_TS = 20         # comm/calc stages per timestep (default 20); the
                            # main runtime lever -- does not move the sphere or
                            # change the mesh size
CHECKSUM_FREQ = 4

# Mesh layout the C writer packs and the Python reader reshapes: one block is
# [NUM_VARS, x, y, z] float64, with a one-cell ghost each side (so NX+2). This
# MUST match the packing in miniAMR/ref/miniamr_ddict.c.
MESH_DIMS = (NUM_VARS, NX + 2, NY + 2, NZ + 2)

# The object starts in a corner and moves diagonally, like the miniAMR README's
# moving-sphere example. type 2 is the surface of a spheroid; bounce 0 lets it
# leave the mesh.
OBJECT_TYPE = 2
OBJECT_BOUNCE = 0
OBJECT_START = -1.71        # center at the start, each axis
OBJECT_INC = 0.0            # no size change over time

# Design-space defaults (the brief can narrow these). Narrowed so the object
# stays inside the mesh at the final timestep: it starts at -1.71 and moves
# speed*NUM_TSTEPS per axis, so these keep its final centre interior, and cap
# the size below the domain width to avoid runs that blow up.
SPEED_RANGE = (0.010, 0.025)   # movement per timestep, each axis; halved to
                               # match the doubled NUM_TSTEPS so the sphere's
                               # total travel (speed*NUM_TSTEPS) is unchanged
SIZE_RANGE = (0.40, 1.00)      # initial half-extent, each axis


def object_args(speed: float, size: float) -> list[str]:
    """The 14 numbers after ``--object`` for a single moving sphere.

    speed sets the movement rate on every axis; size sets the initial extent on
    every axis. Everything else is fixed so the sweep is two-dimensional.
    """
    return [
        str(OBJECT_TYPE), str(OBJECT_BOUNCE),
        str(OBJECT_START), str(OBJECT_START), str(OBJECT_START),
        str(speed), str(speed), str(speed),
        str(size), str(size), str(size),
        str(OBJECT_INC), str(OBJECT_INC), str(OBJECT_INC),
    ]


def ranks_for(npx: int, npy: int, npz: int) -> int:
    """miniAMR needs npx*npy*npz == number of MPI ranks."""
    return npx * npy * npz


def build_argv(dials: list[float], npx: int, npy: int, npz: int) -> list[str]:
    """The full miniAMR command line for one run (the arguments after the program).

    --report_perf 0 silences miniAMR's end-of-run performance report, which it
    otherwise prints to stdout. The result is read from the shared store, not
    from anything printed or written to a file.
    """
    speed, size = float(dials[0]), float(dials[1])
    return [
        "--num_refine", str(NUM_REFINE),
        "--max_blocks", str(MAX_BLOCKS),
        "--npx", str(npx), "--npy", str(npy), "--npz", str(npz),
        "--nx", str(NX), "--ny", str(NY), "--nz", str(NZ),
        "--num_tsteps", str(NUM_TSTEPS),
        "--stages_per_ts", str(STAGES_PER_TS),
        "--num_vars", str(NUM_VARS),
        "--checksum_freq", str(CHECKSUM_FREQ),
        "--report_perf", "0",
        "--num_objects", "1",
        "--object", *object_args(speed, size),
    ]
