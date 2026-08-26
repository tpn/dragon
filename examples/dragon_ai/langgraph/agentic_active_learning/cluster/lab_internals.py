"""
The supporting machinery for the campaign: tuning constants, how design points
are drawn, which machine each run goes to, and the per-process state that lets a
step keep the trained model in memory between rounds.

Imports nothing from campaign.py, or the imports would go in a circle.
"""

from __future__ import annotations

import math
import random
import socket
from typing import Any

from dragon.data.ddict import DDict
from dragon.native.machine import Node, System
from dragon.native.process import current as current_process

from physics import SIZE_RANGE, SPEED_RANGE


# ----------------------------------------------------------------------------
# Tuning. Everything you would reach for first is in this block.
# ----------------------------------------------------------------------------

# A miniAMR run is minutes of real MPI, so budgets are small and waves are few.
MAX_WAVES = 28
DEFAULT_BUDGET = 100        # solver runs the brief allows
RUNS_PER_WAVE = 4           # designs launched per wave (each a multi-node job)
COLLECT_ENOUGH = 2          # per wave, how many results collect waits for before
                            # moving on; the rest keep running and are picked up
                            # in a later wave. Lower than RUNS_PER_WAVE on purpose
                            # so more runs overlap across waves (more in flight)
SCOUT_RUNS = 4              # first wave, spread across the box
CANDIDATE_POOL = 400        # points the ensemble is asked about before picking
MIN_NEW_POINTS = 4          # new results needed before refitting is worth it
# Uncertainty sampling plus an LLM steer can park every run in one corner, and
# then the held-out error only says the model fits that corner.
EXPLORE_PER_WAVE = 2        # of RUNS_PER_WAVE, how many ignore the steer
WAVE_WAIT = 120.0           # seconds; cap on one collect visit (was 1800). Short
                            # so a hung run never stalls the demo; a slow-but-live
                            # run is picked up in a later wave anyway.
DRAIN_WAVES = 120           # rounds of checking before giving up on missing work
DRAIN_POLL = 5.0            # seconds between rounds

# The surrogate learns log(1+blocks); the held-out error is the MEDIAN relative
# error on block counts. TARGET_FRACTION is the tolerance the campaign stops at.
TARGET_FRACTION = 0.10      # tolerance; 0.10 = "typical prediction within 10%"

# A low error on a handful of near-identical runs is luck, not a trained model,
# so hitting the target only ends the campaign once there is this much evidence.
MIN_POINTS_TO_STOP = 32     # training runs before an early "accurate enough"
MIN_VALIDATE_TO_STOP = 6    # held-out points the error must be measured on

DEMAND_QUERIES = 2000       # questions put to the finished surrogate at the end
CHECK_QUERIES = 8           # of those, how many to also run through miniAMR

# Campaign store (state): small Python objects -- the training set, the model,
# the settings. A gigabyte per node is far more than enough.
DDICT_MEM_PER_NODE = 1024 * 1024**2
# Result store (mesh): each run writes hundreds of MB. The reduce deletes a run's
# blocks as soon as it has summarised them, but that delete can land just after
# the next wave's run has already begun writing, so a node briefly holds more
# than one run's mesh. Give generous headroom so a lagging delete never runs
# into a full manager.
MESH_MEM_PER_NODE = 16 * 1024**3

# miniAMR MPI job shape. npx*npy*npz must equal the rank count. On a small
# allocation the job stays on ONE node (JOB_NODES=1) so several can run at once
# alongside the agent hosts and inference. All ranks are cores on that node.
JOB_NPX, JOB_NPY, JOB_NPZ = 2, 2, 2   # 8 ranks, all on one node
JOB_NODES = 1                          # nodes each miniAMR job spans

# DDict keys.
K_TRAIN = "training_set"      # [{"dials": [speed, size], "quantity": float, "run": str}]
K_BRIEF = "brief"             # settings, from the scientist's paragraph
K_MODEL = "surrogate"         # the fitted model, as produced by assemble()
K_VERSION = "surrogate_version"
K_RUNS = "run_index"          # [{"run", "nodes", "results_path", "stdout_path"}]


def default_brief() -> dict[str, Any]:
    """Settings for a run with no model, or a model that gave nonsense.

    bounds are [[speed_lo, speed_hi], [size_lo, size_hi]], taken from physics.
    """
    return {
        "bounds": [list(SPEED_RANGE), list(SIZE_RANGE)],
        "sim_budget": DEFAULT_BUDGET,
        "target_fraction": TARGET_FRACTION,
    }


# ----------------------------------------------------------------------------
# Picking where to simulate next.
# ----------------------------------------------------------------------------


def evenly_spread_dials(count: int, bounds: list[list[float]],
                        seed: int) -> list[list[float]]:
    """Design points spread over the box, for the first wave."""
    rng = random.Random(seed)
    (low_1, high_1), (low_2, high_2) = bounds
    # Lay the points on a roughly square grid: side x side ~= count cells, so
    # side ~= sqrt(count). max(1, ...) avoids a 0 side when count is 0 or 1.
    side = max(1, int(math.isqrt(count)))
    drawn = []
    for i in range(count):
        # Walk the grid cell by cell (first axis cycles fastest), then jitter the
        # point randomly inside its cell so the spread is even but not a rigid lattice.
        cell_1, cell_2 = i % side, (i // side) % side
        drawn.append([
            low_1 + (high_1 - low_1) * (cell_1 + rng.random()) / side,
            low_2 + (high_2 - low_2) * (cell_2 + rng.random()) / side,
        ])
    return drawn


def candidate_dials(count: int, bounds: list[list[float]],
                    seed: int) -> list[list[float]]:
    """A batch of random [speed, size] points for the ensemble to score.

    These are only *candidates*: propose asks the three models to predict each
    one, keeps the few they disagree on most, and simulates only those. The rest
    are thrown away. So this is cheap -- no simulation happens here.

    Unlike evenly_spread_dials (a jittered grid), these are plain uniform-random
    draws over the box, because we want many varied points to probe, not even
    coverage.
    """
    rng = random.Random(seed)
    (low_1, high_1), (low_2, high_2) = bounds
    return [[rng.uniform(low_1, high_1), rng.uniform(low_2, high_2)]
            for _ in range(count)]


def clamp_region(region: list[list[float]],
                 bounds: list[list[float]],
                 min_width_frac: float = 0.3) -> list[list[float]]:
    """Keep the steer region inside the bounds and never narrower than
    min_width_frac of each axis, so steering keeps exploring instead of
    shrinking to a sliver."""
    out = []
    for (lo, hi), (box_lo, box_hi) in zip(region, bounds):
        centre = min(max((lo + hi) / 2, box_lo), box_hi)
        half = max(abs(hi - lo), min_width_frac * (box_hi - box_lo)) / 2
        out.append([max(box_lo, centre - half), min(box_hi, centre + half)])
    return out


def candidate_dials_biased(count: int, bounds: list[list[float]],
                           region: list[list[float]], seed: int,
                           region_fraction: float = 0.5) -> list[list[float]]:
    """Candidate points, biased toward the steered sub-region.

    Half come from the steer, half from the whole box, so a poor steer can be
    escaped and the ensemble keeps a chance to disagree elsewhere.
    """
    n_region = int(round(count * region_fraction))
    n_region = int(round(count * region_fraction))
    in_region = candidate_dials(n_region, region, seed)
    in_box = candidate_dials(count - n_region, bounds, seed + 1)
    return in_region + in_box


def _budget_spent(state: dict[str, Any]) -> bool:
    """Out of solver runs, out of waves, or accurate enough on enough evidence."""
    accurate = (state["validate_error"] is not None
                and state["target_error"] > 0
                and state["validate_error"] <= state["target_error"]
                and state["n_training_points"] >= MIN_POINTS_TO_STOP
                and state["error_n_validate"] >= MIN_VALIDATE_TO_STOP)
    return (state["sims_run"] >= state["budget"]
            or state["wave"] >= MAX_WAVES
            or accurate)


# ----------------------------------------------------------------------------
# Choosing machines. This is what keeps each result next to the code that reads it.
# ----------------------------------------------------------------------------


def allocation_hostnames() -> list[str]:
    """Names of every machine in the allocation, in a stable order."""
    system = System()
    return [Node(h_uid).hostname for h_uid in system.nodes]


def pick_job_nodes(all_nodes: list[str], run_index: int,
                   nodes_per_job: int) -> list[str]:
    """Which machines a given run should use.

    Hands them out round-robin so successive rounds spread over the allocation.
    The choice is recorded, so the code that reads the result can later be sent
    to the same machines.
    """
    if not all_nodes:
        return []
    start = (run_index * nodes_per_job) % len(all_nodes)
    return [all_nodes[(start + k) % len(all_nodes)] for k in range(nodes_per_job)]


def pick_local_manager(store: DDict, run_id: str) -> int:
    """A manager on the node this process is running on."""
    local = store.local_managers
    if not local:
        return store.main_manager
    return local[abs(hash(run_id)) % len(local)]


def manager_host(store: DDict, manager: int) -> str:
    """Hostname of the node a manager lives on."""
    try:
        return str(store.manager_nodes[manager].hostname)
    except Exception:  # noqa: BLE001 -- reporting only
        return "unknown"


def this_host() -> str:
    """Hostname of the node this process is running on."""
    return socket.gethostname()


# ----------------------------------------------------------------------------
# Process-local state. This block is what "warm agent" means.
# ----------------------------------------------------------------------------

_attached: dict[bytes, DDict] = {}   # store address -> attached DDict, one per process
_warm: dict[str, Any] = {"version": "", "model": None, "loads": 0, "reuses": 0}   # the surrogate held in this process's memory, plus load/reuse counts
_tasks: dict[str, Any] = {}   # in-flight batch tasks by key (job:<run>, reduce:<run>, fit-<v>-m<n>)
_runs: dict[str, dict[str, Any]] = {}   # per run: its dials and the machine(s) it was pinned to
_members: dict[str, dict[str, Any]] = {}   # finished ensemble members waiting to be assembled


def _ddict(ser: bytes) -> DDict:
    """Attach to the shared DDict, once per process."""
    store = _attached.get(ser)
    if store is None:
        store = DDict.attach(ser)
        _attached[ser] = store
    return store


def active_surrogate(store: DDict) -> dict[str, Any] | None:
    """The current surrogate, from this process's memory when it can be."""
    version = store[K_VERSION]
    if not version:
        return None
    if version == _warm["version"] and _warm["model"] is not None:
        _warm["reuses"] += 1
        return _warm["model"]   # warm hit: from local RAM, no model fetched

    model = store[K_MODEL]      # miss: fetch weights, only when the version changed
    _warm["version"], _warm["model"] = version, model
    _warm["loads"] += 1
    print(f"[info puid={current_process().puid}] warm     loaded {version} into "
          f"this host's memory (load {_warm['loads']}, reuse {_warm['reuses']})",
          flush=True)
    return model


def warm_stats() -> tuple[int, int]:
    """How often the surrogate was loaded, and how often memory answered."""
    return _warm["loads"], _warm["reuses"]


def _log(state: dict[str, Any], msg: str) -> list[str]:
    """Record a line tagged with the PUID of the process that made it."""
    line = f"[node puid={current_process().puid}] {msg}"
    print(line, flush=True)
    return state["log"] + [line]
