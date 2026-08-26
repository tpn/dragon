"""
Train a small model that predicts what a slow simulation would have produced.

One simulation run takes minutes of compute on many CPUs. We want answers for
thousands of different settings, which we cannot afford to run. So we run a
limited number of real simulations, train a small model on those results, and
then ask the small model instead.

The program picks its own next simulations. After each round it looks at what it
has learned so far and runs the settings it is least sure about.

The loop, eight steps:

    brief -> propose -> simulate -> collect -> analyze -> train -> decide -> publish
               ^                       ^                             |
               |                       +--- drain in-flight work ----+
               +---------------------- keep going --------------------+

    brief     read the request written in plain English, turn it into settings
    propose   choose which simulation settings to run next
    simulate  start those simulation runs, do not wait for them to finish
    collect   pick up finished runs, turn each big result into one number
    analyze   ask the LLM which part of the setting range to try next
    train     train the small model on every result measured so far
    decide    keep going, drain work still running, or stop and hand over the model
    publish   save the model, then answer many questions with it

After decide, one of three edges is taken (see next_step):
    keep going            -> propose : budget left, so start another wave
    drain in-flight work  -> collect : budget spent, but runs from earlier waves
                                       are still finishing; go pick them up and
                                       train on them one more time
    stop                  -> publish : a stopping rule fired, hand over the model

Words used throughout this file:

    surrogate   the small trained model that predicts a simulation result
    dials       the two settings of one run: how fast the object moves through
                the grid, and how big the object is
    quantity    the single number we keep from a finished run. It measures how
                much detail the simulation had to add to follow the object
    ensemble    three small models trained on the same data. Where they disagree,
                we do not have enough measurements yet
    wave        one trip around the loop
    DDict       a key-value store shared by every machine in the job
    reduce      turn one run's large result into one number, on the machine that
                already holds that result, so the data does not travel
    held-out    measured runs kept out of training, used to score the model
    steer       a range of speed and size that the LLM wants the next runs to use

Where Dragon shows up in this file (the integration is small and contained):

    DragonExecutor    build_graph / main -- runs each LangGraph node as a placed
                      process; the graph and its edges are unchanged
    DDict             main -- the shared data plane; each machine's piece lives in
                      that machine's memory, so a result can stay where it was made
    Policy+batch.job  simulate_node -- pin each MPI run to a machine so its
                      200-300 MB result is written there (data locality)
    batch.process     collect_node -- read that result in place and return only a
                      ~1 KB summary; the large data never crosses the network
    batch.function    train_node -- fit the three models at once on free machines
    Dragon inference  main / the LLM nodes -- serve the LLM on the allocation's own
                      GPUs; every model call stays inside the job

Run inside an allocation of several machines. Do not pass -s: that option forces
everything onto a single machine.

    dragon campaign.py --llm <model-path> --gpus 1 --bin <miniAMR.x>
    dragon campaign.py --bin <miniAMR.x>       # no GPU; the LLM steps are skipped

The simulation is miniAMR. It must be built with the DDict hook described in
miniamr/miniamr_ddict.h, so that it writes its results into the shared store
instead of to files. Pass the built program with --bin.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from functools import partial
from typing import Any, TypedDict

import dragon  # noqa: F401  -- must precede multiprocessing
import multiprocessing as mp

from dragon.ai.langgraph import DragonExecutor, get_llm
from dragon.data.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node
from dragon.native.process import ProcessTemplate, current as current_process
from dragon.workflows.batch import Batch, TaskNotReadyError
from langgraph.graph import END, START, StateGraph

import physics
import reduce_worker
from lab_internals import (
    CANDIDATE_POOL, CHECK_QUERIES, COLLECT_ENOUGH, DDICT_MEM_PER_NODE,
    DEFAULT_BUDGET,
    DEMAND_QUERIES, DRAIN_POLL, DRAIN_WAVES, EXPLORE_PER_WAVE, JOB_NODES,
    JOB_NPX, JOB_NPY,
    JOB_NPZ, MAX_WAVES, MESH_MEM_PER_NODE, MIN_NEW_POINTS, MIN_POINTS_TO_STOP,
    MIN_VALIDATE_TO_STOP,
    RUNS_PER_WAVE, SCOUT_RUNS, TARGET_FRACTION,
    WAVE_WAIT,
    K_BRIEF, K_MODEL, K_TRAIN, K_VERSION,
    _budget_spent, _ddict, _log, active_surrogate, allocation_hostnames,
    candidate_dials, candidate_dials_biased, clamp_region, default_brief,
    evenly_spread_dials, pick_job_nodes, warm_stats, _members, _runs, _tasks,
)
from surrogate import ENSEMBLE_SIZE, assemble, predict, save_surrogate, train_member


# English request. The LLM turns this into budget, bounds, and a 10% target.
DEFAULT_BRIEF = (
    "We are studying how hard our adaptive mesh has to refine as an object moves "
    "through the domain. Sweep the object's speed and size. We can afford about "
    "a hundred solver runs. The surrogate is useful if it lands within about ten "
    "percent of the real refinement footprint."
)


# ----------------------------------------------------------------------------
# 1. Training one small model. Dragon runs three of these at the same time.
# ----------------------------------------------------------------------------


def fit_member(training_pairs: list[dict[str, Any]], seed: int,
               member: int, t0: float, wave: int) -> dict[str, Any]:
    """Train one of the three small models.

    All it needs is the list of measured runs, which is a few kilobytes, so
    Dragon can put this on whichever machine happens to be free. `wave` is the
    wave whose results this trains on; because training is submitted without
    waiting, these lines print under the *next* wave's header -- that is the
    training-and-simulation overlap, made visible.
    """
    import socket
    print(f"[batch] t+{time.time() - t0:5.1f}s  wave {wave} fit member {member}: "
          f"started ({socket.gethostname()})", flush=True)
    result = train_member(training_pairs, seed, member)
    print(f"[batch] t+{time.time() - t0:5.1f}s  wave {wave} fit member {member}: done",
          flush=True)
    return result


# ----------------------------------------------------------------------------
# 2. The campaign state. Every step reads this and returns the parts it changed.
# ----------------------------------------------------------------------------


class Campaign(TypedDict):
    wave: int                               # which trip around the loop we are on
    started_at: float
    waves_run: int
    launch_wave: int                        # the wave whose results the current training uses
    budget: int                             # most simulation runs we are allowed
    target_error: float                     # 0.10 means "usually within 10%"
    selected_runs: list[dict[str, Any]]     # settings just chosen: {id, dials}
    running_jobs: list[str]                 # simulations still running
    reducing: list[str]                     # finished runs being turned into one number
    training_members: list[str]             # small models still being trained
    snapshot_key: str                       # the data the current training used
    next_model_version: str
    fitted_at_n_points: int                 # how many runs the last training used
    n_training_points: int                  # measured runs available now
    sims_run: int                           # simulations started so far
    model_version: str
    validate_error: float | None            # how far off the model is on held-out runs
    error_n_validate: int                   # how many held-out runs that came from
    mesh_bytes_read: int                    # result bytes read without moving them
    same_node_reads: int                    # reduce ran on the machine that had the data
    cross_node_reads: int                   # reduce ran somewhere else
    invalid_runs: int                       # ran, but the result was unusable
    missing_runs: int                       # reduce finished but wrote no result
    steering_hint: str                      # the LLM's one-line reason for its choice
    steer_region: list[list[float]]         # speed/size range the LLM wants tried next
    stop_decision: str                      # "" or "publish" if the LLM says stop early
    error_history: list[float]              # the error after each retraining, in order
    warm_loads: int                         # times the model was copied into memory
    warm_reuses: int                        # times the copy already in memory was used
    published: dict[str, Any]
    log: list[str]


# ----------------------------------------------------------------------------
# 3. The three things we ask the LLM to do.
#    Turn the English request into settings, choose where to look next, and say
#    whether to stop. Each answer must come back as JSON so we can act on it.
# ----------------------------------------------------------------------------

_BRIEF_SYSTEM = (
    "Turn a scientist's plain-English description into settings for a "
    "surrogate-training run over an adaptive-mesh simulation.\n"
    "bounds is [[speed_low, speed_high], [size_low, size_high]] inside the "
    "domain you are given.\n"
    "sim_budget is how many solver runs they can afford.\n"
    "target_fraction is the tolerance they will accept, as a FRACTION of the "
    "typical result (e.g. 0.02 for 'within two percent'). The campaign turns it "
    "into an absolute target once it has measured a few runs.\n"
    "Reply with JSON only."
)
_BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "bounds": {"type": "array", "items": {"type": "array",
                   "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                   "minItems": 2, "maxItems": 2},
        "sim_budget": {"type": "integer"},
        "target_fraction": {"type": "number"},
    },
    "required": ["bounds", "sim_budget", "target_fraction"],
    "additionalProperties": False,
}

_TRIAGE_SYSTEM = (
    "You are triaging results from an adaptive-mesh solver campaign. You are "
    "given the newest runs (each with a validity flag, the quantity, and its "
    "speed/size dials), the design-space bounds, and how many runs failed. Pick "
    "the sub-region of speed and size where the next runs should focus -- where "
    "the space is least explored or where runs failed. Stay inside the bounds. "
    "Reply with JSON only: {\"speed\": [low, high], \"size\": [low, high], "
    "\"reason\": \"...\"}."
)
_TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "speed": {"type": "array", "items": {"type": "number"},
                  "minItems": 2, "maxItems": 2},
        "size": {"type": "array", "items": {"type": "number"},
                 "minItems": 2, "maxItems": 2},
        "reason": {"type": "string"},
    },
    "required": ["speed", "size", "reason"], "additionalProperties": False,
}

_DECIDE_SYSTEM = (
    "You decide whether an active-learning campaign should run more simulations "
    "or stop now and publish the surrogate. You are given the held-out relative "
    "error after each retrain (a fraction, e.g. 0.02 = 2%; oldest first, newest "
    "last), the target error, how many solver runs are left, and how many waves "
    "have run. Choose publish if the error has reached the target or has clearly "
    "plateaued so more runs are not worth the cost; otherwise choose refine. "
    "Reply with JSON only: {\"action\": \"refine\"|\"publish\", \"reason\": \"...\"}."
)
_DECIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["refine", "publish"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"], "additionalProperties": False,
}


def _start_inference(model_name: str, num_gpus: int, max_tokens: int,
                     temperature: float):
    """Start serving the LLM on the allocation's own GPUs.

    The model runs here, next to the simulation, so no request goes out to an
    outside service. Returns the queue used to send it work, and the service
    itself so it can be shut down at the end.
    """
    from dragon.ai.inference.config import (
        BatchingConfig, HardwareConfig, InferenceConfig, ModelConfig,
    )
    from dragon.ai.inference.inference_utils import Inference
    from dragon.native.queue import Queue

    queue = Queue(maxsize=256)
    config = InferenceConfig(
        model=ModelConfig(model_name=model_name,
                          hf_token=os.environ.get("HF_TOKEN", ""), tp_size=1,
                          max_tokens=max_tokens, max_model_len=max_tokens * 8,
                          temperature=temperature),
        hardware=HardwareConfig(num_nodes=1, num_gpus=max(1, num_gpus),
                                num_inf_workers_per_cpu=1),
        batching=BatchingConfig(enabled=True, batch_type="dynamic",
                                batch_wait_seconds=0.05, max_batch_size=16),
    )
    service = Inference(config=config, input_queue=queue)
    service.initialize()
    return queue, service


async def _llm_json(inference_queue, system: str, user: dict[str, Any],
                    schema: dict[str, Any]) -> dict[str, Any] | None:
    """Ask the LLM one question and read its answer as JSON.

    Returns None if the LLM is unavailable or its reply cannot be read. Every
    caller must work without an answer, so a bad reply never stops the run.
    """
    try:
        raw = await get_llm(inference_queue).chat(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps(user)}],
            json_schema=schema,
        )
        # Models often wrap JSON in extra text, so take what is between the
        # first { and the last }.
        text = str(raw)
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1]) if start != -1 else None
    except Exception as exc:  # noqa: BLE001
        print(f"[warn puid={current_process().puid}] LLM unusable ({exc})", flush=True)
        return None


# ----------------------------------------------------------------------------
# 4. The eight steps of the loop.
# ----------------------------------------------------------------------------


async def brief_node(state: Campaign, *, ddict_ser: bytes, brief_text: str,
                     inference_queue: Any = None) -> dict[str, Any]:
    """Turn the request written in English into numbers the program can use.

    Without an LLM this uses the built-in defaults instead.
    """
    store = _ddict(ddict_ser)
    settings, source = None, "defaults"
    if inference_queue is not None:
        reply = await _llm_json(inference_queue, _BRIEF_SYSTEM,
                                {"description": brief_text,
                                 "speed_range": list(physics.SPEED_RANGE),
                                 "size_range": list(physics.SIZE_RANGE),
                                 "default_target_fraction": TARGET_FRACTION},
                                _BRIEF_SCHEMA)
        if reply and "bounds" in reply:
            settings, source = reply, "llm"
    if settings is None:
        settings = default_brief()
    store[K_BRIEF] = settings
    lines = _log(state, f"brief    read the campaign brief ({source})")
    # target_error 0.10 means the model is good enough when its guesses are
    # usually within 10% of what the simulation would have produced.
    return {"budget": int(settings.get("sim_budget", DEFAULT_BUDGET)),
            "target_error": float(settings.get("target_fraction", TARGET_FRACTION)),
            "log": lines}


def propose_node(state: Campaign, *, ddict_ser: bytes, seed: int) -> dict[str, Any]:
    """Choose which simulation settings to run next.

    Two things guide the choice: where the three small models disagree with each
    other, and the range the LLM suggested in the analyze step.

    Once a surrogate exists, each wave draws a pool of CANDIDATE_POOL (400)
    random [speed, size] points, asks the three models to predict each, and
    launches RUNS_PER_WAVE (4): the (RUNS_PER_WAVE - EXPLORE_PER_WAVE) == 2 the
    models disagree on most, plus EXPLORE_PER_WAVE (2) spread evenly across the
    box so the search keeps exploring instead of piling into one spot.

    The first wave is the exception: no surrogate yet, so it just spreads
    SCOUT_RUNS (4) points across the whole box and skips the 400-point pool and
    the disagreement step.
    """
    store = _ddict(ddict_ser)
    # The campaign settings (bounds, budget, target); written once at startup in
    # brief_node and never changed after, so it is fixed for the whole run.
    brief = store[K_BRIEF]
    # Every candidate is one pair of dials, [speed, size]. Both stay inside
    # brief["bounds"] == [[speed_low, speed_high], [size_low, size_high]].
    budget_left = max(0, state["budget"] - state["sims_run"])

    # How far off the model is on runs it was never trained on. Fewer than three
    # such runs is not enough to trust the number, so we print how many it used.
    if state["validate_error"] is None:
        error = "no test points yet"
    elif state["error_n_validate"] < 3:
        n = state["error_n_validate"]
        error = f"{100 * state['validate_error']:.0f}% (rough, from {n} test point{'s' if n != 1 else ''})"
    else:
        error = f"{100 * state['validate_error']:.0f}%"
    target = ("--" if state["target_error"] <= 0
              else f"{100 * state['target_error']:.0f}%")
    print(f"\n{'=' * 16} wave {state['wave']} {'=' * 16}  "
          f"t+{time.time() - state['started_at']:.1f}s, "
          f"{state['n_training_points']} point(s), surrogate "
          f"{state['model_version']}, error {error} "
          f"(target {target})", flush=True)
    # The speed/size sub-region the LLM asked for last analyze step, or [] if none.
    region = state["steer_region"]
    if region:
        print(f"[info puid={current_process().puid}] steer    focus speed "
              f"[{region[0][0]:.2f}, {region[0][1]:.2f}] size "
              f"[{region[1][0]:.2f}, {region[1][1]:.2f}]"
              f"{': ' + state['steering_hint'] if state['steering_hint'] else ''}",
              flush=True)

    # Warm state: this node's process is long-lived, so it keeps the trained
    # model in memory between waves (active_surrogate) and reads only a short
    # version string from the store, instead of reloading the model each round.
    model = active_surrogate(store)
    # Per-wave seed: each wave draws different random points.
    wave_seed = seed + state["wave"]
    if model is None:
        # Nothing trained yet, so nothing to learn from. Spread the first runs
        # evenly over the whole range of speeds and sizes.
        n = min(SCOUT_RUNS if state["sims_run"] == 0 else RUNS_PER_WAVE, budget_left)
        # Each entry is a [speed, size] pair laid on a grid over the box.
        chosen = evenly_spread_dials(n, brief["bounds"], wave_seed)
        note = f"{len(chosen)} spread across the box, nothing learned yet"
    else:
        n = min(RUNS_PER_WAVE, budget_left)
        n_explore = min(EXPLORE_PER_WAVE, n)
        # Build a big pool of [speed, size] pairs to score. If the LLM steered
        # us to a sub-region, draw half the pool from there; otherwise draw from
        # the whole speed/size box.
        if region:
            pool = candidate_dials_biased(CANDIDATE_POOL, brief["bounds"], region,
                                          wave_seed)
            where = "in the steered region, "
        else:
            pool = candidate_dials(CANDIDATE_POOL, brief["bounds"], wave_seed)
            where = ""
        # This is the active-learning choice, and it is made by the models, not
        # the LLM: ask all three about many possible settings, then simulate the
        # ones they answered most differently. Disagreement means they are
        # guessing there, so a real run teaches us the most.
        # disagreement[k] is how far the three models spread on pool[k].
        _, disagreement = predict(model, pool)
        # Sort the [speed, size] pairs by disagreement, most first, keep the top.
        ranked = sorted(zip(pool, disagreement), key=lambda p: p[1], reverse=True)
        chosen = [d for d, _ in ranked[:n - n_explore]]
        # Always keep a couple of runs spread out. Otherwise every run piles into
        # one corner and the model never sees the rest of the range.
        spread = evenly_spread_dials(n_explore, brief["bounds"], wave_seed)
        chosen += spread
        note = (f"{n - n_explore} {where}where the ensemble disagrees most, "
                f"{len(spread)} spread across the box")

    # Tag each [speed, size] pair with a run id, w<wave>-p<i>, for tracking.
    selected = [{"id": f"w{state['wave']}-p{i}", "dials": d}
                for i, d in enumerate(chosen)]
    loads, reuses = warm_stats()
    return {
        "selected_runs": selected,
        "waves_run": state["waves_run"] + (1 if selected else 0),
        # Record this wave now; the training it feeds is submitted a few nodes
        # later, after collect has already advanced state["wave"].
        "launch_wave": state["wave"],
        "warm_loads": loads, "warm_reuses": reuses,
        "log": _log(state, f"propose  {note}, {budget_left} run(s) left"),
    }


def simulate_node(state: Campaign, *, batch: Batch, mesh_ser: bytes,
                  bin_path: str) -> dict[str, Any]:
    """Start one simulation per chosen setting, then return without waiting.

    Each run is tied to one machine. The run writes its large result into the
    shared store on that same machine, so the data never crosses the network.
    Here we only record which machine each run went to.
    """
    all_nodes = allocation_hostnames()   # every machine in the allocation, in order
    running = list(state["running_jobs"])   # ongoing simulations from earlier waves
    for k, item in enumerate(state["selected_runs"]):   # each [speed, size] propose picked
        run_id = item["id"]   # its label, w<wave>-p<i>
        nodes = pick_job_nodes(all_nodes, state["sims_run"] + k, JOB_NODES)   # which machine(s) this run uses
        argv = physics.build_argv(item["dials"], JOB_NPX, JOB_NPY, JOB_NPZ)   # miniAMR command-line for these dials
        ranks = physics.ranks_for(JOB_NPX, JOB_NPY, JOB_NPZ)   # MPI rank count (npx*npy*npz)
        # MINIAMR_DDICT tells the simulation where to write its result.
        # DEVNULL drops the simulation's own progress printing, which is long and
        # would bury the campaign's output.
        template = ProcessTemplate(
            target=bin_path, args=tuple(argv),
            env={"MINIAMR_DDICT": _as_str(mesh_ser), "MINIAMR_RUN": run_id},
            stdout=ProcessTemplate.DEVNULL,
            # Data locality: pin the run to one machine (a HOST_NAME Policy) so its
            # 200-300 MB result is written into the store on that same machine,
            # ready to be read there instead of shipped across the network.
            policy=Policy(placement=Policy.Placement.HOST_NAME,
                          host_name=nodes[0]) if nodes else None,
        )
        # batch.job launches the placed MPI run and returns at once -- it does not
        # wait for the run to finish, so waves stay in flight and the graph moves
        # on. Defaults to a Cray machine; elsewhere pass pmi=PMIBackend.PMIX, or
        # pmi=None to turn MPI startup support off.
        _tasks[f"job:{run_id}"] = batch.job([(ranks, template)])
        _runs[run_id] = {"dials": item["dials"], "nodes": nodes}
        running.append(run_id)
        print(f"[node puid={current_process().puid}] simulate {run_id} -> "
              f"{nodes[0] if nodes else '?'} ({ranks} ranks)", flush=True)

    return {
        "running_jobs": running,
        "sims_run": state["sims_run"] + len(state["selected_runs"]),
        "log": _log(state, f"simulate launched {len(state['selected_runs'])} "
                           f"miniAMR job(s), {len(running)} running"),
    }


async def collect_node(state: Campaign, *, batch: Batch, ddict_ser: bytes,
                       mesh_ser: bytes) -> dict[str, Any]:
    """Pick up finished runs and turn each large result into one number.

    The reading is started on the machine that already holds the result, so the
    hundreds of megabytes stay where they are and only one number comes back.

    This never waits for a particular run. It moves on as soon as roughly as many
    results are in as were started, so the next round can begin while slower runs
    are still going.
    """
    store = _ddict(ddict_ser)
    running = list(state["running_jobs"])
    reducing = list(state["reducing"])
    gathered: list[dict[str, Any]] = []
    invalid = same = cross = missing = 0
    mesh_read = 0
    # The reduce reads each rank's blocks by exact key, so it needs the rank count.
    ranks = physics.ranks_for(JOB_NPX, JOB_NPY, JOB_NPZ)

    # Stop waiting after WAVE_WAIT seconds; a merely-slow run is picked up next wave.
    deadline = time.monotonic() + WAVE_WAIT
    # Wait for only COLLECT_ENOUGH results, not the whole wave, so waves overlap.
    want = 0 if _budget_spent(state) else min(COLLECT_ENOUGH, len(running))

    while True:
        for run_id in list(running):
            try:
                _tasks[f"job:{run_id}"].get(block=False)   # finished yet?
            except TaskNotReadyError:
                continue
            _tasks.pop(f"job:{run_id}", None)
            running.remove(run_id)
            # Data locality: read the result where it sits, return only a ~1 KB summary.
            node = (_runs[run_id]["nodes"] or [None])[0]   # the machine that ran this sim
            template = ProcessTemplate(
                target=reduce_worker.reduce,
                args=(run_id, ddict_ser, mesh_ser, physics.MESH_DIMS, ranks),
                policy=Policy(placement=Policy.Placement.HOST_NAME,
                              host_name=node) if node else None,   # pin the reader to that machine
            )
            _tasks[f"reduce:{run_id}"] = batch.process(template)
            reducing.append(run_id)
            # launched wave vs now shows the run outliving its wave: it started a
            # few waves back and only finished now, while newer waves ran.
            print(f"[node puid={current_process().puid}] collect  {run_id} done "
                  f"on {node} (launched wave {_run_wave(run_id)}, now wave "
                  f"{state['wave']}); reading its result on that same machine",
                  flush=True)

        for run_id in list(reducing):
            try:
                _tasks[f"reduce:{run_id}"].get(block=False)   # finished yet?
            except TaskNotReadyError:
                continue
            _tasks.pop(f"reduce:{run_id}", None)
            reducing.remove(run_id)
            desc = store.get(f"desc/{run_id}")
            if desc is None:
                # Reduce finished but wrote no result; counted apart from unusable ones.
                missing += 1
                continue
            mesh_read += desc.get("bytes_read", 0)   # add up mesh bytes read in place (data-locality metric)
            job_node = (_runs[run_id]["nodes"] or [None])[0]
            if desc.get("host") == job_node:
                same += 1
            else:
                cross += 1
            # valid = the run finished with a clean, finite result; invalid = it
            # blew up (NaN/Inf) or wrote no blocks, so it cannot be trained on.
            if desc.get("valid"):
                # One training example: the dials we used and the number the sim produced.
                gathered.append({"dials": _runs[run_id]["dials"],
                                 "quantity": desc["quantity"], "run": run_id})
            else:
                invalid += 1
                print(f"[warn puid={current_process().puid}] collect  {run_id} "
                      f"dropped (launched wave {_run_wave(run_id)}, now wave "
                      f"{state['wave']}): {desc.get('reason', 'unknown')} "
                      f"(dials {_runs[run_id]['dials']}, "
                      f"{desc.get('blocks', 0)} block(s))", flush=True)

        # Stop this visit once enough results are in, nothing is left running, or
        # the deadline passed; otherwise poll again after a short sleep.
        if (len(gathered) >= want or not (running or reducing)
                or time.monotonic() >= deadline):
            break
        await asyncio.sleep(DRAIN_POLL)

    # Append this wave's valid runs to the shared training set for later fits.
    if gathered:
        store[K_TRAIN] = store[K_TRAIN] + gathered

    n_points = len(store[K_TRAIN])
    lines = _log(state, f"collect  {len(gathered)} valid, {invalid} bad, "
                        f"{missing} no-result, {len(running)} still running, "
                        f"{n_points} point(s)")
    if mesh_read:
        lines = _log({"log": lines},
                     f"collect  {mesh_read / 1024**2:.1f} MB of mesh read in "
                     f"place; {same} on-node, {cross} off-node")
    return {
        "wave": state["wave"] + 1,
        "running_jobs": running, "reducing": reducing,
        "n_training_points": n_points,
        "mesh_bytes_read": state["mesh_bytes_read"] + mesh_read,
        "same_node_reads": state["same_node_reads"] + same,
        "cross_node_reads": state["cross_node_reads"] + cross,
        "invalid_runs": state["invalid_runs"] + invalid,
        "missing_runs": state["missing_runs"] + missing,
        "log": lines,
    }


async def analyze_node(state: Campaign, *, ddict_ser: bytes,
                       inference_queue: Any = None) -> dict[str, Any]:
    """Ask the LLM which speed and size range the next runs should use.

    The LLM only sees the small summaries of the last few runs, never the large
    simulation results, so this costs almost nothing and does not slow the loop.

    Without an LLM this does nothing, and the choice of next runs is left to the
    disagreement between the three models.
    """
    if inference_queue is None:
        return {"log": _log(state, "analyze  no LLM; steering on disagreement only")}
    store = _ddict(ddict_ser)
    # The speed/size limits the LLM must keep its suggested region inside.
    bounds = store[K_BRIEF]["bounds"]
    # Only small facts go to the LLM: the last few runs' settings, the number
    # each produced, whether it worked, and how many runs have failed so far.
    # TODO: give analyze a longer memory. Right now it sees only the last
    # RUNS_PER_WAVE runs, so it cannot tell which regions have already been
    # simulated/trained or how those turned out. Pass a summary of the whole
    # history (regions covered so far, their results, and the regions it
    # steered to on earlier waves) so it can avoid re-suggesting places we
    # already know and steer toward genuinely unexplored ones.
    recent = store[K_TRAIN][-RUNS_PER_WAVE:]
    reply = await _llm_json(inference_queue, _TRIAGE_SYSTEM,
                            {"recent_runs": recent, "bounds": bounds,
                             "invalid_so_far": state["invalid_runs"]},
                            _TRIAGE_SCHEMA)
    # No usable answer: clear the range so the next step falls back to using
    # model disagreement on its own.
    if not reply or "speed" not in reply or "size" not in reply:
        return {"steer_region": [], "steering_hint": "",
                "log": _log(state, "analyze  no steer")}
    # The LLM can return a range that is outside the allowed limits or backwards.
    # Trim it to something the simulation can actually run.
    region = clamp_region([reply["speed"], reply["size"]], bounds)
    hint = reply.get("reason", "")
    # The next propose step prefers settings inside this range, so this is how the
    # LLM changes which simulations get run.
    return {"steer_region": region, "steering_hint": hint,
            "log": _log(state, f"analyze  focus speed "
                               f"[{region[0][0]:.2f}, {region[0][1]:.2f}] size "
                               f"[{region[1][0]:.2f}, {region[1][1]:.2f}]"
                               f"{': ' + hint if hint else ''}")}


def train_node(state: Campaign, *, ddict_ser: bytes, batch: Batch,
               seed: int) -> dict[str, Any]:
    """Take in any finished models, then start training the next three.

    This does not sit and wait. If the models are still training it says so and
    the loop comes back here later.
    """
    store = _ddict(ddict_ser)
    out: dict[str, Any] = {}
    lines = state["log"]
    fitting = list(state["training_members"])

    if fitting:   # models from a previous wave are still being trained
        for task_id in list(fitting):   # check each one
            try:
                _members[task_id] = _tasks[task_id].get(block=False)   # done? keep its result
            except TaskNotReadyError:
                continue   # still training, leave it in the list
            _tasks.pop(task_id, None)   # drop the finished task handle
            fitting.remove(task_id)   # and take it off the still-fitting list
        if fitting:   # some are still training, so come back next wave
            return {"training_members": fitting,
                    "log": _log(state, f"train    {len(fitting)} of "
                                       f"{ENSEMBLE_SIZE} member(s) still fitting")}
        # All three are done. Combine them into one model and score it on the
        # runs that were kept out of training.
        model = assemble(store[state["snapshot_key"]], seed, list(_members.values()))   # merge the 3 members, score on held-out runs
        _members.clear()   # done with the finished members
        model["version"] = state["next_model_version"]   # label this build (v<n_points>)
        store[K_MODEL] = model   # promote: this becomes the active surrogate
        store[K_VERSION] = model["version"]   # bump the version other steps watch
        out.update(training_members=[], model_version=model["version"],   # clear the fitting list, record the new version
                   validate_error=model["validate_error"],   # its held-out error
                   error_n_validate=model["n_validate"],   # how many held-out runs that came from
                   error_history=state["error_history"] + [model["validate_error"]])   # append to the error-over-time list
        shown = f"{100 * model['validate_error']:.0f}%"   # error as a percent for the log line
        lines = _log(state, f"train    {model['version']} assembled, error "
                            f"{shown} on {model['n_validate']} held-out point(s)")

    n_points = len(store[K_TRAIN])
    fresh = n_points - state["fitted_at_n_points"]
    merged = {**state, **out}
    # DECIDER 1 (the hard rule, no LLM): train_node itself judges the just-promoted
    # model "good enough". Needs more than a low error -- on a handful of
    # measurements the error can look low by luck, so we also require enough runs
    # in total and enough held-out runs behind the number.
    met = (merged["validate_error"] is not None and merged["target_error"] > 0
           and merged["validate_error"] <= merged["target_error"]
           and merged["n_training_points"] >= MIN_POINTS_TO_STOP
           and merged["error_n_validate"] >= MIN_VALIDATE_TO_STOP)
    # Reasons not to train again: the model is good enough, we have used up the
    # allowed runs, training is still going, or too few new results have arrived
    # to be worth the effort.
    reason = ("target met" if met
              else "budget spent" if _budget_spent(merged)
              else f"{len(fitting)} still fitting" if fitting
              else f"{fresh} new < {MIN_NEW_POINTS}" if fresh < MIN_NEW_POINTS
              else "")
    if reason:
        result = {**out, "log": _log({"log": lines},
                                     f"train    not refitting ({reason})")}
        # When the hard rule (met) fires, train_node -- not the LLM -- sets the
        # publish decision, so the campaign stops even with no LLM in the loop.
        if met:
            result["stop_decision"] = "publish"
        return result

    version = f"v{n_points}"
    snapshot_key = f"train/{version}"
    # Freeze the results used for this training so all three models see the same
    # data even though more results keep arriving.
    store[snapshot_key] = store[K_TRAIN]
    _members.clear()
    ids = []
    # batch.function runs a Python function on the allocation. Submitting all
    # three without waiting trains the ensemble in parallel across free machines,
    # overlapping with simulations from later waves that are still running.
    for member in range(ENSEMBLE_SIZE):
        task_id = f"fit-{version}-m{member}"
        # launch_wave is the wave that produced this data (set in propose). These
        # fits run under the next wave's header -- that gap is the overlap.
        _tasks[task_id] = batch.function(fit_member, store[snapshot_key], seed,
                                         member, state["started_at"],
                                         state["launch_wave"])
        ids.append(task_id)
    return {
        **out, "training_members": ids, "fitted_at_n_points": n_points,
        "snapshot_key": snapshot_key, "next_model_version": version,
        "log": _log({"log": lines}, f"train    submitted {ENSEMBLE_SIZE} "
                                    f"member(s) in parallel on {n_points} point(s)"),
    }


async def decide_node(state: Campaign, *, ddict_ser: bytes,
                      inference_queue: Any = None) -> dict[str, Any]:
    """Let the LLM say whether to keep running simulations or stop now.

    DECIDER 2 (the LLM, optional early stop): unlike train_node's hard rule, this
    looks at the error *history* and can call a stop when the error has stopped
    improving (leveled off), even before the target is hit. It can only make the
    campaign stop sooner, never run longer -- the limit on total simulation runs
    is a fixed rule the LLM cannot override.

    Without an LLM, or before any model has been trained, this does nothing and
    the fixed rule decides on its own.
    """
    if _budget_spent(state) or state["stop_decision"] == "publish":
        return {}   # already stopping, either by the rule or by an earlier answer
    if state["validate_error"] is None or inference_queue is None:
        return {}   # no model to judge yet, or no LLM available
    # The error is measured again after every retraining. With only a few
    # measurements it jumps around, and a drop or rise means nothing yet. Wait
    # until there are five before asking whether it has stopped improving.
    if len(state["error_history"]) < 5:
        return {"log": _log(state, "decide  refine: building error history")}
    reply = await _llm_json(
        inference_queue, _DECIDE_SYSTEM,
        {"error_history": [round(e, 3) for e in state["error_history"]],
         "current_error": round(state["validate_error"], 3),
         "target_error": round(state["target_error"], 3),
         "runs_left": max(0, state["budget"] - state["sims_run"]),
         "waves_run": state["waves_run"]},
        _DECIDE_SCHEMA)
    action = (reply or {}).get("action", "refine")
    reason = (reply or {}).get("reason", "")
    if action == "publish":
        return {"stop_decision": "publish",
                "log": _log(state, f"decide  publish (early): {reason or 'good enough'}")}
    return {"log": _log(state, f"decide  refine: {reason or 'keep going'}")}


async def publish_node(state: Campaign, *, ddict_ser: bytes, out_dir: str,
                       inference_queue: Any = None) -> dict[str, Any]:
    """Save the trained model, write a short note about it, then use it."""
    store = _ddict(ddict_ser)
    model = store[K_MODEL]
    brief = store[K_BRIEF]
    pairs = store[K_TRAIN]
    if model is None:
        return {"log": _log(state, "publish  nothing to publish")}

    # A loud banner so the final result does not get lost among the log lines.
    # Uses '#' (not the wave header's '=') so it does not read as a new wave.
    err = 100 * model["validate_error"]
    tgt = 100 * state["target_error"]
    verdict = "target met" if (tgt > 0 and err <= tgt) else "stopped"
    banner = (f" surrogate {model['version']} published: error {err:.0f}% "
              f"(target {tgt:.0f}%), {verdict} ")
    print(f"\n{'#' * 78}\n{banner:#^78}\n{'#' * 78}\n", flush=True)

    facts = {"n_training_points": len(pairs),
             "held_out_error": round(model["validate_error"], 3),
             "target_error": round(state["target_error"], 3),
             "invalid_runs": state["invalid_runs"]}
    card = "\n".join([f"miniAMR refinement surrogate {model['version']}", "",
                      f"Trained on {len(pairs)} runs, held-out error "
                      f"{facts['held_out_error']} (target {facts['target_error']}).",
                      "", "MEASURED", json.dumps(facts, indent=2)])
    if inference_queue is not None:
        reply = await _llm_json(inference_queue,
                                "Write one plain sentence for the model card's "
                                "'do not trust' line from these numbers. JSON: "
                                "{\"note\": \"...\"}.",
                                facts, {"type": "object",
                                        "properties": {"note": {"type": "string"}},
                                        "required": ["note"],
                                        "additionalProperties": False})
        if reply and reply.get("note"):
            card += f"\n\nDO NOT TRUST\n{reply['note']}"

    model_path, card_path = save_surrogate(out_dir, model, card)
    # The payoff: thousands of answers from the trained model in well under a
    # second, with no further simulation runs.
    queries = candidate_dials(DEMAND_QUERIES, brief["bounds"], 99)
    start = time.perf_counter()
    answers, _ = predict(model, queries)
    seconds = time.perf_counter() - start
    lines = _log(state, f"publish  final error {100 * model['validate_error']:.0f}% "
                        f"(target {100 * state['target_error']:.0f}%)")
    lines = _log({"log": lines}, f"publish  {model_path}")
    lines = _log({"log": lines}, f"publish  answered {len(queries)} design "
                                 f"point(s) in {seconds:.3f}s")
    return {"log": lines,
            "published": {"model": str(model_path), "card": str(card_path),
                          "queries": len(queries), "seconds": seconds,
                          "card_text": card}}


def next_step(state: Campaign) -> str:
    """After decide, choose where the loop goes: round again, wait, or finish.

    This does not decide readiness itself; it just routes on the decision made
    earlier -- by train_node's hard rule, decide_node's LLM, or _budget_spent.
    """
    # If either decider set publish (train_node's rule or the LLM), stop.
    if state["stop_decision"] == "publish":
        return "publish"
    # Simulation runs left, so go round again.
    if not _budget_spent(state):
        return "refine"
    # Budget is used up, so there is nothing new to launch -- but work from earlier
    # waves may still be running (waves overlap, so a run can finish late). Go back
    # to collect and pick up those late results so they still get trained on,
    # unless we have already drained too many times (DRAIN_WAVES).
    if state["running_jobs"] or state["reducing"] or state["training_members"]:
        if state["wave"] - state["waves_run"] < DRAIN_WAVES:
            time.sleep(DRAIN_POLL)
            return "wait"
    return "publish"


# ----------------------------------------------------------------------------
# 5. Putting it together and starting it.
# ----------------------------------------------------------------------------


def _as_str(ser: Any) -> str:
    """The store's address may be bytes or text, but it is passed to the
    simulation as an environment variable, which must be text."""
    return ser.decode() if isinstance(ser, bytes) else str(ser)


def _run_wave(run_id: str) -> int:
    """The wave that launched a run, read back from its id (w<wave>-p<i>)."""
    return int(run_id.split("-")[0][1:])


def build_graph(executor: DragonExecutor):
    """Connect the eight steps into the loop.

    This is an ordinary LangGraph StateGraph. The only Dragon-specific part is
    executor.node(name), which makes each node run as a placed Dragon process
    instead of a call in one process -- the graph and its edges are unchanged.
    """
    g = StateGraph(Campaign)
    for name in ("brief", "propose", "simulate", "collect", "analyze",
                 "train", "decide", "publish"):
        g.add_node(name, executor.node(name))
    g.add_edge(START, "brief")
    g.add_edge("brief", "propose")
    g.add_edge("propose", "simulate")
    g.add_edge("simulate", "collect")
    g.add_edge("collect", "analyze")
    g.add_edge("analyze", "train")
    g.add_edge("train", "decide")
    # The only branch in the loop. next_step picks one of these three.
    g.add_conditional_edges("decide", next_step,
                            {"refine": "propose", "wait": "collect",
                             "publish": "publish"})
    g.add_edge("publish", END)
    return g.compile()


def new_campaign() -> Campaign:
    """The starting state, before anything has been run or learned."""
    return Campaign(
        wave=0, started_at=time.time(), waves_run=0, launch_wave=0,
        budget=DEFAULT_BUDGET,
        target_error=0.0, selected_runs=[], running_jobs=[],
        reducing=[], training_members=[], snapshot_key="",
        next_model_version="", fitted_at_n_points=0, n_training_points=0,
        sims_run=0, model_version="none", validate_error=None,
        error_n_validate=0,
        mesh_bytes_read=0, same_node_reads=0, cross_node_reads=0,
        invalid_runs=0, missing_runs=0, steering_hint="", steer_region=[],
        stop_decision="", error_history=[], warm_loads=0, warm_reuses=0,
        published={}, log=[],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="miniAMR surrogate on Dragon + LangGraph.")
    p.add_argument("--llm", metavar="MODEL", default=None,
                   help="Model to serve on the allocation's own GPUs. "
                        "Omit to run without the LLM steps.")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--brief", default=DEFAULT_BRIEF)
    p.add_argument("--bin", default=physics.MINIAMR_BIN,
                   help="Path to the built miniAMR.x (with miniamr_ddict.c).")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--seed", type=int, default=None,
                   help="Repeat an earlier campaign by passing the seed it printed.")
    return p.parse_args()


async def main(args: argparse.Namespace) -> None:
    seed = args.seed if args.seed is not None else int(time.time()) % 100000
    n_nodes = len(allocation_hostnames()) or 1
    # Two shared DDict stores, one manager per machine -- the shared data plane:
    # any node can reach a store, but each machine's piece lives in that machine's
    # memory.
    # State store: the campaign's own data -- measured runs, the trained model,
    # the settings. Small, ordinary Python objects, free to move between machines.
    state_store = DDict(1, n_nodes, DDICT_MEM_PER_NODE * n_nodes)
    # Result store: the simulation results, 200-300 MB per run, written as plain
    # bytes by miniAMR's C. Each run's result stays in the piece on the machine
    # that produced it -- that node-local placement is what makes the data
    # locality in simulate/collect possible. Sized with headroom (see
    # MESH_MEM_PER_NODE) because a run's mesh is freed just after it is reduced,
    # which can lag the next wave's writes.
    mesh_store = DDict(1, n_nodes, MESH_MEM_PER_NODE * n_nodes)
    batch = inference = inference_queue = None

    try:
        state_store[K_TRAIN] = []
        state_store[K_MODEL] = None
        state_store[K_VERSION] = ""
        state_store[K_BRIEF] = default_brief()
        # The address of each store. Any process can use it to reach the store.
        state_ser = state_store.serialize()
        mesh_ser = mesh_store.serialize()

        batch = Batch()   # can run work on every machine in the allocation
        print(f"[orchestrator] puid={current_process().puid}", flush=True)
        print(f"[orchestrator] {n_nodes} node(s), one DDict manager each",
              flush=True)
        print(f"[orchestrator] seed {seed} (pass --seed {seed} to repeat this run)",
              flush=True)

        if args.llm:
            # Dragon inference service: serve the LLM on the allocation's own
            # GPUs. The LLM nodes reach it through a queue, so every model call
            # stays inside the job -- no external endpoint, no API key.
            print(f"[orchestrator] starting inference for {args.llm} ...", flush=True)
            inference_queue, inference = _start_inference(
                args.llm, args.gpus, args.max_tokens, args.temperature)
            print("[orchestrator] inference ready (in-allocation)", flush=True)

        with DragonExecutor() as executor:
            # The eight steps run in two groups of processes. These five are
            # light: they call the LLM, choose settings, and save files.
            thinkers = {
                "brief": partial(brief_node, ddict_ser=state_ser,
                                 brief_text=args.brief, inference_queue=inference_queue),
                "propose": partial(propose_node, ddict_ser=state_ser, seed=seed),
                "analyze": partial(analyze_node, ddict_ser=state_ser,
                                   inference_queue=inference_queue),
                "decide": partial(decide_node, ddict_ser=state_ser,
                                  inference_queue=inference_queue),
                "publish": partial(publish_node, ddict_ser=state_ser,
                                   out_dir=args.out, inference_queue=inference_queue),
            }
            # These three start and track the heavy work across the machines.
            workers = {
                "simulate": partial(simulate_node, batch=batch, mesh_ser=mesh_ser,
                                    bin_path=args.bin),
                "collect": partial(collect_node, batch=batch, ddict_ser=state_ser,
                                   mesh_ser=mesh_ser),
                "train": partial(train_node, ddict_ser=state_ser, batch=batch,
                                 seed=seed),
            }
            # launch_hosts places each group in its own Dragon host process on the
            # allocation. Splitting light "thinker" nodes from heavy "worker" nodes
            # lets Dragon put each where it belongs.
            hosts = executor.launch_hosts([thinkers, workers])
            # Show which machine each host group landed on (Node maps the host's
            # h_uid to its cluster hostname).
            for label, h in zip(("thinkers", "workers"), hosts):
                print(f"[orchestrator] AgentHost {label}: puid={h.puid} on "
                      f"{Node(h.h_uid).hostname}", flush=True)
            print(flush=True)

            final = await build_graph(executor).ainvoke(
                new_campaign(),
                config={"recursion_limit": (MAX_WAVES + DRAIN_WAVES + 2) * 8})

        published = final.get("published") or {}
        elapsed = time.time() - final["started_at"]
        reads = final["same_node_reads"] + final["cross_node_reads"]

        print("\n=== data stayed where it was produced ===")
        print(f"result data read : {final['mesh_bytes_read'] / 1024**2:.1f} MB "
              f"read on the machine holding it; only one number per run came back")
        print(f"read locally     : {final['same_node_reads']} of {reads} "
              f"read on the same machine that ran the simulation")
        print("why it matters   : sending that much data across the network would "
              "have dominated the run.")
        print("                   The shared store kept each result on its node "
              "and only a one-number")
        print("                   summary ever travelled.")

        print("\n=== runs overlapped instead of queueing ===")
        print(f"simulations      : {final['sims_run']} run(s) across "
              f"{final['waves_run']} round(s) of the loop")
        print(f"total time       : {elapsed:.1f}s")
        print(f"model in memory  : copied in {final['warm_loads']} time(s), "
              f"reused {final['warm_reuses']} time(s) without copying")
        unfinished = len(final["running_jobs"]) + len(final["reducing"])
        print(f"unusable runs    : {final['invalid_runs']} bad result, "
              f"{final['missing_runs']} no result")
        print(f"still going      : {unfinished} not finished when we stopped")
        print("why it matters   : simulations, model training and the LLM ran at "
              "the same time on the")
        print("                   allocation. The loop never waited for one step "
              "to finish before the")
        print("                   next started, and the trained model was reused "
              "from memory, not reloaded.")
        if args.llm:
            print("LLM              : ran on the allocation's own GPUs; "
                  "no data was sent off the machine")

        print("\n=== what came out ===")
        if published:
            print(f"model            : {published['model']}")
            print(f"answered         : {published['queries']} setting(s) in "
                  f"{published['seconds']:.3f}s")
            print("why it matters   : the trained model answers thousands of "
                  "settings in a fraction of a")
            print("                   second; each one as a real simulation is "
                  "minutes on many cores.")
    finally:
        # Always release what we started, even if the run failed partway.
        # Batch is terminated, not joined: when we publish early (target met) some
        # simulations are still running, and the model is already saved, so we
        # force-stop the batch group instead of waiting for work we do not need.
        # Batch first, then the stores it writes to, so no task writes to a
        # destroyed store.
        for label, res, close in (("batch", batch, "terminate"),
                                  ("inference", inference, "destroy"),
                                  ("inference queue", inference_queue, "destroy"),
                                  ("mesh ddict", mesh_store, "destroy"),
                                  ("state ddict", state_store, "destroy")):
            if res is None:
                continue
            try:
                getattr(res, close)()
            except Exception as exc:  # noqa: BLE001
                print(f"[orchestrator] {label} cleanup failed: {exc}", flush=True)


if __name__ == "__main__":
    # Tells Python to start new processes through Dragon, so they can land on any
    # machine in the allocation instead of only this one.
    try:
        mp.set_start_method("dragon")
    except RuntimeError:
        pass
    asyncio.run(main(parse_args()))
