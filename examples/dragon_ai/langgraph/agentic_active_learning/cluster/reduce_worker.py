"""
Turn one simulation's large result into a single number, without moving it.

This is the step that makes the whole campaign affordable. A finished run leaves
200-300 MB in the shared store on the machine it ran on. Dragon starts this
function on that same machine, so it reads those bytes locally and sends back
only a small summary. The large data never crosses the network.

campaign.py starts it with::

    batch.process(ProcessTemplate(target=reduce, policy=HOST_NAME=<machine>))

which is how you run a Python function on a machine of your choosing. A process
started that way reports only whether it succeeded, not a value, so the summary
is written into the shared store under ``desc/<run_id>`` instead of returned.

It takes the addresses of both stores:
  * mesh_ser  -- the results store, holding plain bytes written by the
                 simulation's C code
  * state_ser -- the campaign store, holding ordinary Python objects, where the
                 summary goes
"""

from __future__ import annotations

import socket

from dragon.data.ddict import DDict

import mesh_v2


def reduce(run_id: str, state_ser: bytes, mesh_ser: bytes,
           dims: tuple[int, int, int, int], ranks: int) -> None:
    """Read this machine's share of one run's result and write back a summary.

    Because Dragon started this on the machine that ran the simulation, the
    node-local manager holds that run's blocks. Nothing large leaves the machine.
    """
    mesh = DDict.attach(mesh_ser)
    state = DDict.attach(state_ser)
    try:
        # The simulation's C code wrote plain bytes, so read them as bytes
        # rather than as saved Python objects.
        raw = mesh_v2.raw_bytes_view(mesh)
        # Pin to the node-local manager: the C writer put every block there, so
        # reads AND deletes must target it (the bare store hash-routes elsewhere).
        mstore = raw.manager(raw.local_manager)
        # Name this run's blocks from the writer's per-rank count -- no key
        # listing, which is what streamed keys and raced concurrent deletes.
        block_keys, count_keys = mesh_v2.run_block_keys(mstore, run_id, ranks)
        descriptor = mesh_v2.reduce_local_mesh(mstore, run_id, dims, block_keys)
        descriptor["host"] = socket.gethostname()
        state[f"desc/{run_id}"] = descriptor
        # The run is now one number, so its 200-300 MB of blocks are no longer
        # needed. Free them (and the count keys) by exact key so the space is
        # reused by the next runs instead of piling up until the store fills.
        mesh_v2.free_local_mesh(mstore, block_keys + count_keys)
    finally:
        mesh.detach()
        state.detach()
