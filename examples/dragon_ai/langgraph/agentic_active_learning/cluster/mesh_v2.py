"""
Read a simulation result back out of the shared store, as numpy, without copying.

The simulation's C code (miniAMR/ref/miniamr_ddict.c) writes each active block
as plain float64 bytes, laid out [num_vars, x, y, z] in C order, into the part of
the store on its own machine. This module reads them back.

It is always run on the machine that holds the data. The reader addresses each
block by exact key -- built from a small per-rank count the writer leaves -- so
it never lists the node's keys and the read never touches the network.

Because C wrote plain bytes rather than saved Python objects, the store has to be
told to hand these keys back as bytes. That is what ``raw_bytes_view`` below is
for. The layout is a fixed agreement between the C writer and this reader: if one
changes, the other must change with it.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np


class _RawKeyPickler:
    """Read keys as plain text, exactly as the simulation's C code wrote them.

    Without this the store tries to load the key as a saved Python object and
    fails on the first byte (the 'm' of "mesh/...").
    """

    def dumps(self, key: Any) -> bytes:
        return key.encode("utf-8") if isinstance(key, str) else bytes(key)

    def loads(self, data: bytes) -> str:
        return bytes(data).decode("utf-8")


class _RawValuePickler:
    """Read values as plain float64 bytes, exactly as the C code wrote them."""

    def dump(self, value: Any, file: Any) -> None:
        file.write(bytes(value))

    def load(self, file: Any) -> bytes:
        buf = None
        try:
            while True:
                data = file.read(0)
                if buf is None:
                    buf = bytearray(memoryview(data))
                else:
                    buf.extend(data)
        except EOFError:
            pass
        return bytes(buf) if buf is not None else b""


def raw_bytes_view(store: Any) -> Any:
    """A view of the store that reads C-written keys and values as plain bytes."""
    return store.pickler(_RawKeyPickler(), _RawValuePickler())


def run_block_keys(mstore: Any, run_id: str,
                   ranks: int) -> tuple[list[str], list[str]]:
    """This run's block keys on this node, named directly from the writer's
    manifest instead of listing the node.

    Each rank left a block count under ``mesh/<run>/r<rank>/n`` and wrote its
    blocks under contiguous ``b0..b(n-1)``. Reading those counts (one targeted
    get each) lets us name every block key without calling ``local_keys()`` --
    which streams the whole node's keys back and races concurrent deletes.
    Returns the block keys and the count keys; both are deleted after reducing.
    ``mstore`` is scoped to the node-local manager (see reduce_worker).
    """
    block_keys: list[str] = []
    count_keys: list[str] = []
    for r in range(ranks):
        count_key = f"mesh/{run_id}/r{r}/n"
        try:
            raw = mstore[count_key]
        except KeyError:
            continue   # this rank wrote no blocks on this node
        count_keys.append(count_key)
        n = int(np.frombuffer(raw, dtype=np.int32)[0]) if raw else 0
        block_keys.extend(f"mesh/{run_id}/r{r}/b{i}" for i in range(n))
    return block_keys, count_keys


def free_local_mesh(mstore: Any, keys: list[str]) -> int:
    """Delete this run's blocks (and count keys) from THIS node's store.

    ``mstore`` is scoped to the node-local manager, so each delete lands on the
    manager that actually holds the pinned key. Only targeted deletes are used,
    never a key listing, so nothing is streaming when the deletes happen.
    """
    for key in keys:
        try:
            mstore.pop(key)
        except KeyError:
            pass
    return len(keys)


def read_block(mstore: Any, key: str,
               dims: tuple[int, int, int, int]) -> np.ndarray:
    """One block's field, viewed from the raw bytes. dims = (num_vars, x, y, z)."""
    raw = mstore[key]
    return np.frombuffer(raw, dtype=np.float64).reshape(dims)


def reduce_local_mesh(mstore: Any, run_id: str,
                      dims: tuple[int, int, int, int],
                      keys: list[str]) -> dict[str, Any]:
    """Turn this machine's share of one run's result into a small summary.

    The number we keep is log(1 + block count): how much extra detail the
    simulation had to add to follow the object. Block counts range over several
    orders of magnitude, and taking the log both keeps the value smooth for the
    model to learn and makes a fixed error target mean "within X percent".

    Every byte of the result is still read here, which is what ``bytes_read``
    reports. A run that went unstable leaves NaN or Inf behind and is marked
    invalid so it is not used for training. ``keys`` is the run's block list from
    run_block_keys, and ``mstore`` is scoped to the node-local manager.
    """
    read_bytes = 0
    finite = True
    for key in keys:
        field = read_block(mstore, key, dims)
        read_bytes += field.nbytes
        # Skip the one-cell ghost layer on every side: miniAMR mallocs the block
        # and only fills the interior plus exchanged halos, so boundary ghost
        # cells hold garbage that can read as NaN/Inf. Validity-check the interior.
        interior = field[:, 1:-1, 1:-1, 1:-1]
        if not np.isfinite(interior).all():
            finite = False

    valid = bool(keys) and finite
    reason = ("" if valid
              else "no local blocks" if not keys
              else "field has NaN/Inf -- run blew up")
    return {
        "run": run_id,
        "valid": valid,
        "reason": reason,
        # log of the block count -- footprint spans ~10 to ~5000 blocks, so
        # learning log(blocks) keeps the target smooth and the error relative.
        "quantity": float(np.log1p(len(keys))),
        "blocks": len(keys),
        "bytes_read": read_bytes,
    }
