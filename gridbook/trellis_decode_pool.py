"""One streamed-decode scratch buffer per device, shared by every trellis layer.

A ``streamed`` trellis layer decodes its body into a scratch buffer and consumes
it in the *same* forward, so the contents never have to survive past that
layer's GEMM, and vLLM executes layers sequentially -- inside a graph capture as
well as outside it. Giving every layer its own decode target therefore
multiplies the largest decoded tile by the layer count for no benefit.

That is not a micro-optimisation, it is the mode's whole reason to exist. With
a per-layer target, ``streamed`` held the wire *plus* the prepared clone *plus*
a full decoded tile per layer -- strictly MORE resident memory than ``resident``
mode, which keeps only the tile. A mode whose stated purpose is a smaller
footprint must actually have one, so the pool is part of the contract, not a
tuning knob.

Both halves of that fix live here and in the lanes' finalize hooks:

* the lanes ``del layer.trellis_payload`` in streamed mode too -- the prepared
  wire is a private device clone (``prepare_wire_cuda`` clones every tensor), so
  keeping the parameter as well stored the wire twice;
* this pool holds exactly one buffer per device, sized to the largest tile.

Every reservation happens at load time, so the storage address is fixed before
any capture (``docs/KERNELS.md`` CUDA-graph safety rule 3). Growing the pool
re-slices the buffers already handed out, which is the only reason the clients
are tracked -- a layer finalized before a larger one must not keep a view into
freed storage.
"""
from __future__ import annotations

import weakref
from typing import Dict, List, Tuple

import torch

__all__ = ["reserve", "pool_bytes", "reset_for_tests"]


class _Pool:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.storage: torch.Tensor | None = None
        # WEAK refs: a module-level strong reference to every layer would
        # keep a whole unloaded model alive.
        self.clients: List[Tuple[weakref.ref, int, int]] = []

    def reserve(self, layer, rows: int, columns: int) -> None:
        need = int(rows) * int(columns)
        self.clients = [c for c in self.clients if c[0]() is not None]
        self.clients.append((weakref.ref(layer), int(rows), int(columns)))
        if self.storage is None or self.storage.numel() < need:
            self.storage = torch.empty(need, dtype=torch.uint8,
                                       device=self.device)
            for ref, r, c in self.clients:
                client = ref()
                if client is not None:
                    _bind(client, self.storage, r, c)
        else:
            _bind(layer, self.storage, rows, columns)


def _bind(layer, storage: torch.Tensor, rows: int, columns: int) -> None:
    # A prefix slice of a 1-D buffer reshaped is contiguous, which is what
    # ``decode_native_packed_out`` requires of its target.
    view = storage[: rows * columns].view(rows, columns)
    layer.register_buffer("decode_buf", view, persistent=False)


_POOLS: Dict[torch.device, _Pool] = {}


def reserve(layer, rows: int, columns: int, device) -> None:
    """Bind ``layer.decode_buf`` to a ``[rows, columns]`` uint8 view of the pool.

    Call once per streamed layer at finalize. Safe to call for layers of
    different sizes and in any order: a reservation larger than the current
    storage grows it and re-binds every earlier client.
    """
    device = torch.device(device)
    pool = _POOLS.get(device)
    if pool is None:
        pool = _Pool(device)
        _POOLS[device] = pool
    pool.reserve(layer, rows, columns)


def pool_bytes(device) -> int:
    """Total scratch bytes held for ``device`` -- one tile, not one per layer."""
    pool = _POOLS.get(torch.device(device))
    return 0 if pool is None or pool.storage is None else pool.storage.numel()


def reset_for_tests() -> None:
    """Drop every pool. Tests only; production reserves once per load."""
    _POOLS.clear()
