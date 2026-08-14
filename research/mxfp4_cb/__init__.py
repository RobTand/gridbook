"""MXFP4-CB research prototype — CPU-only, not Gridbook.

Exports the small surface needed by tests.
"""
from .format import Mxfp4CbFormat, type_size_for_k, effective_bpw, SUPERBLOCK
from .e2m1 import (
    e2m1_encode,
    e2m1_decode,
    e2m1_nibbles_to_packed,
    e2m1_packed_to_nibbles,
    e8m0_encode_amax,
    e8m0_decode,
    quantize_mxfp4_block,
    dequant_mxfp4_block,
)
from .codec import encode_mxfp4_cb, decode_mxfp4_cb, pack_indices, unpack_indices

__all__ = [
    "Mxfp4CbFormat",
    "type_size_for_k",
    "effective_bpw",
    "SUPERBLOCK",
    "e2m1_encode",
    "e2m1_decode",
    "e2m1_nibbles_to_packed",
    "e2m1_packed_to_nibbles",
    "e8m0_encode_amax",
    "e8m0_decode",
    "quantize_mxfp4_block",
    "dequant_mxfp4_block",
    "encode_mxfp4_cb",
    "decode_mxfp4_cb",
    "pack_indices",
    "unpack_indices",
]
