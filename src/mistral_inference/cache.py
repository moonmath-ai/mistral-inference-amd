from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from xformers.ops.fmha.attn_bias import (  # type: ignore
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)


def get_cache_sizes(n_layers: int, max_seq_len: int, sliding_window: Optional[int] | Optional[List[int]]) -> List[int]:
    if sliding_window is None:
        return n_layers * [max_seq_len]
    elif isinstance(sliding_window, int):
        return n_layers * [sliding_window]
    else:
        assert isinstance(sliding_window, list), f"Expected list, got {type(sliding_window)}"
        assert (
            n_layers % len(sliding_window) == 0
        ), f"Expected n_layers % len(sliding_window) == 0, got {n_layers} % {len(sliding_window)}"
        num_repeats = n_layers // len(sliding_window)
        return num_repeats * [w if w is not None else max_seq_len for w in sliding_window]


@dataclass
class CacheInputMetadata:
    # # rope absolute positions
    # positions: torch.Tensor
    # # where tokens should go in the cache
    # cache_positions: torch.Tensor

    # # if prefill, use block diagonal causal mask
    # # else use causal with padded key mask
    # prefill: bool
    # mask: AttentionBias
    # seqlens: List[int]
    # rope absolute positions
    positions: torch.Tensor
    # which elements in the sequences need to be cached
    to_cache_mask: torch.Tensor
    # how many elements are cached per sequence
    cached_elements: torch.Tensor
    # where tokens should go in the cache
    cache_positions: torch.Tensor
    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask (or tensor for decode, Phase 5.5)
    prefill: bool
    mask: Union[AttentionBias, torch.Tensor]
    seqlens: List[int]


class DecodeMetadataBuffers:
    """
    Persistent device buffers for decode metadata (Phase 5.3).
    Same storage every step so graph replay can read updated values.
    """

    def __init__(self, n_layers: int, max_batch_size: int, device: torch.device) -> None:
        self.n_layers = n_layers
        self.max_batch_size = max_batch_size
        self.device = device
        self.positions_buf = torch.empty(
            (n_layers, max_batch_size), device=device, dtype=torch.long
        )
        self.to_cache_mask_buf = torch.empty(
            (n_layers, max_batch_size), device=device, dtype=torch.bool
        )
        self.cached_elements_buf = torch.empty(
            (n_layers, max_batch_size), device=device, dtype=torch.long
        )
        self.cache_positions_buf = torch.empty(
            (n_layers, max_batch_size), device=device, dtype=torch.long
        )

    def update_from_metadata(self, metadata: List[CacheInputMetadata], B: int) -> None:
        """Copy tensor fields from metadata into persistent buffers."""
        assert len(metadata) == self.n_layers and B <= self.max_batch_size
        for i, m in enumerate(metadata):
            self.positions_buf[i, :B].copy_(m.positions)
            self.to_cache_mask_buf[i, :B].copy_(m.to_cache_mask)
            self.cached_elements_buf[i, :B].copy_(m.cached_elements)
            self.cache_positions_buf[i, :B].copy_(m.cache_positions)

    def get_metadata_list(
        self, B: int, source_metadata: List[CacheInputMetadata], seqlens: List[int]
    ) -> List[CacheInputMetadata]:
        """Return CacheInputMetadata list that uses buffer views; masks from source_metadata."""
        assert len(source_metadata) == self.n_layers and B <= self.max_batch_size
        out: List[CacheInputMetadata] = []
        for i in range(self.n_layers):
            m = source_metadata[i]
            out.append(
                CacheInputMetadata(
                    positions=self.positions_buf[i, :B],
                    to_cache_mask=self.to_cache_mask_buf[i, :B],
                    cached_elements=self.cached_elements_buf[i, :B],
                    cache_positions=self.cache_positions_buf[i, :B],
                    prefill=m.prefill,
                    mask=m.mask,
                    seqlens=seqlens,
                )
            )
        return out


def _decode_mask_tensor(
    device: torch.device,
    B: int,
    cache_size: int,
    kv_seqlen_for_layer: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build (1, 1, B, k_padded) mask for decode: 0.0 valid, -inf elsewhere. No sync.
    Returns full k_padded so CK kernel gets stride(-2) % 8 == 0; attention layer must pad K/V to k_padded."""
    total_k = B * cache_size
    k_padded = (total_k + 7) // 8 * 8
    offset = torch.arange(B, device=device, dtype=torch.long) * cache_size
    valid_end = offset + kv_seqlen_for_layer
    k = torch.arange(total_k, device=device)
    mask_valid = (k.unsqueeze(0) >= offset.unsqueeze(1)) & (k.unsqueeze(0) < valid_end.unsqueeze(1))
    mask_padded = torch.zeros(1, 1, B, k_padded, device=device, dtype=torch.bool)
    mask_padded[:, :, :, :total_k] = mask_valid.unsqueeze(0).unsqueeze(0)
    neg_inf = torch.finfo(dtype).min if dtype.is_floating_point else float("-inf")
    out = torch.zeros(1, 1, B, k_padded, device=device, dtype=dtype)
    out.masked_fill_(~mask_padded, neg_inf)
    return out


def interleave_list(l1: List[torch.Tensor], l2: List[torch.Tensor]) -> List[torch.Tensor]:
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


def unrotate(cache: torch.Tensor, seqlen: int) -> torch.Tensor:
    assert cache.ndim == 3  # (W, H, D)
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        return torch.cat([cache[position:], cache[:position]], dim=0)


class CacheView:
    def __init__(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        metadata: CacheInputMetadata,
        kv_seqlens: torch.Tensor,
    ):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor) -> None:
        """
        to_cache_mask masks the last [max_seq_len] tokens in each sequence
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)

        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

    def interleave_kv(self, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a naive implementation and not optimized for speed.
        """
        assert xk.ndim == xv.ndim == 3  # (B * T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Make it a list of [(T, H, D)]
        xk: Tuple[torch.Tensor] = torch.split(xk, self.metadata.seqlens)  # type: ignore
        xv: Tuple[torch.Tensor] = torch.split(xv, self.metadata.seqlens)  # type: ignore
        assert len(xk) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        # Order elements in cache by position by unrotating
        cache_k = [unrotate(t, s) for t, s in zip(self.cache_k, self.kv_seqlens)]
        cache_v = [unrotate(t, s) for t, s in zip(self.cache_v, self.kv_seqlens)]

        interleaved_k = interleave_list(cache_k, list(xk))
        interleaved_v = interleave_list(cache_v, list(xv))

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def max_seq_len(self) -> int:
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[: len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[: len(self.kv_seqlens)]

    @property
    def prefill(self) -> bool:
        return self.metadata.prefill

    @property
    def mask(self) -> AttentionBias:
        return self.metadata.mask


class BufferCache:
    """
    This is an example that implements a buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        sliding_window: Optional[int] | Optional[List[int]] = None,
    ):
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_layers = n_layers

        self.cache_sizes: List[int] = get_cache_sizes(n_layers, max_seq_len, sliding_window)
        assert len(self.cache_sizes) == n_layers, f"Expected {n_layers} cache sizes, got {len(self.cache_sizes)}"

        self.cache_k = {}
        self.cache_v = {}
        for i, cache_size in enumerate(self.cache_sizes):
            self.cache_k[i] = torch.empty((max_batch_size, cache_size, n_kv_heads, head_dim))
            self.cache_v[i] = torch.empty((max_batch_size, cache_size, n_kv_heads, head_dim))

        # holds the valid length for each batch element in the cache
        self.kv_seqlens: Optional[torch.Tensor] = None

    def get_view(self, layer_id: int, metadata: CacheInputMetadata) -> CacheView:
        assert self.kv_seqlens is not None
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens)

    def reset(self) -> None:
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int) -> None:
        self.kv_seqlens = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    @property
    def device(self) -> torch.device:
        return self.cache_k[0].device

    def to(self, device: torch.device, dtype: torch.dtype) -> "BufferCache":
        for i in range(self.n_layers):
            self.cache_k[i] = self.cache_k[i].to(device=device, dtype=dtype)
            self.cache_v[i] = self.cache_v[i].to(device=device, dtype=dtype)

        return self

    def update_seqlens(self, seqlens: List[int]) -> None:
        assert self.kv_seqlens is not None
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> List[CacheInputMetadata]:
        """
        input = seqlens [5,7,2] // seqpos [0, 1, 3] // sliding_window 3
        --> only cache last 3 tokens in each sequence
        - to_cache_mask = [0 0 1 1 1 | 0 0 0 0 1 1 1 | 1 1]
        - cached_elements = [3 | 3 | 2]
        --> absolute positions are used for rope
        - positions = [0 1 2 3 4 | 1 2 3 4 5 6 7 | 3 4]
        --> cache positions are positions cache_masked, modulo sliding_window + batch_idx * sliding_window
        - cache_positions = [2 0 1 | 5 3 4 | 6 7]
        """
        metadata: List[CacheInputMetadata] = []

        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))

        assert self.kv_seqlens is not None
        assert len(seqlens) == len(
            self.kv_seqlens
        ), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"
        assert len(seqlens) > 0, seqlens

        dev = self.device
        seqlens_t = torch.tensor(seqlens, device=dev, dtype=torch.long)
        total_len = seqlens_t.sum().item()

        first_prefill = (self.kv_seqlens[0] == 0).item()
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)
        kv_seqlen_lists: Optional[List[List[int]]] = None
        if not first_prefill:
            cache_sizes_t = torch.tensor(self.cache_sizes, device=dev, dtype=torch.long)
            B = len(seqlens)
            if subsequent_prefill:
                kv_clamped = self.kv_seqlens.unsqueeze(0).clamp(max=cache_sizes_t.unsqueeze(1))
                kv_seqlen_2d = seqlens_t.unsqueeze(0) + kv_clamped
            else:
                cached_elements_2d = torch.minimum(
                    seqlens_t.unsqueeze(0), cache_sizes_t.unsqueeze(1)
                )
                kv_seqlen_2d = (
                    self.kv_seqlens.unsqueeze(0) + cached_elements_2d
                ).clamp(max=cache_sizes_t.unsqueeze(1))
            kv_seqlen_lists = kv_seqlen_2d.tolist()

        for layer_idx, cache_size in enumerate(self.cache_sizes):
            metadata.append(
                self._get_input_metadata_layer(
                    cache_size,
                    seqlens,
                    seqlens_t,
                    total_len,
                    self.kv_seqlens,
                    first_prefill,
                    subsequent_prefill,
                    kv_seqlen_lists[layer_idx] if kv_seqlen_lists is not None else None,
                )
            )

        return metadata

    def get_input_metadata_decode(
        self, B: int, use_tensor_mask: bool = False
    ) -> List[CacheInputMetadata]:
        """Decode step: seqlens = [1]*B. use_tensor_mask=False (default): one .tolist() sync, correct output.
        use_tensor_mask=True: no sync (Phase 5.5); mask matches xformers materialized, but on some backends
        (e.g. xformers CK with padded K/V) the attention output can be NaN; keep default for correctness."""
        assert self.kv_seqlens is not None, "cache not initialized"
        assert len(self.kv_seqlens) == B, (len(self.kv_seqlens), B)
        dev = self.device
        seqlens = [1] * B
        seqlens_t = torch.ones(B, device=dev, dtype=torch.long)
        total_len = B
        first_prefill = False
        subsequent_prefill = False
        cache_sizes_t = torch.tensor(self.cache_sizes, device=dev, dtype=torch.long)
        cached_elements_2d = torch.minimum(
            seqlens_t.unsqueeze(0), cache_sizes_t.unsqueeze(1)
        )
        kv_seqlen_2d = (
            self.kv_seqlens.unsqueeze(0) + cached_elements_2d
        ).clamp(max=cache_sizes_t.unsqueeze(1))
        if use_tensor_mask:
            kv_seqlen_lists = None
        else:
            kv_seqlen_lists = kv_seqlen_2d.tolist()
        metadata: List[CacheInputMetadata] = []
        for layer_idx, cache_size in enumerate(self.cache_sizes):
            metadata.append(
                self._get_input_metadata_layer(
                    cache_size,
                    seqlens,
                    seqlens_t,
                    total_len,
                    self.kv_seqlens,
                    first_prefill,
                    subsequent_prefill,
                    kv_seqlen_list_for_layer=kv_seqlen_lists[layer_idx] if kv_seqlen_lists is not None else None,
                    kv_seqlen_for_layer_tensor=kv_seqlen_2d[layer_idx] if use_tensor_mask else None,
                )
            )
        return metadata

    def _get_input_metadata_layer(
        self,
        cache_size: int,
        seqlens: List[int],
        seqlens_t: torch.Tensor,
        total_len: int,
        kv_seqlens_tensor: torch.Tensor,
        first_prefill: bool,
        subsequent_prefill: bool,
        kv_seqlen_list_for_layer: Optional[List[int]],
        kv_seqlen_for_layer_tensor: Optional[torch.Tensor] = None,
    ) -> CacheInputMetadata:
        dev = self.device
        B = len(seqlens)

        segment_offsets = torch.cat(
            [torch.zeros(1, device=dev, dtype=torch.long), torch.cumsum(seqlens_t, 0)[:-1]]
        )
        segment_start_per_token = segment_offsets.repeat_interleave(seqlens_t)
        positions = kv_seqlens_tensor.repeat_interleave(seqlens_t) + (
            torch.arange(total_len, device=dev, dtype=torch.long) - segment_start_per_token
        )

        threshold_per_token = (seqlens_t.repeat_interleave(seqlens_t) - cache_size).clamp(min=0)
        local_pos = torch.arange(total_len, device=dev, dtype=torch.long) - segment_start_per_token
        to_cache_mask = local_pos >= threshold_per_token

        cached_elements = torch.minimum(seqlens_t, torch.tensor(cache_size, device=dev, dtype=torch.long))

        batch_idx = torch.arange(B, device=dev, dtype=torch.long).repeat_interleave(seqlens_t)
        cache_positions = positions % cache_size + batch_idx * cache_size
        cache_positions_masked = cache_positions[to_cache_mask]

        if first_prefill:
            assert (kv_seqlens_tensor == 0).all().item(), "expected all seqpos == 0 on first prefill"
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(cache_size)
        elif subsequent_prefill:
            assert kv_seqlen_list_for_layer is not None
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,
                kv_seqlen=kv_seqlen_list_for_layer,
            ).make_local_attention_from_bottomright(cache_size)
        elif kv_seqlen_for_layer_tensor is not None:
            # Decode with tensor mask (Phase 5.5): no .tolist(), no sync.
            mask = _decode_mask_tensor(
                dev, B, cache_size, kv_seqlen_for_layer_tensor, dtype=self.cache_k[0].dtype
            )
        else:
            assert kv_seqlen_list_for_layer is not None
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=cache_size,
                kv_seqlen=kv_seqlen_list_for_layer,
            )
        return CacheInputMetadata(
            positions=positions,
            to_cache_mask=to_cache_mask,
            cached_elements=cached_elements,
            cache_positions=cache_positions_masked,
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens,
        )
