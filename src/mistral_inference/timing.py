from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, Optional

import torch


@dataclass
class StageTiming:
    tokenization_ms: float = 0.0
    prefill_ms: float = 0.0
    prefill_tokens: int = 0
    ttft_ms: float = 0.0
    decode_ms: float = 0.0
    decode_tokens: int = 0
    attn_prefill_ms: float = 0.0
    attn_decode_ms: float = 0.0
    moe_dispatch_combine_decode_ms: float = 0.0
    moe_expert_gemm_decode_ms: float = 0.0

    @property
    def prefill_tps(self) -> float:
        if self.prefill_ms <= 0:
            return 0.0
        return self.prefill_tokens / (self.prefill_ms / 1000.0)

    @property
    def decode_tps(self) -> float:
        if self.decode_ms <= 0:
            return 0.0
        return self.decode_tokens / (self.decode_ms / 1000.0)


_CURRENT_TIMING: ContextVar[Optional[StageTiming]] = ContextVar("_CURRENT_TIMING", default=None)
_CURRENT_PHASE: ContextVar[str] = ContextVar("_CURRENT_PHASE", default="")


def get_current_timing() -> Optional[StageTiming]:
    return _CURRENT_TIMING.get()


def get_current_phase() -> str:
    return _CURRENT_PHASE.get()


@contextmanager
def timing_scope(timing: Optional[StageTiming], phase: Optional[str] = None) -> Iterator[None]:
    timing_token = _CURRENT_TIMING.set(timing)
    phase_token = _CURRENT_PHASE.set(phase if phase is not None else _CURRENT_PHASE.get())
    try:
        yield
    finally:
        _CURRENT_TIMING.reset(timing_token)
        _CURRENT_PHASE.reset(phase_token)


def measure_gpu_ms_if_available(fn):
    """
    Measure callable duration in ms.
    - Uses CUDA events on GPU tensors (accurate kernel time).
    - Falls back to wall-clock on CPU.
    Returns (result, elapsed_ms).
    """
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        return result, float(start.elapsed_time(end))
    start_s = perf_counter()
    result = fn()
    elapsed_ms = (perf_counter() - start_s) * 1000.0
    return result, elapsed_ms
