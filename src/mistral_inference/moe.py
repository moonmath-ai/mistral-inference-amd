import dataclasses
from typing import List
from time import perf_counter

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn

from mistral_inference.timing import get_current_phase, get_current_timing, measure_gpu_ms_if_available


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        timing = get_current_timing()
        phase = get_current_phase()
        timed_decode = timing is not None and phase == "decode"
        use_gpu_events = timed_decode and torch.cuda.is_available() and inputs.is_cuda
        total_start_evt = torch.cuda.Event(enable_timing=True) if use_gpu_events else None
        total_end_evt = torch.cuda.Event(enable_timing=True) if use_gpu_events else None
        total_start_s = perf_counter() if timed_decode and not use_gpu_events else 0.0
        if total_start_evt is not None:
            total_start_evt.record()
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        expert_ms = 0.0
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if timed_decode:
                expert_out, e_ms = measure_gpu_ms_if_available(lambda: expert(inputs[batch_idx]))
                expert_ms += e_ms
            else:
                expert_out = expert(inputs[batch_idx])
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert_out
        if timed_decode and timing is not None:
            if total_end_evt is not None and total_start_evt is not None:
                total_end_evt.record()
                total_end_evt.synchronize()
                total_ms = float(total_start_evt.elapsed_time(total_end_evt))
            else:
                total_ms = (perf_counter() - total_start_s) * 1000.0
            timing.moe_expert_gemm_decode_ms += expert_ms
            timing.moe_dispatch_combine_decode_ms += max(0.0, total_ms - expert_ms)
        return results
