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
        if timed_decode:
            gate_logits, gate_ms = measure_gpu_ms_if_available(lambda: self.gate(inputs))
            timing.moe_gate_decode_ms += gate_ms
            (weights, selected_experts), topk_ms = measure_gpu_ms_if_available(
                lambda: torch.topk(gate_logits, self.args.num_experts_per_tok)
            )
            timing.moe_topk_decode_ms += topk_ms
            weights, softmax_ms = measure_gpu_ms_if_available(
                lambda: F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
            )
            timing.moe_softmax_decode_ms += softmax_ms
        else:
            gate_logits = self.gate(inputs)
            weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
            weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)

        # Route once in flattened form.
        # We avoid global sort (can be expensive for decode) and still keep
        # combine as index_add_ for efficient accumulation.
        num_tokens = inputs.shape[0]
        k = self.args.num_experts_per_tok
        if timed_decode:
            def _route_extract():
                _flat_experts = selected_experts.reshape(-1)
                _flat_tokens = (
                    torch.arange(num_tokens, device=inputs.device, dtype=torch.long)
                    .unsqueeze(1)
                    .expand(num_tokens, k)
                    .reshape(-1)
                )
                _flat_weights = weights.reshape(-1)
                return _flat_experts, _flat_tokens, _flat_weights

            (flat_experts, flat_tokens, flat_weights), route_ms = measure_gpu_ms_if_available(_route_extract)
            timing.moe_route_extract_decode_ms += route_ms
        else:
            flat_experts = selected_experts.reshape(-1)
            flat_tokens = (
                torch.arange(num_tokens, device=inputs.device, dtype=torch.long)
                .unsqueeze(1)
                .expand(num_tokens, k)
                .reshape(-1)
            )
            flat_weights = weights.reshape(-1)

        expert_ms = 0.0
        for i, expert in enumerate(self.experts):
            # Keep expert loop static (no tensor->python sync for dynamic ids),
            # but use index_select for faster gathers on ROCm.
            route_pos = torch.where(flat_experts == i)[0]
            if timed_decode:
                (token_idx, token_weights), gather_ms = measure_gpu_ms_if_available(
                    lambda: (
                        flat_tokens.index_select(0, route_pos),
                        flat_weights.index_select(0, route_pos),
                    )
                )
                timing.moe_gather_decode_ms += gather_ms
            else:
                token_idx = flat_tokens.index_select(0, route_pos)
                token_weights = flat_weights.index_select(0, route_pos)
            if token_idx.numel() == 0:
                continue
            if timed_decode:
                expert_out, e_ms = measure_gpu_ms_if_available(lambda: expert(inputs.index_select(0, token_idx)))
                expert_ms += e_ms
            else:
                expert_out = expert(inputs.index_select(0, token_idx))
            # Combine expert outputs with weighted index accumulation.
            expert_out.mul_(token_weights.unsqueeze(1))
            if timed_decode:
                _, combine_ms = measure_gpu_ms_if_available(
                    lambda: results.index_add_(0, token_idx, expert_out)
                )
                timing.moe_combine_decode_ms += combine_ms
            else:
                results.index_add_(0, token_idx, expert_out)
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
