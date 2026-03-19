from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import argparse
import json
from pathlib import Path
from types import MethodType

from utils import get_fake_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"






def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sum-seq-len", type=int, default=1024)
    parser.add_argument("--gen-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./results/torch_profile_finegrained")
    parser.add_argument(
        "--show-detail",
        action="store_true",
        help="Show per-event latency lists in output (prefill_all_ms/decode_all_ms).",
    )
    return parser.parse_args()


def get_output_folder(args):
    output_dir = Path(args.output_dir)
    output_dir = output_dir / args.model_name.replace("/", "-") \
                 / f"batch-size-{args.batch_size}" \
                 / f"sum-seq-len-{args.sum_seq_len}" \
                 / f"gen-seq-len-{args.gen_seq_len}"
 
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(args):
    torch.manual_seed(args.seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(args.seed)
    
    # Memory measurement setup
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    base_mem_loadmodel = torch.cuda.memory_allocated()
    
    # Load model & tokenizer
    tokenizer = None
    model = None
    if "Qwen" in args.model_name:
        from transformers import AutoModelForCausalLM
        # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.float16,
            device_map=DEVICE
        )
    elif "mamba" in args.model_name:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, device_map=DEVICE,)
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    model.eval()
    torch.cuda.synchronize()
    peak_mem_loadmodel = torch.cuda.max_memory_allocated() - base_mem_loadmodel # Byte

    # ---- Fine-grained profiling setup for Mamba ----
    component_names = [
        "in_proj",
        "conv1d",
        "x_proj",
        "dt_proj",
        "softplus",
        "discrete_A",
        "discrete_B",
        "deltaB_u",
        "scan_ssm_update",
        "scan_matmul",
        "scan_add_D",
        "scan_act_gate",
        "update_ssm_state",
        "out_proj",
    ]
    layer_events = {
        "prefill": {k: [] for k in component_names},
        "decode": {k: [] for k in component_names},
    }
    profile_state = {"phase": None}

    def record_op(component, fn, *args, **kwargs):
        if profile_state["phase"] is None or DEVICE != "cuda":
            return fn(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        layer_events[profile_state["phase"]][component].append((start, end))
        return out

    def update_ssm_state(cache_params, layer_idx, ssm_state):
        if hasattr(cache_params, "update_ssm_state"):
            return cache_params.update_ssm_state(layer_idx, ssm_state)
        cache_params.ssm_states[layer_idx].copy_(ssm_state)
        return cache_params.ssm_states[layer_idx]

    def summarize_phase(phase: str, include_detail: bool):
        torch.cuda.synchronize()
        averages = {}
        counts = {}
        all_ms = {} if include_detail else None
        for comp in component_names:
            events = layer_events[phase].get(comp, [])
            counts[comp] = len(events)
            if not events:
                averages[comp] = None
                if include_detail:
                    all_ms[comp] = []
                continue
            total = 0.0
            per_event = [] if include_detail else None
            for start, end in events:
                dt = start.elapsed_time(end)
                total += dt
                if include_detail:
                    per_event.append(dt)
            averages[comp] = total / len(events)
            if include_detail:
                all_ms[comp] = per_event
        return averages, counts, all_ms

    def make_profiled_slow_forward():
        def profiled_slow_forward(
            self,
            input_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
        ):
            batch_size, seq_len, _ = input_states.shape
            dtype = input_states.dtype

            projected_states = record_op(
                "in_proj", lambda: self.in_proj(input_states).transpose(1, 2)
            )
            hidden_states, gate = projected_states.chunk(2, dim=1)
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask[:, None, :]
                hidden_states = hidden_states * attention_mask[..., :seq_len].to(hidden_states.dtype)

            if cache_params is not None:
                if cache_position is None:
                    raise ValueError("cache_position must be provided when cache_params is supplied")
                ssm_state = cache_params.ssm_states[self.layer_idx][:batch_size].clone()
                ssm_state = ssm_state.to(hidden_states.device)
                if cache_position.shape[0] == self.conv_kernel_size:
                    conv_input = F.pad(
                        hidden_states,
                        (self.conv_kernel_size - hidden_states.shape[-1], 0),
                    )
                    cache_params.update_conv_state(self.layer_idx, conv_input, cache_position)
                    conv_out = record_op(
                        "conv1d",
                        lambda: self.conv1d(hidden_states.to(self.conv1d.weight.dtype))[..., :seq_len],
                    )
                    hidden_states = self.act(conv_out).to(dtype)
                else:
                    conv_state = cache_params.update_conv_state(
                        self.layer_idx, hidden_states, cache_position
                    )
                    conv_state = conv_state.to(self.conv1d.weight.device)
                    conv_out = record_op(
                        "conv1d",
                        lambda: (conv_state * self.conv1d.weight[:, 0, :]).sum(-1),
                    )
                    if self.use_conv_bias:
                        conv_out = conv_out + self.conv1d.bias
                    hidden_states = self.act(conv_out).to(dtype).unsqueeze(-1)
            else:
                ssm_state = torch.zeros(
                    (batch_size, self.intermediate_size, self.ssm_state_size),
                    device=hidden_states.device,
                    dtype=dtype,
                )
                conv_out = record_op(
                    "conv1d",
                    lambda: self.conv1d(hidden_states.to(self.conv1d.weight.dtype))[..., :seq_len],
                )
                hidden_states = self.act(conv_out).to(dtype)

            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask[..., :seq_len].to(hidden_states.dtype)

            ssm_parameters = record_op(
                "x_proj", lambda: self.x_proj(hidden_states.transpose(1, 2))
            )
            time_step, B, C = torch.split(
                ssm_parameters,
                [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
                dim=-1,
            )
            discrete_time_step = record_op("dt_proj", lambda: self.dt_proj(time_step))
            discrete_time_step = record_op(
                "softplus",
                lambda: F.softplus(discrete_time_step).transpose(1, 2),
            )

            A = -torch.exp(self.A_log.float())
            discrete_A = record_op(
                "discrete_A",
                lambda: torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]),
            )
            discrete_B = record_op(
                "discrete_B",
                lambda: discrete_time_step[:, :, :, None] * B[:, None, :, :].float(),
            )
            deltaB_u = record_op(
                "deltaB_u",
                lambda: discrete_B * hidden_states[:, :, :, None].float(),
            )

            scan_outputs = []
            for i in range(seq_len):
                ssm_state = record_op(
                    "scan_ssm_update",
                    lambda: discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :],
                )
                c_vector = C[:, i, :].unsqueeze(-1)
                scan_output = record_op(
                    "scan_matmul",
                    lambda: torch.matmul(ssm_state.to(dtype), c_vector),
                )
                scan_outputs.append(scan_output[:, :, 0])

            scan_output = torch.stack(scan_outputs, dim=-1)
            scan_output = record_op(
                "scan_add_D",
                lambda: scan_output + hidden_states * self.D[None, :, None],
            )
            scan_output = record_op(
                "scan_act_gate",
                lambda: scan_output * self.act(gate),
            )

            if cache_params is not None:
                record_op(
                    "update_ssm_state",
                    lambda: update_ssm_state(cache_params, self.layer_idx, ssm_state),
                )

            contextualized_states = record_op(
                "out_proj",
                lambda: self.out_proj(scan_output.transpose(1, 2)),
            )
            return contextualized_states

        return profiled_slow_forward

    profiled_slow_forward = make_profiled_slow_forward()
    for module in model.modules():
        if module.__class__.__name__ == "MambaMixer" and hasattr(module, "slow_forward"):
            module.slow_forward = MethodType(profiled_slow_forward, module)
            # Keep forward as-is; it should call slow_forward when kernels aren't available.

    # helpers
    @torch.inference_mode()
    def do_prefill(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    @torch.inference_mode()
    def do_decode_step(input_ids, past_key_values):
        return model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)

    # ---- Build input ids and attention mask ----
    input_ids, attention_mask = get_fake_prompt(tokenizer, args.batch_size, args.sum_seq_len, device=DEVICE)
    # ---- Warmup ----
    print("Warmup...")
    for _ in range(args.warmup_iters):
        do_prefill(input_ids, attention_mask)
    torch.cuda.synchronize()

    # ---- Prefill phase ----
    print("Prefill phase...")
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    base_mem_prefill = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    include_detail = args.show_detail
    profile_state["phase"] = "prefill"
    start.record()
    outputs = do_prefill(input_ids, attention_mask)
    end.record()
    profile_state["phase"] = None
    torch.cuda.synchronize()
    TTFT_time = start.elapsed_time(end)
    prefill_peak_mem = torch.cuda.max_memory_allocated() - base_mem_prefill # Byte
    prefill_component_avg, prefill_component_counts, prefill_component_all = summarize_phase(
        "prefill", include_detail
    )

    print(f"Prompt length: {args.sum_seq_len}")
    print(f"    - TTFT time: {TTFT_time:.2f} ms")
    print("    - Prefill component avg (ms):")
    for comp in component_names:
        val = prefill_component_avg.get(comp)
        if val is None:
            continue
        print(f"        * {comp}: {val:.4f}")


    # ---- Decode phase ----
    print("Decode phase...")
    # Initialize for decoding
    cache_params = outputs.cache_params
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    base_mem_decode = torch.cuda.memory_allocated()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    profile_state["phase"] = "decode"
    start.record()
    for _ in range(args.gen_seq_len):
        outputs = do_decode_step(pred_token_idx, cache_params)
        cache_params = outputs.cache_params
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)    
    end.record()
    profile_state["phase"] = None
    torch.cuda.synchronize()
    decoding_time = start.elapsed_time(end) # ms
    decoding_throughput = args.gen_seq_len / decoding_time # tokens/ms
    decode_component_avg, decode_component_counts, decode_component_all = summarize_phase(
        "decode", include_detail
    )

    decode_peak_mem = torch.cuda.max_memory_allocated() - base_mem_decode # Byte

    print(f"    - Decoding time: {decoding_time:.2f} ms")
    print(f"    - Decoding throughput: {decoding_throughput:.2f} tokens/ms")
    print("    - Decode component avg (ms):")
    for comp in component_names:
        val = decode_component_avg.get(comp)
        if val is None:
            continue
        print(f"        * {comp}: {val:.4f}")
    print(f"    - Load model peak memory usage: {peak_mem_loadmodel / 1024**3} GB")
    print(f"    - Prefill peak memory usage:    {prefill_peak_mem / 1024**3} GB")
    print(f"    - Decoding peak memory usage:   {decode_peak_mem / 1024**3} GB")


    finegrained = {
        "components": component_names,
        "prefill": prefill_component_avg,
        "prefill_counts": prefill_component_counts,
        "decode": decode_component_avg,
        "decode_counts": decode_component_counts,
        "unit": {
            "time": "ms",
        },
        "detail_enabled": include_detail,
    }
    if include_detail:
        finegrained["prefill_all"] = prefill_component_all
        finegrained["decode_all"] = decode_component_all

    output_folder = get_output_folder(args)
    with open(output_folder / "results.json", "w") as f:
        results = {
            "args": {
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "sum_seq_len": args.sum_seq_len,
                "gen_seq_len": args.gen_seq_len,
                "warmup_iters": args.warmup_iters,
                "seed": args.seed,
            },
            "performance": {
                "unit": {
                    "time": "ms",
                    "throughput": "tokens/s",
                    "memory": "GB",
                },
                "TTFT_time": TTFT_time,
                "decoding_time": decoding_time,
                "decoding_throughput": decoding_throughput * 1e3,
            },
            "memory_usage": {
                "load_model_peak_memory": peak_mem_loadmodel / 1024**3,
                "prefill_peak_memory": prefill_peak_mem / 1024**3,
                "decoding_peak_memory": decode_peak_mem / 1024**3,
            },
            "finegrained": finegrained,
        }
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_folder / 'results.json'}")




    


if __name__ == "__main__":
    args = parse_args()
    main(args)
