from transformers import AutoTokenizer
import torch
import argparse
import json
from pathlib import Path

from utils import get_fake_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"






def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sum-seq-len", type=int, default=1024)
    parser.add_argument("--gen-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./results/torch_profile_finegrained/")
    parser.add_argument(
        "--show-detail",
        action="store_true",
        help="Show per-event latency lists in output (prefill_all/decode_all).",
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
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    model.eval()
    torch.cuda.synchronize()
    peak_mem_loadmodel = torch.cuda.max_memory_allocated() - base_mem_loadmodel # Byte

    # ---- Fine-grained profiling setup ----
    component_names = [
        "wq", "wk", "wv", "wo",
        "qK", "softmax", "sV",
        "w1", "w2", "w3",
        "qK_softmax_sV_fused",
    ]
    layer_events = {
        "prefill": {k: [] for k in component_names},
        "decode": {k: [] for k in component_names},
    }
    profile_state = {
        "phase": None,
        "in_attention": 0,
        "matmul_index": 0,
    }
    hook_handles = []

    def is_transformer_block_module(name: str) -> bool:
        return any(
            token in name
            for token in (
                ".layers.", ".h.", ".blocks.", ".block.",
                ".decoder.layers.", ".encoder.layers.", ".transformer.h.",
            )
        )

    def is_attention_module(name: str, module: torch.nn.Module) -> bool:
        if not is_transformer_block_module(name):
            return False
        return any(
            hasattr(module, attr)
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "W_pack")
        )

    def make_pre_hook(component):
        def pre_hook(module, inputs):
            if profile_state["phase"] is None or DEVICE != "cuda":
                return
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            if not hasattr(module, "_profile_starts"):
                module._profile_starts = []
            module._profile_starts.append(start)
        return pre_hook

    def make_post_hook(component):
        def post_hook(module, inputs, output):
            if profile_state["phase"] is None or DEVICE != "cuda":
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            starts = getattr(module, "_profile_starts", None)
            if not starts:
                return
            start = starts.pop()
            layer_events[profile_state["phase"]][component].append((start, end))
        return post_hook

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

    class TorchOpPatcher:
        def __init__(self):
            self.orig_matmul = torch.matmul
            self.orig_bmm = torch.bmm
            self.orig_softmax = torch.nn.functional.softmax
            self.orig_sdpa = getattr(torch.nn.functional, "scaled_dot_product_attention", None)

        def __enter__(self):
            def wrapped_matmul(a, b, *args, **kwargs):
                if profile_state["phase"] is None or DEVICE != "cuda" or profile_state["in_attention"] == 0:
                    return self.orig_matmul(a, b, *args, **kwargs)
                component = None
                if profile_state["matmul_index"] == 0:
                    component = "qK"
                elif profile_state["matmul_index"] == 1:
                    component = "sV"
                profile_state["matmul_index"] += 1
                if component is None:
                    return self.orig_matmul(a, b, *args, **kwargs)
                return record_op(component, self.orig_matmul, a, b, *args, **kwargs)

            def wrapped_bmm(a, b, *args, **kwargs):
                if profile_state["phase"] is None or DEVICE != "cuda" or profile_state["in_attention"] == 0:
                    return self.orig_bmm(a, b, *args, **kwargs)
                component = None
                if profile_state["matmul_index"] == 0:
                    component = "qK"
                elif profile_state["matmul_index"] == 1:
                    component = "sV"
                profile_state["matmul_index"] += 1
                if component is None:
                    return self.orig_bmm(a, b, *args, **kwargs)
                return record_op(component, self.orig_bmm, a, b, *args, **kwargs)

            def wrapped_softmax(input, *args, **kwargs):
                if profile_state["phase"] is None or DEVICE != "cuda" or profile_state["in_attention"] == 0:
                    return self.orig_softmax(input, *args, **kwargs)
                return record_op("softmax", self.orig_softmax, input, *args, **kwargs)

            torch.matmul = wrapped_matmul
            torch.bmm = wrapped_bmm
            torch.nn.functional.softmax = wrapped_softmax
            if self.orig_sdpa is not None:
                def wrapped_sdpa(*args, **kwargs):
                    if profile_state["phase"] is None or DEVICE != "cuda" or profile_state["in_attention"] == 0:
                        return self.orig_sdpa(*args, **kwargs)
                    return record_op("qK_softmax_sV_fused", self.orig_sdpa, *args, **kwargs)
                torch.nn.functional.scaled_dot_product_attention = wrapped_sdpa
            return self

        def __exit__(self, exc_type, exc, tb):
            torch.matmul = self.orig_matmul
            torch.bmm = self.orig_bmm
            torch.nn.functional.softmax = self.orig_softmax
            if self.orig_sdpa is not None:
                torch.nn.functional.scaled_dot_product_attention = self.orig_sdpa

    # Register hooks for linear components inside transformer blocks
    module_component_map = {
        "q_proj": "wq",
        "k_proj": "wk",
        "v_proj": "wv",
        "o_proj": "wo",
        "gate_proj": "w1",
        "down_proj": "w2",
        "up_proj": "w3",
    }
    for name, module in model.named_modules():
        if not is_transformer_block_module(name):
            continue
        leaf = name.split(".")[-1]
        component = module_component_map.get(leaf)
        if component is not None:
            hook_handles.append(module.register_forward_pre_hook(make_pre_hook(component)))
            hook_handles.append(module.register_forward_hook(make_post_hook(component)))
        if is_attention_module(name, module):
            def attn_pre_hook(_m, _inp):
                if profile_state["phase"] is None:
                    return
                profile_state["in_attention"] += 1
                profile_state["matmul_index"] = 0
            def attn_post_hook(_m, _inp, _out):
                if profile_state["phase"] is None:
                    return
                profile_state["in_attention"] -= 1
            hook_handles.append(module.register_forward_pre_hook(attn_pre_hook))
            hook_handles.append(module.register_forward_hook(attn_post_hook))

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
    for _ in range(args.warmup_iters):
        do_prefill(input_ids, attention_mask)
    torch.cuda.synchronize()

    # ---- Summarization phase ----
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    base_mem_prefill = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with TorchOpPatcher():
        profile_state["phase"] = "prefill"
        start.record()
        outputs = do_prefill(input_ids, attention_mask)
        end.record()
        profile_state["phase"] = None
    torch.cuda.synchronize()
    TTFT_time = start.elapsed_time(end)
    prefill_peak_mem = torch.cuda.max_memory_allocated() - base_mem_prefill # Byte
    include_detail = args.show_detail
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


    # ---- Generation phase ----
    # Initialize for decoding
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    base_mem_decode = torch.cuda.memory_allocated()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with TorchOpPatcher():
        profile_state["phase"] = "decode"
        start.record()
        for _ in range(args.gen_seq_len):
            outputs = do_decode_step(pred_token_idx, past_key_values)
            past_key_values = outputs.past_key_values
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

    for h in hook_handles:
        h.remove()

    finegrained = {
        "unit": {
            "time": "ms",
            "throughput": "tokens/s",
        },
        "components": component_names,
        "prefill": prefill_component_avg,
        "prefill_counts": prefill_component_counts,
        "decode": decode_component_avg,
        "decode_counts": decode_component_counts,
        "fused_attention": (
            (prefill_component_counts.get("qK_softmax_sV_fused", 0) or 0) > 0
            or (decode_component_counts.get("qK_softmax_sV_fused", 0) or 0) > 0
        ),
        "note": "If qK/softmax/sV are fused (e.g., SDPA/FlashAttention), timings appear under qK_softmax_sV_fused.",
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




    


if __name__ == "__main__":
    args = parse_args()
    main(args)
