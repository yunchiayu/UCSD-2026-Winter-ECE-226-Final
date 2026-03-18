from transformers import AutoTokenizer
import torch
import argparse
import json
from pathlib import Path

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
    parser.add_argument("--output-dir", type=str, default="./results/torch_profile")
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
    start.record()
    outputs = do_prefill(input_ids, attention_mask)
    end.record()
    torch.cuda.synchronize()
    TTFT_time = start.elapsed_time(end)
    prefill_peak_mem = torch.cuda.max_memory_allocated() - base_mem_prefill # Byte

    print(f"Prompt length: {args.sum_seq_len}")
    print(f"    - TTFT time: {TTFT_time:.2f} ms")


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
    start.record()
    for _ in range(args.gen_seq_len):
        outputs = do_decode_step(pred_token_idx, cache_params)
        cache_params = outputs.cache_params
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)    
    end.record()
    torch.cuda.synchronize()
    decoding_time = start.elapsed_time(end) # ms
    decoding_throughput = args.gen_seq_len / decoding_time # tokens/ms

    decode_peak_mem = torch.cuda.max_memory_allocated() - base_mem_decode # Byte

    print(f"    - Decoding time: {decoding_time:.2f} ms")
    print(f"    - Decoding throughput: {decoding_throughput:.2f} tokens/ms")
    print(f"    - Load model peak memory usage: {peak_mem_loadmodel / 1024**3} GB")
    print(f"    - Prefill peak memory usage:    {prefill_peak_mem / 1024**3} GB")
    print(f"    - Decoding peak memory usage:   {decode_peak_mem / 1024**3} GB")


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
                "TTFT_time": TTFT_time,
                "decoding_time": decoding_time,
                "latency_unit": "ms",
                "decoding_throughput": decoding_throughput * 1e3,
                "throughput_unit": "tokens/s",
            },
            "memory_usage": {
                "load_model_peak_memory": peak_mem_loadmodel / 1024**3,
                "prefill_peak_memory": prefill_peak_mem / 1024**3,
                "decoding_peak_memory": decode_peak_mem / 1024**3,
                "memory_unit": "GB",
            }
        }
        json.dump(results, f, indent=4)




    


if __name__ == "__main__":
    args = parse_args()
    main(args)