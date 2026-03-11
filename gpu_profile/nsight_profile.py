from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from contextlib import contextmanager
from torch.cuda import nvtx
import argparse
from pathlib import Path

from utils import get_fake_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sum-seq-len", type=int, default=1024)
    parser.add_argument("--gen-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_output_folder(args):
    output_dir = Path(args.output_dir)
    output_dir = output_dir / args.model.replace("/", "-") \
                 / f"batch-size-{args.batch_size}" \
                 / f"sum-seq-len-{args.sum_seq_len}" \
                 / f"gen-seq-len-{args.gen_seq_len}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

@contextmanager
def nvtx_ops():
    # Annotate each PyTorch op with NVTX (appears in Nsight Systems)
    with torch.autograd.profiler.emit_nvtx():
        yield



def add_nvtx_module_bands(model):
    handles = []
    for name, mod in model.named_modules():
        def pre_hook(_m, _inp, n=name):
            torch.cuda.nvtx.range_push(f"{n}::{_m.__class__.__name__}")
        def post_hook(_m, _inp, _out):
            torch.cuda.nvtx.range_pop()
        handles.append(mod.register_forward_pre_hook(pre_hook))
        handles.append(mod.register_forward_hook(post_hook))
    return handles



def main(args):
    torch.manual_seed(args.seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(args.seed)

    # ---- Load model & tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=DEVICE
    )
    model.eval()
    torch.cuda.synchronize()

    handles = add_nvtx_module_bands(model)

    # ---- Helpers ----
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
    nvtx.range_push("prefill")
    with nvtx_ops():
    # with torch.autograd.profiler.emit_nvtx():
        outputs = do_prefill(input_ids, attention_mask)
        torch.cuda.synchronize()
    nvtx.range_pop()

    # ---- Generation phase ----
    # Initialize for decoding
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    nvtx.range_push("decode")
    with nvtx_ops():
        for _ in range(args.gen_seq_len):
            outputs = do_decode_step(pred_token_idx, past_key_values)
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        torch.cuda.synchronize()
    nvtx.range_pop()

    # Remove NVTX handles
    for h in handles: h.remove()
    
    


if __name__ == "__main__":
    args = parse_args()
    main(args)