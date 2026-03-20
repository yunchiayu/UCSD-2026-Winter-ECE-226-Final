

import argparse
from pathlib import Path
import json
import yaml
from typing import Any, Dict


from evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sum-seq-len", type=int, default=1024)
    parser.add_argument("--gen-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-sim-dir", type=str, default="./results/gpu_sim")
    parser.add_argument("--torch-profile-finegrained-dir", type=str, default="./results/torch_profile_finegrained")
    parser.add_argument("--output-dir", type=str, default="./results/roofline_data")
    return parser.parse_args()


def load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)

def load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)

def get_leaf_dir(root_dir: str, model_name, batch_size, sum_seq_len, gen_seq_len):
    root_dir = Path(root_dir)
    leaf_dir = root_dir \
                / model_name.replace("/", "-") \
                / f"batch-size-{batch_size}" \
                / f"sum-seq-len-{sum_seq_len}" \
                / f"gen-seq-len-{gen_seq_len}"
    return leaf_dir

def save_json(data: dict, path: str):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)

def main(args):


    gpu_sim_folder = get_leaf_dir(args.gpu_sim_dir, args.model_name, args.batch_size, args.sum_seq_len, args.gen_seq_len)
    torch_profile_finegrained_folder = get_leaf_dir(args.torch_profile_finegrained_dir, args.model_name, args.batch_size, args.sum_seq_len, args.gen_seq_len)
    output_folder = get_leaf_dir(args.output_dir, args.model_name, args.batch_size, args.sum_seq_len, args.gen_seq_len)
    output_folder.mkdir(parents=True, exist_ok=True)

    gpu_sim_results = load_json(gpu_sim_folder / "results.json")
    torch_profile_finegrained_results = load_json(torch_profile_finegrained_folder / "results.json")

    performance_per_layer = gpu_sim_results["performance"]["performance_per_layer"]
    kernels = []
    for phase in ["prefill", "decode"]:
        # overall
        overall_flops = performance_per_layer[phase]["overall"]["flops"]
        overall_arithmetic_intensity = performance_per_layer[phase]["overall"]["arithmetic_intensity"]
        overall_optimal_time = performance_per_layer[phase]["overall"]["time"]
        overall_optimal_throughput = performance_per_layer[phase]["overall"]["throughput"]
        overall_dict = {
            "phase": phase,
            "name": "overall",
            "flops": overall_flops,
            "arithmetic_intensity": overall_arithmetic_intensity,
            "time": 0.0,
            "throughput": 0.0,
            "optimal_time": performance_per_layer[phase]["overall"]["time"],
            "optimal_throughput": performance_per_layer[phase]["overall"]["throughput"],
        }
        overall_time = 0.0
        
        kernels.append(overall_dict)
        # breakdown
        for kernel_name, kernel_data in performance_per_layer[phase]["breakdown"].items():
            kernel_profile_time = torch_profile_finegrained_results["finegrained"][phase][kernel_name] # ms
            kernel_flops = kernel_data["flops"] # GFLOP
            kernel_arithmetic_intensity = kernel_data["arithmetic_intensity"] # FLOPS/byte
            throughput = kernel_flops / kernel_profile_time   # TFLOPS 
            kernel_type = kernel_data.get("type", None)
            kernels.append({
                "phase": phase,
                "name": kernel_name,
                "type": kernel_type,
                "flops": kernel_data["flops"],
                "arithmetic_intensity": kernel_arithmetic_intensity,
                "time": kernel_profile_time,
                "throughput": throughput,
                "optimal_time": kernel_data["time"],
                "optimal_throughput": kernel_data["throughput"],
            })
            overall_time += kernel_profile_time
        overall_throughput = overall_flops / overall_time   # TFLOPS 
        overall_dict["time"] = overall_time
        overall_dict["throughput"] = overall_throughput

    

    output_dict = {
        "args": gpu_sim_results["args"],
        "unit": gpu_sim_results["performance"]["performance_per_layer"]["unit"],
        "kernels": kernels,
    }


    save_json(output_dict, output_folder / "results.json")
    print(f"Saved results to {output_folder / 'results.json'}")
    



if __name__ == "__main__":
    args = parse_args()
    main(args)