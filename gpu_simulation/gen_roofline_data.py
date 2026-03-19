

import argparse
from pathlib import Path
import json
import yaml
from typing import Any, Dict


from evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sum-seq-len", type=int, default=1024)
    parser.add_argument("--gen-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-sim-dir", type=str, default="./results/gpu_sim")
    parser.add_argument("--torch-profile-finegrained-dir", type=str, default="./results/torch_profile_finegrained")
    return parser.parse_args()


def load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)

def load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)

def get_gpu_sim_folder(args):
    gpu_sim_dir = Path(args.gpu_sim_dir)
    gpu_sim_dir = gpu_sim_dir / args.model.replace("/", "-") \
                 / f"batch-size-{args.batch_size}" \
                 / f"sum-seq-len-{args.sum_seq_len}" \
                 / f"gen-seq-len-{args.gen_seq_len}"
    return gpu_sim_dir

def get_torch_profile_finegrained_folder(args):
    torch_profile_finegrained_dir = Path(args.torch_profile_finegrained_dir)
    torch_profile_finegrained_dir = torch_profile_finegrained_dir / args.model.replace("/", "-") \
                 / f"batch-size-{args.batch_size}" \
                 / f"sum-seq-len-{args.sum_seq_len}" \
                 / f"gen-seq-len-{args.gen_seq_len}"
    return torch_profile_finegrained_dir

def main(args):


    gpu_sim_folder = get_gpu_sim_folder(args)
    torch_profile_finegrained_folder = get_torch_profile_finegrained_folder(args)

    simulation_results = load_json(gpu_sim_folder / "results.json")
    torch_profile_finegrained_results = load_json(torch_profile_finegrained_folder / "results.json")
    
    



if __name__ == "__main__":
    args = parse_args()
    main(args)