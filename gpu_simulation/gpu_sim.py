

import argparse
from pathlib import Path
import json
import yaml
from typing import Any, Dict


from evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model-config-path", type=str, default="./gpu_simulation/model_config/Qwen2.5-3B-Instruct.json")
    parser.add_argument("--hardware-config-path", type=str, default="./gpu_simulation/hardware_config/RTX4090.yaml")
    parser.add_argument("--sum-seq-len", type=int, default=1024)
    parser.add_argument("--gen-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="./results/gpu_sim")
    return parser.parse_args()


def load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)

def load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)

def get_output_folder(args):
    output_dir = Path(args.output_dir)
    output_dir = output_dir / args.model.replace("/", "-") \
                 / f"batch-size-{args.batch_size}" \
                 / f"sum-seq-len-{args.sum_seq_len}" \
                 / f"gen-seq-len-{args.gen_seq_len}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(args):
    # Load model config
    model_config_path = Path(args.model_config_path)
    raw_config = load_json(model_config_path)
    if 'Qwen2ForCausalLM' in raw_config["architectures"]:
        from Llama import LlamaModelArgs, LlamaModel
        model_args = LlamaModelArgs.init_from_config(raw_config)
        model = LlamaModel(model_args=model_args)
    else:
        raise ValueError(f"Model {args.model_config_path} not supported")

    # Load hardware config
    hardware_config_path = Path(args.hardware_config_path)
    hardware_config = load_yaml(hardware_config_path)

    evaluator = Evaluator(hardware_config=hardware_config, element_size=2)
    performance = evaluator.evaluate_model(model=model, batch_size=args.batch_size, sum_seq_len=args.sum_seq_len, gen_seq_len=args.gen_seq_len)
    # print(performance)

    output_folder = get_output_folder(args)
    with open(output_folder / "results.json", "w") as f:
        results = {
            "args": {
                "model_config_path": args.model_config_path,
                "hardware_config_path": args.hardware_config_path,
                "sum_seq_len": args.sum_seq_len,
                "gen_seq_len": args.gen_seq_len,
                "batch_size": args.batch_size,
            },
            "performance": performance,
        }
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_folder / 'results.json'}")

    

    









if __name__ == "__main__":
    args = parse_args()
    main(args)