import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="./results/gpu_sim")
    parser.add_argument("--roofline-data-dir", dest="results_dir", type=str, help=argparse.SUPPRESS)  # backward compat
    parser.add_argument("--model-name", type=str, default=None)  # backward compat (single model)
    parser.add_argument(
        "--model-name-list",
        type=str,
        nargs="+",
        default=["Qwen-Qwen2.5-3B-Instruct", "state-spaces-mamba-2.8b-hf"],
    )
    parser.add_argument("--sum-seq-len-list", type=int, nargs="+", default=[1024, 2048, 4096, 8192])  # 1024, 2048, 4096, 8192
    parser.add_argument("--gen-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="./figures/model_flops_and_latency_breakdown")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()

def load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)

def get_leaf_dir(root_dir: str, model_name, batch_size, sum_seq_len, gen_seq_len):
    root_dir = Path(root_dir)
    leaf_dir = root_dir \
                / model_name.replace("/", "-") \
                / f"batch-size-{batch_size}" \
                / f"sum-seq-len-{sum_seq_len}" \
                / f"gen-seq-len-{gen_seq_len}"
    return leaf_dir


def _infer_special_type(model_name: str, prefill_breakdown: Dict) -> str:
    types = {entry.get("type") for entry in prefill_breakdown.values() if isinstance(entry, dict)}
    names = set(prefill_breakdown.keys())
    if "qK_softmax_sV_fused" in types or "qK_softmax_sV_fused" in names:
        return "qK_softmax_sV_fused"
    if "mamba_elementwise" in types or "mamba_elementwise" in names:
        return "mamba_elementwise"

    name_l = model_name.lower()
    if "qwen" in name_l:
        return "qK_softmax_sV_fused"
    if "mamba" in name_l:
        return "mamba_elementwise"

    raise ValueError("Unable to infer special op type; please check input data or model name.")


def _build_type_map(breakdown: Dict) -> Dict[str, str]:
    type_map = {}
    for name, entry in breakdown.items():
        if isinstance(entry, dict) and "type" in entry:
            type_map[name] = entry["type"]
    return type_map


def _sum_by_type(
    breakdown: Dict,
    metric_key: str,
    special_type: str,
    fallback_type_map: Dict[str, str],
) -> Tuple[float, float, float]:
    gemm_sum = 0.0
    special_sum = 0.0
    other_sum = 0.0

    for name, entry in breakdown.items():
        if not isinstance(entry, dict):
            continue
        value = entry.get(metric_key)
        if value is None:
            continue

        op_type = entry.get("type")
        if op_type is None:
            op_type = fallback_type_map.get(name)
        if op_type is None and name == special_type:
            op_type = special_type

        if op_type == special_type:
            special_sum += float(value)
        elif op_type == "gemm":
            gemm_sum += float(value)
        else:
            other_sum += float(value)

    return gemm_sum, special_sum, other_sum


def _load_breakdown(
    results_dir: str,
    model_name: str,
    batch_size: int,
    sum_seq_len: int,
    gen_seq_len: int,
    phase: str,
) -> Dict:
    results_path = get_leaf_dir(results_dir, model_name, batch_size, sum_seq_len, gen_seq_len) / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json: {results_path}")
    results = load_json(results_path)
    return results["performance"]["performance_per_layer"][phase]["breakdown"]

def _find_project_root(results_dir: str) -> Path:
    results_path = Path(results_dir).resolve()
    for parent in [results_path] + list(results_path.parents):
        if (parent / "gpu_simulation" / "model_config").exists():
            return parent
    return results_path


def _load_num_layers(
    results_dir: str,
    model_name: str,
    batch_size: int,
    sum_seq_len: int,
    gen_seq_len: int,
) -> int:
    results_path = get_leaf_dir(results_dir, model_name, batch_size, sum_seq_len, gen_seq_len) / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json: {results_path}")
    results = load_json(results_path)
    model_config_path = results.get("args", {}).get("model_config_path")
    if not model_config_path:
        raise KeyError(f"results.json missing args.model_config_path: {results_path}")

    config_path = Path(model_config_path)
    if not config_path.is_absolute():
        project_root = _find_project_root(results_dir)
        config_path = (project_root / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    config = load_json(config_path)
    num_layers = config.get("num_hidden_layers", config.get("n_layer"))
    if not isinstance(num_layers, int):
        raise KeyError(f"num_hidden_layers/n_layer not found in {config_path}")
    return num_layers


def _prepare_phase_arrays(
    results_dir: str,
    model_name: str,
    sum_seq_len_list: List[int],
    batch_size: int,
    gen_seq_len: int,
    phase: str,
    special_type: str,
    num_layers: Optional[int] = None,
    flops_unit_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gemm_flops = []
    special_flops = []
    gemm_time = []
    special_time = []

    other_flops_total = 0.0
    other_time_total = 0.0

    for sum_seq_len in sum_seq_len_list:
        breakdown = _load_breakdown(results_dir, model_name, batch_size, sum_seq_len, gen_seq_len, phase)
        prefill_breakdown = _load_breakdown(results_dir, model_name, batch_size, sum_seq_len, gen_seq_len, "prefill")
        fallback_type_map = _build_type_map(prefill_breakdown)

        g_flops, s_flops, o_flops = _sum_by_type(breakdown, "flops", special_type, fallback_type_map)
        g_time, s_time, o_time = _sum_by_type(breakdown, "time", special_type, fallback_type_map)

        if num_layers is not None:
            g_flops *= num_layers
            s_flops *= num_layers
            g_time *= num_layers
            s_time *= num_layers

        if flops_unit_scale != 1.0:
            g_flops *= flops_unit_scale
            s_flops *= flops_unit_scale

        gemm_flops.append(g_flops)
        special_flops.append(s_flops)
        gemm_time.append(g_time)
        special_time.append(s_time)

        other_flops_total += o_flops
        other_time_total += o_time

    if other_flops_total > 0 or other_time_total > 0:
        print(
            f"Warning: {phase} has ops outside GEMM/{special_type}. "
            f"Other flops sum={other_flops_total:.6f}, other time sum={other_time_total:.6f}."
        )

    return (
        np.array(gemm_flops, dtype=float),
        np.array(special_flops, dtype=float),
        np.array(gemm_time, dtype=float),
        np.array(special_time, dtype=float),
    )

def main(args):
    results_dir = args.results_dir or "./results/gpu_sim"
    model_name_list = [args.model_name] if args.model_name else args.model_name_list
    sum_seq_len_list = args.sum_seq_len_list

    model_payloads = []
    special_types = []
    for model_name in model_name_list:
        sample_prefill = _load_breakdown(
            results_dir,
            model_name,
            args.batch_size,
            sum_seq_len_list[0],
            args.gen_seq_len,
            "prefill",
        )
        special_type = _infer_special_type(model_name, sample_prefill)
        special_types.append(special_type)
        num_layers = _load_num_layers(
            results_dir,
            model_name,
            args.batch_size,
            sum_seq_len_list[0],
            args.gen_seq_len,
        )
        prefill_arrays = _prepare_phase_arrays(
            results_dir,
            model_name,
            sum_seq_len_list,
            args.batch_size,
            args.gen_seq_len,
            "prefill",
            special_type,
            num_layers=num_layers,
            flops_unit_scale=1.0 / 1000.0,
        )
        model_payloads.append((model_name, special_type, prefill_arrays))

    colors = {
        "gemm": "#1f77b4",
        "qK_softmax_sV_fused": "#2ca02c",
        "mamba_elementwise": "#ff7f0e",
    }

    # fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig, axes = plt.subplots(1, 2, figsize=(5, 4.5))
    # fig, axes = plt.subplots(2, 1 ,figsize=(4.5, 12))

    group_x = np.arange(len(sum_seq_len_list), dtype=float)
    group_gap = 0.25
    bar_gap = 0.06
    total_group_width = 1.0 - group_gap
    n_models = len(model_payloads)
    bar_w = (total_group_width - (n_models - 1) * bar_gap) / n_models
    bar_offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * (bar_w + bar_gap)

    ax_flops = axes[0]
    ax_time = axes[1]

    for idx, (model_name, special_type, prefill_arrays) in enumerate(model_payloads):
        gemm_flops, special_flops, gemm_time, special_time = prefill_arrays
        x = group_x + bar_offsets[idx]
        special_color = colors.get(special_type, "#ff7f0e")

        ax_flops.bar(x, gemm_flops, width=bar_w, color=colors["gemm"], edgecolor="black", linewidth=0.8)
        ax_flops.bar(
            x,
            special_flops,
            width=bar_w,
            bottom=gemm_flops,
            color=special_color,
            edgecolor="black",
            linewidth=0.8,
        )

        ax_time.bar(x, gemm_time, width=bar_w, color=colors["gemm"], edgecolor="black", linewidth=0.8)
        ax_time.bar(
            x,
            special_time,
            width=bar_w,
            bottom=gemm_time,
            color=special_color,
            edgecolor="black",
            linewidth=0.8,
        )

    ax_flops.set_title("FLOPs (Prefill)")
    ax_flops.set_xticks(group_x)
    ax_flops.set_xticklabels([str(v) for v in sum_seq_len_list])
    # ax_flops.set_xlabel("Sum Seq Len")
    ax_flops.set_xlabel("Prompt Length")
    # ax_flops.set_ylabel("FLOPs (TFLOPs)")
    ax_flops.set_ylabel("TFLOPs")
    ax_flops.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_flops.set_xlim(group_x[0] - 0.6, group_x[-1] + 0.6)

    ax_time.set_title("Latency (Prefill)")
    ax_time.set_xticks(group_x)
    ax_time.set_xticklabels([str(v) for v in sum_seq_len_list])
    # ax_time.set_xlabel("Sum Seq Len")
    ax_time.set_xlabel("Prompt Length")
    ax_time.set_ylabel("Latency (ms)")
    ax_time.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_time.set_xlim(group_x[0] - 0.6, group_x[-1] + 0.6)

    label_table = {
        "gemm": "GEMM",
        "qK_softmax_sV_fused": "Fused Attention",
        "mamba_elementwise": "Elementwise Operations",
    }

    handles = [Patch(facecolor=colors["gemm"], edgecolor="black", label=label_table["gemm"])]
    if "qK_softmax_sV_fused" in special_types:
        handles.append(Patch(facecolor=colors["qK_softmax_sV_fused"], edgecolor="black", label=label_table["qK_softmax_sV_fused"]))
    if "mamba_elementwise" in special_types:
        handles.append(Patch(facecolor=colors["mamba_elementwise"], edgecolor="black", label=label_table["mamba_elementwise"]))

    # title = "Prefill FLOPs/Latency Breakdown"
    title = ""
    if len(model_payloads) == 1:
        title = model_payloads[0][0]
    fig.suptitle(title)
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(model_payloads) == 1:
        base_name = f"{model_payloads[0][0].replace('/', '-')}_flops_latency_breakdown"
    else:
        base_name = "multi_model_flops_latency_breakdown"
    base_path = output_dir / base_name
    for suffix in [".png", ".pdf"]:
        fig.savefig(str(base_path.with_suffix(suffix)), dpi=300, bbox_inches="tight")
    print(f"Saved figure to {base_path.with_suffix('.png')}")
    print(f"Saved figure to {base_path.with_suffix('.pdf')}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
