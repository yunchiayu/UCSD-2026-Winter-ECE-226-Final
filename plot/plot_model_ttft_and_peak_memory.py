import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-profile-dir", type=str, default="./results/torch_profile")
    parser.add_argument("--output-dir", type=str, default="./figures/model_ttft_and_peak_memory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct") # Qwen/Qwen2.5-3B-Instruct, state-spaces/mamba-2.8b-hf, fla-hub/rwkv7-2.9B-world
    return parser.parse_args()


def get_leaf_dir(root_dir: str, model_name, batch_size, sum_seq_len, gen_seq_len):
    root_dir = Path(root_dir)
    leaf_dir = root_dir \
                / model_name.replace("/", "-") \
                / f"batch-size-{batch_size}" \
                / f"sum-seq-len-{sum_seq_len}" \
                / f"gen-seq-len-{gen_seq_len}"
    return leaf_dir


def plot_peak_memory_and_latency(args):
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    model = args.model
    sum_seq_len_list = [1024, 2048, 4096, 8192]

    batch_size = 1
    gen_seq_len = 64

    load_model_peak_memory = []
    prefill_peak_memory = []
    decoding_peak_memory = []
    ttft_time = []
    missing = []

    for sum_seq_len in sum_seq_len_list:
        results_dir = get_leaf_dir(args.torch_profile_dir, model, batch_size, sum_seq_len, gen_seq_len)
        results_path = results_dir / "results.json"
        if not results_path.exists():
            missing.append(str(results_path))
            continue

        with open(results_path, "r") as f:
            results = json.load(f)

        memory = results.get("memory_usage", {})
        perf = results.get("performance", {})

        load_model_peak_memory.append(memory.get("load_model_peak_memory"))
        prefill_peak_memory.append(memory.get("prefill_peak_memory"))
        decoding_peak_memory.append(memory.get("decoding_peak_memory"))
        ttft_time.append(perf.get("TTFT_time"))

    if missing:
        missing_list = "\n".join(missing)
        raise FileNotFoundError(f"Missing results.json files:\n{missing_list}")

    x = np.arange(len(sum_seq_len_list))

    model_weight_arr = np.array(load_model_peak_memory, dtype=float)
    
    prefill_arr = np.array(prefill_peak_memory, dtype=float)
    decode_arr = np.array(decoding_peak_memory, dtype=float)
    KV_cache_memory = [ prefill_peak_memory[i] + decoding_peak_memory[i] for i in range(len(sum_seq_len_list))]
    KV_cache_arr = np.array(KV_cache_memory, dtype=float)




    ttft_arr = np.array(ttft_time, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig, ax = plt.subplots(figsize=(6, 6))

    bar_colors = {
        "Model Weight": "#4C78A8",
        "Cache": "#f2cf41",
        # "prefill": "#F58518",
        # "decode": "#54A24B",
    }

    ax.bar(x, model_weight_arr, label="Model Weight", color=bar_colors["Model Weight"])
    ax.bar(x, prefill_arr, bottom=model_weight_arr, label="Cache", color=bar_colors["Cache"])
    # ax.bar(
    #     x,
    #     decode_arr,
    #     bottom=model_weight_arr + prefill_arr,
    #     label="Decoding peak memory",
    #     color=bar_colors["decode"],
    # )

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in sum_seq_len_list])
    ax.set_xlabel("prompt sequence length")
    ax.set_ylabel("Peak memory (GB)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.6)

    ax2 = ax.twinx()
    ax2.plot(x, ttft_arr, color="#E45756", marker="o", linewidth=2, label="TTFT")
    ax2.set_ylabel("TTFT (ms)")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=10)

    fig.tight_layout()




    save_dir = Path(args.output_dir) / model.replace("/", "-")
    save_dir.mkdir(parents=True, exist_ok=True)

    model_name_table = {
        "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
        "state-spaces/mamba-2.8b-hf": "Mamba-2.8b-hf",
        "fla-hub/rwkv7-2.9B-world": "RWKV-2.9B-World",
    }

    save_path = save_dir / f"peak_memory_ttft_{model_name_table[model]}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {save_path}")
    # plt.show()

    return fig


def main(args):
    plot_peak_memory_and_latency(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
