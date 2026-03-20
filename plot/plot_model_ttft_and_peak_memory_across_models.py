import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-profile-dir", type=str, default="./results/torch_profile")
    parser.add_argument("--output-dir", type=str, default="./figures/model_ttft_and_peak_memory")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "Qwen/Qwen2.5-3B-Instruct",
            "fla-hub/rwkv7-2.9B-world",
            "state-spaces/mamba-2.8b-hf",
        ],
    )
    return parser.parse_args()


def get_leaf_dir(root_dir: str, model_name, batch_size, sum_seq_len, gen_seq_len):
    root_dir = Path(root_dir)
    return (
        root_dir
        / model_name.replace("/", "-")
        / f"batch-size-{batch_size}"
        / f"sum-seq-len-{sum_seq_len}"
        / f"gen-seq-len-{gen_seq_len}"
    )


def load_results(root_dir, model, sum_seq_len_list, batch_size, gen_seq_len):
    import json
    import numpy as np

    model_weight = []
    cache = []
    ttft = []
    missing = []

    for sum_seq_len in sum_seq_len_list:
        results_dir = get_leaf_dir(root_dir, model, batch_size, sum_seq_len, gen_seq_len)
        results_path = results_dir / "results.json"
        if not results_path.exists():
            missing.append(str(results_path))
            continue

        with open(results_path, "r") as f:
            results = json.load(f)

        memory = results.get("memory_usage", {})
        perf = results.get("performance", {})

        model_weight.append(memory.get("load_model_peak_memory"))
        cache.append(memory.get("prefill_peak_memory"))
        ttft.append(perf.get("TTFT_time"))

    if missing:
        raise FileNotFoundError("Missing results.json files:\n" + "\n".join(missing))

    return {
        "model_weight": np.array(model_weight, dtype=float),
        "cache": np.array(cache, dtype=float),
        "ttft": np.array(ttft, dtype=float),
    }


def build_axes(models, figsize_unit=4.2):
    import matplotlib.pyplot as plt

    if len(models) == 3:
        fig = plt.figure(figsize=(figsize_unit * len(models) + 2.2, figsize_unit))
        gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.5, 1, 0.5, 1], wspace=0.0)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 2], sharey=ax0)
        ax2 = fig.add_subplot(gs[0, 4], sharey=ax0)
        axes = [ax0, ax1, ax2]
    else:
        fig, axes = plt.subplots(
            1,
            len(models),
            figsize=(figsize_unit * len(models), figsize_unit),
            sharey=True,
            gridspec_kw={"wspace": 0.0},
        )
        if len(models) == 1:
            axes = [axes]
    return fig, axes


def plot_peak_memory_and_latency(args):
    import numpy as np
    import matplotlib.pyplot as plt

    sum_seq_len_list = [1024, 2048, 4096, 8192]
    batch_size = 1
    gen_seq_len = 64
    models = args.models

    model_name_table = {
        "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
        "state-spaces/mamba-2.8b-hf": "Mamba-2.8b-hf",
        "fla-hub/rwkv7-2.9B-world": "RWKV-2.9B-World",
    }

    results_by_model = {}
    max_memory = 0.0
    for model in models:
        results = load_results(args.torch_profile_dir, model, sum_seq_len_list, batch_size, gen_seq_len)
        results_by_model[model] = results
        max_memory = max(max_memory, float(np.max(results["model_weight"] + results["cache"])))

    if len(models) > 1:
        shared_ttft_max = max(
            float(np.max(results_by_model[m]["ttft"])) for m in models[:-1]
        )
        last_ttft_max = float(np.max(results_by_model[models[-1]]["ttft"]))
    else:
        shared_ttft_max = float(np.max(results_by_model[models[0]]["ttft"]))
        last_ttft_max = shared_ttft_max

    fig, axes = build_axes(models)

    bar_colors = {
        "Model Weight": "#4C78A8",
        "Cache": "#f2cf41",
    }

    x = np.arange(len(sum_seq_len_list))
    twin_axes = []

    for idx, (ax, model) in enumerate(zip(axes, models)):
        r = results_by_model[model]

        ax.bar(x, r["model_weight"], label="Model Weight", color=bar_colors["Model Weight"])
        ax.bar(x, r["cache"], bottom=r["model_weight"], label="Cache", color=bar_colors["Cache"])

        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in sum_seq_len_list])
        ax.set_xlabel("prompt sequence length")
        ax.set_ylabel("Peak memory (GB)", labelpad=10)
        ax.tick_params(labelleft=True)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(model_name_table.get(model, model))

        ax2 = ax.twinx()
        ax2.plot(x, r["ttft"], color="#E45756", marker="o", linewidth=2, label="TTFT")
        ax2.set_ylabel("TTFT (ms)", labelpad=10)
        ax2.tick_params(labelright=True)
        if len(models) > 1 and idx < len(models) - 1:
            ax2.set_ylim(0, shared_ttft_max * 1.1)
        else:
            ax2.set_ylim(0, last_ttft_max * 1.1)
        twin_axes.append(ax2)

    for ax in axes:
        ax.set_ylim(0, max_memory * 1.1)

    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = twin_axes[0].get_legend_handles_labels()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        fontsize=12,
        frameon=False,
    )

    fig.subplots_adjust(wspace=0.0, top=0.82, left=0.08, right=0.92)

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "peak_memory_ttft_across_models.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {save_path}")

    return fig


def main(args):
    plot_peak_memory_and_latency(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
