import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import argparse
import yaml
import json
from pathlib import Path
from typing import List

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-config-path", type=str, default="./gpu_simulation/hardware_config/RTX4090.yaml")
    parser.add_argument("--roofline-data-dir", type=str, default="./results/roofline_data")
    parser.add_argument("--model-name-list", type=List[str], default=["state-spaces-mamba-2.8b-hf"]) # ["Qwen-Qwen2.5-3B-Instruct", "state-spaces-mamba-2.8b-hf"]
    parser.add_argument("--sum-seq-len-list", type=List[int], default=[1024, 2048, 4096, 8192]) # 1024, 2048, 4096, 8192
    parser.add_argument("--gen-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-path", type=str, default="./figures/roofline.png")
    return parser.parse_args()

def load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)

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

def plot_roofline(args, hardware_configs, point_datas, output_path: str, plot_config: dict, legend_config: dict):
    # ========== Set global style from config ==========
    fontsize = plot_config.get("fontsize", {})
    plt.rc('axes', labelsize=fontsize.get("labelsize", 18))
    plt.rc('axes', titlesize=fontsize.get("titlesize", 20))
    plt.rc('xtick', labelsize=fontsize.get("tick_labelsize", 16))
    plt.rc('ytick', labelsize=fontsize.get("tick_labelsize", 16))
    plt.rc('legend', fontsize=fontsize.get("legend_fontsize", 16))

    # ========== Set figure size from config ==========
    figsize = plot_config.get("figsize", (10, 10))
    fig, ax = plt.subplots(figsize=figsize)

    # ========== Set axis from config ==========
    axis = plot_config.get("axis", {})
    xscale = axis.get("xscale", "log")
    yscale = axis.get("yscale", "log")
    xlim = axis.get("xlim", (0.01, 1200))
    ylim = axis.get("ylim", (0.01, 1200))
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])

    # ========== Store handles/labels for separate legends ==========
    roofline_handles = []
    roofline_labels = []
    point_handles = []
    point_labels = []

    # Plot each roofline
    for hw in hardware_configs:
        pf = hw['peak_compute']
        pbw = hw['peak_bandwidth']
        knee = pf / pbw
        xs = np.logspace(np.log10(ax.get_xlim()[0]), np.log10(ax.get_xlim()[1]), 5000)
        ys = np.minimum(pbw * xs, pf)
        color = hw['color']
        roofline = ax.plot(xs, ys, color=color, linewidth=2, label=hw["name"])[0]
        roofline_handles.append(roofline)
        roofline_labels.append(hw["name"]) 
        ymin, _ = ax.get_ylim()
        ax.plot([knee, knee], [ymin, pf], linestyle='--', color=color)
        ax.axhline(pbw * knee, linestyle='--', color=color)
        ax.text(knee * 1.02, ymin * 1.1, f'{knee:.2f}', rotation=0, va='bottom', ha='center', color=color, zorder=5)
        ax.text(0.085, pf * 1.04, f'{pf:.1f}', rotation=0, va='bottom', ha='right',
                color=color, transform=ax.get_yaxis_transform(), zorder=5, clip_on=False, fontsize=fontsize.get("labelsize", 18))

    # Plot each point
    for point_data in point_datas:
        sc = ax.scatter(point_data["x-axis"], point_data["y-axis"], color=point_data["color"], marker=point_data["symbol"], label=point_data["label"], s=point_data["symbol_size"], linewidth=0.5)
        point_handles.append(sc)
        point_labels.append(point_data["label"])

    # Labels
    ax.set_xlabel(plot_config.get("xlabel", 'Arithmetic Intensity (FLOPs/Byte)'))
    ax.set_ylabel(plot_config.get("ylabel", 'Performance (TFLOPS)'))
    ax.grid(plot_config.get("grid", True), which='both', linestyle=':', linewidth=0.5)


    # ========== Legend Setting ==========
    legend_cfg = plot_config.get("legend", {})
    legend_points_cfg = plot_config.get("legend_points", {})  # you can add this to your config!
    
    # Draw the legends (only if handles present)
    if roofline_handles:
        leg1 = ax.legend(roofline_handles, roofline_labels,
                        loc=legend_cfg.get('loc', 'upper left'),
                        bbox_to_anchor=legend_cfg.get('bbox_to_anchor', (0.02, 0.98)),
                        ncol=legend_cfg.get('ncol', 1),
                        frameon=legend_cfg.get('frameon', False),
                        fontsize=fontsize.get("legend_fontsize", 16),
                        title_fontsize=fontsize.get("legend_title_fontsize", 18))
        ax.add_artist(leg1)

    # Add legend below figure for phase symbols and model color palettes
    fig.subplots_adjust(bottom=0.20)

    phase_handles = legend_config["phase_legend"]["handles"]
    phase_leg = fig.legend(
        phase_handles, ['Prefill', 'Decode'],
        loc='lower left',
        bbox_to_anchor=(0.06, 0.03),
        ncol=2,
        frameon=False,
        title='Phase',
        fontsize=fontsize.get("legend_fontsize", 16),
        title_fontsize=fontsize.get("legend_title_fontsize", 18),
    )
    fig.add_artist(phase_leg)

    model_handles = legend_config["model_legend"]["handles"]
    model_labels = legend_config["model_legend"]["labels"]

    if model_handles:
        model_leg = fig.legend(
            model_handles, model_labels,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
            loc='lower right',
            bbox_to_anchor=(0.88, 0.03),
            ncol=min(2, len(model_handles)),
            frameon=False,
            title='Models (Prompt Length: 1024, 2048, 4096, 8192)',
            fontsize=fontsize.get("legend_fontsize", 16),
            title_fontsize=fontsize.get("legend_title_fontsize", 18),
        )
        fig.add_artist(model_leg)

    plt.savefig(output_path, dpi=plot_config.get('dpi', 300), bbox_inches='tight')
    return fig, ax




def main(args):

    plot_config = {
        "figsize": (13, 10),
        "fontsize": {
            "labelsize": 18,
            "titlesize": 18,
            "tick_labelsize": 18,
            "legend_fontsize": 18,
            "legend_title_fontsize": 18
        },
        "axis":{
            "xscale": "log",
            "yscale": "log",
            "xlim": (0.01, 10000),
            "ylim": (0.01, 10000),
        },
        "color": {
            "roofline": "#4A8462",
        }
    }

    # Load hardware config
    hardware_config_dict = load_yaml(args.hardware_config_path)
    hw = hardware_config_dict['GPU']
    hw["color"] = "#4A8462"
    hardware_configs = [hw]

    # Load Model roofline data
    point_datas = []
    # model_colors = ["#9467bd", "#8c564b"]
    model_colors = {
        "Qwen-Qwen2.5-3B-Instruct": ["#BFDBFE", "#60A5FA", "#2563EB", "#1E3A8A"],
        "state-spaces-mamba-2.8b-hf": ["#E8F5E9", "#A5D6A7", "#66BB6A", "#2E7D32"]
    }
    symbols = ["o", "*", "o", "s"] #{'triangle': '^', 'circle': 'o', 'square': 's'},
    
    for model_idx, model_name in enumerate(args.model_name_list): 
        for sum_seq_len_idx, sum_seq_len in enumerate(args.sum_seq_len_list):
            roofline_data_dir = get_leaf_dir(args.roofline_data_dir, model_name, args.batch_size, sum_seq_len, args.gen_seq_len)
            roofline_data_path = roofline_data_dir / "results.json"
            roofline_data = load_json(roofline_data_path)
            
            for kernel in roofline_data["kernels"]:
                point_label = ""
                # point_label = f"{kernel['name']} ({kernel['phase']})"

                if kernel["name"] == "overall": continue

                if kernel["phase"] == "prefill":
                    symbol = "o"
                elif kernel["phase"] == "decode":
                    symbol = "^"
                symbol_size = 72

                # color = model_colors[model_idx][sum_seq_len_idx]
                color = model_colors[model_name][sum_seq_len_idx]
                point = {
                    "y-axis": kernel["throughput"], 
                    "x-axis": kernel["arithmetic_intensity"],
                    "label": point_label,
                    "color": color,
                    "symbol": symbol,
                    "symbol_size": symbol_size,
                }
                point_datas.append(point)

                # # rooflien check
                # roof = min(hw["peak_compute"], hw["peak_bandwidth"]*kernel["arithmetic_intensity"] )
                # if kernel["throughput"] > roof:
                #     print(f"Out of roofline: {kernel['name']} ({kernel['phase']})")
                # else:
                #     print(f"Within roofline: {kernel['name']:10s} ({kernel['phase']}): throughput {kernel['throughput']:.2f} TFLOPS, arithmetic intensity {kernel['arithmetic_intensity']:.2f} FLOPs/Byte")


                # if kernel["throughput"] < 10 and kernel["arithmetic_intensity"] > 661:
                #     print(f"{kernel['name']} ({kernel['phase']}): throughput {kernel['throughput']:.2f} TFLOPS, arithmetic intensity {kernel['arithmetic_intensity']:.2f} FLOPs/Byte")

                # Kernel Check
                if sum_seq_len == 8192:
                    if model_name == "Qwen-Qwen2.5-3B-Instruct":
                        if kernel["phase"] == "prefill":
                            if kernel["arithmetic_intensity"] < 10:
                                print(f"{kernel['name']} ({kernel['phase']}): throughput {kernel['throughput']:.2f} TFLOPS, arithmetic intensity {kernel['arithmetic_intensity']:.2f} FLOPs/Byte")
                    elif model_name == "state-spaces-mamba-2.8b-hf":
                        if kernel["phase"] == "prefill":
                            if 5 < kernel["arithmetic_intensity"] < 10:
                                print(f"{kernel['name']} ({kernel['phase']}): throughput {kernel['throughput']:.2f} TFLOPS, arithmetic intensity {kernel['arithmetic_intensity']:.2f} FLOPs/Byte")
                            elif kernel["arithmetic_intensity"] < 5:
                                print(f"{kernel['name']} ({kernel['phase']}): throughput {kernel['throughput']:.2f} TFLOPS, arithmetic intensity {kernel['arithmetic_intensity']:.2f} FLOPs/Byte")
                    

    legends = {
        "phase_legend": {},
        "model_legend": {},
    }
    # Legend setting
    model_handles = []
    model_labels = []
    for model_idx, model_name in enumerate(args.model_name_list):
        # if model_idx >= len(model_colors):
        #     break
        # palette = model_colors[model_idx]
        palette = model_colors[model_name]
        handle_tuple = tuple(
            Line2D([0], [0], marker='s', linestyle='None', markersize=6,
                   markerfacecolor=c, markeredgecolor='none')
            for c in palette
        )

        # label
        if model_name == "Qwen-Qwen2.5-3B-Instruct":
            model_label = "Qwen2.5-3B-Instruct"
        elif model_name == "state-spaces-mamba-2.8b-hf":
            model_label = "Mamba-2.8B"
        else:
            model_label = model_name
        model_handles.append(handle_tuple)
        model_labels.append(model_label)
    phase_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='black',
               markeredgecolor='black', markersize=7, label='Prefill'),
        Line2D([0], [0], marker='^', linestyle='None', markerfacecolor='black',
               markeredgecolor='black', markersize=7, label='Decode'),
    ]
    legend_config = {
        "phase_legend": {
            "handles": phase_handles,
            "labels": ['Prefill', 'Decode'],
        },
        "model_legend": {
            "handles": model_handles,
            "labels": model_labels,
        },
    }
    



    plot_roofline(args, hardware_configs, point_datas, args.output_path, plot_config, legend_config)
    print(f"Saved figure to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
