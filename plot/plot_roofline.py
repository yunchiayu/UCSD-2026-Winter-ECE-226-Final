import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-config-path", type=str, default="./gpu_simulation/hardware_config/RTX4090.yaml")
    parser.add_argument("--output-path", type=str, default="./figures/roofline.png")
    return parser.parse_args()

def load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)

def plot_roofline(args, hardware_configs, output_path: str, plot_config: dict):
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
    xlim = axis.get("xlim", (0.1, 1200))
    ylim = axis.get("ylim", (0.1, 1200))
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
        xs = np.logspace(np.log10(ax.get_xlim()[0]), np.log10(ax.get_xlim()[1]), 200)
        ys = np.minimum(pbw * xs, pf)
        color = plot_config['color']['roofline']
        roofline = ax.plot(xs, ys, color=color, linewidth=2, label=hw["name"])[0]
        roofline_handles.append(roofline)
        roofline_labels.append(hw["name"]) 
    ymin, _ = ax.get_ylim()
    ax.plot([knee, knee], [ymin, pf], linestyle='--', color=color)
    ax.axhline(pbw * knee, linestyle='--', color=color)
    ax.text(knee * 1.02, ymin * 1.1, f'{knee:.2f}', rotation=0, va='bottom', ha='center', color=color, zorder=5)
    ax.text(0.085, pf * 1.04, f'{pf:.1f}', rotation=0, va='bottom', ha='right',
            color=color, transform=ax.get_yaxis_transform(), zorder=5, clip_on=False)
    
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
    
    plt.savefig(output_path, dpi=plot_config.get('dpi', 300), bbox_inches='tight')
    return fig, ax




def main(args):

    plot_config = {
        "figsize": (10, 10),
        "fontsize": {
            "labelsize": 12,
            "titlesize": 12,
            "tick_labelsize": 12,
            "legend_fontsize": 12,
            "legend_title_fontsize": 12
        },
        "axis":{
            "xscale": "log",
            "yscale": "log",
            "xlim": (0.1, 10000),
            "ylim": (0.1, 10000),
        },
        "symbol_map": {'triangle': '^', 'circle': 'o', 'square': 's'},
        "color": {
            "roofline": "#4A8462",
        }
    }

    hardware_config_dict = load_yaml(args.hardware_config_path)
    hardware_configs = [hardware_config_dict['GPU']]
    plot_roofline(args, hardware_configs, args.output_path, plot_config)
    print(f"Saved figure to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)