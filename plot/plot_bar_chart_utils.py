from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


Number = Union[int, float]
MaybeNumber = Optional[Number]
NormalizeSetting = Optional[Union[float, int, str]]

# --------------------------------
# Bar Chart
# --------------------------------

@dataclass
class BarChartData:
    """
    3-level nested data:
      data[subplot_id][group_id][bar_id] -> value (float or None)

    Minimal naming convention (recommended):
      - subplots: models / configurations shown as separate panels
      - groups  : x-axis groups inside each subplot (e.g., batch sizes)
      - bars    : categories within each group (e.g., num_pimlets values)
    """
    # core data
    bar_data: List[List[List[MaybeNumber]]]

    # labels
    subplot_labels: List[str]                  # len == N_subplot
    group_labels: List[str]                    # len == N_group (shared across all subplots)
    bar_labels: List[str]                      # len == N_bar   (shared across all subplots)

    # styling / mapping
    bar_colors: Optional[List[str]] = None     # len == N_bar, if None -> matplotlib default cycle
    bar_hatches: Optional[List[str]] = None    # len == N_bar, optional
    bar_edgecolor: str = "black"
    bar_linewidth: float = 1.0
    bar_alpha: float = 1.0

    # optional overlay line (e.g., energy efficiency)
    # line_data[subplot_id][group_id] -> y (float or None)
    line_data: Optional[List[List[List[MaybeNumber]]]] = None
    line_label: str = ""
    line_color: str = "tab:red"
    line_marker: str = "o"
    line_linestyle: str = "-"
    line_linewidth: float = 2.0
    line_markersize: float = 5.0

    # y-axis labels
    left_ylabel: str = ""
    right_ylabel: str = ""

    # units / scaling / normalization
    # normalize_by: Optional[Number] = None      # if set: value /= normalize_by
    # normalize_per_subplot: bool = False        # if True: each subplot normalized by its own baseline
    normalize: NormalizeSetting = None  # applies to both axes unless overridden below
    # baseline selector: (group_id, bar_id)
    normalize_baseline: Tuple[int, int] = (0, 0)

    # Optional per-axis normalization overrides.
    # - left axis == bars
    # - right axis == line (twinx)
    # Use "inherit" to reuse `normalize` / `normalize_baseline`.
    # Use None to disable normalization on that axis.
    normalize_left: NormalizeSetting = "inherit"
    normalize_baseline_left: Optional[Tuple[int, int]] = None
    normalize_right: NormalizeSetting = "inherit"
    normalize_baseline_right: Optional[Tuple[int, int]] = None

    # misc
    title: str = ""
    legend: bool = True
    legend_loc: str = "upper right"
    legend_ncol: int = 1

    def shape(self) -> Tuple[int, int, int]:
        ns = len(self.bar_data)
        ng = len(self.bar_data[0]) if ns else 0
        nb = len(self.bar_data[0][0]) if (ns and ng) else 0
        return ns, ng, nb

    def validate(self) -> None:
        ns, ng, nb = self.shape()
        assert ns == len(self.subplot_labels), "subplot_labels length mismatch"
        assert ng == len(self.group_labels), "group_labels length mismatch"
        assert nb == len(self.bar_labels), "bar_labels length mismatch"
        for s in range(ns):
            assert len(self.bar_data[s]) == ng, f"bar_data[{s}] group length mismatch"
            for g in range(ng):
                assert len(self.bar_data[s][g]) == nb, f"bar_data[{s}][{g}] bar length mismatch"
        if self.bar_colors is not None:
            assert len(self.bar_colors) == nb, f"bar_colors length mismatch, bar_colors length: {len(self.bar_colors)}, nb: {nb}"
        if self.bar_hatches is not None:
            assert len(self.bar_hatches) == nb, "bar_hatches length mismatch"
        if self.line_data is not None:
            assert len(self.line_data) == ns, "line_data subplot length mismatch"
            for s in range(ns):
                assert len(self.line_data[s]) == ng, f"line_data[{s}] group length mismatch"

        # line data validation
        if self.line_data is not None:
            ns, ng, nb = self.shape()
            assert len(self.line_data) == ns, "line_data subplot length mismatch"
            for s in range(ns):
                assert len(self.line_data[s]) == ng, f"line_data[{s}] group length mismatch"
                for g in range(ng):
                    assert len(self.line_data[s][g]) == nb, f"line_data[{s}][{g}] bar length mismatch"


def _apply_normalization(
    data: np.ndarray,
    normalize: NormalizeSetting,
    baseline: Tuple[int, int],
) -> np.ndarray:
    if normalize is None:
        return data

    if isinstance(normalize, (float, int)):
        return data / float(normalize)

    if normalize == "per_subplot":
        bg, bb = baseline
        baseline_vals = data[:, bg, bb]  # shape (ns,)
        baseline_vals = np.where((baseline_vals == 0) | np.isnan(baseline_vals), np.nan, baseline_vals)
        return data / baseline_vals[:, None, None]

    if normalize == "per_group":
        _bg, bb = baseline
        baseline_vals = data[:, :, bb]  # shape (ns, ng)
        baseline_vals = np.where((baseline_vals == 0) | np.isnan(baseline_vals), np.nan, baseline_vals)
        return data / baseline_vals[:, :, None]

    raise ValueError(
        f"Unsupported normalize setting: {normalize!r}. Use a number, 'per_subplot', 'per_group', or None."
    )


def plot_barchart_grid(
    payload: BarChartData,
    *,
    figsize: Tuple[float, float] = (12, 3.0),
    sharey: bool = True,
    y_lim_right: Optional[Tuple[float, float]] = None,
    y_lim_left: Optional[Tuple[float, float]] = None,
    left_yscale: str = "linear",  # e.g., "linear" or "log"
    grid: bool = True,
    grid_axis: str = "y",
    grid_linestyle: str = "--",
    grid_linewidth: float = 1.0,
    x_tick_rotation: float = 0.0,
    group_gap: float = 0.25,     # gap between groups
    bar_gap: float = 0.0,        # gap between bars within a group
    show: bool = True,
    savepath: Optional[Union[str, Path]] = None,
    font_sizes: Optional[Dict[str, float]] = None,
    offsets: Optional[Dict[str, float]] = None,
    tight_layout_kwargs: Optional[Dict[str, float]] = None,
    subplots_adjust_kwargs: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """
    Horizontal subplots. Each subplot shows grouped bars, optionally plus a right-axis line.
    """
    payload.validate()
    ns, ng, nb = payload.shape()


    if font_sizes is None:
        font_sizes = {
            "title": 14,
            "subplot_title": 18,
            "bar_legend": 14,
            "xticks": 14,
            "yticks": 14,
            "left_ylabel": 16,
            "right_ylabel": 14
        }
    
    if offsets is None:
        offsets = {
            "subplot_title_y": 0.82,
            "bar_legend_y": 1.1,
        }

    # ---- normalization ----
    bar_data = np.array(payload.bar_data, dtype=float)  # None -> nan
    bar_data = np.where(np.isfinite(bar_data), bar_data, np.nan)  # convert None to nan

    normalize_left = payload.normalize if payload.normalize_left == "inherit" else payload.normalize_left
    baseline_left = payload.normalize_baseline_left or payload.normalize_baseline
    bar_data = _apply_normalization(bar_data, normalize_left, baseline_left)

    if left_yscale not in {"linear", "log"}:
        raise ValueError(f"Unsupported left_yscale: {left_yscale!r}. Use 'linear' or 'log'.")

    min_positive_left = None
    if left_yscale == "log":
        positive = bar_data[np.isfinite(bar_data) & (bar_data > 0)]
        if positive.size:
            min_positive_left = float(np.nanmin(positive))

    # line normalization
    line_data = None
    if payload.line_data is not None:
        line_data = np.array(payload.line_data, dtype=float)  # None -> nan
        line_data = np.where(np.isfinite(line_data), line_data, np.nan)  # convert None to nan

        normalize_right = payload.normalize if payload.normalize_right == "inherit" else payload.normalize_right
        baseline_right = payload.normalize_baseline_right or payload.normalize_baseline
        line_data = _apply_normalization(line_data, normalize_right, baseline_right)

    # ---- layout ----
    fig, axes = plt.subplots(1, ns, figsize=figsize, sharey=sharey)
    if ns == 1:
        axes = [axes]

    # x positions for groups
    group_x = np.arange(ng, dtype=float)

    # bar width computation
    # total width occupied by bars in a group: nb*bar_w + (nb-1)*bar_gap
    total_group_width = 1.0 - group_gap
    bar_w = (total_group_width - (nb - 1) * bar_gap) / nb
    # bar offsets centered around group_x
    bar_offsets = (np.arange(nb) - (nb - 1) / 2.0) * (bar_w + bar_gap)

    colors = payload.bar_colors  # may be None -> matplotlib default
    hatches = payload.bar_hatches

    for s, ax in enumerate(axes):
        if left_yscale == "log":
            ax.set_yscale("log")

        # bars
        for b in range(nb):
            y = bar_data[s, :, b]  # shape (ng,)
            ax.bar(
                group_x + bar_offsets[b],
                y,
                width=bar_w,
                label=payload.bar_labels[b] if s == 0 else None,  # one legend for the whole fig
                color=None if colors is None else colors[b],
                alpha=payload.bar_alpha,
                edgecolor=payload.bar_edgecolor,
                linewidth=payload.bar_linewidth,
                hatch=None if (hatches is None) else hatches[b],
            )
        ax.set_title(payload.subplot_labels[s], fontsize=font_sizes["subplot_title"], y=offsets["subplot_title_y"])
        ax.set_xticks(group_x)
        ax.set_xticklabels(payload.group_labels, rotation=x_tick_rotation, fontsize=font_sizes["xticks"])
        ax.tick_params(axis="y", labelsize=font_sizes["yticks"])
        if grid:
            ax.grid(True, axis=grid_axis, linestyle=grid_linestyle, linewidth=grid_linewidth)
        if y_lim_left is not None:
            y0, y1 = y_lim_left
            if left_yscale == "log" and y0 <= 0:
                if min_positive_left is None:
                    raise ValueError("left_yscale='log' requires positive y data/limits.")
                y0 = min_positive_left * 0.8
            ax.set_ylim(y0, y1)
        if left_yscale == "log":
            # Force *only* powers-of-two major y-ticks (0.5, 1, 2, 4, 8, ...).
            ax.yaxis.set_major_locator(mticker.LogLocator(base=2, subs=(1.0,)))
            ax.yaxis.set_minor_locator(mticker.NullLocator())

            def _fmt_pow2(y, _pos):
                if not np.isfinite(y):
                    return ""
                if abs(y - round(y)) < 1e-9:
                    return str(int(round(y)))
                return f"{y:g}"

            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_pow2))
        

        # optional line on right axis (like your figure)
        if line_data is not None:
            ax2 = ax.twinx()
            for g in range(ng):
                x = group_x[g] + bar_offsets
                y = line_data[s, g, :]
                label = payload.line_label if (s == ns - 1 and g == 0) else None

                ax2.plot(
                    x,
                    y,
                    label=label,
                    color=payload.line_color,
                    marker=payload.line_marker,
                    linestyle=payload.line_linestyle,
                    linewidth=payload.line_linewidth,
                    markersize=payload.line_markersize,
                )
            if payload.right_ylabel and (s == ns - 1):
                ax2.set_ylabel(payload.right_ylabel, fontsize=font_sizes["right_ylabel"])
                ax2.tick_params(axis="y", labelsize=font_sizes["yticks"])
            elif payload.right_ylabel and (s != ns - 1):
                ax2.set_ylabel("")
                ax2.set_yticklabels([])
                ax2.tick_params(axis="y", left=False)
            
            if y_lim_right is not None:
                ax2.set_ylim(*y_lim_right)

        if payload.left_ylabel and (s == 0):
            ax.set_ylabel(payload.left_ylabel, fontsize=font_sizes["left_ylabel"])

    if payload.title:
        fig.suptitle(payload.title, fontsize=font_sizes["title"])

    # legend (bars + optional line)
    if payload.legend:
        handles, labels = [], []
        # collect from first axis (bars)
        h0, l0 = axes[0].get_legend_handles_labels()
        handles += h0
        labels += l0
        # optional line legend: pull from last axis' twin if exists
        if line_data is not None:
            # if twin exists, it is the last created Axes on the same subplot
            # robust approach: scan fig axes for lines with matching label
            for a in fig.axes:
                h, l = a.get_legend_handles_labels()
                for hh, ll in zip(h, l):
                    if ll and ll not in labels:
                        handles.append(hh)
                        labels.append(ll)
                        
        legend_kwargs = dict(
            loc=payload.legend_loc,
            ncol=payload.legend_ncol,
            fontsize=font_sizes.get("bar_legend"),
            bbox_to_anchor=(0.5, offsets["bar_legend_y"]),
        )
        fig.legend(handles, labels, **legend_kwargs)

        # fig.legend(handles, labels, loc=payload.legend_loc, ncol=payload.legend_ncol, fontsize=font_sizes["bar_legend"])

    fig.tight_layout(**(tight_layout_kwargs or {}))
    if subplots_adjust_kwargs:
        fig.subplots_adjust(**subplots_adjust_kwargs)
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        # Save both PNG and PDF by default for paper-ready figures.
        # - If `savepath` ends with `.png` -> also save `.pdf`
        # - If `savepath` ends with `.pdf` -> also save `.png`
        # - Otherwise -> save `<savepath>.png` and `<savepath>.pdf`
        suffix = savepath.suffix.lower()
        if suffix in {".png", ".pdf"}:
            savepaths = [savepath, savepath.with_suffix(".pdf" if suffix == ".png" else ".png")]
        else:
            savepaths = [savepath.with_suffix(".png"), savepath.with_suffix(".pdf")]

        seen = set()
        for sp in savepaths:
            if sp in seen:
                continue
            seen.add(sp)
            fig.savefig(str(sp), dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    return fig



def main():
    pass


if __name__ == "__main__":
    main()