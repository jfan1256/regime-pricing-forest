# analyze_full.py
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter

from util.analyze import (
    alpha_tstat,
    filter_sample,
    sharpe_by_z,
    sharpe_pivot,
    series_by_levels,
    series_by_z,
)
from util.load import load_sdfs
from util.plot import (
    make_figure,
    savefig,
    lineplot,
    format_legend,
    get_model_color,
    get_line_color,
    get_shrinkage_color,
)
from util.system import get_result_lr, get_result_rlr, get_result_rpf


def format_depth_label(depth: int) -> str:
    """
    Format depth labels using plain numeric values.
    """
    return f"Depth = {depth}"


def format_depth_tick(depth: int) -> str:
    """
    Format depth tick labels using plain numeric values.
    """
    return f"{depth}"


def format_shrinkage_label(z: float) -> str:
    """
    Format shrinkage values for legend labels as 10^k.
    """
    exponent = int(round(np.log10(z)))
    if not np.isclose(z, 10.0**exponent):
        return f"z = {z:g}"
    return rf"$z = 10^{{{exponent}}}$"


def format_shrinkage_tick(z: float) -> str:
    """
    Format shrinkage values for x-axis tick labels as 10^k.
    """
    exponent = int(round(np.log10(z)))
    if not np.isclose(z, 10.0**exponent):
        return f"{z:g}"
    return rf"$10^{{{exponent}}}$"


def apply_log_z_ticks(ax, z_values: list[float]) -> None:
    """
    Apply log-scale x-axis with fixed 10^k tick labels.
    """
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(z_values))
    ax.xaxis.set_major_formatter(
        FixedFormatter([format_shrinkage_tick(z) for z in z_values])
    )
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)


if __name__ == "__main__":
    start = "1993-01-31"
    end = "2024-12-31"

    lr = filter_sample(load_sdfs("lr_full.yml", get_result_lr()), start=start, end=end)
    rlr = filter_sample(load_sdfs("rlr_full.yml", get_result_rlr()), start=start, end=end)
    rpf = filter_sample(load_sdfs("rpf_full_depth.yml", get_result_rpf()), start=start, end=end)

    if lr.empty:
        raise ValueError("No loaded LR results found.")
    if rlr.empty:
        raise ValueError("No loaded RLR results found.")
    if rpf.empty:
        raise ValueError("No loaded RPF results found.")

    lr_sr = sharpe_by_z(lr)
    rlr_sr = sharpe_by_z(rlr)
    rpf_sr = sharpe_pivot(rpf, index_level="max_depth", column_level="z")

    lr_series = series_by_z(lr)
    rlr_series = series_by_z(rlr)
    rpf_series = series_by_levels(rpf, ["max_depth", "z"])

    common_z = sorted(set(lr_sr.index) & set(rlr_sr.index) & set(rpf_sr.columns))
    depths = [int(depth) for depth in rpf_sr.index.to_list()]
    z_colors = {z: get_shrinkage_color(z) for z in common_z}

    plot_depth = 3

    alpha_vs_lr = pd.Series(
        {
            z: alpha_tstat(rpf_series[(plot_depth, z)], lr_series[z])
            for z in common_z
        }
    ).sort_index()

    alpha_vs_rlr = pd.Series(
        {
            z: alpha_tstat(rpf_series[(plot_depth, z)], rlr_series[z])
            for z in common_z
        }
    ).sort_index()

    fig, axes = make_figure(nrows=1, ncols=2, width=6.8, height=3.5)
    ax_left, ax_right = axes

    lineplot(
        ax=ax_left,
        x=common_z,
        y=lr_sr.loc[common_z].to_numpy(),
        label="LR",
        color=get_model_color("LR"),
        marker="o",
        xlabel="Shrinkage z",
        ylabel="Sharpe",
    )
    lineplot(
        ax=ax_left,
        x=common_z,
        y=rlr_sr.loc[common_z].to_numpy(),
        label="RLR",
        color=get_model_color("RLR"),
        marker="o",
    )
    lineplot(
        ax=ax_left,
        x=common_z,
        y=rpf_sr.loc[plot_depth, common_z].to_numpy(),
        label="RPF",
        color=get_model_color("RPF"),
        marker="o",
    )
    apply_log_z_ticks(ax_left, common_z)
    format_legend(ax_left, outside=False)

    lineplot(
        ax=ax_right,
        x=common_z,
        y=alpha_vs_lr.to_numpy(),
        label="vs LR",
        color=get_model_color("vs LR"),
        marker="o",
        xlabel="Shrinkage z",
        ylabel="Alpha t-stat",
    )
    lineplot(
        ax=ax_right,
        x=common_z,
        y=alpha_vs_rlr.to_numpy(),
        label="vs RLR",
        color=get_model_color("vs RLR"),
        marker="o",
    )
    apply_log_z_ticks(ax_right, common_z)
    format_legend(ax_right, outside=False)

    savefig(fig, "full_sharpe_alpha_shrinkage")

    fig, ax = make_figure(width=6.5, height=3.5)
    for i, depth in enumerate(depths):
        lineplot(
            ax=ax,
            x=common_z,
            y=rpf_sr.loc[depth, common_z].to_numpy(),
            label=format_depth_label(depth),
            color=get_line_color(i),
            marker="o",
            xlabel="Shrinkage z",
            ylabel="Sharpe",
        )
    apply_log_z_ticks(ax, common_z)
    format_legend(ax, outside=False)
    savefig(fig, "full_sharpe_depth_by_shrinkage")

    fig, ax = make_figure(width=6.5, height=3.5)
    for z in common_z:
        lineplot(
            ax=ax,
            x=depths,
            y=rpf_sr.loc[depths, z].to_numpy(),
            label=format_shrinkage_label(z),
            color=z_colors[z],
            marker="o",
            xlabel="Depth",
            ylabel="Sharpe",
        )
    ax.set_xticks(depths)
    ax.set_xticklabels([format_depth_tick(depth) for depth in depths])
    format_legend(ax, outside=False)
    savefig(fig, "full_sharpe_depth")