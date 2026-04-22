# analyze_num.py
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter

from util.analyze import filter_sample, sharpe_pivot
from util.load import load_sdfs
from util.plot import (
    make_figure,
    savefig,
    lineplot,
    format_legend,
    get_shrinkage_color,
)
from util.system import get_result_rpf


def format_power_of_two_ticks(values: list[int]) -> list[str]:
    """
    Format integer powers of two as 2^k.
    """
    labels: list[str] = []
    for v in values:
        k = np.log2(v)
        if np.isclose(k, round(k)):
            labels.append(rf"$2^{{{int(round(k))}}}$")
        else:
            labels.append(str(v))
    return labels


def format_shrinkage_label(z: float) -> str:
    """
    Format shrinkage parameter as 10^k for legend labels.
    """
    exponent = int(round(np.log10(z)))
    if not np.isclose(z, 10.0**exponent):
        return f"z = {z:g}"
    return rf"$z = 10^{{{exponent}}}$"


if __name__ == "__main__":
    start = "1993-01-31"
    end = "2024-12-31"
    plot_depth = 3

    rpf = filter_sample(load_sdfs("rpf_full_num.yml", get_result_rpf()), start=start, end=end)

    if rpf.empty:
        raise ValueError("No loaded RPF results found.")

    rpf_depth = rpf.loc[
        :,
        rpf.columns.get_level_values("max_depth").astype(int) == plot_depth,
    ]
    rpf_sr = sharpe_pivot(rpf_depth, index_level="num_tree", column_level="z")

    num_trees = [int(v) for v in rpf_sr.index.to_list()]
    z_list = [float(v) for v in rpf_sr.columns.to_list()]
    z_colors = {z: get_shrinkage_color(z) for z in z_list}

    fig, ax = make_figure(width=6.8, height=3.5)
    for z in z_list:
        lineplot(
            ax=ax,
            x=num_trees,
            y=rpf_sr.loc[num_trees, z].to_numpy(),
            label=format_shrinkage_label(z),
            color=z_colors[z],
            marker="o",
            xlabel="Number of Trees",
            ylabel="Sharpe",
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(num_trees))
    ax.xaxis.set_major_formatter(FixedFormatter(format_power_of_two_ticks(num_trees)))

    format_legend(ax, outside=False)
    savefig(fig, "full_sharpe_num_tree")