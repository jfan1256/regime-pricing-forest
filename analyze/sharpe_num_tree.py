import numpy as np
import pandas as pd

from util.load import load_sdfs
from util.plot import (
    make_figure,
    savefig,
    lineplot,
    format_legend,
    get_line_color,
)
from util.system import get_result_rpf


def sharpe(df: pd.DataFrame) -> pd.Series:
    """
    Annualized Sharpe ratio per column for monthly returns.
    """
    return np.sqrt(12.0) * df.mean(axis=0) / df.std(axis=0)


def filter_sample(
    df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Restrict a panel to a date interval.
    """
    if df.empty:
        return df

    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out.loc[(out.index >= start) & (out.index <= end)]


def sharpe_by_z_num_tree(
    df: pd.DataFrame,
    depth: int,
) -> pd.DataFrame:
    """
    Return RMST Sharpe table with index=num_tree and columns=z at a fixed depth.

    Assumes one column per (max_depth, z, num_tree).
    """
    sr = sharpe(df)

    depth_mask = sr.index.get_level_values("max_depth").astype(int) == depth
    sr = sr.loc[depth_mask]

    if sr.empty:
        raise ValueError(f"No RMST results found for depth {depth}.")

    zs = sr.index.get_level_values("z").astype(float)
    num_trees = sr.index.get_level_values("num_tree").astype(int)

    out = pd.DataFrame(
        {
            "z": zs,
            "num_tree": num_trees,
            "sharpe": sr.to_numpy(dtype=float),
        }
    )
    out = out.pivot(index="num_tree", columns="z", values="sharpe")
    return out.sort_index().sort_index(axis=1)


if __name__ == "__main__":
    start = "1993-01-31"
    end = "2024-12-31"
    plot_depth = 3

    rmst = filter_sample(load_sdfs("rmst.yml", get_result_rpf()), start=start, end=end)

    if rmst.empty:
        raise ValueError("No loaded RMST results found.")

    rmst_sr = sharpe_by_z_num_tree(rmst, depth=plot_depth)

    num_trees = rmst_sr.index.to_list()
    z_list = rmst_sr.columns.to_list()

    fig, ax = make_figure(width=6.8, height=4.1)
    for i, z in enumerate(z_list):
        lineplot(
            ax=ax,
            x=num_trees,
            y=rmst_sr.loc[num_trees, z].to_numpy(),
            label=f"z = {z:g}",
            color=get_line_color(i),
            marker="o",
            xlabel="Number of Trees",
            ylabel="Annualized Sharpe",
            title=f"RMST Sharpe by Number of Trees (Depth = {plot_depth})",
        )

    ax.set_xticks(num_trees)
    format_legend(ax, outside=True)
    savefig(fig, "sharpe_num_tree")