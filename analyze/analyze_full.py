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
from util.system import get_result_lr, get_result_rlr, get_result_rpf


def sharpe(df: pd.DataFrame) -> pd.Series:
    """
    Annualized Sharpe ratio per column for monthly returns.
    """
    return np.sqrt(12.0) * df.mean(axis=0) / df.std(axis=0)


def alpha_tstat(
    target: pd.Series,
    benchmark: pd.Series,
) -> float:
    """
    t-statistic of the intercept from the regression:
        target_t = alpha + beta * benchmark_t + eps_t
    """
    work = pd.concat(
        [target.rename("target"), benchmark.rename("benchmark")],
        axis=1,
    ).dropna()

    y = work["target"].to_numpy(dtype=float)
    x = work["benchmark"].to_numpy(dtype=float)

    X = np.column_stack([np.ones(len(x), dtype=float), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)

    resid = y - X @ beta
    n_obs, n_param = X.shape
    sigma2 = float(resid @ resid) / (n_obs - n_param)
    vcov = sigma2 * xtx_inv

    alpha = float(beta[0])
    alpha_se = float(np.sqrt(vcov[0, 0]))
    return alpha / alpha_se


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


def sharpe_by_z(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Return Sharpe indexed by z for models with one column per z.
    """
    sr = sharpe(df)
    sr.index = pd.Index(sr.index.get_level_values("z").astype(float), name="z")
    return sr.sort_index()


def sharpe_by_depth_z(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return rpf Sharpe table with index=max_depth and columns=z.

    Assumes one column per (max_depth, z).
    """
    sr = sharpe(df)
    depths = sr.index.get_level_values("max_depth").astype(int)
    zs = sr.index.get_level_values("z").astype(float)

    out = pd.DataFrame(
        {"max_depth": depths, "z": zs, "sharpe": sr.to_numpy(dtype=float)}
    )
    out = out.pivot(index="max_depth", columns="z", values="sharpe")
    return out.sort_index().sort_index(axis=1)


def rpf_series_by_depth_z(
    df: pd.DataFrame,
) -> dict[tuple[int, float], pd.Series]:
    """
    Return rpf return series keyed by (max_depth, z).

    Assumes one column per (max_depth, z).
    """
    out: dict[tuple[int, float], pd.Series] = {}
    names = list(df.columns.names)
    depth_idx = names.index("max_depth")
    z_idx = names.index("z")

    for col in df.columns:
        depth = int(col[depth_idx])
        z = float(col[z_idx])
        out[(depth, z)] = df[col]

    return out


def series_by_z(
    df: pd.DataFrame,
) -> dict[float, pd.Series]:
    """
    Return return series keyed by z for LR/RLR.

    Assumes one column per z.
    """
    out: dict[float, pd.Series] = {}
    names = list(df.columns.names)
    z_idx = names.index("z")

    for col in df.columns:
        z = float(col[z_idx])
        out[z] = df[col]

    return out


def format_depth_label(depth: int) -> str:
    """
    Format depth as a power-of-2 label, e.g. 1 -> 2^0, 2 -> 2^1, 4 -> 2^2.
    Assumes depth is a power of 2.
    """
    exponent = int(np.log2(depth))
    return rf"Depth = $2^{{{exponent}}}$"


def format_depth_tick(depth: int) -> str:
    """
    Format depth tick labels as powers of 2.
    """
    exponent = int(np.log2(depth))
    return rf"$2^{{{exponent}}}$"


def format_shrinkage_label(z: float) -> str:
    """
    Format shrinkage values for legend labels.
    """
    return f"z = {z:g}"


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
    rpf_sr = sharpe_by_depth_z(rpf)

    lr_series = series_by_z(lr)
    rlr_series = series_by_z(rlr)
    rpf_series = rpf_series_by_depth_z(rpf)

    common_z = sorted(set(lr_sr.index) & set(rlr_sr.index) & set(rpf_sr.columns))
    depths = rpf_sr.index.to_list()

    plot_depth = 1

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

    fig, axes = make_figure(nrows=1, ncols=2, width=13.6, height=4.1)
    ax_left, ax_right = axes

    lineplot(
        ax=ax_left,
        x=common_z,
        y=lr_sr.loc[common_z].to_numpy(),
        label="LR",
        color=get_line_color(0),
        marker="o",
        xlabel="Shrinkage z",
        ylabel="Annualized Sharpe",
        title="Sharpe as a Function of Shrinkage",
    )
    lineplot(
        ax=ax_left,
        x=common_z,
        y=rlr_sr.loc[common_z].to_numpy(),
        label="RLR",
        color=get_line_color(1),
        marker="o",
    )
    lineplot(
        ax=ax_left,
        x=common_z,
        y=rpf_sr.loc[plot_depth, common_z].to_numpy(),
        label="RPF",
        color=get_line_color(2),
        marker="o",
    )
    ax_left.set_xscale("log")
    ax_left.tick_params(axis="x", which="minor", bottom=False, top=False)
    format_legend(ax_left, outside=True)

    lineplot(
        ax=ax_right,
        x=common_z,
        y=alpha_vs_lr.to_numpy(),
        label="Bench: LR",
        color=get_line_color(0),
        marker="o",
        xlabel="Shrinkage z",
        ylabel="Alpha t-stat",
        title="RPF Alpha t-stat by Shrinkage",
        zero=False,
    )
    lineplot(
        ax=ax_right,
        x=common_z,
        y=alpha_vs_rlr.to_numpy(),
        label="Bench: RLR",
        color=get_line_color(1),
        marker="o",
    )
    ax_right.axhline(
        2.576,
        color="0.35",
        linewidth=0.9,
        linestyle="--",
        label="1% threshold",
    )
    ax_right.set_xscale("log")
    ax_right.tick_params(axis="x", which="minor", bottom=False, top=False)
    format_legend(ax_right, outside=True)

    savefig(fig, "sharpe_jkp_summary")

    print(rpf_sr)

    fig, ax = make_figure(width=6.8, height=4.1)
    for i, depth in enumerate(depths):
        lineplot(
            ax=ax,
            x=common_z,
            y=rpf_sr.loc[depth, common_z].to_numpy(),
            label=format_depth_label(depth),
            color=get_line_color(i),
            marker="o",
            xlabel="Shrinkage z",
            ylabel="Annualized Sharpe",
            title="RPF Sharpe as a Function of Shrinkage",
        )
    ax.set_xscale("log")
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    format_legend(ax, outside=True)
    savefig(fig, "sharpe_jkp_by_depth")

    fig, ax = make_figure(width=6.8, height=4.1)
    for i, z in enumerate(common_z):
        lineplot(
            ax=ax,
            x=depths,
            y=rpf_sr.loc[depths, z].to_numpy(),
            label=format_shrinkage_label(z),
            color=get_line_color(i),
            marker="o",
            xlabel="Depth",
            ylabel="Annualized Sharpe",
            title="RPF Sharpe as a Function of Depth",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(depths)
    ax.set_xticklabels([format_depth_tick(depth) for depth in depths])
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    format_legend(ax, outside=True)
    savefig(fig, "sharpe_jkp_by_shrinkage")