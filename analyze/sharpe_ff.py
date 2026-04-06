import numpy as np
import pandas as pd

from util.load import load_sdfs
from util.plot import (
    make_figure,
    savefig,
    barplot,
    lineplot,
    get_bar_colors,
    get_line_color,
)
from util.system import get_result_lr, get_result_rlr, get_result_rpf, get_result_rpt


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


def best_model(
    df: pd.DataFrame,
) -> tuple[tuple, pd.Series, float]:
    """
    Return best column key, series, and Sharpe.
    """
    if df.empty:
        raise ValueError("No loaded results found for this model.")

    sr = sharpe(df)
    key = sr.idxmax()
    return key, df[key], float(sr.loc[key])


def best_sharpe_by_depth(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Return the best Sharpe within each RMST depth.
    """
    if df.empty:
        raise ValueError("No loaded RMST results found.")

    sr = sharpe(df)
    depths = sorted(sr.index.get_level_values("max_depth").unique())

    out: dict[int, float] = {}
    for depth in depths:
        mask = sr.index.get_level_values("max_depth") == depth
        out[int(depth)] = float(sr.loc[mask].max())

    return pd.Series(out).sort_index()


def vol_target_series(
    s: pd.Series,
    target_vol: float = 0.10,
) -> pd.Series:
    """
    Scale monthly returns to a target annualized volatility over the full sample.
    """
    s = s.dropna().copy()
    ann_vol = float(s.std() * np.sqrt(12.0))
    if ann_vol == 0.0:
        return s * 0.0
    return s * (target_vol / ann_vol)


def cumulative_sum_return(
    s: pd.Series,
    target_vol: float = 0.10,
) -> pd.Series:
    """
    Arithmetic cumulative return after full-sample volatility targeting.
    """
    return vol_target_series(s, target_vol=target_vol).cumsum()


if __name__ == "__main__":
    start = "1993-01-31"
    end = "2024-12-31"

    lr = filter_sample(load_sdfs("lr.yml", get_result_lr()), start=start, end=end)
    rlr = filter_sample(load_sdfs("rlr.yml", get_result_rlr()), start=start, end=end)
    rmst = filter_sample(load_sdfs("rmst.yml", get_result_rpf()), start=start, end=end)

    lr_key, lr_best, lr_sr = best_model(lr)
    rlr_key, rlr_best, rlr_sr = best_model(rlr)
    rmst_key, rmst_best, rmst_sr = best_model(rmst)

    tstat_vs_lr = alpha_tstat(rmst_best, lr_best)
    tstat_vs_rlr = alpha_tstat(rmst_best, rlr_best)

    lr_cum = cumulative_sum_return(lr_best, target_vol=0.10)
    rlr_cum = cumulative_sum_return(rlr_best, target_vol=0.10)
    rmst_cum = cumulative_sum_return(rmst_best, target_vol=0.10)

    fig, axes = make_figure(nrows=1, ncols=3, width=13.0, height=3.6)
    ax_left, ax_mid, ax_right = axes

    barplot(
        ax=ax_left,
        x=["LR", "RLR", "RMST"],
        y=[lr_sr, rlr_sr, rmst_sr],
        colors=get_bar_colors(3),
        ylabel="Annualized Sharpe",
        title="Model Sharpe Ratios",
        zero=True,
    )

    barplot(
        ax=ax_mid,
        x=["vs LR", "vs RLR"],
        y=[tstat_vs_lr, tstat_vs_rlr],
        colors=get_bar_colors(2),
        ylabel="Alpha t-stat",
        title="RMST Alpha t-stat Relative to Benchmarks",
        zero=True,
    )

    lineplot(
        ax=ax_right,
        x=lr_cum.index,
        y=lr_cum.to_numpy(),
        label="LR",
        color=get_line_color(0),
    )
    lineplot(
        ax=ax_right,
        x=rlr_cum.index,
        y=rlr_cum.to_numpy(),
        label="RLR",
        color=get_line_color(1),
    )
    lineplot(
        ax=ax_right,
        x=rmst_cum.index,
        y=rmst_cum.to_numpy(),
        label="RMST",
        color=get_line_color(2),
    )
    ax_right.set_ylabel("Cumulative Return")
    ax_right.set_title("Cumulative Return at 10% Vol")
    ax_right.legend()

    savefig(fig, "sharpe_ff_summary")

    depth_sr = best_sharpe_by_depth(rmst)

    fig, ax = make_figure(width=6.5, height=4.0)
    lineplot(
        ax=ax,
        x=depth_sr.index.to_list(),
        y=depth_sr.to_numpy(),
        color=get_line_color(0),
        marker="o",
        xlabel="Depth",
        ylabel="Annualized Sharpe",
        title="RMST Sharpe by Depth",
    )
    ax.set_xticks(depth_sr.index.to_list())

    savefig(fig, "sharpe_ff_depth")