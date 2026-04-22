# analyze_full_real.py
import matplotlib.dates as mdates

from util.analyze import (
    alpha_tstat,
    cumulative_sum_return,
    filter_column_levels,
    filter_sample,
    inverse_vol_ensemble_return,
    series_by_z,
    sharpe,
)
from util.load import load_sdfs
from util.plot import (
    make_figure,
    savefig,
    barplot,
    lineplot,
    get_model_color,
)
from util.system import get_result_lr, get_result_rlr, get_result_rpf


def format_date_axis(ax) -> None:
    """
    Format a compact date axis without xlabel or rotated ticks.
    """
    ax.xaxis.set_major_locator(mdates.YearLocator(base=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)


if __name__ == "__main__":
    start = "1993-01-31"
    end = "2024-12-31"
    vol_window = 12
    plot_depth = 3
    plot_num_tree = 64

    lr = filter_sample(load_sdfs("lr_full.yml", get_result_lr()), start=start, end=end)
    rlr = filter_sample(load_sdfs("rlr_full.yml", get_result_rlr()), start=start, end=end)
    rpf = filter_sample(load_sdfs("rpf_full_num.yml", get_result_rpf()), start=start, end=end)

    if lr.empty:
        raise ValueError("No loaded LR results found.")
    if rlr.empty:
        raise ValueError("No loaded RLR results found.")
    if rpf.empty:
        raise ValueError("No loaded RPF results found.")

    rpf_fixed = filter_column_levels(
        rpf,
        filters={"max_depth": plot_depth, "num_tree": plot_num_tree},
        drop_filtered_levels=True,
    )

    lr_real = inverse_vol_ensemble_return(series_by_z(lr), window=vol_window)
    rlr_real = inverse_vol_ensemble_return(series_by_z(rlr), window=vol_window)
    rpf_real = inverse_vol_ensemble_return(series_by_z(rpf_fixed), window=vol_window)

    common_index = (
        lr_real.index.intersection(rlr_real.index)
        .intersection(rpf_real.index)
        .sort_values()
    )
    lr_real = lr_real.loc[common_index]
    rlr_real = rlr_real.loc[common_index]
    rpf_real = rpf_real.loc[common_index]

    lr_sr = float(sharpe(lr_real.to_frame("x")).iloc[0])
    rlr_sr = float(sharpe(rlr_real.to_frame("x")).iloc[0])
    rpf_sr = float(sharpe(rpf_real.to_frame("x")).iloc[0])

    tstat_vs_lr = alpha_tstat(rpf_real, lr_real)
    tstat_vs_rlr = alpha_tstat(rpf_real, rlr_real)

    lr_cum = cumulative_sum_return(lr_real, target_vol=0.10)
    rlr_cum = cumulative_sum_return(rlr_real, target_vol=0.10)
    rpf_cum = cumulative_sum_return(rpf_real, target_vol=0.10)

    fig, axes = make_figure(nrows=1, ncols=3, width=6.8, height=3.5)
    ax_left, ax_mid, ax_right = axes

    barplot(
        ax=ax_left,
        x=["LR", "RLR", "RPF"],
        y=[lr_sr, rlr_sr, rpf_sr],
        colors=[
            get_model_color("LR"),
            get_model_color("RLR"),
            get_model_color("RPF"),
        ],
        ylabel="Sharpe",
        zero=True,
    )

    barplot(
        ax=ax_mid,
        x=["vs LR", "vs RLR"],
        y=[tstat_vs_lr, tstat_vs_rlr],
        colors=[
            get_model_color("vs LR"),
            get_model_color("vs RLR"),
        ],
        ylabel="Alpha t-stat",
        zero=True,
    )

    lineplot(
        ax=ax_right,
        x=lr_cum.index,
        y=lr_cum.to_numpy(),
        label="LR",
        color=get_model_color("LR"),
    )
    lineplot(
        ax=ax_right,
        x=rlr_cum.index,
        y=rlr_cum.to_numpy(),
        label="RLR",
        color=get_model_color("RLR"),
    )
    lineplot(
        ax=ax_right,
        x=rpf_cum.index,
        y=rpf_cum.to_numpy(),
        label="RPF",
        color=get_model_color("RPF"),
    )
    ax_right.set_ylabel("Cumulative Return")
    format_date_axis(ax_right)
    ax_right.legend()

    savefig(fig, "full_ens_summary_model")