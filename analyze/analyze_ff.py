# analyze_ff.py
import matplotlib.dates as mdates

from util.analyze import (
    alpha_tstat,
    best_model,
    best_sharpe_by_level,
    cumulative_sum_return,
    filter_sample,
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
    Format a date axis to avoid overlapping tick labels in compact subplots.
    """
    ax.xaxis.set_major_locator(mdates.YearLocator(base=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)


if __name__ == "__main__":
    start = "1993-01-31"
    end = "2024-12-31"

    lr = filter_sample(load_sdfs("lr_ff.yml", get_result_lr()), start=start, end=end)
    rlr = filter_sample(load_sdfs("rlr_ff.yml", get_result_rlr()), start=start, end=end)
    rpf = filter_sample(load_sdfs("rpf_ff.yml", get_result_rpf()), start=start, end=end)

    _, lr_best, lr_sr = best_model(lr)
    _, rlr_best, rlr_sr = best_model(rlr)
    _, rpf_best, rpf_sr = best_model(rpf)

    tstat_vs_lr = alpha_tstat(rpf_best, lr_best)
    tstat_vs_rlr = alpha_tstat(rpf_best, rlr_best)

    lr_cum = cumulative_sum_return(lr_best, target_vol=0.10)
    rlr_cum = cumulative_sum_return(rlr_best, target_vol=0.10)
    rpf_cum = cumulative_sum_return(rpf_best, target_vol=0.10)

    fig, axes = make_figure(nrows=1, ncols=3, width=6.5, height=2.6)
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

    savefig(fig, "ff_summary_model")

    depth_sr = best_sharpe_by_level(rpf, level="max_depth")
    depth_values = depth_sr.index.to_list()

    fig, ax = make_figure(width=6.8, height=3.5)
    lineplot(
        ax=ax,
        x=depth_values,
        y=depth_sr.to_numpy(),
        color=get_model_color("RPF"),
        marker="o",
        xlabel="Depth",
        ylabel="Sharpe",
    )
    ax.set_xticks(depth_values)
    ax.set_xticklabels([str(depth) for depth in depth_values])

    savefig(fig, "ff_sharpe_depth")