# analyze_full_size_real.py
from util.analyze import (
    alpha_tstat,
    filter_column_levels,
    filter_sample,
    inverse_vol_ensemble_return,
    series_by_z,
    sharpe,
    subset_level,
)
from util.load import load_sdfs
from util.plot import (
    make_figure,
    savefig,
    grouped_barplot,
    format_legend,
)
from util.system import get_result_lr, get_result_rlr, get_result_rpf


REALTIME_SIZE_ORDER = ["micro", "small", "large", "mega"]
REALTIME_SIZE_LABELS = ["Micro", "Small", "Large", "Mega"]


def get_size_panel(
    size_df,
    size: str,
):
    """
    Return the relevant panel for one size group.
    """
    return subset_level(size_df, level="size", value=size, normalize_str=True, drop_level=True)


if __name__ == "__main__":
    start = "1993-01-31"
    end = "2024-12-31"
    vol_window = 12
    plot_depth = 3
    plot_num_tree = 64

    lr_size = filter_sample(load_sdfs("lr_full_size.yml", get_result_lr()), start=start, end=end)
    rlr_size = filter_sample(load_sdfs("rlr_full_size.yml", get_result_rlr()), start=start, end=end)
    rpf_size = filter_sample(load_sdfs("rpf_full_size.yml", get_result_rpf()), start=start, end=end)

    if lr_size.empty:
        raise ValueError("No loaded LR size results found.")
    if rlr_size.empty:
        raise ValueError("No loaded RLR size results found.")
    if rpf_size.empty:
        raise ValueError("No loaded RPF size results found.")

    sharpe_panel = {"LR": [], "RLR": [], "RPF": []}
    alpha_panel = {"vs LR": [], "vs RLR": []}

    for size in REALTIME_SIZE_ORDER:
        lr_panel = get_size_panel(lr_size, size)
        rlr_panel = get_size_panel(rlr_size, size)
        rpf_panel = get_size_panel(rpf_size, size)

        rpf_panel = filter_column_levels(
            rpf_panel,
            filters={"max_depth": plot_depth, "num_tree": plot_num_tree},
            drop_filtered_levels=True,
        )

        lr_real = inverse_vol_ensemble_return(series_by_z(lr_panel), window=vol_window)
        rlr_real = inverse_vol_ensemble_return(series_by_z(rlr_panel), window=vol_window)
        rpf_real = inverse_vol_ensemble_return(series_by_z(rpf_panel), window=vol_window)

        common_index = (
            lr_real.index.intersection(rlr_real.index)
            .intersection(rpf_real.index)
            .sort_values()
        )
        lr_real = lr_real.loc[common_index]
        rlr_real = rlr_real.loc[common_index]
        rpf_real = rpf_real.loc[common_index]

        sharpe_panel["LR"].append(float(sharpe(lr_real.to_frame("x")).iloc[0]))
        sharpe_panel["RLR"].append(float(sharpe(rlr_real.to_frame("x")).iloc[0]))
        sharpe_panel["RPF"].append(float(sharpe(rpf_real.to_frame("x")).iloc[0]))

        alpha_panel["vs LR"].append(alpha_tstat(rpf_real, lr_real))
        alpha_panel["vs RLR"].append(alpha_tstat(rpf_real, rlr_real))

    fig, axes = make_figure(nrows=1, ncols=2, width=6.8, height=3.5)
    ax_left, ax_right = axes

    grouped_barplot(
        ax=ax_left,
        groups=REALTIME_SIZE_LABELS,
        series=sharpe_panel,
        ylabel="Sharpe",
        xlabel="Size",
        zero=True,
    )
    format_legend(ax_left, outside=False)

    grouped_barplot(
        ax=ax_right,
        groups=REALTIME_SIZE_LABELS,
        series=alpha_panel,
        ylabel="Alpha t-stat",
        xlabel="Size",
        zero=True,
    )
    format_legend(ax_right, outside=False)

    savefig(fig, "size_ens_summary_size")