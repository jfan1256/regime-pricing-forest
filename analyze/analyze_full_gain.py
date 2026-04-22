# analyze_ff_gain.py
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from util.load import load_macro
from util.plot import (
    make_figure,
    savefig,
    barplot,
    lineplot,
    rotate_xticklabels,
)
from util.system import get_result_rpf
from util.interpret import (
    flatten_single_run,
    top_macro_from_gain,
    average_prefixed,
    rolling_prefixed,
    macro_label,
    macro_color_map,
)


# NBER recession windows, aligned at month end.
RECESSIONS = [
    ("1969-12-31", "1970-11-30"),
    ("1973-11-30", "1975-03-31"),
    ("1980-01-31", "1980-07-31"),
    ("1981-07-31", "1982-11-30"),
    ("1990-07-31", "1991-03-31"),
    ("2001-03-31", "2001-11-30"),
    ("2007-12-31", "2009-06-30"),
    ("2020-02-29", "2020-04-30"),
]


def format_date_axis(ax) -> None:
    """
    Format a denser date axis and rotate labels for readability.
    """
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=40)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def add_recession_bands(ax, index: pd.DatetimeIndex) -> None:
    """
    Add NBER recession shading over the visible sample span.
    """
    if len(index) == 0:
        return

    start = pd.Timestamp(index.min())
    end = pd.Timestamp(index.max())

    for rec_start, rec_end in RECESSIONS:
        left = pd.Timestamp(rec_start)
        right = pd.Timestamp(rec_end)

        if right < start or left > end:
            continue

        ax.axvspan(
            max(left, start),
            min(right, end),
            color="#D8DEE9",
            alpha=0.45,
            zorder=0,
        )


def raise_top_ylim(
    ax,
    values: np.ndarray,
    pad_frac: float = 0.28,
) -> None:
    """
    Increase the upper y-limit to create room for an in-panel legend.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return

    y_min = float(finite.min())
    y_max = float(finite.max())
    span = y_max - y_min

    if span <= 0.0:
        span = max(abs(y_max), 1.0)

    top = y_max + pad_frac * span
    bottom = min(0.0, y_min - 0.05 * span)
    ax.set_ylim(bottom, top)


if __name__ == "__main__":
    top_k = 5
    roll_window = 60

    macro = load_macro("rpf_full_interpret.yml", get_result_rpf())
    macro = flatten_single_run(macro)

    if macro.empty:
        raise ValueError("No FF macro interpretation output found.")

    leaders = top_macro_from_gain(macro, top_k=top_k)
    colors = macro_color_map(leaders)

    avg_gain = average_prefixed(macro, prefix="gain_share_").loc[leaders]
    avg_gain.index = [macro_label(name) for name in avg_gain.index]

    gain_roll = rolling_prefixed(macro, prefix="gain_share_", window=roll_window)
    gain_roll = gain_roll.loc[:, leaders]

    fig, axes = make_figure(nrows=1, ncols=2, width=6.8, height=3.2)
    ax_left, ax_right = axes

    barplot(
        ax=ax_left,
        x=avg_gain.index.to_list(),
        y=avg_gain.to_list(),
        colors=[colors[name] for name in leaders],
        ylabel="Average Gain Share",
        zero=True,
    )
    rotate_xticklabels(ax_left, rotation=40.0)

    add_recession_bands(ax_right, gain_roll.index)

    for name in leaders:
        lineplot(
            ax=ax_right,
            x=gain_roll.index,
            y=gain_roll[name],
            label=macro_label(name),
            color=colors[name],
            ylabel=f"{roll_window}-Month Rolling Gain Share",
        )

    raise_top_ylim(ax_right, gain_roll.to_numpy(dtype=float), pad_frac=0.5)

    format_date_axis(ax_right)
    ax_right.legend(
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.9,
        handlelength=1.8,
        borderaxespad=0.2,
    )

    savefig(fig, "full_gain_share")