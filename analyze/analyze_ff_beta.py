# analyze_ff_beta.py
import matplotlib.dates as mdates

from util.load import load_char
from util.plot import make_figure, savefig, lineplot, format_legend, get_line_color
from util.system import get_result_rpf
from util.interpret import (
    flatten_single_run,
    prefixed_columns,
    strip_prefix,
    factor_label,
)


def format_date_axis(ax) -> None:
    """
    Format a compact date axis for active-beta plots.
    """
    ax.xaxis.set_major_locator(mdates.YearLocator(base=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)


if __name__ == "__main__":
    top_k = 3
    roll_window = 12

    char = load_char("rpf_ff_interpret.yml", get_result_rpf())
    char = flatten_single_run(char)

    if char.empty:
        raise ValueError("No FF characteristic interpretation output found.")

    beta_cols = prefixed_columns(char, "beta_oos_")
    if not beta_cols:
        raise ValueError("No active OOS beta columns found.")

    avg_abs = char[beta_cols].abs().mean(axis=0).sort_values(ascending=False)
    top_factors = strip_prefix(list(avg_abs.index[: min(top_k, len(avg_abs))]), "beta_oos_")

    beta_roll = (
        char[[f"beta_oos_{name}" for name in top_factors]]
        .rolling(roll_window, min_periods=roll_window)
        .mean()
        .rename(columns={f"beta_oos_{name}": name for name in top_factors})
    )

    fig, ax = make_figure(width=6.8, height=3.5)

    for i, name in enumerate(top_factors):
        lineplot(
            ax=ax,
            x=beta_roll.index,
            y=beta_roll[name],
            label=factor_label(name),
            color=get_line_color(i),
            ylabel="12-Month Rolling Active Beta",
            zero=True,
        )

    format_date_axis(ax)
    format_legend(ax, outside=False)

    savefig(fig, "ff_active_beta")