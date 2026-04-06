import pandas as pd

from util.load import load_char
from util.plot import (
    make_figure,
    savefig,
    barplot,
    lineplot,
    rotate_xticklabels,
    format_legend,
    get_bar_colors,
    get_line_color,
)
from util.system import get_result_rpf

FACTOR_LABELS = {
    "market_equity": "Size",
    "be_me": "Value",
    "ope_be": "Profitability",
    "at_gr1": "Investment",
    "ret_12_1": "Momentum",
}

MACRO_LABELS = {
    "ip_growth_yoy": "IP Growth",
    "housing_starts_growth_yoy": "Housing Growth",
    "unemployment_rate": "Unemployment",
    "real_income_ex_transfers_yoy": "Income Growth",
    "capacity_utilization": "Capacity Util.",
    "inflation_yoy": "Inflation",
    "m2_growth_yoy": "M2 Growth",
    "fed_funds": "Fed Funds",
    "term_spread": "Term Spread",
    "credit_spread_baa10y": "Credit Spread",
    "consumer_sentiment": "Sentiment",
    "sp500_vol_12m": "Equity Vol.",
}

def flatten_single_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract one run from a sweep-loaded DataFrame.
    """
    if df.empty or df.columns.nlevels <= 1:
        return df.copy()

    run_key = df.columns[0][:-1]
    cols = [col for col in df.columns if col[:-1] == run_key]

    out = df.loc[:, cols].copy()
    out.columns = [col[-1] for col in cols]
    return out


def prefixed_columns(
    df: pd.DataFrame,
    prefix: str,
) -> list[str]:
    """
    Return all columns with a given prefix.
    """
    return [col for col in df.columns if col.startswith(prefix)]


def factor_label(name: str) -> str:
    """
    Map raw factor code to display label.
    """
    return FACTOR_LABELS.get(name, name)


def macro_label(name: str) -> str:
    """
    Map raw macro code to display label.
    """
    return MACRO_LABELS.get(name, name)


def split_factor_macro(name: str) -> tuple[str, str]:
    """
    Split a combined interaction name into factor and macro keys.

    Handles names like:
        ope_be__inflation_yoy
        at_gr1_unemployment_rate
    by matching the factor prefix first, then stripping any separator
    underscores before mapping the macro part.
    """
    factor_keys = sorted(FACTOR_LABELS, key=len, reverse=True)
    for factor in factor_keys:
        if name.startswith(factor):
            macro = name[len(factor):].lstrip("_")
            return factor, macro

    raise ValueError(f"Could not split factor-macro interaction: {name}")


def spread_label(name: str) -> str:
    """
    Map raw factor-macro interaction code to display label.
    """
    factor, macro = split_factor_macro(name)
    return f"{factor_label(factor)} + {macro_label(macro)}"


if __name__ == "__main__":
    char = load_char("rmst.yml", get_result_rpf())
    char = flatten_single_run(char)

    if char.empty:
        raise ValueError("No RMST char export found.")

    dispersion_cols = prefixed_columns(char, "beta_dispersion_")
    oos_cols = prefixed_columns(char, "beta_oos_")
    spread_cols = prefixed_columns(char, "beta_spread_")

    avg_dispersion_raw = char[dispersion_cols].mean(axis=0).sort_values(ascending=False)
    avg_dispersion = avg_dispersion_raw.copy()
    avg_dispersion.index = [
        factor_label(col.removeprefix("beta_dispersion_"))
        for col in avg_dispersion.index
    ]

    top_k = min(10, len(avg_dispersion))

    fig, ax = make_figure(width=7.0, height=4.0)
    barplot(
        ax=ax,
        x=avg_dispersion.head(top_k).index.to_list(),
        y=avg_dispersion.head(top_k).to_list(),
        colors=get_bar_colors(top_k),
        ylabel="Average Beta Dispersion",
        title="Top Factors by Cross-Leaf Beta Dispersion",
        zero=True,
    )
    rotate_xticklabels(ax)
    savefig(fig, "interpret_char_dispersion")

    leaders_raw = (
        avg_dispersion_raw.index.str.removeprefix("beta_dispersion_")
        .to_list()[: min(5, len(avg_dispersion_raw))]
    )
    char_roll = char.rolling(12, min_periods=12).mean()

    fig, ax = make_figure(width=8.2, height=4.2)
    for i, name in enumerate(leaders_raw):
        lineplot(
            ax=ax,
            x=char_roll.index,
            y=char_roll[f"beta_oos_{name}"],
            label=factor_label(name),
            color=get_line_color(i),
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("12-Month Rolling Active OOS Beta")
    ax.set_title("Out-of-Sample Active RMST Betas")
    format_legend(ax, outside=True)
    savefig(fig, "interpret_char_beta_oos_ts")

    avg_abs_spread_raw = char[spread_cols].abs().mean(axis=0).sort_values(ascending=False)
    avg_abs_spread = avg_abs_spread_raw.copy()
    avg_abs_spread.index = [
        spread_label(col.removeprefix("beta_spread_"))
        for col in avg_abs_spread.index
    ]

    top_spreads = avg_abs_spread.head(min(10, len(avg_abs_spread)))

    fig, ax = make_figure(width=8.0, height=4.2)
    barplot(
        ax=ax,
        x=top_spreads.index.to_list(),
        y=top_spreads.to_list(),
        colors=get_bar_colors(len(top_spreads)),
        ylabel="Mean Absolute Beta Spread",
        title="Largest Factor-Macro Beta Spread Interactions",
        zero=True,
    )
    rotate_xticklabels(ax, rotation=50.0)
    savefig(fig, "interpret_char_beta_spread")