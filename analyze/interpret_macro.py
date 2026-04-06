import pandas as pd

from util.load import load_macro
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


def macro_label(name: str) -> str:
    """
    Map raw macro code to display label.
    """
    return MACRO_LABELS.get(name, name)


if __name__ == "__main__":
    macro = load_macro("rmst.yml", get_result_rpf())
    macro = flatten_single_run(macro)

    if macro.empty:
        raise ValueError("No RMST macro export found.")

    gain_cols = prefixed_columns(macro, "gain_share_")
    root_cols = prefixed_columns(macro, "root_share_")

    avg_gain_raw = macro[gain_cols].mean(axis=0).sort_values(ascending=False)
    avg_root_raw = macro[root_cols].mean(axis=0).sort_values(ascending=False)

    avg_gain = avg_gain_raw.copy()
    avg_root = avg_root_raw.copy()

    avg_gain.index = [macro_label(col.removeprefix("gain_share_")) for col in avg_gain.index]
    avg_root.index = [macro_label(col.removeprefix("root_share_")) for col in avg_root.index]

    top_k = min(10, len(avg_gain))

    fig, ax = make_figure(width=7.0, height=4.0)
    barplot(
        ax=ax,
        x=avg_gain.head(top_k).index.to_list(),
        y=avg_gain.head(top_k).to_list(),
        colors=get_bar_colors(top_k),
        ylabel="Average Gain Share",
        title="Top Macro Variables by Average Gain Share",
        zero=True,
    )
    rotate_xticklabels(ax)
    savefig(fig, "interpret_macro_gain_share")

    fig, ax = make_figure(width=7.0, height=4.0)
    barplot(
        ax=ax,
        x=avg_root.head(top_k).index.to_list(),
        y=avg_root.head(top_k).to_list(),
        colors=get_bar_colors(top_k),
        ylabel="Average Root Split Share",
        title="Top Macro Variables by Root Split Share",
        zero=True,
    )
    rotate_xticklabels(ax)
    savefig(fig, "interpret_macro_root_share")

    leaders_raw = (
        avg_gain_raw.index.str.removeprefix("gain_share_")
        .to_list()[: min(5, len(avg_gain_raw))]
    )
    macro_roll = macro.rolling(12, min_periods=12).mean()

    fig, ax = make_figure(width=8.2, height=4.2)
    for i, name in enumerate(leaders_raw):
        lineplot(
            ax=ax,
            x=macro_roll.index,
            y=macro_roll[f"gain_share_{name}"],
            label=macro_label(name),
            color=get_line_color(i),
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("12-Month Rolling Gain Share")
    ax.set_title("Leading Macro Split Importance")
    format_legend(ax, outside=True)
    savefig(fig, "interpret_macro_gain_share_ts")