# util/interpret.py
import numpy as np
import pandas as pd


FACTOR_LABELS = {
    "market_equity": "Size",
    "be_me": "Value",
    "ope_be": "Profitability",
    "at_gr1": "Investment",
    "ret_12_1": "Momentum",
}

THEME_LABELS = {
    "Accruals": "Accruals",
    "Debt Issuance": "Debt Issuance",
    "Investment": "Investment",
    "Low Leverage": "Low Leverage",
    "Low Risk": "Low Risk",
    "Momentum": "Momentum",
    "Profit Growth": "Profit Growth",
    "Profitability": "Profitability",
    "Quality": "Quality",
    "Seasonality": "Seasonality",
    "Short-Term Reversal": "ST Reversal",
    "Size": "Size",
    "Value": "Value",
}

MACRO_LABELS = {
    "ip_growth_yoy": "IP Growth",
    "unemployment_rate": "Unemployment",
    "housing_starts_growth_yoy": "Housing Growth",
    "capacity_utilization": "Capacity Util.",
    "real_personal_income_ex_transfers_growth_yoy": "Income Growth",
    "inflation_yoy": "Inflation",
    "fed_funds": "Fed Funds",
    "term_spread": "Term Spread",
    "credit_spread_baa10y": "Credit Spread",
    "m2_growth_yoy": "M2 Growth",
    "market_vol_12m": "Equity Vol.",
    "dividend_yield_1m": "Dividend Yield",
}

FACTOR_ORDER = [
    "market_equity",
    "be_me",
    "ope_be",
    "at_gr1",
    "ret_12_1",
]

MACRO_ORDER = [
    "ip_growth_yoy",
    "unemployment_rate",
    "housing_starts_growth_yoy",
    "capacity_utilization",
    "real_personal_income_ex_transfers_growth_yoy",
    "inflation_yoy",
    "fed_funds",
    "term_spread",
    "credit_spread_baa10y",
    "m2_growth_yoy",
    "market_vol_12m",
    "dividend_yield_1m",
]

# Fixed macro palette so each macro variable keeps the same color in every figure.
# Chosen to stay in the muted blue/green/purple family used elsewhere.
MACRO_COLORS = {
    "ip_growth_yoy": "#7A6BA8",
    "unemployment_rate": "#6574A8",
    "housing_starts_growth_yoy": "#5A7FA6",
    "capacity_utilization": "#6E9E9F",
    "real_personal_income_ex_transfers_growth_yoy": "#86B39A",
    "inflation_yoy": "#4C6A91",
    "fed_funds": "#5A8C7A",
    "term_spread": "#8D7AAF",
    "credit_spread_baa10y": "#6B8FB3",
    "m2_growth_yoy": "#7DAA95",
    "market_vol_12m": "#8A7FB2",
    "dividend_yield_1m": "#5D8FA8",
}


def flatten_single_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract one run from a sweep-loaded DataFrame.

    Assumes the final column level contains the artifact field names, while all
    preceding levels identify the run configuration.
    """
    if df.empty or df.columns.nlevels <= 1:
        return df.copy()

    run_key = df.columns[0][:-1]
    cols = [col for col in df.columns if col[:-1] == run_key]

    out = df.loc[:, cols].copy()
    out.columns = [col[-1] for col in cols]
    return out.sort_index()


def prefixed_columns(
    df: pd.DataFrame,
    prefix: str,
) -> list[str]:
    """
    Return all columns beginning with a shared prefix.
    """
    return [col for col in df.columns if col.startswith(prefix)]


def strip_prefix(
    values: list[str] | pd.Index,
    prefix: str,
) -> list[str]:
    """
    Remove a shared prefix from a list of strings.
    """
    return [str(value).removeprefix(prefix) for value in values]


def factor_label(name: str) -> str:
    """
    Map a raw factor code to a display label.
    """
    return FACTOR_LABELS.get(name, name)

def theme_label(name: str) -> str:
    """
    Map a raw theme name to a display label.
    """
    return THEME_LABELS.get(name, name)

def macro_label(name: str) -> str:
    """
    Map a raw macro code to a display label.
    """
    return MACRO_LABELS.get(name, name)


def factor_order_key(name: str) -> tuple[int, str]:
    """
    Stable sort key for factor names.
    """
    if name in FACTOR_ORDER:
        return (FACTOR_ORDER.index(name), name)
    return (len(FACTOR_ORDER), name)


def macro_order_key(name: str) -> tuple[int, str]:
    """
    Stable sort key for macro names.
    """
    if name in MACRO_ORDER:
        return (MACRO_ORDER.index(name), name)
    return (len(MACRO_ORDER), name)


def macro_color(name: str) -> str:
    """
    Return the fixed color for one macro variable.
    """
    if name in MACRO_COLORS:
        return MACRO_COLORS[name]
    raise ValueError(f"No fixed macro color configured for {name!r}.")


def macro_color_map(names: list[str]) -> dict[str, str]:
    """
    Return a fixed macro-color map for the requested macro variables.
    """
    return {name: macro_color(name) for name in names}


def average_prefixed(
    df: pd.DataFrame,
    prefix: str,
) -> pd.Series:
    """
    Time average of all columns sharing a prefix, indexed by stripped names.
    """
    cols = prefixed_columns(df, prefix)
    if not cols:
        return pd.Series(dtype=float)

    out = df[cols].mean(axis=0)
    out.index = pd.Index(strip_prefix(list(out.index), prefix))
    return out.sort_values(ascending=False)


def rolling_prefixed(
    df: pd.DataFrame,
    prefix: str,
    window: int = 12,
) -> pd.DataFrame:
    """
    Rolling mean of all columns sharing a prefix, with stripped column names.
    """
    cols = prefixed_columns(df, prefix)
    if not cols:
        return pd.DataFrame(index=df.index)

    out = df[cols].rolling(window, min_periods=window).mean()
    out = out.rename(columns={col: col.removeprefix(prefix) for col in cols})
    return out


def top_names_by_average(
    df: pd.DataFrame,
    prefix: str,
    top_k: int,
) -> list[str]:
    """
    Return the top-k stripped names ranked by time-averaged level.
    """
    avg = average_prefixed(df, prefix)
    return avg.index.to_list()[: min(top_k, len(avg))]


def parse_spread_column(name: str) -> tuple[str, str]:
    """
    Parse a beta-spread column of the form:

        beta_spread_<factor>__<macro>

    into its factor and macro components.
    """
    raw = name.removeprefix("beta_spread_")
    factor, macro = raw.split("__", maxsplit=1)
    return factor, macro


def spread_columns_to_matrix(
    df: pd.DataFrame,
    spread_columns: list[str],
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate beta-spread columns into a factor-by-macro matrix.

    Parameters
    ----------
    df : DataFrame
        Character interpretation panel.
    spread_columns : list[str]
        Columns to include, each of form beta_spread_<factor>__<macro>.
    agg : str
        One of {"mean", "abs_mean"}.

    Returns
    -------
    DataFrame
        Index is macro name, columns are factor name.
    """
    if not spread_columns:
        return pd.DataFrame()

    if agg not in {"mean", "abs_mean"}:
        raise ValueError(f"Unknown agg={agg!r}.")

    rows: list[dict[str, object]] = []

    for col in spread_columns:
        factor, macro = parse_spread_column(col)
        s = df[col].dropna()

        if s.empty:
            value = np.nan
        elif agg == "mean":
            value = float(s.mean())
        else:
            value = float(s.abs().mean())

        rows.append(
            {
                "macro": macro,
                "factor": factor,
                "value": value,
            }
        )

    out = pd.DataFrame(rows).pivot(index="macro", columns="factor", values="value")

    macro_index = sorted(out.index.to_list(), key=macro_order_key)
    factor_index = sorted(out.columns.to_list(), key=factor_order_key)
    return out.reindex(index=macro_index, columns=factor_index)


def top_macro_from_gain(
    macro_df: pd.DataFrame,
    top_k: int,
) -> list[str]:
    """
    Return the top-k macro variables ranked by average gain share.
    """
    return top_names_by_average(macro_df, prefix="gain_share_", top_k=top_k)


def top_factor_from_active_beta(
    char_df: pd.DataFrame,
    top_k: int,
) -> list[str]:
    """
    Return the top-k factors ranked by average absolute active OOS beta.
    """
    cols = prefixed_columns(char_df, "beta_oos_")
    if not cols:
        return []

    avg_abs = char_df[cols].abs().mean(axis=0).sort_values(ascending=False)
    avg_abs.index = pd.Index(strip_prefix(list(avg_abs.index), "beta_oos_"))
    return avg_abs.index.to_list()[: min(top_k, len(avg_abs))]