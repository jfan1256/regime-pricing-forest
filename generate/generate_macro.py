import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from util.system import get_data

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_series(series_id: str) -> pd.Series:
    """Download a single FRED series indexed by observation_date."""
    url = FRED_CSV_URL.format(series_id=series_id)
    df = pd.read_csv(url)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    s = df.set_index("observation_date")[series_id].sort_index()
    s.index = pd.DatetimeIndex(s.index)
    s.name = series_id
    return s


def infer_freq(index: pd.DatetimeIndex) -> str:
    """Infer native sampling frequency from observation_date spacing."""
    index = index.sort_values().unique()
    gaps = pd.Series(index).diff().dropna().dt.days
    median_gap = gaps.median()

    if median_gap <= 7:
        return "daily"
    if median_gap <= 40:
        return "monthly"
    if median_gap <= 100:
        return "quarterly"
    return "monthly"


def lag_freq(index: pd.DatetimeIndex) -> int:
    """
    Map native frequency to a monthly lag:
    - daily-like: 0
    - monthly: 1
    - quarterly: 4
    """
    freq = infer_freq(index)

    if freq == "daily":
        return 0
    if freq == "monthly":
        return 1
    if freq == "quarterly":
        return 4
    return 1


def to_month_end(s: pd.Series) -> pd.Series:
    """Convert a series to month-end frequency using the last observation in each month."""
    out = s.resample("ME").last()
    out.index = out.index.to_period("M").to_timestamp("M")
    return out


def prep_series(series_id: str) -> pd.Series:
    """
    Fetch a FRED series, infer native frequency from observation_date spacing,
    convert to month-end, and apply the corresponding lag.
    """
    s = fetch_series(series_id)
    lag = lag_freq(s.index)

    s = to_month_end(s)
    if lag > 0:
        s = s.shift(lag)

    return s


def yoy_log_growth(s: pd.Series) -> pd.Series:
    """Year-over-year log growth in percent."""
    return 100.0 * (np.log(s) - np.log(s.shift(12)))


def load_crsp_market_indexes() -> pd.DataFrame:
    """
    Load CRSP monthly stock market index data.

    Returns
    -------
    DataFrame indexed by month-end with:
    - market_vol_12m: rolling 12m std of monthly S&P 500 price returns in percent
    - dividend_yield_1m: monthly dividend yield proxy in percent

    Notes
    -----
    - No lag is applied to any CRSP-derived series.
    - No logs are used on CRSP-derived data.
    - price return is computed from spindx.
    - dividend yield proxy is total return minus price return.
    """
    path = get_data() / "crsp" / "crsp_monthly_stock_market_indexes.csv"
    df = pd.read_csv(path)

    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    df = df.sort_values("MthCalDt").copy()

    numeric_cols = [
        "vwretd",
        "vwretx",
        "vwTotVal",
        "vwUsdVal",
        "vwTotCnt",
        "vwUsdCnt",
        "ewretd",
        "ewretx",
        "ewTotVal",
        "ewUsdVal",
        "ewTotCnt",
        "ewUsdCnt",
        "sprtrn",
        "spindx",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("MthCalDt")
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp("M")
    df.index.name = "date"

    out = pd.DataFrame(index=df.index)

    # Monthly S&P 500 price return from the index level.
    sp500_price_return_1m = 100.0 * df["spindx"].pct_change()

    # Rolling 12-month volatility of monthly price returns.
    out["market_vol_12m"] = sp500_price_return_1m.rolling(12).std()

    # Monthly dividend yield proxy = total return - price return.
    out["dividend_yield_1m"] = 100.0 * df["sprtrn"] - sp500_price_return_1m

    return out.sort_index()


def build_macro() -> pd.DataFrame:
    # Real activity / labor / household demand
    indpro = prep_series("INDPRO")
    unrate = prep_series("UNRATE")
    houst = prep_series("HOUST")
    cumfns = prep_series("CUMFNS")
    w875rx1 = prep_series("W875RX1")

    # Inflation / monetary / policy / rates / credit
    cpi = prep_series("CPIAUCSL")
    fedfunds = prep_series("FEDFUNDS")
    gs10 = prep_series("GS10")
    tb3ms = prep_series("TB3MS")
    baa = prep_series("BAA")
    m2 = prep_series("M2SL")

    # Market data (not lagged)
    crsp = load_crsp_market_indexes()

    df = pd.DataFrame(
        {
            "ip_growth_yoy": yoy_log_growth(indpro),
            "unemployment_rate": unrate,
            "housing_starts_growth_yoy": yoy_log_growth(houst),
            "capacity_utilization": cumfns,
            "real_personal_income_ex_transfers_growth_yoy": yoy_log_growth(w875rx1),
            "inflation_yoy": yoy_log_growth(cpi),
            "fed_funds": fedfunds,
            "term_spread": gs10 - tb3ms,
            "credit_spread_baa10y": baa - gs10,
            "m2_growth_yoy": yoy_log_growth(m2),
            "market_vol_12m": crsp["market_vol_12m"],
            "dividend_yield_1m": crsp["dividend_yield_1m"],
        }
    )

    df = df[
        [
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
    ]

    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "date"
    return df


def sum_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness, coverage, and sample start/end for each column."""
    rows = []

    for col in df.columns:
        s = df[col]
        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()

        if first_valid is not None and last_valid is not None:
            in_sample = s.loc[first_valid:last_valid]
            internal_nan_ratio = in_sample.isna().mean()
            non_nan_count = s.notna().sum()
        else:
            internal_nan_ratio = np.nan
            non_nan_count = 0

        rows.append(
            {
                "column": col,
                "nan_ratio": s.isna().mean(),
                "nan_count": s.isna().sum(),
                "non_nan_count": non_nan_count,
                "first_valid": first_valid,
                "last_valid": last_valid,
                "internal_nan_ratio": internal_nan_ratio,
                "starts_late": first_valid != df.index.min() if first_valid is not None else True,
                "ends_early": last_valid != df.index.max() if last_valid is not None else True,
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["nan_ratio", "first_valid"],
        ascending=[False, True],
    )
    return out


if __name__ == "__main__":
    macro = build_macro()

    missing_stats = sum_nan(macro)
    print("\nMissingness summary:\n")
    print(missing_stats.to_string(index=False))

    macro = macro.ffill().dropna()
    macro.to_parquet(get_data() / "fred" / "macro_m.pq")


#
#
# import warnings
# warnings.filterwarnings("ignore")
#
# import numpy as np
# import pandas as pd
#
# from util.system import get_data
#
# FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
#
# def fetch_series(series_id: str) -> pd.Series:
#     """Download a single FRED series and return it indexed by observation_date."""
#     url = FRED_CSV_URL.format(series_id=series_id)
#     df = pd.read_csv(url)
#     df["observation_date"] = pd.to_datetime(df["observation_date"])
#     df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
#
#     s = df.set_index("observation_date")[series_id].sort_index()
#     s.index = pd.DatetimeIndex(s.index)
#     s.name = series_id
#     return s
#
# def infer_freq(index: pd.DatetimeIndex) -> str:
#     """Infer native sampling frequency from observation_date spacing."""
#     index = index.sort_values().unique()
#     gaps = pd.Series(index).diff().dropna().dt.days
#     median_gap = gaps.median()
#
#     if median_gap <= 7:
#         return "daily"
#     if median_gap <= 40:
#         return "monthly"
#     if median_gap <= 100:
#         return "quarterly"
#     return "monthly"
#
# def lag_freq(index: pd.DatetimeIndex) -> int:
#     """
#     Map native frequency to a monthly lag:
#     - daily-like: 0
#     - monthly: 1
#     - quarterly: 4
#     """
#     freq = infer_freq(index)
#
#     if freq == "daily":
#         return 0
#     if freq == "monthly":
#         return 1
#     if freq == "quarterly":
#         return 4
#     return 1
#
# def to_month_end(s: pd.Series) -> pd.Series:
#     """Convert a series to month-end frequency using the last observation in each month."""
#     out = s.resample("M").last()
#     out.index = out.index.to_period("M").to_timestamp("M")
#     return out
#
# def prep_series(series_id: str) -> pd.Series:
#     """
#     Fetch a series, infer its native frequency from observation_date spacing,
#     convert to month-end, and apply the corresponding lag.
#     """
#     s = fetch_series(series_id)
#     lag = lag_freq(s.index)
#
#     s = to_month_end(s)
#     if lag > 0:
#         s = s.shift(lag)
#
#     return s
#
# def yoy_log_growth(s: pd.Series) -> pd.Series:
#     """Year-over-year log growth in percent."""
#     return 100.0 * (np.log(s) - np.log(s.shift(12)))
#
# def build_macro() -> pd.DataFrame:
#     indpro = prep_series("INDPRO")
#     payems = prep_series("PAYEMS")
#     houst = prep_series("HOUST")
#     pcecc96 = prep_series("PCECC96")
#     dspic96 = prep_series("DSPIC96")
#
#     unrate = prep_series("UNRATE")
#     cpi = prep_series("CPIAUCSL")
#     fedfunds = prep_series("FEDFUNDS")
#
#     gs10 = prep_series("GS10")
#     tb3ms = prep_series("TB3MS")
#     baa = prep_series("BAA")
#     aaa = prep_series("AAA")
#
#     sp500 = prep_series("SP500")
#     vix = prep_series("VIXCLS")
#     consumer_sentiment = prep_series("UMCSENT")
#
#     df = pd.DataFrame(
#         {
#             "ip_growth_yoy": yoy_log_growth(indpro),
#             "payroll_growth_yoy": yoy_log_growth(payems),
#             "housing_starts_growth_yoy": yoy_log_growth(houst),
#             "real_pce_growth_yoy": yoy_log_growth(pcecc96),
#             "real_income_growth_yoy": yoy_log_growth(dspic96),
#             "unemployment_rate": unrate,
#             "inflation_yoy": yoy_log_growth(cpi),
#             "fed_funds": fedfunds,
#             "term_spread": gs10 - tb3ms,
#             "default_spread": baa - aaa,
#             "credit_spread_baa10y": baa - gs10,
#             "market_return_1m": 100.0 * sp500.pct_change(1),
#             "vix": vix,
#             "consumer_sentiment": consumer_sentiment,
#         }
#     )
#
#     df["market_vol_12m"] = df["market_return_1m"].rolling(12).std()
#
#     df = df[
#         [
#             "ip_growth_yoy",
#             "payroll_growth_yoy",
#             "housing_starts_growth_yoy",
#             "real_pce_growth_yoy",
#             "real_income_growth_yoy",
#             "unemployment_rate",
#             "inflation_yoy",
#             "fed_funds",
#             "term_spread",
#             "default_spread",
#             "credit_spread_baa10y",
#             # "market_return_1m",
#             # "market_vol_12m",
#             # "vix",
#             "consumer_sentiment",
#         ]
#     ]
#
#     df.index = pd.DatetimeIndex(df.index)
#     df.index.name = "date"
#     df = df.ffill()
#     return df
#
# def sum_nan(df: pd.DataFrame) -> pd.DataFrame:
#     """Summarize missingness, coverage, and sample start/end for each column."""
#     rows = []
#
#     for col in df.columns:
#         s = df[col]
#         first_valid = s.first_valid_index()
#         last_valid = s.last_valid_index()
#
#         if first_valid is not None and last_valid is not None:
#             in_sample = s.loc[first_valid:last_valid]
#             internal_nan_ratio = in_sample.isna().mean()
#             non_nan_count = s.notna().sum()
#         else:
#             internal_nan_ratio = np.nan
#             non_nan_count = 0
#
#         rows.append(
#             {
#                 "column": col,
#                 "nan_ratio": s.isna().mean(),
#                 "nan_count": s.isna().sum(),
#                 "non_nan_count": non_nan_count,
#                 "first_valid": first_valid,
#                 "last_valid": last_valid,
#                 "internal_nan_ratio": internal_nan_ratio,
#                 "starts_late": first_valid != df.index.min() if first_valid is not None else True,
#                 "ends_early": last_valid != df.index.max() if last_valid is not None else True,
#             }
#         )
#
#     out = pd.DataFrame(rows).sort_values(
#         ["nan_ratio", "first_valid"],
#         ascending=[False, True],
#     )
#     return out
#
#
# if __name__ == "__main__":
#     macro = build_macro()
#
#     missing_stats = sum_nan(macro)
#     print("\nMissingness summary:\n")
#     print(missing_stats.to_string(index=False))
#     print("\nColumns with highest NaN ratios:\n")
#     print(
#         missing_stats.loc[:, ["column", "nan_ratio", "first_valid", "last_valid"]]
#         .head(10)
#         .to_string(index=False)
#     )
#
#     macro = macro.ffill().dropna()
#     print(macro)
#     macro.to_parquet(get_data() / "fred" / "macro_m.pq")