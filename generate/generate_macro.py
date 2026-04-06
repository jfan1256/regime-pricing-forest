import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from util.system import get_data

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_series(series_id: str) -> pd.Series:
    """Download a single FRED series and return it indexed by observation_date."""
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
    out = s.resample("M").last()
    out.index = out.index.to_period("M").to_timestamp("M")
    return out


def prep_series(series_id: str) -> pd.Series:
    """
    Fetch a series, infer its native frequency from observation_date spacing,
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


def load_consumer_sentiment() -> pd.Series:
    """
    Build a continuous Michigan consumer sentiment series by splicing:
    - UMCSENT1 for history through 1977-11
    - UMCSENT from 1978-01 onward
    """
    old = prep_series("UMCSENT1")
    new = prep_series("UMCSENT")

    s = old.combine_first(new)
    s.loc[new.index] = new
    s.name = "consumer_sentiment"
    return s


def load_crsp_sp500() -> pd.DataFrame:
    """
    Load CRSP Monthly Index and Portfolios on S&P 500 data for INDNO 1000502.

    Returns
    -------
    DataFrame indexed by month-end with:
    - sp500_price_return_1m: monthly price return in percent
    - sp500_vol_12m: rolling 12m std of monthly price returns in percent
    """
    path = get_data() / "crsp" / "crsp_monthly_index_and_portfolios_on_sp500.csv"
    df = pd.read_csv(path)

    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    df = df.sort_values("MthCalDt").copy()

    if "INDNO" in df.columns:
        df = df[df["INDNO"] == 1000502].copy()

    numeric_cols = [
        "MthTotRet",
        "MthTotInd",
        "MthPrcRet",
        "MthPrcInd",
        "MthIncRet",
        "MthIncInd",
        "MthUsdCnt",
        "MthUsdVal",
        "MthTotCnt",
        "MthTotVal",
        "MthEligCnt",
        "MthWgtAmt",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("MthCalDt")
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp("M")
    df.index.name = "date"

    out = pd.DataFrame(index=df.index)
    out["sp500_price_return_1m"] = 100.0 * df["MthPrcRet"]
    out["sp500_vol_12m"] = out["sp500_price_return_1m"].rolling(12).std()

    return out.sort_index()


def build_macro() -> pd.DataFrame:
    # Real activity / labor
    indpro = prep_series("INDPRO")
    houst = prep_series("HOUST")
    unrate = prep_series("UNRATE")
    w875rx1 = prep_series("W875RX1")
    cumfns = prep_series("CUMFNS")

    # Inflation / monetary / policy / rates / credit
    cpi = prep_series("CPIAUCSL")
    m2sl = prep_series("M2SL")
    fedfunds = prep_series("FEDFUNDS")
    gs10 = prep_series("GS10")
    tb3ms = prep_series("TB3MS")
    baa = prep_series("BAA")

    # Sentiment
    consumer_sentiment = load_consumer_sentiment()

    # Market data
    crsp = load_crsp_sp500()

    df = pd.DataFrame(
        {
            "ip_growth_yoy": yoy_log_growth(indpro),
            "housing_starts_growth_yoy": yoy_log_growth(houst),
            "unemployment_rate": unrate,
            "real_income_ex_transfers_yoy": yoy_log_growth(w875rx1),
            "capacity_utilization": cumfns,
            "inflation_yoy": yoy_log_growth(cpi),
            "m2_growth_yoy": yoy_log_growth(m2sl),
            "fed_funds": fedfunds,
            "term_spread": gs10 - tb3ms,
            "credit_spread_baa10y": baa - gs10,
            "consumer_sentiment": consumer_sentiment,
            "sp500_vol_12m": crsp["sp500_vol_12m"],
        }
    )

    df = df[
        [
            "ip_growth_yoy",
            "housing_starts_growth_yoy",
            "unemployment_rate",
            "real_income_ex_transfers_yoy",
            "capacity_utilization",
            "inflation_yoy",
            "m2_growth_yoy",
            "fed_funds",
            "term_spread",
            "credit_spread_baa10y",
            "consumer_sentiment",
            "sp500_vol_12m",
        ]
    ]

    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "date"
    df = df.ffill()
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
    print(macro)
    macro.to_parquet(get_data() / "fred" / "macro_m.pq")