# util/analyze.py
import numpy as np
import pandas as pd

from collections.abc import Iterable
from typing import Any


def sharpe(df: pd.DataFrame) -> pd.Series:
    """
    Annualized Sharpe ratio per column for monthly returns.
    """
    return np.sqrt(12.0) * df.mean(axis=0) / df.std(axis=0)


def alpha_tstat(
    target: pd.Series,
    benchmark: pd.Series,
) -> float:
    """
    Return the t-statistic of the intercept in:

        target_t = alpha + beta * benchmark_t + eps_t
    """
    work = pd.concat(
        [target.rename("target"), benchmark.rename("benchmark")],
        axis=1,
    ).dropna()

    y = work["target"].to_numpy(dtype=float)
    x = work["benchmark"].to_numpy(dtype=float)

    # Design matrix shape: (n_obs, 2) with intercept and benchmark.
    X = np.column_stack([np.ones(len(x), dtype=float), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)

    resid = y - X @ beta
    n_obs, n_param = X.shape
    sigma2 = float(resid @ resid) / (n_obs - n_param)
    vcov = sigma2 * xtx_inv

    alpha = float(beta[0])
    alpha_se = float(np.sqrt(vcov[0, 0]))
    return alpha / alpha_se


def filter_sample(
    df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Restrict a panel to the inclusive date interval [start, end].
    """
    if df.empty:
        return df

    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out.loc[(out.index >= start) & (out.index <= end)]


def subset_level(
    df: pd.DataFrame,
    level: str,
    value: Any,
    normalize_str: bool = False,
    drop_level: bool = True,
) -> pd.DataFrame:
    """
    Restrict a panel to one column level value.
    """
    if level not in df.columns.names:
        raise ValueError(f"{level!r} not found in columns: {df.columns.names}")

    values = pd.Index(df.columns.get_level_values(level))
    if normalize_str:
        lhs = values.map(lambda x: str(x).strip().lower())
        rhs = str(value).strip().lower()
        mask = lhs == rhs
    else:
        mask = values == value

    if not mask.any():
        raise ValueError(f"Value {value!r} not found in level {level!r}.")

    out = df.loc[:, mask]
    if drop_level:
        out.columns = out.columns.droplevel(level)
    return out


def filter_column_levels(
    df: pd.DataFrame,
    filters: dict[str, Any],
    normalize_str_levels: Iterable[str] | None = None,
    drop_filtered_levels: bool = False,
) -> pd.DataFrame:
    """
    Restrict a panel to a set of column-level values.
    """
    if df.empty or not isinstance(df.columns, pd.MultiIndex):
        return df

    normalize_levels = set(normalize_str_levels or [])
    out = df

    for level, value in filters.items():
        out = subset_level(
            out,
            level=level,
            value=value,
            normalize_str=level in normalize_levels,
            drop_level=drop_filtered_levels,
        )

    return out


def series_by_z(
    df: pd.DataFrame,
) -> dict[float, pd.Series]:
    """
    Return return series keyed by z.

    Works for either:
    - single-level columns containing z directly
    - MultiIndex columns with a 'z' level
    """
    out: dict[float, pd.Series] = {}

    if isinstance(df.columns, pd.MultiIndex):
        names = list(df.columns.names)
        z_idx = names.index("z")
        for col in df.columns:
            out[float(col[z_idx])] = df[col]
    else:
        for col in df.columns:
            out[float(col)] = df[col]

    return dict(sorted(out.items()))


def sharpe_by_z(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Return Sharpe indexed by z.
    """
    sr = sharpe(df)

    if isinstance(sr.index, pd.MultiIndex):
        z_index = sr.index.get_level_values("z").astype(float)
    else:
        z_index = pd.Index(sr.index.astype(float), name="z")

    sr.index = pd.Index(z_index, name="z")
    return sr.sort_index()


def sharpe_pivot(
    df: pd.DataFrame,
    index_level: str,
    column_level: str,
) -> pd.DataFrame:
    """
    Pivot Sharpe ratios into a table indexed by one level and columned by another.
    """
    sr = sharpe(df)

    if not isinstance(sr.index, pd.MultiIndex):
        raise ValueError("sharpe_pivot requires MultiIndex columns.")

    out = pd.DataFrame(
        {
            index_level: sr.index.get_level_values(index_level),
            column_level: sr.index.get_level_values(column_level),
            "sharpe": sr.to_numpy(dtype=float),
        }
    )
    out = out.pivot(index=index_level, columns=column_level, values="sharpe")
    return out.sort_index().sort_index(axis=1)


def series_by_levels(
    df: pd.DataFrame,
    levels: list[str],
) -> dict[tuple[Any, ...], pd.Series]:
    """
    Return return series keyed by a tuple of MultiIndex column levels.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("series_by_levels requires MultiIndex columns.")

    names = list(df.columns.names)
    idx = [names.index(level) for level in levels]

    out: dict[tuple[Any, ...], pd.Series] = {}
    for col in df.columns:
        key = tuple(col[i] for i in idx)
        out[key] = df[col]

    return out


def best_model(
    df: pd.DataFrame,
) -> tuple[Any, pd.Series, float]:
    """
    Return the best column key, return series, and Sharpe ratio.
    """
    if df.empty:
        raise ValueError("No loaded results found for this model.")

    sr = sharpe(df)
    key = sr.idxmax()
    return key, df[key], float(sr.loc[key])


def best_sharpe_by_level(
    df: pd.DataFrame,
    level: str,
) -> pd.Series:
    """
    Return the best Sharpe ratio within each specified column level.
    """
    if df.empty:
        raise ValueError("No loaded results found.")

    sr = sharpe(df)
    if not isinstance(sr.index, pd.MultiIndex):
        raise ValueError("best_sharpe_by_level requires MultiIndex columns.")

    values = pd.Index(sr.index.get_level_values(level))
    out: dict[Any, float] = {}

    for value in values.unique():
        mask = values == value
        out[value] = float(sr.loc[mask].max())

    return pd.Series(out).sort_index()


def vol_target_series(
    s: pd.Series,
    target_vol: float = 0.10,
) -> pd.Series:
    """
    Scale monthly returns to a target annualized volatility over the full sample.
    """
    s = s.dropna().copy()
    ann_vol = float(s.std() * np.sqrt(12.0))
    if ann_vol == 0.0:
        return s * 0.0
    return s * (target_vol / ann_vol)


def cumulative_sum_return(
    s: pd.Series,
    target_vol: float = 0.10,
) -> pd.Series:
    """
    Arithmetic cumulative return after full-sample volatility targeting.
    """
    return vol_target_series(s, target_vol=target_vol).cumsum()


def inverse_vol_ensemble_return(
    candidates: dict[float, pd.Series],
    window: int = 12,
) -> pd.Series:
    """
    Real-time inverse-volatility ensemble across z candidates.

    For the first `window` months, use equal weights across available z.
    Starting at time t = window, use trailing `window`-month standard deviation
    computed from observations up to t-1, then apply those inverse-vol weights
    to returns at time t.
    """
    if not candidates:
        raise ValueError("No candidate series provided.")

    panel = pd.concat(candidates, axis=1).sort_index()
    panel = panel.dropna(how="all")

    out = pd.Series(index=panel.index, dtype=float)

    for i in range(len(panel)):
        current = panel.iloc[i].dropna()
        if current.empty:
            continue

        if i < window:
            weights = pd.Series(
                1.0 / len(current),
                index=current.index,
                dtype=float,
            )
        else:
            hist = panel.iloc[i - window:i]
            vol = hist.std(axis=0).replace(0.0, np.nan)
            vol = vol.reindex(current.index)
            inv_vol = 1.0 / vol
            inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()

            if inv_vol.empty:
                weights = pd.Series(
                    1.0 / len(current),
                    index=current.index,
                    dtype=float,
                )
            else:
                weights = inv_vol / inv_vol.sum()

        common = current.index.intersection(weights.index)
        out.iloc[i] = float((current.loc[common] * weights.loc[common]).sum())

    return out.dropna()