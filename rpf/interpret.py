import numpy as np
import pandas as pd

from rpf.model import RPF

# -------------------------------------------------
# Splits
# -------------------------------------------------
def interpret_splits(
    splits: pd.DataFrame,
    macro_columns: list[str],
    date,
) -> pd.DataFrame:
    """
    Summarize split importance by macro variable for one rolling window.
    """
    if splits.empty:
        row = {"date": date}
        for name in macro_columns:
            row[f"gain_total_{name}"] = 0.0
            row[f"gain_share_{name}"] = 0.0
            row[f"gain_depth_weighted_{name}"] = 0.0
            row[f"root_share_{name}"] = 0.0
        return pd.DataFrame([row]).set_index("date")

    work = splits.copy()
    work["depth_weighted_gain"] = work["gain"] / (1.0 + work["depth"].astype(float))

    gain_total = (
        work.groupby("split_variable", observed=False)["gain"]
        .sum()
        .reindex(macro_columns, fill_value=0.0)
    )
    gain_depth_weighted = (
        work.groupby("split_variable", observed=False)["depth_weighted_gain"]
        .sum()
        .reindex(macro_columns, fill_value=0.0)
    )

    total_gain = float(gain_total.sum())
    if total_gain > 0.0:
        gain_share = gain_total / total_gain
    else:
        gain_share = pd.Series(0.0, index=macro_columns, dtype=float)

    root = work.loc[work["is_root"]]
    if len(root) > 0:
        root_share = (
            root.groupby("split_variable", observed=False).size()
            .reindex(macro_columns, fill_value=0)
            .astype(float)
            / float(root["tree_id"].nunique())
        )
    else:
        root_share = pd.Series(0.0, index=macro_columns, dtype=float)

    row = {"date": date}
    for name in macro_columns:
        row[f"gain_total_{name}"] = float(gain_total.loc[name])
        row[f"gain_share_{name}"] = float(gain_share.loc[name])
        row[f"gain_depth_weighted_{name}"] = float(gain_depth_weighted.loc[name])
        row[f"root_share_{name}"] = float(root_share.loc[name])

    return pd.DataFrame([row]).set_index("date")

# -------------------------------------------------
# Leaves
# -------------------------------------------------
def interpret_leaves(
    leaves: pd.DataFrame,
    factor_columns: list[str],
    date,
) -> pd.DataFrame:
    """
    Summarize leaf-level beta means and dispersions for one rolling window.
    """
    row = {"date": date}

    if leaves.empty:
        for name in factor_columns:
            row[f"beta_mean_{name}"] = np.nan
            row[f"beta_dispersion_{name}"] = np.nan
        return pd.DataFrame([row]).set_index("date")

    weights = leaves["count"].to_numpy(dtype=float)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        weights = np.ones_like(weights, dtype=float) / float(len(weights))
    else:
        weights = weights / weight_sum

    for name in factor_columns:
        beta = leaves[f"beta_{name}"].to_numpy(dtype=float)
        beta_mean = float(np.sum(weights * beta))
        beta_dispersion = float(np.sum(weights * (beta - beta_mean) ** 2))

        row[f"beta_mean_{name}"] = beta_mean
        row[f"beta_dispersion_{name}"] = beta_dispersion

    return pd.DataFrame([row]).set_index("date")

# -------------------------------------------------
# Regimes
# -------------------------------------------------
def interpret_regimes(
    model: RPF,
    dates_is: pd.DatetimeIndex,
    F_is: np.ndarray,
    M_is: np.ndarray,
    F_oos: np.ndarray,
    M_oos: np.ndarray,
    date_oos,
    factor_columns: list[str],
    macro_columns: list[str],
) -> pd.DataFrame:
    """
    Summarize active and state-conditioned betas for one rolling window.
    """
    row = {"date": date_oos}

    # Active OOS beta: average beta across trees for the out-of-sample month.
    regimes_oos = model.export_regimes(
        dates=pd.DatetimeIndex([date_oos]),
        F=F_oos[None, :],
        M=M_oos[None, :],
        factor_columns=factor_columns,
        macro_columns=macro_columns,
    )

    for factor_name in factor_columns:
        row[f"beta_oos_{factor_name}"] = float(regimes_oos[f"beta_{factor_name}"].mean())

    # State-conditioned beta spreads within the in-sample rolling window.
    regimes_is = model.export_regimes(
        dates=dates_is,
        F=F_is,
        M=M_is,
        factor_columns=factor_columns,
        macro_columns=macro_columns,
    )

    for macro_name in macro_columns:
        macro_col = f"macro_{macro_name}"
        threshold = float(regimes_is[macro_col].median())

        high = regimes_is.loc[regimes_is[macro_col] >= threshold]
        low = regimes_is.loc[regimes_is[macro_col] < threshold]

        for factor_name in factor_columns:
            beta_col = f"beta_{factor_name}"

            if high.empty or low.empty:
                spread = np.nan
            else:
                spread = float(high[beta_col].mean() - low[beta_col].mean())

            row[f"beta_spread_{factor_name}__{macro_name}"] = spread

    return pd.DataFrame([row]).set_index("date")