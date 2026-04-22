import numpy as np
import pandas as pd

from rpf.model import RPF
from util.system import get_data

def load_theme_map() -> dict[str, str]:
    """
    Load the JKP characteristic-to-theme mapping.
    """
    path = get_data() / "jkp" / "jkp_theme_list.csv"
    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError("jkp_theme_list.csv must contain factor and theme columns.")

    factor_col = df.columns[0]
    theme_col = df.columns[1]

    theme_map = (
        df[[factor_col, theme_col]]
        .dropna()
        .drop_duplicates(subset=[factor_col])
        .set_index(factor_col)[theme_col]
        .astype(str)
        .to_dict()
    )
    return theme_map

def aggregate_factor_theme(
    values: dict[str, float],
    theme_map: dict[str, str] | None,
    factor_columns: list[str],
) -> dict[str, float]:
    """
    Aggregate factor-level scalar values to theme-level means.
    """
    if theme_map is None:
        return values

    missing = [name for name in factor_columns if name not in theme_map]
    if missing:
        raise ValueError(f"Missing theme mapping for factors: {missing[:10]}")

    work = pd.DataFrame(
        {
            "factor": factor_columns,
            "value": [float(values[name]) for name in factor_columns],
            "theme": [theme_map[name] for name in factor_columns],
        }
    )

    out = (
        work.groupby("theme", observed=False)["value"]
        .mean()
        .sort_index()
        .to_dict()
    )
    return {str(k): float(v) for k, v in out.items()}

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
    theme_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Summarize leaf-level beta means and dispersions for one rolling window.
    """
    row = {"date": date}

    if leaves.empty:
        output_names = (
            sorted({theme_map[name] for name in factor_columns})
            if theme_map is not None
            else factor_columns
        )
        for name in output_names:
            row[f"beta_mean_{name}"] = np.nan
            row[f"beta_dispersion_{name}"] = np.nan
        return pd.DataFrame([row]).set_index("date")

    weights = leaves["count"].to_numpy(dtype=float)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        weights = np.ones_like(weights, dtype=float) / float(len(weights))
    else:
        weights = weights / weight_sum

    beta_mean_by_factor: dict[str, float] = {}
    beta_dispersion_by_factor: dict[str, float] = {}

    for name in factor_columns:
        beta = leaves[f"beta_{name}"].to_numpy(dtype=float)
        beta_mean = float(np.sum(weights * beta))
        beta_dispersion = float(np.sum(weights * (beta - beta_mean) ** 2))

        beta_mean_by_factor[name] = beta_mean
        beta_dispersion_by_factor[name] = beta_dispersion

    beta_mean_out = aggregate_factor_theme(beta_mean_by_factor, theme_map, factor_columns)
    beta_dispersion_out = aggregate_factor_theme(beta_dispersion_by_factor, theme_map, factor_columns)

    for name in beta_mean_out:
        row[f"beta_mean_{name}"] = float(beta_mean_out[name])
        row[f"beta_dispersion_{name}"] = float(beta_dispersion_out[name])

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
    theme_map: dict[str, str] | None = None,
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

    beta_oos_by_factor: dict[str, float] = {}
    for factor_name in factor_columns:
        beta_oos_by_factor[factor_name] = float(
            regimes_oos[f"beta_{factor_name}"].mean()
        )

    beta_oos_out = aggregate_factor_theme(beta_oos_by_factor, theme_map, factor_columns)
    for name in beta_oos_out:
        row[f"beta_oos_{name}"] = float(beta_oos_out[name])

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

        spread_by_factor: dict[str, float] = {}
        for factor_name in factor_columns:
            beta_col = f"beta_{factor_name}"

            if high.empty or low.empty:
                spread = np.nan
            else:
                spread = float(high[beta_col].mean() - low[beta_col].mean())

            spread_by_factor[factor_name] = spread

        spread_out = aggregate_factor_theme(spread_by_factor, theme_map, factor_columns)
        for name in spread_out:
            row[f"beta_spread_{name}__{macro_name}"] = float(spread_out[name])

    return pd.DataFrame([row]).set_index("date")