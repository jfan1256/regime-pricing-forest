import numpy as np
import pandas as pd

from tqdm import tqdm

from util.data import load_jkp_data, format_jkp_size_data, load_ctff_data, format_ctff_data
from util.system import load_yaml, get_config, get_data


def map_t_to_rows(T_all: np.ndarray) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """Map each integer date code to its contiguous row block in the stacked panel."""
    t_vals, first = np.unique(T_all, return_index=True)
    last = np.concatenate([first[1:], np.array([len(T_all)])])
    t_cross_start = {int(t): int(s) for t, s in zip(t_vals, first)}
    t_cross_end = {int(t): int(e) for t, e in zip(t_vals, last)}
    return t_vals, t_cross_start, t_cross_end


def compute_factors(
    X_all: np.ndarray,
    R_all: np.ndarray,
    T_all: np.ndarray,
    X_cols: list[str],
) -> pd.DataFrame:
    """
    Precompute characteristic-managed factors:

        F_t = X_t' R_{t+1} / sqrt(N_t)

    Parameters
    ----------
    X_all : (sum_t N_t, d) array
    R_all : (sum_t N_t,) array
    T_all : (sum_t N_t,) array of integer YYYYMMDD date codes
    X_cols : list[str]
        Names of the characteristic columns in X_all, length d.

    Returns
    -------
    DataFrame
        Index is calendar date, columns are factor names from X_cols.
    """
    t_vals, t_cross_start, t_cross_end = map_t_to_rows(T_all)
    d = X_all.shape[1]

    if len(X_cols) != d:
        raise ValueError(f"len(X_cols)={len(X_cols)} does not match X_all.shape[1]={d}")

    F = np.zeros((len(t_vals), d), dtype=X_all.dtype)

    for i, t in enumerate(tqdm(t_vals, desc="Compute LR factors")):
        s = t_cross_start[int(t)]
        e = t_cross_end[int(t)]
        Xt = X_all[s:e]      # (N_t, d)
        Rt1 = R_all[s:e]     # (N_t,)
        Nt = e - s
        F[i] = Xt.T @ Rt1 / np.sqrt(Nt)  # (d,)

    dates = pd.to_datetime(pd.Index(t_vals.astype(str)), format="%Y%m%d")

    return pd.DataFrame(
        F,
        index=pd.DatetimeIndex(dates, name="date"),
        columns=pd.Index(X_cols, name="feature"),
    )

if __name__ == "__main__":
    DATA_TYPE = 'ctff'

    out_dir = get_data() / "factor" / "lr"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = {"dtype": "float32"}

    if DATA_TYPE == 'jkp':
        base_config["size_list"] = ["all", "micro", "small", "large", "mega"]
        data = load_jkp_data()
        X_cols = [c for c in data.columns if c not in {"r_1", "size_grp", "excntry"}]
        size_data = format_jkp_size_data(data, base_config)
        for size in base_config["size_list"]:
            X_all, R_all, T_all, P_all = size_data[size]
            factors = compute_factors(X_all, R_all, T_all, X_cols)
            factors.to_parquet(out_dir / f"lr_{DATA_TYPE}_{size}_m.pq")

    elif DATA_TYPE == 'ctff':
        base_config["size_list"] = ["all", "small", "large", "mega"]
        for size in base_config["size_list"]:
            data = load_ctff_data(size)
            X_cols = [c for c in data.columns if c not in {"ret_exc_lead1m"}]
            X_all, R_all, T_all, P_all = format_ctff_data(data, base_config)
            factors = compute_factors(X_all, R_all, T_all, X_cols)
            factors.to_parquet(out_dir / f"lr_{DATA_TYPE}_{size}_m.pq")