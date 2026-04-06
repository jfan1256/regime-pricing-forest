import pandas as pd

from tqdm import tqdm

from rlr.model import RLR

def train_roll(
    factors: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Train the rolling random linear regression ensemble and return out-of-sample SDFs.

    Parameters
    ----------
    factors : DataFrame
        Monthly factor matrix indexed by date.
    config : dict
        RLR configuration.

    Returns
    -------
    sdfs : DataFrame
        Out-of-sample SDF series.
    """
    window = int(config["window"])
    dtype = config["dtype"]

    dates = pd.DatetimeIndex(factors.index)
    F = factors.to_numpy(dtype=dtype, copy=True)

    n_windows = len(dates) - window
    start_w = max(0, int(360 - window))
    start_w = min(start_w, n_windows)

    sdfs: list[pd.DataFrame] = []

    for w in tqdm(range(start_w, n_windows), desc="Rolling Window"):
        is_start = w
        is_end = w + window
        oos = w + window

        F_is = F[is_start:is_end]
        F_oos = F[oos]
        date_oos = dates[oos]

        model = RLR(config=config)
        model.fit(F_is)

        sdf_row = model.predict(F_oos, date_oos)
        sdfs.append(sdf_row)

    return pd.concat(sdfs, axis=0).sort_index()


def train_fix(
    factors: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Train the random linear regression ensemble once on a fixed train split and
    predict over a fixed test split.

    Parameters
    ----------
    factors : DataFrame
        Monthly factor matrix indexed by date.
    config : dict
        RLR configuration. Must contain:
        split = {
            "train": [start_date, end_date],
            "test": [start_date, end_date],
        }

    Returns
    -------
    sdfs : DataFrame
        Out-of-sample SDF series.
    """
    dtype = config["dtype"]

    train_start, train_end = config["split"]["train"]
    test_start, test_end = config["split"]["test"]

    factors_train = factors.loc[train_start:train_end]
    factors_test = factors.loc[test_start:test_end]

    F_train = factors_train.to_numpy(dtype=dtype, copy=True)
    F_test = factors_test.to_numpy(dtype=dtype, copy=True)
    test_dates = pd.DatetimeIndex(factors_test.index)

    model = RLR(config=config)
    model.fit(F_train)

    sdfs: list[pd.DataFrame] = []

    for i in range(len(test_dates)):
        sdf_row = model.predict(F_test[i], test_dates[i])
        sdfs.append(sdf_row)

    return pd.concat(sdfs, axis=0).sort_index()


def train(
    factors: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Dispatch RLR training based on config["train"].

    Parameters
    ----------
    factors : DataFrame
        Monthly factor matrix indexed by date.
    config : dict
        RLR configuration.

    Returns
    -------
    sdfs : DataFrame
        Out-of-sample SDF series.
    """
    train_mode = config["train"]

    if train_mode == "roll":
        return train_roll(factors, config)

    if train_mode == "fix":
        return train_fix(factors, config)

    raise ValueError(f"unknown train mode: {train_mode}")