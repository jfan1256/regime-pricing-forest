import pandas as pd

from tqdm import tqdm
from pandas import DataFrame

from rpt.model import RPT

def train_roll(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    config: dict,
) -> DataFrame:
    """
    Train the rolling market-state tree and return out-of-sample SDFs.

    Parameters
    ----------
    factors : DataFrame
        Monthly factor matrix indexed by date.
    macro : DataFrame
        Monthly macro-state matrix indexed by date and aligned to factors.
    config : dict
        MST configuration.

    Returns
    -------
    DataFrame
        Out-of-sample SDF series indexed by date.
    """
    window = int(config["window"])
    dtype = config["dtype"]

    dates = pd.DatetimeIndex(factors.index)
    F = factors.to_numpy(dtype=dtype, copy=True)
    M = macro.to_numpy(dtype=dtype, copy=True)
    FF = F[:, :, None] * F[:, None, :]

    n_windows = len(dates) - window
    start_w = max(0, int(360 - window))
    start_w = min(start_w, n_windows)

    sdfs: list[DataFrame] = []

    for w in tqdm(range(start_w, n_windows), desc="Rolling Window"):
        is_start = w
        is_end = w + window
        oos = w + window

        F_is = F[is_start:is_end]
        FF_is = FF[is_start:is_end]
        M_is = M[is_start:is_end]

        F_oos = F[oos]
        M_oos = M[oos]
        date_oos = dates[oos]

        model = RPT(config=config)
        model.fit(F_is, FF_is, M_is)

        sdf_row = model.predict(F_oos, M_oos, date_oos)
        sdfs.append(sdf_row)

    return pd.concat(sdfs, axis=0)

def train_fix(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    config: dict,
) -> DataFrame:
    """
    Train MST once on a fixed train split and predict over a fixed test split.

    Parameters
    ----------
    factors : DataFrame
        Monthly factor matrix indexed by date.
    macro : DataFrame
        Monthly macro-state matrix indexed by date and aligned to factors.
    config : dict
        MST configuration. Must contain:
        split = {
            "train": [start_date, end_date],
            "test": [start_date, end_date],
        }

    Returns
    -------
    DataFrame
        Out-of-sample SDF series indexed by test dates.
    """
    dtype = config["dtype"]

    train_start, train_end = config["split"]["train"]
    test_start, test_end = config["split"]["test"]

    factors_train = factors.loc[train_start:train_end]
    macro_train = macro.loc[train_start:train_end]

    factors_test = factors.loc[test_start:test_end]
    macro_test = macro.loc[test_start:test_end]

    F_train = factors_train.to_numpy(dtype=dtype, copy=True)
    M_train = macro_train.to_numpy(dtype=dtype, copy=True)
    FF_train = F_train[:, :, None] * F_train[:, None, :]

    F_test = factors_test.to_numpy(dtype=dtype, copy=True)
    M_test = macro_test.to_numpy(dtype=dtype, copy=True)
    test_dates = pd.DatetimeIndex(factors_test.index)

    model = RPT(config=config)
    model.fit(F_train, FF_train, M_train)

    sdfs: list[DataFrame] = []

    for i in range(len(test_dates)):
        sdf_row = model.predict(F_test[i], M_test[i], test_dates[i])
        sdfs.append(sdf_row)

    return pd.concat(sdfs, axis=0)

def train(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    config: dict,
) -> DataFrame:
    """
    Dispatch MST training based on config["train"].

    Parameters
    ----------
    factors : DataFrame
        Monthly factor matrix indexed by date.
    macro : DataFrame
        Monthly macro-state matrix indexed by date and aligned to factors.
    config : dict
        MST configuration.

    Returns
    -------
    DataFrame
        Out-of-sample SDF series indexed by date.
    """
    if config["train"] == "roll":
        return train_roll(factors, macro, config)

    if config["train"] == "fix":
        return train_fix(factors, macro, config)

    raise ValueError(f'unknown train mode: {config["train"]}')