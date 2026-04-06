import numpy as np
import pandas as pd

from tqdm import tqdm
from pandas import DataFrame
from typing import List, Tuple

from lr.model import LR

def train_roll(
    factors: pd.DataFrame,
    config: dict,
) -> DataFrame:
    """
    Train the rolling linear ridge model on precomputed factors.

    Parameters
    ----------
    factors : DataFrame
        Monthly factor matrix indexed by date.
    config : dict
        LR configuration.

    Returns
    -------
    sdfs : DataFrame
        Out-of-sample SDF series by z.
    lams : DataFrame
        Out-of-sample coefficient path by z and feature.
    """
    window = int(config["window"])
    dates = pd.DatetimeIndex(factors.index)
    F = factors.to_numpy(dtype=config["dtype"], copy=True)

    n_windows = len(dates) - window
    start_w = max(0, int(360 - window))
    start_w = min(start_w, n_windows)

    sdfs: List[DataFrame] = []

    for w in tqdm(range(start_w, n_windows), desc="Rolling Window"):
        is_start = w
        is_end = w + window
        oos = w + window

        F_is = F[is_start:is_end]
        F_oos = F[oos]
        date_oos = dates[oos]

        model = LR(config=config)
        model.fit(F_is)

        sdf_row = model.predict(F_oos, date_oos)
        sdfs.append(sdf_row)

    return pd.concat(sdfs, axis=0)

def train_fix(
    factors: pd.DataFrame,
    config: dict,
) -> DataFrame:
    """
    Train the linear ridge model once on a fixed train split and predict over a fixed test split.

    Parameters
    ----------
    factors : DataFrame
        Monthly factor matrix indexed by date.
    config : dict
        LR configuration. Must contain:
        split = {
            "train": [start_date, end_date],
            "test": [start_date, end_date],
        }

    Returns
    -------
    sdfs : DataFrame
        Out-of-sample SDF series by z.
    lams : DataFrame
        Out-of-sample coefficient path by z and feature.
    """
    train_start, train_end = config["split"]["train"]
    test_start, test_end = config["split"]["test"]

    factors_train = factors.loc[train_start:train_end]
    factors_test = factors.loc[test_start:test_end]

    F_train = factors_train.to_numpy(dtype=config["dtype"], copy=True)
    F_test = factors_test.to_numpy(dtype=config["dtype"], copy=True)
    test_dates = pd.DatetimeIndex(factors_test.index)

    model = LR(config=config)
    model.fit(F_train)

    sdfs: List[DataFrame] = []

    for i in range(len(test_dates)):
        sdf_row = model.predict(F_test[i], test_dates[i])
        sdfs.append(sdf_row)

    return pd.concat(sdfs, axis=0)

def train(
    factors: pd.DataFrame,
    config: dict,
) -> DataFrame:
    """
    Dispatch LR training based on config["train"].

    Parameters
    ----------
    factors : DataFrame
        Monthly factor matrix indexed by date.
    config : dict
        LR configuration.

    Returns
    -------
    sdfs : DataFrame
        Out-of-sample SDF series by z.
    lams : DataFrame
        Out-of-sample coefficient path by z and feature.
    """
    if config["train"] == "roll":
        return train_roll(factors, config)

    if config["train"] == "fix":
        return train_fix(factors, config)

    raise ValueError(f'unknown train mode: {config["train"]}')