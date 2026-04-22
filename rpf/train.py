import pandas as pd

from tqdm import tqdm

from rpf.model import RPF
from rpf.interpret import interpret_splits, interpret_leaves, interpret_regimes, load_theme_map

def train_roll(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """
    Train the rolling random market-state tree ensemble and return out-of-sample SDFs.

    When save=True, rolling mode exports compact per-window summaries rather than
    raw splits, leaves, and regimes for every window.
    """
    window = int(config["window"])
    dtype = config["dtype"]
    save = bool(config.get("save", False))

    dates = pd.DatetimeIndex(factors.index)
    factor_columns = factors.columns.tolist()
    macro_columns = macro.columns.tolist()

    theme_map = None
    if save and config.get("factors") == "all":
        theme_map = load_theme_map()

    F = factors.to_numpy(dtype=dtype, copy=True)
    M = macro.to_numpy(dtype=dtype, copy=True)
    FF = (F[:, :, None] * F[:, None, :]).astype(dtype, copy=False)

    n_windows = len(dates) - window
    start_w = max(0, int(360 - window))
    start_w = min(start_w, n_windows)

    plan_model = RPF(config=config)
    sample_indices, seeds = plan_model.make_tree_plan(window)

    sdfs: list[pd.DataFrame] = []
    macro_rows: list[pd.DataFrame] = []
    char_rows: list[pd.DataFrame] = []

    for w in tqdm(range(start_w, n_windows), desc="Rolling Window"):
        is_start = w
        is_end = w + window
        oos = w + window

        F_is = F[is_start:is_end]
        FF_is = FF[is_start:is_end]
        M_is = M[is_start:is_end]
        dates_is = dates[is_start:is_end]

        F_oos = F[oos]
        M_oos = M[oos]
        date_oos = dates[oos]

        model = RPF(config=config)
        model.fit(F_is, FF_is, M_is, sample_indices=sample_indices, seeds=seeds)

        sdf_row = model.predict(F_oos, M_oos, date_oos)
        sdfs.append(sdf_row)

        if save:
            splits = model.export_splits(macro_columns=macro_columns)
            leaves = model.export_leaves(
                macro_columns=macro_columns,
                factor_columns=factor_columns,
            )

            macro_rows.append(
                interpret_splits(
                    splits=splits,
                    macro_columns=macro_columns,
                    date=date_oos,
                )
            )

            char_leaf_row = interpret_leaves(
                leaves=leaves,
                factor_columns=factor_columns,
                date=date_oos,
                theme_map=theme_map,
            )
            char_regime_row = interpret_regimes(
                model=model,
                dates_is=dates_is,
                F_is=F_is,
                M_is=M_is,
                F_oos=F_oos,
                M_oos=M_oos,
                date_oos=date_oos,
                factor_columns=factor_columns,
                macro_columns=macro_columns,
                theme_map=theme_map,
            )
            char_rows.append(char_leaf_row.join(char_regime_row, how="inner"))

    result = {
        "sdfs": pd.concat(sdfs, axis=0).sort_index(),
        "splits": pd.DataFrame(),
        "leaves": pd.DataFrame(),
        "regimes": pd.DataFrame(),
        "macro": pd.DataFrame(),
        "char": pd.DataFrame(),
    }

    if save:
        result["macro"] = pd.concat(macro_rows, axis=0).sort_index()
        result["char"] = pd.concat(char_rows, axis=0).sort_index()

    return result

def train_fix(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """
    Train RMST once on a fixed train split and predict over a fixed test split.

    Parameters
    ----------
    factors : DataFrame
        Monthly factor matrix indexed by date.
    macro : DataFrame
        Monthly macro-state matrix indexed by date and aligned to factors.
    config : dict
        RMST configuration. Must contain:
        split = {
            "train": [start_date, end_date],
            "test": [start_date, end_date],
        }

    Returns
    -------
    dict[str, DataFrame]
        Dictionary with sdfs and, if requested, interpretation tables.
    """
    dtype = config["dtype"]
    save = bool(config.get("save", False))

    train_start, train_end = config["split"]["train"]
    test_start, test_end = config["split"]["test"]

    factors_train = factors.loc[train_start:train_end]
    macro_train = macro.loc[train_start:train_end]

    factors_test = factors.loc[test_start:test_end]
    macro_test = macro.loc[test_start:test_end]

    F_train = factors_train.to_numpy(dtype=dtype, copy=True)
    M_train = macro_train.to_numpy(dtype=dtype, copy=True)
    FF_train = (F_train[:, :, None] * F_train[:, None, :]).astype(dtype, copy=False)

    F_test = factors_test.to_numpy(dtype=dtype, copy=True)
    M_test = macro_test.to_numpy(dtype=dtype, copy=True)
    test_dates = pd.DatetimeIndex(factors_test.index)

    factor_columns = factors.columns.tolist()
    macro_columns = macro.columns.tolist()

    theme_map = None
    if save and config.get("factors") == "all":
        theme_map = load_theme_map()

    model = RPF(config=config)
    model.fit(F_train, FF_train, M_train)

    sdfs: list[pd.DataFrame] = []

    for i in range(len(test_dates)):
        sdf_row = model.predict(F_test[i], M_test[i], test_dates[i])
        sdfs.append(sdf_row)

    result = {
        "sdfs": pd.concat(sdfs, axis=0).sort_index(),
        "splits": pd.DataFrame(),
        "leaves": pd.DataFrame(),
        "regimes": pd.DataFrame(),
        "macro": pd.DataFrame(),
        "char": pd.DataFrame(),
    }

    if save:
        result["splits"] = model.export_splits(macro_columns=macro_columns)
        result["leaves"] = model.export_leaves(
            macro_columns=macro_columns,
            factor_columns=factor_columns,
        )
        result["regimes"] = model.export_regimes(
            dates=test_dates,
            F=F_test,
            M=M_test,
            factor_columns=factor_columns,
            macro_columns=macro_columns,
        )

        result["macro"] = interpret_splits(
            splits=result["splits"],
            macro_columns=macro_columns,
            date=test_dates[-1],
        )

        char_leaf_row = interpret_leaves(
            leaves=result["leaves"],
            factor_columns=factor_columns,
            date=test_dates[-1],
            theme_map=theme_map,
        )
        char_regime_row = interpret_regimes(
            model=model,
            dates_is=pd.DatetimeIndex(factors_train.index),
            F_is=F_train,
            M_is=M_train,
            F_oos=F_test[-1],
            M_oos=M_test[-1],
            date_oos=test_dates[-1],
            factor_columns=factor_columns,
            macro_columns=macro_columns,
            theme_map=theme_map,
        )
        result["char"] = char_leaf_row.join(char_regime_row, how="inner")

    return result

def train(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """
    Dispatch RMST training based on config["train"].

    Parameters
    ----------
    factors : DataFrame
        Monthly factor matrix indexed by date.
    macro : DataFrame
        Monthly macro-state matrix indexed by date and aligned to factors.
    config : dict
        RMST configuration.

    Returns
    -------
    dict[str, DataFrame]
        Output tables from the selected training mode.
    """
    train_mode = config["train"]

    if train_mode == "roll":
        return train_roll(factors, macro, config)

    if train_mode == "fix":
        return train_fix(factors, macro, config)

    raise ValueError(f"unknown train mode: {train_mode}")