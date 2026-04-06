import pandas as pd

from typing import Any
from pathlib import Path

from util.run import load_run_dir, iter_sweep
from util.system import load_yaml, get_config

def infer_sweep(base_config: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """
    Infer sweep_spec and order from config keys ending in '_list'.

    Example
    -------
    'window_list' -> sweep key 'window'
    'max_depth_list' -> sweep key 'max_depth'
    """
    order = [key[:-5] for key in base_config if key.endswith("_list")]
    sweep_spec = {key: f"{key}_list" for key in order}
    return sweep_spec, order

def to_multiindex_columns(
    df: pd.DataFrame,
    key: tuple[Any, ...],
    names: list[str],
) -> pd.DataFrame:
    """
    Prefix DataFrame columns with sweep-key MultiIndex levels.
    """
    if isinstance(df.columns, pd.MultiIndex):
        suffixes = list(df.columns)
        suffix_names = list(df.columns.names)
    else:
        suffixes = [(col,) for col in df.columns]
        suffix_names = [df.columns.name]

    out = df.copy()
    out.columns = pd.MultiIndex.from_tuples(
        [key + suffix for suffix in suffixes],
        names=names + suffix_names,
    )
    return out

def load_artifact(
    yml: str,
    result_dir: str | Path,
    artifact: str = "sdfs",
) -> pd.DataFrame:
    """
    Load one saved artifact across all runs implied by the YAML sweep.

    Each run is expected to contain:
        <run_config_dir>/{artifact}.pq

    Sweep dimensions are inferred from all top-level YAML keys ending in '_list'.
    Missing runs are skipped.
    """
    base_config = load_yaml(get_config() / yml)
    base_run_dir = load_run_dir(result_dir, base_config)
    sweep_spec, order = infer_sweep(base_config)

    frames: list[pd.DataFrame] = []

    for config, _ in iter_sweep(base_config, sweep_spec, order):
        path = load_run_dir(base_run_dir, config) / f"{artifact}.pq"
        if not path.exists():
            print(f"Skipping {path}")
            continue

        df = pd.read_parquet(path)
        key = tuple(config[name] for name in order)
        frames.append(to_multiindex_columns(df, key=key, names=order))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1).sort_index(axis=1)

def load_sdfs(
    yml: str,
    result_dir: str | Path,
) -> pd.DataFrame:
    """
    Load saved SDF outputs across all YAML-implied runs.
    """
    return load_artifact(
        yml=yml,
        result_dir=result_dir,
        artifact="sdfs",
    )

def load_macro(
    yml: str,
    result_dir: str | Path,
) -> pd.DataFrame:
    """
    Load saved macro interpretation outputs across all YAML-implied runs.
    """
    return load_artifact(
        yml=yml,
        result_dir=result_dir,
        artifact="macro",
    )

def load_char(
    yml: str,
    result_dir: str | Path,
) -> pd.DataFrame:
    """
    Load saved characteristic interpretation outputs across all YAML-implied runs.
    """
    return load_artifact(
        yml=yml,
        result_dir=result_dir,
        artifact="char",
    )