import yaml
import copy
import hashlib

from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

# Hash config
def hash_config(config: dict) -> str:
    config = copy.deepcopy(config)
    return hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode()).hexdigest()[:10]

# Make run directory
def make_run_dir(
    root: Path,
    config: Dict[str, Any],
    filename: str = "config.yml",
) -> Path:
    dump_config = copy.deepcopy(config)
    if "device" in dump_config:
        dump_config["device"] = str(dump_config["device"])
    if "use_dtype" in dump_config:
        dump_config["use_dtype"] = str(dump_config["use_dtype"])

    h = hash_config(dump_config)
    run_dir = root / str(h)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / filename, "w") as f:
        yaml.safe_dump(dump_config, f, sort_keys=True)

    return run_dir

# Load run directory
def load_run_dir(
    root: Path,
    config: Dict[str, Any],
) -> Path:
    dump_config = copy.deepcopy(config)
    if "device" in dump_config:
        dump_config["device"] = str(dump_config["device"])
    if "use_dtype" in dump_config:
        dump_config["use_dtype"] = str(dump_config["use_dtype"])

    h = hash_config(dump_config)
    run_dir = root / str(h)
    return run_dir

# Make iter sweep
def iter_sweep(
    base_config: Dict[str, Any],
    sweep_spec: Dict[str, str],
    order: List[str],
) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Yields (config, choices) for each point in the sweep.

    sweep_spec maps config_key -> list_key in base_config
      e.g. {"size": "size_list", "window": "window_list", "z": "z_list"}
    order sets nesting order, e.g. ["size", "window", "z"] or ["window", "size", "z"]
    """
    def rec(i: int, cfg: Dict[str, Any], choices: Dict[str, Any]):
        if i == len(order):
            yield cfg, choices
            return
        k = order[i]
        list_key = sweep_spec[k]
        values = cfg[list_key]
        for v in values:
            cfg2 = copy.deepcopy(cfg)
            cfg2[k] = v
            choices2 = dict(choices)
            choices2[k] = v
            yield from rec(i + 1, cfg2, choices2)

    yield from rec(0, copy.deepcopy(base_config), {})