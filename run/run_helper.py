import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd

from lr.train import train as train_lr
from rlr.train import train as train_rlr
from rpf.train import train as train_rpf
from rpt.train import train as train_rpt
from util.run import make_run_dir, iter_sweep
from util.system import load_yaml, get_config, get_result_lr, get_result_rlr, get_result_rpf, get_result_rpt, get_data

def run_lr(config_name):
    base_config = load_yaml(get_config() / f"{config_name}.yml")
    result_dir = get_result_lr()
    result_dir.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(result_dir, base_config, filename=f"{config_name}.yml")
    print("*" * 30 + f"HASH={run_dir.name}" + "*" * 30 + "\n")

    factor_dir = get_data() / "factor" / "lr"

    sweep_spec = {
        "size": "size_list",
    }
    order = ["size"]

    for config, choices in iter_sweep(base_config, sweep_spec, order):
        size = choices["size"]

        run_config_dir = make_run_dir(run_dir, config, filename=f"{config_name}_run.yml")
        output_path = run_config_dir / "sdfs.pq"

        print("*" * 30 + f"SIZE={size}" + "*" * 30 + "\n")

        if output_path.exists():
            print(f"Skipping existing output: {output_path}")
            continue

        factors = pd.read_parquet(factor_dir / f"lr_{config['data']}_{size}_m.pq").sort_index()
        if config["factors"] != 'all':
            factors = factors[config["factors"]]
        sdfs = train_lr(factors, config)

        sdfs.to_parquet(output_path)

def run_rlr(config_name):
    base_config = load_yaml(get_config() / f"{config_name}.yml")
    result_dir = get_result_rlr()
    result_dir.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(result_dir, base_config, filename=f"{config_name}.yml")
    print("*" * 30 + f"HASH={run_dir.name}" + "*" * 30 + "\n")

    factor_dir = get_data() / "factor" / "lr"

    sweep_spec = {
        "size": "size_list",
        "z": "z_list",
    }
    order = ["size", "z"]

    for config, choices in iter_sweep(base_config, sweep_spec, order):
        size = choices["size"]
        z = choices["z"]

        run_config_dir = make_run_dir(run_dir, config, filename=f"{config_name}_run.yml")
        output_path = run_config_dir / "sdfs.pq"

        print(
            "*" * 30
            + f"SIZE={size} | Z={z}"
            + "*" * 30
            + "\n"
        )

        if output_path.exists():
            print(f"Skipping existing output: {output_path}")
            continue

        factors = pd.read_parquet(factor_dir / f"lr_{config['data']}_{size}_m.pq").sort_index()
        if config["factors"] != 'all':
            factors = factors[config["factors"]]
        sdfs = train_rlr(factors, config)

        sdfs.to_parquet(output_path)

def run_rpt(config_name):
    base_config = load_yaml(get_config() / f"{config_name}.yml")
    result_dir = get_result_rpt()
    result_dir.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(result_dir, base_config, filename=f"{config_name}.yml")
    print("*" * 30 + f"HASH={run_dir.name}" + "*" * 30 + "\n")

    factor_dir = get_data() / "factor" / "lr"
    macro_path = get_data() / "fred" / "macro_m.pq"

    sweep_spec = {
        "size": "size_list",
        "max_depth": "max_depth_list",
        "z": "z_list",
    }
    order = ["size", "max_depth", "z"]

    for config, choices in iter_sweep(base_config, sweep_spec, order):
        size = choices["size"]
        max_depth = choices["max_depth"]
        z = choices["z"]

        run_config_dir = make_run_dir(run_dir, config, filename=f"{config_name}_run.yml")
        output_path = run_config_dir / "sdfs.pq"

        print(
            "*" * 30
            + f"SIZE={size} | DEPTH={max_depth} | Z={z}"
            + "*" * 30
            + "\n"
        )

        if output_path.exists():
            print(f"Skipping existing output: {output_path}")
            continue

        factors = pd.read_parquet(factor_dir / f"lr_{config['data']}_{size}_m.pq").sort_index()
        if config["factors"] != 'all':
            factors = factors[config["factors"]]
        macro = pd.read_parquet(macro_path).sort_index()
        macro = macro.loc[factors.index]

        sdfs = train_rpt(factors, macro, config)
        sdfs.to_parquet(output_path)

def run_rpf(config_name):
    base_config = load_yaml(get_config() / f"{config_name}.yml")
    result_dir = get_result_rpf()
    result_dir.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(result_dir, base_config, filename=f"{config_name}.yml")
    print("*" * 30 + f"HASH={run_dir.name}" + "*" * 30 + "\n")

    factor_dir = get_data() / "factor" / "lr"
    macro_path = get_data() / "fred" / "macro_m.pq"

    sweep_spec = {
        "size": "size_list",
        "num_tree": "num_tree_list",
        "max_depth": "max_depth_list",
        "z": "z_list",
    }
    order = ["size", "num_tree", "max_depth", "z"]

    for config, choices in iter_sweep(base_config, sweep_spec, order):
        size = choices["size"]
        num_tree = choices["num_tree"]
        max_depth = choices["max_depth"]
        z = choices["z"]

        run_config_dir = make_run_dir(run_dir, config, filename=f"{config_name}_run.yml")
        output_path = run_config_dir / "sdfs.pq"

        print(
            "*" * 30
            + f"SIZE={size} | NUM_TREE={num_tree} | DEPTH={max_depth} | Z={z}"
            + "*" * 30
            + "\n"
        )

        if output_path.exists():
            print(f"Skipping existing output: {output_path}")
            continue

        factors = pd.read_parquet(factor_dir / f"lr_{config['data']}_{size}_m.pq").sort_index()
        if config["factors"] != 'all':
            factors = factors[config["factors"]]
        macro = pd.read_parquet(macro_path).sort_index()
        macro = macro.loc[factors.index]

        result = train_rpf(factors, macro, config)

        result["sdfs"].to_parquet(run_config_dir / "sdfs.pq")

        if config["save"]:
            if "splits" in result and not result["splits"].empty:
                result["splits"].to_parquet(run_config_dir / "splits.pq")

            if "leaves" in result and not result["leaves"].empty:
                result["leaves"].to_parquet(run_config_dir / "leaves.pq")

            if "regimes" in result and not result["regimes"].empty:
                result["regimes"].to_parquet(run_config_dir / "regimes.pq")

            if "macro" in result and not result["macro"].empty:
                result["macro"].to_parquet(run_config_dir / "macro.pq")

            if "char" in result and not result["char"].empty:
                result["char"].to_parquet(run_config_dir / "char.pq")