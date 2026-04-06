import re
import glob
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from util.system import get_data

# Load data
def load_ctff_data(size: str) -> pd.DataFrame:
    data = pd.read_parquet(get_data() / "ctff" / f"ctff_m_rank_{size}.pq")
    data = data.loc[data.index.get_level_values("eom") >= 19630101]
    return data

# Data
def format_ctff_data(
    data_all: pd.DataFrame,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtype_np = np.float32 if config["dtype"] == "float32" else np.float64
    data_all = data_all.reorder_levels(["eom", "id"]).sort_index()
    R_col = "ret_exc_lead1m"
    drop_cols = {"ret_exc_lead1m"}
    X_cols = [c for c in data_all.columns if c not in drop_cols]
    X_all = data_all[X_cols].to_numpy(dtype=dtype_np, copy=False)
    R_all = data_all[R_col].to_numpy(dtype=dtype_np, copy=False)
    T_all = data_all.index.get_level_values("eom").astype("int64").to_numpy()
    P_all = data_all.index.get_level_values("id").astype("int64").to_numpy()
    return X_all, R_all, T_all, P_all

# Load data
def load_jkp_data() -> pd.DataFrame:
    data = pd.read_pickle(get_data() / "lfm" / "usa_132_per_size_ranks_False_permno_False_float32_sorted.pkl").set_index(["date", "id"])
    return data

# Filter size data
def format_jkp_size_data(data: pd.DataFrame, config: dict) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    data_size_map = {}
    for size in config["size_list"]:
        if size != "all":
            data_size = data.loc[data.size_grp == size].copy(deep=True)
        else:
            data_size = data.copy(deep=True)
        data_tuple_size = format_jkp_data(data_size, config)
        data_size_map[size] = data_tuple_size
    return data_size_map

# Data
def format_jkp_data(
    data_all: pd.DataFrame,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtype_np = np.float32 if config["dtype"] == "float32" else np.float64
    data_all = data_all.rename(index=lambda d: int(d.strftime("%Y%m%d")), level="date")
    data_all = data_all.reorder_levels(["date", "id"]).sort_index()
    R_col = "r_1"
    drop_cols = {"r_1", "size_grp", "excntry"}
    X_cols = [c for c in data_all.columns if c not in drop_cols]
    X_all = data_all[X_cols].to_numpy(dtype=dtype_np, copy=False)
    R_all = data_all[R_col].to_numpy(dtype=dtype_np, copy=False)
    T_all = data_all.index.get_level_values("date").astype("int64").to_numpy()
    P_all = data_all.index.get_level_values("id").astype("int64").to_numpy()
    return X_all, R_all, T_all, P_all

# Save file in chunks
def save_parquet_chunk(df: pd.DataFrame, folder_path: str, file_pattern: str, num_chunk: int):
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    chunks = np.array_split(df, num_chunk)
    for i, chunk in tqdm(enumerate(chunks, 1), total=num_chunk, desc='Chunk'):
        filename = folder_path / f"{file_pattern}_{i:06d}.pq"
        chunk.to_parquet(filename, compression="brotli")

# Read file in chunks
def read_parquet_chunk(folder_path: str, file_pattern: str, columns: List[str]=None, num_file: int=None) -> pd.DataFrame:
    folder_path = Path(folder_path)
    full_pattern = str(folder_path / file_pattern)
    file_list = glob.glob(full_pattern)

    rx = re.compile(re.escape(file_pattern).replace(r"\*", r"(\d+)"))
    file_list = [f for f in file_list if rx.search(Path(f).name)]
    file_list.sort(key=lambda x: int(rx.search(Path(x).name).group(1)))

    if num_file is not None:
        file_list = file_list[:num_file]

    df_dict = {}
    with ThreadPoolExecutor() as exe:
        futures = [exe.submit(lambda f: (f, pd.read_parquet(f, columns=columns)), f) for f in file_list]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Chunk"):
            f, df = fut.result()
            df_dict[f] = df

    return pd.concat([df_dict[f] for f in file_list], axis=0)