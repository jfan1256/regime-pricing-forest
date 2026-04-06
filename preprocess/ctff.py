import wrds
import yaml
import pandas as pd
from tqdm import tqdm

from util.system import get_config, get_data

def month_ranges(start: str, end: str):
    start_dt = pd.to_datetime(start).normalize()
    end_dt = pd.to_datetime(end).normalize()
    return pd.date_range(start_dt, end_dt, freq="MS")

def fetch_monthly_wrds(db, table: str, date_col: str, start: str, end: str, out_dir, prefix: str, compression: str = "brotli"):
    out_dir.mkdir(parents=True, exist_ok=True)
    months = month_ranges(start, end)

    for ms in tqdm(months, desc=f"Fetching {prefix} monthly"):
        me = ms + pd.offsets.MonthBegin(1)
        ym = f"{ms.year}{ms.month:02d}"
        out_path = out_dir / f"{prefix}_{ym}.parquet"

        if out_path.exists():
            tqdm.write(f"Skip {prefix} {ym} (exists)")
            continue

        sql = f"""
            SELECT *
            FROM {table}
            WHERE {date_col} >= DATE '{ms.date()}'
              AND {date_col} <  DATE '{me.date()}'
        """

        df = db.raw_sql(sql)
        if df.empty:
            tqdm.write(f"Empty {prefix} {ym}")
            continue

        df.to_parquet(out_path, index=False, compression=compression)

if __name__ == "__main__":
    api_token = yaml.safe_load(open(get_config() / "api.yaml"))["wrds"]
    db = wrds.Connection(wrds_username=api_token)

    data_dir = get_data()
    (data_dir / "ctff").mkdir(parents=True, exist_ok=True)

    print("Fetching ctff_features...")
    ctff_features = db.raw_sql("SELECT * FROM contrib_global_factor.ctff_features;")
    ctff_features.to_csv(data_dir / "ctff" / "ctff_list.csv", index=False)

    start = "1952-01-01"
    end = "2024-01-01"

    print("Fetching ctff_chars monthly...")
    fetch_monthly_wrds(
        db,
        table="contrib_global_factor.ctff_chars",
        date_col="eom",
        start=start,
        end=end,
        out_dir=data_dir / "ctff" / "ctff_m",
        prefix="ctff_m",
        compression="brotli",
    )

    print("Fetching ctff_daily_ret monthly...")
    fetch_monthly_wrds(
        db,
        table="contrib_global_factor.ctff_daily_ret",
        date_col="date",
        start=start,
        end=end,
        out_dir=data_dir / "ctff" / "ctff_d",
        prefix="ctff_d",
        compression="brotli",
    )