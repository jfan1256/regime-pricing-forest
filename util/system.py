import yaml
from pathlib import Path

# Get root dir
def get_root():
    return Path(__file__).resolve().parent.parent

# Get root dir
def get_parent_root():
    return Path(__file__).resolve().parent.parent.parent

# Create dir if needed and return path
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

# Get data dir
def get_data():
    return ensure_dir(get_parent_root() / "data")

# Get config dir
def get_config():
    return ensure_dir(get_root() / "config")

# Get result dir
def get_result():
    return ensure_dir(get_parent_root() / "result" / "rpf")

# Get plot dir
def get_plot():
    return ensure_dir(get_result() / "plot")

# Get lr result dir
def get_result_lr():
    return ensure_dir(get_result() / "lr")

# Get rlr dir
def get_result_rlr():
    return ensure_dir(get_result() / "rlr")

# Get mst result dir
def get_result_rpt():
    return ensure_dir(get_result() / "rpt")

# Get rmst result dir
def get_result_rpf():
    return ensure_dir(get_result() / "rpf")

# Load yaml
def load_yaml(file_path):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config