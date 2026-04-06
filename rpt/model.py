import numpy as np
import pandas as pd

from rpt.tree import Tree

class RPT:
    def __init__(self, config: dict):
        self.config = config
        self.dtype = config["dtype"]
        self.tree = None

    def fit(self, F: np.ndarray, FF: np.ndarray, M: np.ndarray) -> None:
        """
        Fit MST on one in-sample window.
        """
        tree = Tree(self.config)
        tree.fit(F, FF, M)
        self.tree = tree

    def predict(self, F_row: np.ndarray, M_row: np.ndarray, date) -> pd.DataFrame:
        """
        Predict one out-of-sample SDF value.
        """
        sdf_value = self.tree.predict(F_row, M_row).item()
        idx = pd.Index([date], name="date")
        return pd.DataFrame({"sdf": [sdf_value]}, index=idx)