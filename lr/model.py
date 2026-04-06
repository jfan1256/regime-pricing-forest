import numpy as np
import pandas as pd

class LR:
    def __init__(self, config: dict):
        self.config = config
        self.z_list = [float(z) for z in self.config["z_list_na"]]
        self.dtype = self.config["dtype"]
        self.model = None

    def fit(self, F: np.ndarray) -> None:
        """
        Fit the ridge path using the precomputed factor matrix.

        Parameters
        ----------
        F : (T, d) array
            Characteristic-managed factors over the in-sample window.
        """
        T, d = F.shape
        y = np.ones((T,), dtype=self.dtype)

        if d <= T:
            G = F.T @ F
            scale = np.trace(G) / G.shape[0]
            I = np.eye(d, dtype=self.dtype)
            b = F.T @ y

            lam_path = {}
            for z in self.z_list:
                z_eff = np.asarray(z, dtype=self.dtype) * scale
                lam_path[z] = np.linalg.solve(G + z_eff * I, b)
        else:
            H = F @ F.T
            scale = np.trace(H) / H.shape[0]
            I = np.eye(T, dtype=self.dtype)

            lam_path = {}
            for z in self.z_list:
                z_eff = np.asarray(z, dtype=self.dtype) * scale
                alpha = np.linalg.solve(H + z_eff * I, y)
                lam_path[z] = F.T @ alpha

        self.model = {"lam_path": lam_path}

    def predict(self, F_row: np.ndarray, date) -> pd.DataFrame:
        """
        Predict the SDF and save the ridge coefficients for one out-of-sample date.

        Parameters
        ----------
        F_row : (d,) array
            Precomputed factor row for the OOS date.
        date : scalar
            Calendar date for the OOS prediction.

        Returns
        -------
        sdf : DataFrame
            One-row DataFrame of SDF values indexed by date and z.
        lam : DataFrame
            One-row DataFrame of coefficient vectors indexed by date.
        """
        z_list = self.z_list
        lam_path = self.model["lam_path"]

        sdfs = np.stack([F_row.dot(lam_path[z]) for z in z_list])

        idx = pd.Index([date], name="date")
        sdf = pd.DataFrame([sdfs], index=idx, columns=pd.Index(z_list, name="z"))
        return sdf