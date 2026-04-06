import numpy as np

class Leaf:
    def __init__(self, config: dict):
        self.config = config
        self.dtype = config["dtype"]
        self.z = float(config["z"])

        self.beta = None
        self.loss = None
        self.count = None
        self.gram = None
        self.rhs = None
        self.scale = None
        self.lam = None

    def fit(
        self,
        count: int,
        gram: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """
        Fit one leaf from sufficient statistics.

        Parameters
        ----------
        count : int
            Number of observations in the leaf.
        gram : (d, d) array
            F'F for the leaf.
        rhs : (d,) array
            F'1 for the leaf, i.e. the sum of factor rows.
        """
        d = gram.shape[0]

        scale = np.trace(gram) / d
        lam = np.asarray(self.z, dtype=self.dtype) * scale
        eye = np.eye(d, dtype=self.dtype)

        beta = np.linalg.solve(gram + lam * eye, rhs)
        loss = np.asarray(float(count), dtype=self.dtype) - rhs @ beta

        self.beta = beta
        self.loss = loss
        self.count = int(count)
        self.gram = gram
        self.rhs = rhs
        self.scale = scale
        self.lam = lam

    def predict(self, F_row: np.ndarray) -> np.ndarray:
        """
        Predict beta'F for one factor row.

        Parameters
        ----------
        F_row : (d,) array
            Out-of-sample factor row.

        Returns
        -------
        array
            Scalar prediction.
        """
        return F_row @ self.beta