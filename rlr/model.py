import numpy as np
import pandas as pd

from joblib import Parallel, delayed


class RLR:
    def __init__(self, config: dict):
        self.config = config
        self.dtype = config["dtype"]

        self.z = float(config["z"])
        self.num_tree = int(config["num_tree"])
        self.sample_frac = float(config["sample_frac"])
        self.random_state = int(config["random_state"])
        self.n_jobs = int(config.get("n_jobs", 1))

        self.betas: list[np.ndarray] = []

    def sample_rows(self, n_obs: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample in-sample rows for one ensemble member without replacement.
        """
        sample_size = max(1, int(self.sample_frac * n_obs))
        sample_size = min(sample_size, n_obs)
        return np.sort(rng.choice(n_obs, size=sample_size, replace=False))

    def make_model_plan(self, n_obs: int) -> tuple[list[np.ndarray], list[int]]:
        """
        Precompute deterministic sampled row indices and seeds for all models.
        """
        seed_seq = np.random.SeedSequence(self.random_state)
        child_seeds = seed_seq.spawn(self.num_tree)

        sample_indices: list[np.ndarray] = []
        seeds: list[int] = []

        for child_seed in child_seeds:
            seed = int(child_seed.generate_state(1, dtype=np.uint32)[0])
            rng = np.random.default_rng(seed)
            sample_indices.append(self.sample_rows(n_obs, rng))
            seeds.append(seed)

        return sample_indices, seeds

    @staticmethod
    def fit_one_model(
        config: dict,
        F: np.ndarray,
        sample_index: np.ndarray,
    ) -> np.ndarray:
        """
        Fit one ridge model on a sampled in-sample subset.

        Parameters
        ----------
        config : dict
            RLR configuration with scalar z and dtype.
        F : (T, d) array
            Full in-sample factor matrix.
        sample_index : (Ts,) array
            Selected row indices for this ensemble member.

        Returns
        -------
        beta : (d,) array
            Ridge coefficients for one sampled fit.
        """
        dtype = config["dtype"]
        z = float(config["z"])

        F_sub = F[sample_index]
        T_sub, d = F_sub.shape
        y_sub = np.ones((T_sub,), dtype=dtype)

        if d <= T_sub:
            gram = F_sub.T @ F_sub
            scale = np.trace(gram) / gram.shape[0]
            lam = np.asarray(z, dtype=dtype) * scale
            eye = np.eye(d, dtype=dtype)
            rhs = F_sub.T @ y_sub
            beta = np.linalg.solve(gram + lam * eye, rhs)
            return beta

        gram_dual = F_sub @ F_sub.T
        scale = np.trace(gram_dual) / gram_dual.shape[0]
        lam = np.asarray(z, dtype=dtype) * scale
        eye = np.eye(T_sub, dtype=dtype)

        alpha = np.linalg.solve(gram_dual + lam * eye, y_sub)
        beta = F_sub.T @ alpha
        return beta

    def fit(self, F: np.ndarray) -> None:
        """
        Fit the random linear regression ensemble on one in-sample window.
        """
        sample_indices, seeds = self.make_model_plan(F.shape[0])

        self.betas = list(
            Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(self.fit_one_model)(
                    self.config,
                    F,
                    sample_index,
                )
                for sample_index, seed in zip(sample_indices, seeds)
            )
        )

    def predict(self, F_row: np.ndarray, date) -> pd.DataFrame:
        """
        Predict one out-of-sample SDF value by averaging ensemble-member predictions.
        """
        preds = np.asarray(
            [F_row @ beta for beta in self.betas],
            dtype=self.dtype,
        )
        sdf_value = preds.mean().item()
        idx = pd.Index([date], name="date")
        return pd.DataFrame({"sdf": [sdf_value]}, index=idx)