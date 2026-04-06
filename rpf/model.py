# rmst/model.py
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from rpf.tree import Tree
from rpf.export import export_leaves, export_regimes, export_splits


class RPF:
    def __init__(self, config: dict):
        self.config = config
        self.dtype = config["dtype"]

        self.num_tree = int(config["num_tree"])
        self.sample_frac = float(config["sample_frac"])
        self.random_state = int(config["random_state"])
        self.n_jobs = int(config["n_jobs"])
        self.parallel = str(config["parallel"])

        self.trees: list[Tree] = []

    def sample_rows(self, n_obs: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample in-sample rows for one tree without replacement.
        """
        min_size = 2 * int(self.config["min_leaf_size"])
        sample_size = max(min_size, int(self.sample_frac * n_obs))
        sample_size = min(sample_size, n_obs)
        return np.sort(rng.choice(n_obs, size=sample_size, replace=False))

    def make_tree_plan(self, n_obs: int) -> tuple[list[np.ndarray], list[int]]:
        """
        Precompute deterministic sampled row indices and seeds for all trees.
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
    def fit_one_tree(
        config: dict,
        F: np.ndarray,
        FF: np.ndarray,
        M: np.ndarray,
        sample_index: np.ndarray,
        seed: int,
    ) -> Tree:
        """
        Fit one randomized tree with a deterministic seed.
        """
        rng = np.random.default_rng(seed)
        tree = Tree(config, rng=rng)
        tree.fit(F, FF, M, sample_index)
        return tree

    def fit(
        self,
        F: np.ndarray,
        FF: np.ndarray,
        M: np.ndarray,
        sample_indices: list[np.ndarray] | None = None,
        seeds: list[int] | None = None,
    ) -> None:
        """
        Fit RMST on one in-sample window.
        """
        if sample_indices is None or seeds is None:
            sample_indices, seeds = self.make_tree_plan(F.shape[0])

        self.trees = list(
            Parallel(n_jobs=self.n_jobs, prefer=self.parallel)(
                delayed(self.fit_one_tree)(
                    self.config,
                    F,
                    FF,
                    M,
                    sample_index,
                    seed,
                )
                for sample_index, seed in zip(sample_indices, seeds)
            )
        )

    def predict(self, F_row: np.ndarray, M_row: np.ndarray, date) -> pd.DataFrame:
        """
        Predict one out-of-sample SDF value by averaging tree predictions.
        """
        preds = np.asarray(
            [tree.predict(F_row, M_row) for tree in self.trees],
            dtype=self.dtype,
        )
        sdf_value = preds.mean().item()
        idx = pd.Index([date], name="date")
        return pd.DataFrame({"sdf": [sdf_value]}, index=idx)

    def export_splits(self, macro_columns: list[str]) -> pd.DataFrame:
        """
        Export all internal-node split summaries across trees.
        """
        return export_splits(self.trees, macro_columns)

    def export_leaves(
        self,
        macro_columns: list[str],
        factor_columns: list[str],
    ) -> pd.DataFrame:
        """
        Export all terminal leaves across trees.
        """
        return export_leaves(self.trees, macro_columns, factor_columns)

    def export_regimes(
        self,
        dates: pd.DatetimeIndex,
        F: np.ndarray,
        M: np.ndarray,
        factor_columns: list[str],
        macro_columns: list[str],
    ) -> pd.DataFrame:
        """
        Export test-period regime routing across trees.
        """
        return export_regimes(self.trees, dates, F, M, factor_columns, macro_columns)