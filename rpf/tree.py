import numpy as np

from dataclasses import dataclass

from rpf.leaf import Leaf

@dataclass
class TreeNode:
    index: np.ndarray
    depth: int
    node_id: int

    count: int | None = None
    gram: np.ndarray | None = None
    rhs: np.ndarray | None = None

    beta: np.ndarray | None = None
    loss: np.ndarray | None = None

    split_feature: int | None = None
    split_value: float | None = None
    split_gain: float | None = None

    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

class Tree:
    def __init__(self, config: dict, rng: np.random.Generator):
        self.config = config
        self.dtype = config["dtype"]

        self.max_depth = int(config["max_depth"])
        self.min_leaf_size = int(config["min_leaf_size"])
        self.bin = int(config["bin"])
        self.max_features = int(config["max_features"])

        self.rng = rng
        self.root = None
        self.next_id = 0
        self.eye = None

    def next_node_id(self) -> int:
        """
        Allocate a deterministic node id within one tree.
        """
        node_id = self.next_id
        self.next_id += 1
        return node_id

    def fit(
        self,
        F: np.ndarray,
        FF: np.ndarray,
        M: np.ndarray,
        sample_index: np.ndarray,
    ) -> None:
        """
        Fit the randomized tree on one in-sample window.
        """
        self.next_id = 0
        self.eye = np.eye(F.shape[1], dtype=self.dtype)
        self.root = self.build_node(F, FF, M, sample_index, depth=0)

    def build_node(
        self,
        F: np.ndarray,
        FF: np.ndarray,
        M: np.ndarray,
        index: np.ndarray,
        depth: int,
    ) -> TreeNode:
        """
        Recursively build one node.
        """
        count, gram, rhs = self.compute_stats(F, FF, index)

        node = TreeNode(
            index=index,
            depth=depth,
            node_id=self.next_node_id(),
            count=count,
            gram=gram,
            rhs=rhs,
        )

        leaf = Leaf(self.config, eye=self.eye)
        leaf.fit(count=count, gram=gram, rhs=rhs)

        node.beta = leaf.beta
        node.loss = leaf.loss

        can_split = depth < self.max_depth and count >= 2 * self.min_leaf_size
        if not can_split:
            return node

        split = self.find_best_split(F, FF, M, node)
        if split is None:
            return node

        node.split_feature = split["feature"]
        node.split_value = split["threshold"]
        node.split_gain = split["gain"]
        node.left = self.build_node(F, FF, M, split["left_index"], depth + 1)
        node.right = self.build_node(F, FF, M, split["right_index"], depth + 1)
        return node

    def compute_stats(
        self,
        F: np.ndarray,
        FF: np.ndarray,
        index: np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Compute sufficient statistics for one node.
        """
        F_node = F[index]
        FF_node = FF[index]

        count = int(F_node.shape[0])
        gram = FF_node.sum(axis=0)
        rhs = F_node.sum(axis=0)
        return count, gram, rhs

    def fit_leaf(
        self,
        count: int,
        gram: np.ndarray,
        rhs: np.ndarray,
    ) -> Leaf:
        """
        Fit a leaf directly from sufficient statistics.
        """
        leaf = Leaf(self.config, eye=self.eye)
        leaf.fit(count=count, gram=gram, rhs=rhs)
        return leaf

    def make_bins(self, values: np.ndarray) -> np.ndarray | None:
        """
        Build sorted bin edges from node values using empirical quantiles.
        """
        unique_values = np.unique(values)

        if unique_values.size < 2:
            return None

        n_edge = min(self.bin, unique_values.size) - 1
        if n_edge <= 0:
            return None

        q = np.linspace(0.0, 1.0, num=n_edge + 2, dtype=self.dtype)[1:-1]
        edges = np.unique(np.quantile(values, q))

        if edges.size == 0:
            return None

        return edges

    def assign_bins(self, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Assign values to bins defined by interior edges.
        """
        return np.searchsorted(edges, values, side="right")

    def sample_split_features(self, n_macro: int) -> np.ndarray:
        """
        Sample the candidate macro features for one node split.
        """
        n_feature = min(self.max_features, n_macro)
        return self.rng.choice(n_macro, size=n_feature, replace=False)

    def find_best_split(
        self,
        F: np.ndarray,
        FF: np.ndarray,
        M: np.ndarray,
        node: TreeNode,
    ) -> dict | None:
        """
        Find the best split for one node using binned sufficient statistics
        and random feature subsampling.
        """
        index = node.index
        count_parent = node.count
        gram_parent = node.gram
        rhs_parent = node.rhs
        loss_parent = node.loss

        F_node = F[index]
        FF_node = FF[index]

        d = F_node.shape[1]
        n_macro = M.shape[1]

        best_gain = None
        best_feature = None
        best_threshold = None

        candidate_features = self.sample_split_features(n_macro)
        ones = np.ones((F_node.shape[0],), dtype=self.dtype)

        for feature in candidate_features:
            values = M[index, feature]
            edges = self.make_bins(values)
            if edges is None:
                continue

            bin_id = self.assign_bins(values, edges)
            n_bin = int(bin_id.max().item()) + 1

            count_bin = np.zeros((n_bin,), dtype=self.dtype)
            rhs_bin = np.zeros((n_bin, d), dtype=self.dtype)
            gram_bin = np.zeros((n_bin, d, d), dtype=self.dtype)

            np.add.at(count_bin, bin_id, ones)
            np.add.at(rhs_bin, bin_id, F_node)
            np.add.at(gram_bin, bin_id, FF_node)

            count_left = 0
            rhs_left = np.zeros((d,), dtype=self.dtype)
            gram_left = np.zeros((d, d), dtype=self.dtype)

            for b in range(n_bin - 1):
                count_left += int(count_bin[b].item())
                rhs_left = rhs_left + rhs_bin[b]
                gram_left = gram_left + gram_bin[b]

                count_right = count_parent - count_left
                if count_left < self.min_leaf_size or count_right < self.min_leaf_size:
                    continue

                rhs_right = rhs_parent - rhs_left
                gram_right = gram_parent - gram_left

                # Skip degenerate children with no factor variation / no information.
                if np.trace(gram_left) <= 1.0e-12 or np.trace(gram_right) <= 1.0e-12:
                    continue

                left_leaf = self.fit_leaf(count=count_left, gram=gram_left, rhs=rhs_left)
                right_leaf = self.fit_leaf(count=count_right, gram=gram_right, rhs=rhs_right)

                gain = loss_parent - left_leaf.loss - right_leaf.loss

                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(edges[b].item())

        if best_gain is None or best_feature is None or best_threshold is None:
            return None

        if best_gain <= 0:
            return None

        values = M[index, best_feature]
        left_mask = values < best_threshold
        right_mask = ~left_mask

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left_index": index[left_mask].copy(),
            "right_index": index[right_mask].copy(),
            "gain": float(best_gain),
        }

    def predict(self, F_row: np.ndarray, M_row: np.ndarray) -> np.ndarray:
        """
        Route one state row to its terminal leaf and return beta(M_t)'F_{t+1}.
        """
        node = self.root

        while not node.is_leaf:
            if M_row[node.split_feature] < node.split_value:
                node = node.left
            else:
                node = node.right

        return F_row @ node.beta