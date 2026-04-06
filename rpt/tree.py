import numpy as np

from dataclasses import dataclass

from rpt.leaf import Leaf

@dataclass
class TreeNode:
    index: np.ndarray
    depth: int

    count: int | None = None
    gram: np.ndarray | None = None
    rhs: np.ndarray | None = None

    beta: np.ndarray | None = None
    loss: np.ndarray | None = None

    split_feature: int | None = None
    split_value: float | None = None

    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class Tree:
    def __init__(self, config: dict):
        self.config = config
        self.dtype = config["dtype"]

        self.max_depth = int(config["max_depth"])
        self.min_leaf_size = int(config["min_leaf_size"])
        self.bin = int(config["bin"])

        self.root = None

    def fit(self, F: np.ndarray, FF: np.ndarray, M: np.ndarray) -> None:
        """
        Fit the tree on one in-sample window.

        Parameters
        ----------
        F : (T, d) array
            In-sample factor matrix.
        FF : (T, d, d) array
            Row-level outer products, where FF[t] = outer(F[t], F[t]).
        M : (T, m) array
            In-sample macro-state matrix aligned to F.
        """
        root_index = np.arange(F.shape[0], dtype=np.int64)
        self.root = self.build_node(F, FF, M, root_index, depth=0)

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
            count=count,
            gram=gram,
            rhs=rhs,
        )

        leaf = Leaf(self.config)
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
        leaf = Leaf(self.config)
        leaf.fit(count=count, gram=gram, rhs=rhs)
        return leaf

    def make_bins(self, values: np.ndarray) -> np.ndarray | None:
        """
        Build sorted bin edges from node values using empirical quantiles.

        Parameters
        ----------
        values : (n_node,) array
            One macro feature restricted to the current node.

        Returns
        -------
        array or None
            Sorted interior bin edges. If fewer than 2 distinct values exist,
            returns None.
        """
        unique_values = np.unique(values)
        unique_values = np.sort(unique_values)

        if unique_values.size < 2:
            return None

        n_edge = min(self.bin, unique_values.size) - 1
        if n_edge <= 0:
            return None

        q = np.linspace(0.0, 1.0, num=n_edge + 2, dtype=self.dtype)[1:-1]
        edges = np.quantile(values, q)
        edges = np.unique(edges)
        edges = np.sort(edges)

        if edges.size == 0:
            return None

        return edges

    def assign_bins(self, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Assign values to bins defined by interior edges.

        Parameters
        ----------
        values : (n_node,) array
            Feature values.
        edges : (n_edge,) array
            Sorted interior edges.

        Returns
        -------
        array
            Integer bin ids in {0, ..., n_edge}.
        """
        return np.searchsorted(edges, values, side="right")

    def find_best_split(
        self,
        F: np.ndarray,
        FF: np.ndarray,
        M: np.ndarray,
        node: TreeNode,
    ) -> dict | None:
        """
        Find the best split for one node using binned sufficient statistics.
        """
        index = node.index
        count_parent = node.count
        gram_parent = node.gram
        rhs_parent = node.rhs
        loss_parent = node.loss

        F_node = F[index]
        FF_node = FF[index]
        M_node = M[index]

        d = F_node.shape[1]
        n_macro = M_node.shape[1]

        best_gain = None
        best_split = None

        for feature in range(n_macro):
            values = M_node[:, feature]
            edges = self.make_bins(values)
            if edges is None:
                continue

            bin_id = self.assign_bins(values, edges)
            n_bin = int(bin_id.max().item()) + 1

            count_bin = np.zeros((n_bin,), dtype=self.dtype)
            rhs_bin = np.zeros((n_bin, d), dtype=self.dtype)
            gram_bin = np.zeros((n_bin, d, d), dtype=self.dtype)

            ones = np.ones((F_node.shape[0],), dtype=self.dtype)

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

                left_leaf = self.fit_leaf(count=count_left, gram=gram_left, rhs=rhs_left)
                right_leaf = self.fit_leaf(count=count_right, gram=gram_right, rhs=rhs_right)

                gain = loss_parent - left_leaf.loss - right_leaf.loss

                if best_gain is None or gain > best_gain:
                    threshold = float(edges[b].item())
                    left_mask = values < threshold
                    right_mask = ~left_mask

                    best_gain = gain
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "left_index": index[left_mask].copy(),
                        "right_index": index[right_mask].copy(),
                    }

        if best_split is None:
            return None

        if best_gain is not None and best_gain <= 0:
            return None

        return best_split

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