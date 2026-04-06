import numpy as np
import pandas as pd

from rpf.tree import Tree, TreeNode

# -------------------------------------------------
# Splits
# -------------------------------------------------
def render_rule(rule_parts: list[str]) -> str:
    """
    Render one readable rule string from a list of branch conditions.
    """
    if not rule_parts:
        return "root"
    return " and ".join(rule_parts)

def append_split_rows(
    node: TreeNode,
    tree_id: int,
    macro_columns: list[str],
    rows: list[dict],
) -> None:
    """
    Append one tree's internal split nodes to rows using depth-first traversal.
    """
    if node is None or node.is_leaf:
        return

    loss_left = float(node.left.loss) if node.left is not None else np.nan
    loss_right = float(node.right.loss) if node.right is not None else np.nan

    rows.append(
        {
            "tree_id": tree_id,
            "node_id": node.node_id,
            "depth": node.depth,
            "count": int(node.count),
            "split_feature": int(node.split_feature),
            "split_variable": macro_columns[int(node.split_feature)],
            "split_value": float(node.split_value),
            "loss_parent": float(node.loss),
            "loss_left": loss_left,
            "loss_right": loss_right,
            "gain": float(node.split_gain),
            "is_root": node.depth == 0,
        }
    )

    append_split_rows(node.left, tree_id, macro_columns, rows)
    append_split_rows(node.right, tree_id, macro_columns, rows)

def export_splits(trees: list[Tree], macro_columns: list[str]) -> pd.DataFrame:
    """
    Build one DataFrame with all internal split nodes across trees.
    """
    rows: list[dict] = []

    for tree_id, tree in enumerate(trees):
        append_split_rows(tree.root, tree_id, macro_columns, rows)

    return pd.DataFrame(rows).reindex(
        columns=[
            "tree_id",
            "node_id",
            "depth",
            "count",
            "split_feature",
            "split_variable",
            "split_value",
            "loss_parent",
            "loss_left",
            "loss_right",
            "gain",
            "is_root",
        ]
    )

# -------------------------------------------------
# Leaves
# -------------------------------------------------
def append_leaf_rows(
    node: TreeNode,
    tree_id: int,
    factor_columns: list[str],
    path_bits: list[str],
    rule_parts: list[str],
    macro_columns: list[str],
    rows: list[dict],
) -> None:
    """
    Append one tree's terminal leaves to rows using depth-first traversal.
    """
    if node is None:
        return

    if node.is_leaf:
        row = {
            "tree_id": tree_id,
            "leaf_id": node.node_id,
            "leaf_path": "".join(path_bits) if path_bits else "root",
            "depth": node.depth,
            "count": int(node.count),
            "rule": render_rule(rule_parts),
        }
        for j, name in enumerate(factor_columns):
            row[f"beta_{name}"] = float(node.beta[j])
        rows.append(row)
        return

    feature = int(node.split_feature)
    name = macro_columns[feature]
    threshold = float(node.split_value)

    append_leaf_rows(
        node.left,
        tree_id,
        factor_columns,
        path_bits + ["L"],
        rule_parts + [f"{name} < {threshold:.6g}"],
        macro_columns,
        rows,
    )
    append_leaf_rows(
        node.right,
        tree_id,
        factor_columns,
        path_bits + ["R"],
        rule_parts + [f"{name} >= {threshold:.6g}"],
        macro_columns,
        rows,
    )

def export_leaves(
    trees: list[Tree],
    macro_columns: list[str],
    factor_columns: list[str],
) -> pd.DataFrame:
    """
    Build one DataFrame with all terminal leaves across trees.
    """
    rows: list[dict] = []

    for tree_id, tree in enumerate(trees):
        append_leaf_rows(
            tree.root,
            tree_id,
            factor_columns,
            path_bits=[],
            rule_parts=[],
            macro_columns=macro_columns,
            rows=rows,
        )

    columns = ["tree_id", "leaf_id", "leaf_path", "depth", "count", "rule"]
    columns.extend([f"beta_{name}" for name in factor_columns])

    return pd.DataFrame(rows).reindex(columns=columns)

# -------------------------------------------------
# Regimes
# -------------------------------------------------
def append_regime_rows(
    tree: Tree,
    tree_id: int,
    date,
    F_row: np.ndarray,
    M_row: np.ndarray,
    factor_columns: list[str],
    macro_columns: list[str],
    rows: list[dict],
) -> None:
    """
    Route one out-of-sample row through one tree and append the active regime row.
    """
    node = tree.root
    path_bits: list[str] = []
    rule_parts: list[str] = []

    while not node.is_leaf:
        feature = int(node.split_feature)
        threshold = float(node.split_value)
        name = macro_columns[feature]

        if M_row[feature] < threshold:
            path_bits.append("L")
            rule_parts.append(f"{name} < {threshold:.6g}")
            node = node.left
        else:
            path_bits.append("R")
            rule_parts.append(f"{name} >= {threshold:.6g}")
            node = node.right

    row = {
        "date": date,
        "tree_id": tree_id,
        "leaf_id": node.node_id,
        "leaf_path": "".join(path_bits) if path_bits else "root",
        "rule": render_rule(rule_parts),
        "prediction": float(F_row @ node.beta),
    }

    for j, name in enumerate(macro_columns):
        row[f"macro_{name}"] = float(M_row[j])

    for j, name in enumerate(factor_columns):
        row[f"factor_{name}"] = float(F_row[j])
        row[f"beta_{name}"] = float(node.beta[j])

    rows.append(row)

def export_regimes(
    trees: list[Tree],
    dates: pd.DatetimeIndex,
    F: np.ndarray,
    M: np.ndarray,
    factor_columns: list[str],
    macro_columns: list[str],
) -> pd.DataFrame:
    """
    Build one DataFrame with test-period regime assignments across trees.
    """
    rows: list[dict] = []

    for i, date in enumerate(dates):
        F_row = F[i]
        M_row = M[i]

        for tree_id, tree in enumerate(trees):
            append_regime_rows(
                tree=tree,
                tree_id=tree_id,
                date=date,
                F_row=F_row,
                M_row=M_row,
                factor_columns=factor_columns,
                macro_columns=macro_columns,
                rows=rows,
            )

    columns = ["date", "tree_id", "leaf_id", "leaf_path", "rule", "prediction"]
    columns.extend([f"macro_{name}" for name in macro_columns])
    columns.extend([f"factor_{name}" for name in factor_columns])
    columns.extend([f"beta_{name}" for name in factor_columns])

    return pd.DataFrame(rows).reindex(columns=columns)