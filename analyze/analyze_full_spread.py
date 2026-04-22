# analyze_full_spread.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from util.load import load_macro, load_char
from util.plot import make_figure, savefig
from util.system import get_result_rpf
from util.interpret import (
    flatten_single_run,
    top_macro_from_gain,
    spread_columns_to_matrix,
    prefixed_columns,
    theme_label,
    macro_label,
)


def make_rpf_diverging_cmap() -> LinearSegmentedColormap:
    """
    Build a higher-contrast diverging colormap in the same family as the paper palette.

    Negative values lean more strongly purple, positive values lean more strongly
    green, and the center remains white.
    """
    return LinearSegmentedColormap.from_list(
        "rpf_diverging",
        [
            "#5B4B8A",  # deeper purple
            "#7A6BA8",  # purple
            "#9B8CC2",  # light purple
            "#FFFFFF",  # center
            "#A7CBB8",  # light green
            "#86B39A",  # green
            "#5F9478",  # deeper green
        ],
        N=256,
    )


def annotate_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
) -> None:
    """
    Add numeric annotations to a heatmap.
    """
    finite = matrix[np.isfinite(matrix)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 0.0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue

            color = "white" if max_abs > 0.0 and abs(value) >= 0.55 * max_abs else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )


def top_themes_from_spread(
    spread_mat,
    top_k: int,
) -> list[str]:
    """
    Rank themes by average absolute beta spread across all macros.
    """
    score = spread_mat.abs().mean(axis=0).sort_values(ascending=False)
    return score.index[:top_k].tolist()


if __name__ == "__main__":
    top_k_macro = 5
    top_k_theme = 5

    macro = load_macro("rpf_full_interpret.yml", get_result_rpf())
    char = load_char("rpf_full_interpret.yml", get_result_rpf())

    macro = flatten_single_run(macro)
    char = flatten_single_run(char)

    if macro.empty:
        raise ValueError("No full-model macro interpretation output found.")
    if char.empty:
        raise ValueError("No full-model characteristic interpretation output found.")

    leaders = top_macro_from_gain(macro, top_k=top_k_macro)
    spread_cols = prefixed_columns(char, "beta_spread_")

    spread_mat_full = spread_columns_to_matrix(char, spread_cols, agg="mean")
    if spread_mat_full.empty:
        raise ValueError("No full-model beta spread columns found.")

    theme_leaders = top_themes_from_spread(spread_mat_full, top_k=top_k_theme)

    spread_mat = spread_mat_full.loc[
        [name for name in leaders if name in spread_mat_full.index],
        [name for name in theme_leaders if name in spread_mat_full.columns],
    ]

    if spread_mat.empty:
        raise ValueError("No theme columns remain after top_k filtering.")

    x_labels = [theme_label(name) for name in spread_mat.columns]
    y_labels = [macro_label(name) for name in spread_mat.index]
    values = spread_mat.to_numpy(dtype=float)

    finite = values[np.isfinite(values)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax == 0.0:
        vmax = 1.0

    fig, ax = make_figure(width=6.8, height=3.5)
    im = ax.imshow(
        values,
        cmap=make_rpf_diverging_cmap(),
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("")
    ax.set_ylabel("")

    annotate_heatmap(ax, values)

    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Average Beta Spread (High − Low)")

    savefig(fig, "full_beta_spread")