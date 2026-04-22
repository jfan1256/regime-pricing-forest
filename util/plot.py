# util/plot.py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from pathlib import Path
from collections.abc import Iterable

from util.system import get_plot


FULL_WIDTH = 6.5
HALF_WIDTH = 3.2

FIG_DPI = 150
SAVE_DPI = 300
SAVE_EXT = ".png"

FONT_SIZE = 9
AXIS_LABEL_SIZE = 9
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 9
TITLE_SIZE = 9

LINE_WIDTH = 1.7
AXIS_LINE_WIDTH = 0.8
TICK_WIDTH = 0.8
GRID_WIDTH = 0.6
BAR_EDGE_WIDTH = 0
MARKER_SIZE = 3.8

COLOR_GRID = "#D8DEE9"
COLOR_SPINE = "#2B2B2B"
COLOR_ZERO = "#6B7280"
COLOR_SIG = "#8B8B8B"

# Fixed shrinkage palette taken directly from the preferred shrinkage-line style.
SHRINKAGE_COLORS = {
    1e-4: "#7A6BA8",  # dark purple
    1e-3: "#6574A8",  # indigo
    1e-2: "#5A7FA6",  # blue
    1e-1: "#6E9E9F",  # blue-green
    1e0: "#86B39A",   # green
}

# Use the same fixed colors everywhere.
MODEL_COLORS = {
    "LR": SHRINKAGE_COLORS[1e-4],
    "RLR": SHRINKAGE_COLORS[1e-2],
    "RPF": SHRINKAGE_COLORS[1e0],
    "Bench: LR": SHRINKAGE_COLORS[1e-4],
    "Bench: RLR": SHRINKAGE_COLORS[1e-2],
    "vs LR": SHRINKAGE_COLORS[1e-4],
    "vs RLR": SHRINKAGE_COLORS[1e-2],
    "1% threshold": COLOR_SIG,
}

LINE_COLORS = [
    SHRINKAGE_COLORS[1e-4],
    SHRINKAGE_COLORS[1e-3],
    SHRINKAGE_COLORS[1e-2],
    SHRINKAGE_COLORS[1e-1],
    SHRINKAGE_COLORS[1e0],
    "#4C6A91",
    "#5A8C7A",
    "#8D7AAF",
]

BAR_COLORS = [
    MODEL_COLORS["LR"],
    MODEL_COLORS["RLR"],
    MODEL_COLORS["RPF"],
    SHRINKAGE_COLORS[1e-3],
    SHRINKAGE_COLORS[1e-1],
]

ALPHA_LINE = 0.98
ALPHA_BAR = 1.00
LEGEND_FRAMEON = False


def set_plot_style() -> None:
    """
    Set the shared plotting style for all figures.
    """
    mpl.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": SAVE_DPI,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.02,
            "figure.figsize": (FULL_WIDTH, 4.0),
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "axes.edgecolor": COLOR_SPINE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "xtick.major.width": TICK_WIDTH,
            "ytick.major.width": TICK_WIDTH,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": LEGEND_SIZE,
            "legend.frameon": LEGEND_FRAMEON,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
        }
    )


def iter_axes(axes) -> Iterable[plt.Axes]:
    """
    Iterate over axes for either a single axis or an array of axes.
    """
    if isinstance(axes, plt.Axes):
        yield axes
        return

    try:
        for ax in axes.flat:
            yield ax
    except AttributeError:
        for ax in axes:
            yield ax


def normalize_ylabel(label: str | None) -> str | None:
    """
    Normalize common y-axis labels.
    """
    if label == "Annualized Sharpe":
        return "Sharpe"
    return label


def apply_axis_style(ax: plt.Axes) -> plt.Axes:
    """
    Apply shared axis formatting.
    """
    ax.spines["left"].set_color(COLOR_SPINE)
    ax.spines["bottom"].set_color(COLOR_SPINE)
    ax.tick_params(axis="both", which="major", length=3.5, width=TICK_WIDTH)
    ax.tick_params(axis="both", which="minor", length=2.0, width=TICK_WIDTH)
    return ax


def add_ygrid(ax: plt.Axes) -> plt.Axes:
    """
    Add a subtle horizontal grid.
    """
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=GRID_WIDTH)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    return ax


def add_zero_line(ax: plt.Axes) -> plt.Axes:
    """
    Add a zero reference line.
    """
    ax.axhline(0.0, color=COLOR_ZERO, linewidth=0.9, zorder=0)
    return ax


def get_bar_colors(n: int) -> list[str]:
    """
    Return n colors from the fixed bar palette.
    """
    return [BAR_COLORS[i % len(BAR_COLORS)] for i in range(n)]


def get_line_color(i: int) -> str:
    """
    Return the i-th line color from the fixed palette.
    """
    return LINE_COLORS[i % len(LINE_COLORS)]


def get_model_color(label: str, fallback_index: int = 0) -> str:
    """
    Return a fixed color for model-level plots.
    """
    return MODEL_COLORS.get(label, get_line_color(fallback_index))


def get_shrinkage_color(z: float) -> str:
    """
    Return the fixed color for a shrinkage value.
    """
    key = float(z)
    if key in SHRINKAGE_COLORS:
        return SHRINKAGE_COLORS[key]
    raise ValueError(f"No fixed shrinkage color configured for z={z}.")


def format_legend(
    ax: plt.Axes,
    loc: str = "best",
    ncol: int = 1,
    outside: bool = False,
) -> None:
    """
    Add a legend if labeled handles exist.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    if outside:
        ax.legend(
            loc="upper left",
            ncol=ncol,
            frameon=LEGEND_FRAMEON,
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
    else:
        ax.legend(loc=loc, ncol=ncol, frameon=LEGEND_FRAMEON)


def rotate_xticklabels(
    ax: plt.Axes,
    rotation: float = 45.0,
    ha: str = "right",
) -> None:
    """
    Rotate x-axis tick labels.
    """
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha(ha)


def make_figure(
    nrows: int = 1,
    ncols: int = 1,
    width: float = FULL_WIDTH,
    height: float = 4.0,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[plt.Figure, plt.Axes | object]:
    """
    Create a styled figure and axes.
    """
    set_plot_style()
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width, height),
        sharex=sharex,
        sharey=sharey,
        constrained_layout=True,
    )

    for ax in iter_axes(axes):
        apply_axis_style(ax)

    return fig, axes


def normalize_plot_path(filename: str | Path) -> Path:
    """
    Normalize a plot filename to the configured output extension.
    """
    path = Path(filename)
    if path.suffix != SAVE_EXT:
        path = path.with_suffix(SAVE_EXT)
    return path


def savefig(
    fig: plt.Figure,
    filename: str | Path,
    dpi: int = SAVE_DPI,
) -> Path:
    """
    Save a figure under get_plot() using the configured raster format.
    """
    plot_dir = get_plot()
    plot_dir.mkdir(parents=True, exist_ok=True)

    path = normalize_plot_path(filename)
    path = plot_dir / path.name

    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def set_categorical_ticks(
    ax: plt.Axes,
    x: np.ndarray,
    labels: list[str],
) -> None:
    """
    Apply fixed categorical ticks.
    """
    ax.set_xticks(x)
    ax.set_xticklabels(labels)


def barplot(
    ax: plt.Axes,
    x: list[str],
    y: list[float],
    colors: list[str] | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    zero: bool = False,
) -> plt.Axes:
    """
    Draw a standardized bar plot.
    """
    if colors is None:
        colors = get_bar_colors(len(x))

    ax.bar(
        x,
        y,
        color=colors,
        alpha=ALPHA_BAR,
        edgecolor=COLOR_SPINE,
        linewidth=BAR_EDGE_WIDTH,
    )

    ylabel = normalize_ylabel(ylabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)

    add_ygrid(ax)
    if zero:
        add_zero_line(ax)
    return ax


def grouped_barplot(
    ax: plt.Axes,
    groups: list[str],
    series: dict[str, list[float]],
    ylabel: str | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    zero: bool = False,
) -> plt.Axes:
    """
    Draw a standardized grouped bar plot.
    """
    labels = list(series.keys())
    values = np.asarray(list(series.values()), dtype=float)
    x = np.arange(len(groups), dtype=float)
    width = 0.8 / len(labels)

    for i, label in enumerate(labels):
        offset = (i - (len(labels) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values[i],
            width=width,
            label=label,
            color=get_model_color(label, i),
            alpha=ALPHA_BAR,
            edgecolor=COLOR_SPINE,
            linewidth=BAR_EDGE_WIDTH,
        )

    ylabel = normalize_ylabel(ylabel)

    set_categorical_ticks(ax, x=x, labels=groups)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)

    add_ygrid(ax)
    if zero:
        add_zero_line(ax)
    return ax


def lineplot(
    ax: plt.Axes,
    x,
    y,
    label: str | None = None,
    color: str | None = None,
    marker: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    zero: bool = False,
) -> plt.Axes:
    """
    Draw a standardized line plot.
    """
    ylabel = normalize_ylabel(ylabel)

    line, = ax.plot(
        x,
        y,
        label=label,
        color=color,
        marker=marker,
        alpha=ALPHA_LINE,
    )
    # line.set_path_effects([
    #     pe.Stroke(linewidth=LINE_WIDTH + BAR_EDGE_WIDTH, foreground=COLOR_SPINE),
    #     pe.Normal(),
    # ])

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)

    add_ygrid(ax)
    if zero:
        add_zero_line(ax)
    return ax