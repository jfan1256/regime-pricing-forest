import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from collections.abc import Iterable

from util.system import get_plot


# -------------------------------------------------
# Global style parameters
# -------------------------------------------------
FIG_WIDTH = 6.5
FIG_HEIGHT = 4.0

FIG_DPI = 150
SAVE_DPI = 300
SAVE_EXT = ".png"

FONT_SIZE = 10
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 9
TITLE_SIZE = 11

LINE_WIDTH = 1.8
AXIS_LINE_WIDTH = 0.8
TICK_WIDTH = 0.8
GRID_WIDTH = 0.6
BAR_EDGE_WIDTH = 0.5

MARKER_SIZE = 4.0

COLOR_GRID = "0.88"
COLOR_SPINE = "0.15"

BAR_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
]

LINE_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#EECA3B",
    "#FF9DA6",
]

ALPHA_LINE = 0.95
ALPHA_BAR = 0.90

LEGEND_FRAMEON = False
TIGHT_LAYOUT_PAD = 0.2


def set_plot_style() -> None:
    """
    Set a compact journal-style plotting theme for all analysis figures.
    """
    mpl.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": SAVE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),
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
    Iterate through axes regardless of subplot layout shape.
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
    ax.axhline(0.0, color="0.45", linewidth=0.9, zorder=0)
    return ax


def get_bar_colors(n: int) -> list[str]:
    """
    Return n bar colors from the fixed palette.
    """
    return [BAR_COLORS[i % len(BAR_COLORS)] for i in range(n)]


def get_line_color(i: int) -> str:
    """
    Return the i-th line color from the fixed palette.
    """
    return LINE_COLORS[i % len(LINE_COLORS)]


def format_legend(
    ax: plt.Axes,
    loc: str = "upper left",
    ncol: int = 1,
    outside: bool = False,
) -> None:
    """
    Add a clean legend if labeled handles exist.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    if outside:
        ax.legend(
            loc=loc,
            ncol=ncol,
            frameon=LEGEND_FRAMEON,
            bbox_to_anchor=(1.02, 1.0),
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
    width: float = FIG_WIDTH,
    height: float = FIG_HEIGHT,
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
    )

    for ax in iter_axes(axes):
        apply_axis_style(ax)

    return fig, axes


def normalize_plot_path(filename: str | Path) -> Path:
    """
    Normalize a plot filename to the configured output extension.
    """
    path = Path(filename)
    if path.suffix == "":
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

    fig.tight_layout(pad=TIGHT_LAYOUT_PAD)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


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
    ax.plot(x, y, label=label, color=color, marker=marker, alpha=ALPHA_LINE)

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