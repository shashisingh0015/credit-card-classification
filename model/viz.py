"""Shared visual tokens for the EDA notebook and the Streamlit app.

Keeping these in one module means the notebook figures and the app's confusion
matrix look like one system instead of two, and the palette is defined once.

Colour choices follow the job each encoding does:

* **Categorical** (class identity: "no default" vs "default") -> two fixed hues,
  assigned in a fixed order and never cycled. Blue/orange was validated for
  colour-vision deficiency: worst pair separation is ΔE 24.7 under protanopia and
  33.6 for normal vision, comfortably above the ΔE 8 / 15 floors.
* **Sequential** (magnitude, all one sign: collinearity, confusion-matrix counts)
  -> a single blue hue ramped light to dark.
* **Diverging** (polarity: correlations that can be negative or positive) -> blue
  and red poles around a *neutral grey* midpoint, so "no correlation" reads as
  nothing rather than as a colour. Never a rainbow colormap: on a rainbow the
  midpoint becomes a vivid hue and implies significance where there is none.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------
# Chrome & ink. Marks carry colour; text stays in ink tokens.
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# --------------------------------------------------------------------------
# Categorical: slot order is the CVD-safety mechanism, so do not reorder.
# --------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# Semantic aliases for the two target classes.
NO_DEFAULT = SERIES[0]
DEFAULT = SERIES[1]
CLASS_COLORS = {0: NO_DEFAULT, 1: DEFAULT}
CLASS_LABELS = {0: "No default", 1: "Default"}

# --------------------------------------------------------------------------
# Ramps
# --------------------------------------------------------------------------
_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", _BLUE_RAMP)

# Equal step count per arm, neutral grey at the centre.
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#2a78d6", "#86b6ef", "#f0efec",
                     "#f0a2a1", "#e34948", "#a52322"]
)


def apply_style() -> None:
    """Set matplotlib rcParams: recessive chrome, thin marks, no chart junk."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Grid is a hairline reference, not a feature: it sits behind the marks.
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.bottom": False,
        "ytick.left": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.titlesize": 13,
    })


def grid_axis(ax, axis: str = "y") -> None:
    """Restrict the grid to one axis -- the one the reader measures against."""
    ax.grid(False)
    ax.grid(True, axis=axis, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
