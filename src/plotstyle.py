"""
plotstyle.py — one look for every second-half figure.


Colours come from a colourblind-validated palette: non-disease is always
blue, disease is always orange, in every figure, so the slides read as one
system. Text is never drawn in a series colour.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures" / "disease"

NON_DISEASE = "#2a78d6"   # palette slot 1
DISEASE = "#eb6834"       # palette slot 2
THIRD = "#1baf7a"         # palette slot 3, for a third series only
CLASS_COLOURS = {0: NON_DISEASE, 1: DISEASE}
CLASS_NAMES = {0: "non-disease", 1: "disease"}

TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#a8a7a2"       # reference lines, chance level, "not significant"


def setup():
    """Apply the shared matplotlib style. Returns pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
    })
    return plt


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path
