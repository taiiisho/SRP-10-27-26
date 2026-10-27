"""Shared plotting style: colour-blind-safe colours and headless matplotlib.

Okabe & Ito's palette stays distinguishable for the common forms of colour
blindness. Every plot is also saved as a CSV of the plotted numbers, so a
figure can always be checked or re-drawn.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed (works over SSH)

import matplotlib.pyplot as plt  # noqa: E402

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

# Fixed colour per model so every figure uses the same colours.
MODEL_COLOURS = {
    "minimum_jerk_euler": OKABE_ITO["blue"],
    "minimum_jerk_closed": OKABE_ITO["sky_blue"],
    "minimum_jerk_replan": OKABE_ITO["purple"],
    "minimum_jerk_free_target": OKABE_ITO["yellow"],
    "constant_velocity": OKABE_ITO["green"],
    "euler_constant_acceleration": OKABE_ITO["orange"],
    "zero_motion": "#7F7F7F",
    "gru": OKABE_ITO["vermillion"],
    "mlp": OKABE_ITO["black"],
    "gp": "#8C564B",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def model_colour(name: str, index: int = 0) -> str:
    base = name.split("@")[0]
    if base in MODEL_COLOURS:
        return MODEL_COLOURS[base]
    palette = list(OKABE_ITO.values())
    return palette[index % len(palette)]


def new_figure(width: float = 7.0, height: float = 4.5, **kwargs):
    fig, ax = plt.subplots(figsize=(width, height), **kwargs)
    return fig, ax


def save_figure(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
