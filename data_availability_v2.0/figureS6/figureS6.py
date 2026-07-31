#!/usr/bin/env python3
"""Plot the TC-intensity validation figure from one NetCDF file.

Input
-----
figureS6_data.nc

The NetCDF file already contains the storm-level intensity samples, the
30-year CTL and GGW annual means, and the statistical comparison shown in the
figure. This script performs no raw-track extraction and writes no intermediate
data files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy import stats


# -----------------------------------------------------------------------------
# Files
# -----------------------------------------------------------------------------
WORK = Path(__file__).resolve().parent
INPUT_NC = WORK / "figureS6_data.nc"
OUTPUT_DIR = WORK
OUTPUT_STEM = "figureS6"


# -----------------------------------------------------------------------------
# Figure settings
# -----------------------------------------------------------------------------
FONT_SIZE = 16
BORDER_WIDTH = 0.75
COLORS = {
    "IBTrACS": "#2b2b2b",
    "CTL": "#2166ac",
    "GGW": "#d73027",
}

WIND_THRESHOLD_KT = 34.0
KT_TO_MS = 0.514444

ECDF_X_MIN = 0.0
ECDF_X_MAX = 90.0
ECDF_X_MAJOR_STEP = 10.0
ECDF_X_MINOR_DIVISIONS = 2
ECDF_Y_MIN = 0.0
ECDF_Y_MAX = 1.0
ECDF_Y_MAJOR_STEP = 0.2

PAIR_Y_MIN = 6.0
PAIR_Y_MAX = 12.0
PAIR_Y_MAJOR_STEP = 1.0
PAIR_Y_MINOR_DIVISIONS = 2

REQUIRED_VARIABLES = {
    "storms__source",
    "storms__lifetime_max_wind_ms",
    "annual__ctl_annual_mean_ms",
    "annual__ggw_annual_mean_ms",
    "comparisons__comparison",
    "comparisons__test",
    "comparisons__difference_ms",
    "comparisons__p_value",
}


def setup_style() -> None:
    """Use Times New Roman and bold only the subplot titles."""
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.weight": "normal",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE-4,
            "axes.labelweight": "normal",
            "axes.titlesize": FONT_SIZE-4,
            "axes.titleweight": "normal",
            "xtick.labelsize": FONT_SIZE-4,
            "ytick.labelsize": FONT_SIZE-4,
            "legend.fontsize": FONT_SIZE-4,
            "axes.linewidth": BORDER_WIDTH,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_figure_data() -> xr.Dataset:
    """Load the only input file and verify the required variables."""
    if not INPUT_NC.is_file():
        raise FileNotFoundError(f"Missing figure input: {INPUT_NC}")

    with xr.open_dataset(INPUT_NC) as source:
        dataset = source.load()

    missing = sorted(REQUIRED_VARIABLES.difference(dataset.data_vars))
    if missing:
        raise KeyError(f"Missing required variables in {INPUT_NC.name}: {missing}")
    return dataset


def style_axis(ax) -> None:
    """Apply the shared grid, ticks, and 0.75-pt black border."""
    ax.grid(
        which="major",
        linestyle="--",
        linewidth=0.5,
        color="0.72",
        zorder=0,
    )
    ax.grid(which="minor", visible=False)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.0,
        width=BORDER_WIDTH,
        top=False,
        right=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=0,
        width=0,
        top=False,
        right=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(BORDER_WIDTH)


def add_panel_title(ax, title: str) -> None:
    """Place the only bold figure text above a subplot."""
    ax.text(
        0.0,
        1.035,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE,
        fontweight="bold",
        clip_on=False,
    )


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted finite values and their empirical cumulative frequency."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot calculate an ECDF from an empty sample")
    x_values = np.sort(finite)
    probabilities = np.arange(1, finite.size + 1, dtype=float) / finite.size
    return x_values, probabilities


def mean_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided Student-t interval for one annual-mean series."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        raise ValueError("At least two finite annual means are required")
    mean = float(np.mean(finite))
    standard_error = float(stats.sem(finite))
    critical_value = float(
        stats.t.ppf((1.0 + confidence) / 2.0, finite.size - 1)
    )
    return (
        mean - critical_value * standard_error,
        mean + critical_value * standard_error,
    )


def storm_intensity_by_source(
    dataset: xr.Dataset,
    source_name: str,
) -> np.ndarray:
    """Read one source's storm-level lifetime-maximum intensity samples."""
    sources = np.asarray(dataset["storms__source"].values).astype(str)
    intensity = np.asarray(
        dataset["storms__lifetime_max_wind_ms"].values,
        dtype=float,
    )
    selected = intensity[sources == source_name]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        raise ValueError(f"No finite storm intensities found for {source_name}")
    return selected


def paired_t_comparison(dataset: xr.Dataset) -> tuple[float, float]:
    """Read the saved GGW-minus-CTL paired-t result from the NetCDF file."""
    comparisons = np.asarray(dataset["comparisons__comparison"].values).astype(str)
    tests = np.asarray(dataset["comparisons__test"].values).astype(str)
    mask = (
        (comparisons == "GGW_minus_CTL_paired_annual_means")
        & (np.char.find(tests, "paired t-test") >= 0)
        & (np.char.find(tests, "sign-flip") < 0)
    )
    indices = np.flatnonzero(mask)
    if indices.size != 1:
        raise ValueError(
            "Expected exactly one saved GGW-minus-CTL paired t-test result"
        )
    index = int(indices[0])
    difference = float(dataset["comparisons__difference_ms"].values[index])
    p_value = float(dataset["comparisons__p_value"].values[index])
    return difference, p_value


def plot_absolute_intensity(ax, dataset: xr.Dataset) -> None:
    """Plot storm-level ECDFs for IBTrACS, CTL, and GGW."""
    for source_name in ("IBTrACS", "CTL", "GGW"):
        values = storm_intensity_by_source(dataset, source_name)
        x_values, probabilities = empirical_cdf(values)
        ax.step(
            x_values,
            probabilities,
            where="post",
            color=COLORS[source_name],
            linewidth=1.8,
            label=f"{source_name} (n={values.size})",
            zorder=3,
        )

    threshold_ms = WIND_THRESHOLD_KT * KT_TO_MS
    ax.axvline(
        threshold_ms,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    )
    ax.text(
        threshold_ms + 1.0,
        0.05,
        "34 kt",
        rotation=90,
        ha="left",
        va="bottom",
        color="0.35",
        fontsize=FONT_SIZE-4,
        fontweight="normal",
    )

    ax.set_xlim(ECDF_X_MIN, ECDF_X_MAX)
    ax.set_ylim(ECDF_Y_MIN, ECDF_Y_MAX)
    ax.xaxis.set_major_locator(MultipleLocator(ECDF_X_MAJOR_STEP))
    ax.xaxis.set_minor_locator(AutoMinorLocator(ECDF_X_MINOR_DIVISIONS))
    ax.yaxis.set_major_locator(MultipleLocator(ECDF_Y_MAJOR_STEP))
    ax.set_xlabel(r"Lifetime-maximum wind or model proxy (m s$^{-1}$)")
    ax.set_ylabel("Cumulative probability")
    ax.legend(
        loc="lower right",
        frameon=False,
        facecolor="white",
        handlelength=2.2,
    )
    add_panel_title(ax, "(a) Absolute intensity diagnostic")
    style_axis(ax)


def plot_relative_response(ax, dataset: xr.Dataset) -> None:
    """Plot paired CTL and GGW annual-mean model intensity proxies."""
    ctl = np.asarray(dataset["annual__ctl_annual_mean_ms"].values, dtype=float)
    ggw = np.asarray(dataset["annual__ggw_annual_mean_ms"].values, dtype=float)
    if ctl.shape != ggw.shape:
        raise ValueError(f"CTL and GGW annual arrays differ: {ctl.shape}, {ggw.shape}")
    if ctl.size != 30:
        raise ValueError(f"Expected 30 annual means, found {ctl.size}")

    offsets = np.linspace(-0.045, 0.045, ctl.size)
    for index in range(ctl.size):
        ax.plot(
            [offsets[index], 1.0 + offsets[index]],
            [ctl[index], ggw[index]],
            color="0.76",
            linewidth=0.65,
            alpha=0.75,
            zorder=1,
        )
    ax.scatter(
        offsets,
        ctl,
        s=20,
        color=COLORS["CTL"],
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )
    ax.scatter(
        1.0 + offsets,
        ggw,
        s=20,
        color=COLORS["GGW"],
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )

    means = np.asarray([np.mean(ctl), np.mean(ggw)])
    ctl_ci = mean_confidence_interval(ctl)
    ggw_ci = mean_confidence_interval(ggw)
    errors = np.asarray(
        [
            [means[0] - ctl_ci[0], means[1] - ggw_ci[0]],
            [ctl_ci[1] - means[0], ggw_ci[1] - means[1]],
        ]
    )
    ax.errorbar(
        [0.0, 1.0],
        means,
        yerr=errors,
        fmt="D",
        markersize=7.0,
        color="black",
        markerfacecolor="white",
        markeredgewidth=1.0,
        elinewidth=1.1,
        capsize=4,
        capthick=1.1,
        zorder=5,
    )

    difference, p_value = paired_t_comparison(dataset)
    ax.text(
        0.5,
        0.98,
        (
            f"CTL={means[0]:.2f}, GGW={means[1]:.2f}\n"
            f"difference={difference:+.2f}, p={p_value:.3f}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FONT_SIZE-4,
        fontweight="normal",
    )

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(PAIR_Y_MIN, PAIR_Y_MAX)
    ax.set_xticks([0.0, 1.0], labels=["CTL", "GGW"])
    ax.yaxis.set_major_locator(MultipleLocator(PAIR_Y_MAJOR_STEP))
    ax.yaxis.set_minor_locator(AutoMinorLocator(PAIR_Y_MINOR_DIVISIONS))
    ax.set_ylabel(r"Mean lifetime-maximum proxy (m s$^{-1}$)")
    add_panel_title(ax, "(b) Relative model response")
    style_axis(ax)


def create_figure(dataset: xr.Dataset):
    """Create the two-panel diagnostic figure."""
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), dpi=300)
    plot_absolute_intensity(axes[0], dataset)
    plot_relative_response(axes[1], dataset)
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.19,
        top=0.90,
        wspace=0.27,
    )
    return figure


def save_figure(figure) -> None:
    """Save PNG, PDF, SVG, and 600-dpi TIFF outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": "tight", "pad_inches": 0.02}
    figure.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.png", dpi=300, **common)
    figure.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.pdf", **common)
    figure.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.svg", **common)
    figure.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.tiff", dpi=600, **common)


def main() -> None:
    plt.switch_backend("Agg")
    setup_style()
    dataset = load_figure_data()
    try:
        figure = create_figure(dataset)
        save_figure(figure)
        plt.close(figure)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
