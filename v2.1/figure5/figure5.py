#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot five response fields and add 1000-hPa wind vectors only to panel (b)."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr


# ------------------ Style Config ------------------
FONT_NAME = "Times New Roman"
FONT_SIZE = 24
mpl.rcParams.update(
    {
        "font.family": FONT_NAME,
        "font.serif": ["Times New Roman", "Times"],
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 24,
        "figure.dpi": 100,
        "savefig.dpi": 600,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

AXES_FRAME_LINEWIDTH = 2.0
SPINE_ZORDER = 100
CONTOUR_ZORDER = 1
COAST_ZORDER = 50

DATA_DIR = Path(__file__).resolve().parent
INPUT_NC = DATA_DIR / "figure5_data.nc"
OUTPUT_PREFIX = DATA_DIR / "figure5"

LAT_SLICE = (-10, 50)
LON_SLICE = (-90, 50)
FIGSIZE = (16, 50)

ALPHA = 0.05
MDR_LON_MIN = -70.0
MDR_LON_MAX = -17.5
MDR_LAT_MIN = 10.0
MDR_LAT_MAX = 20.0
MDR_BOX_LINEWIDTH = 2.0

# Retain the original 1000-hPa difference-vector settings.
WIND_SKIP = 2
WIND_SCALE = 30
WIND_WIDTH = 0.004
WIND_KEY_VALUE = 1.0
WIND_KEY_X = 0.80
WIND_KEY_Y = 1.07
WIND_KEY_LABEL = r"1 m s$^{-1}$"


def make_axes_bold(ax, linewidth=AXES_FRAME_LINEWIDTH):
    """Retain the axis-frame treatment of the current plotting script."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_edgecolor("black")
        spine.set_zorder(SPINE_ZORDER)


def format_lon_label(value):
    rounded = int(np.round(value))
    if rounded == 0:
        return "0°"
    return f"{abs(rounded)}°W" if rounded < 0 else f"{rounded}°E"


def format_lat_label(value):
    rounded = int(np.round(value))
    if rounded == 0:
        return "0°"
    return f"{abs(rounded)}°S" if rounded < 0 else f"{rounded}°N"


def apply_style(ax):
    major_xticks = np.arange(LON_SLICE[0], LON_SLICE[1] + 0.1, 30.0)
    major_yticks = np.arange(-10, 51, 20.0)
    minor_xticks = (major_xticks[:-1] + major_xticks[1:]) / 2.0
    minor_yticks = (major_yticks[:-1] + major_yticks[1:]) / 2.0

    ax.set_xticks(major_xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(major_yticks, crs=ccrs.PlateCarree())
    ax.set_xticks(minor_xticks, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(minor_yticks, minor=True, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: format_lon_label(x))
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, pos: format_lat_label(y))
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=8,
        width=AXES_FRAME_LINEWIDTH,
        top=False,
        bottom=True,
        left=True,
        right=False,
        labelsize=FONT_SIZE - 4,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=4,
        width=1.0,
        top=False,
        bottom=True,
        left=True,
        right=False,
    )

    ax.set_extent(
        [LON_SLICE[0], LON_SLICE[1], LAT_SLICE[0], LAT_SLICE[1]],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="110m", linewidth=1.2, zorder=COAST_ZORDER)
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.4,
        zorder=CONTOUR_ZORDER + 0.1,
    )
    ax.gridlines(
        draw_labels=False,
        linewidth=0.35,
        color="gray",
        linestyle="--",
        zorder=2,
    )
    make_axes_bold(ax)


def format_colorbar_tick(value):
    if abs(value) < 1e-12:
        return "0"
    if abs(value) >= 1:
        return str(int(round(value)))
    return f"{value:.1f}"


def style_colorbar(colorbar, unit_label):
    colorbar.ax.tick_params(labelsize=FONT_SIZE - 4)
    ticks = colorbar.get_ticks()
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([format_colorbar_tick(tick) for tick in ticks])
    colorbar.set_label(unit_label, fontsize=FONT_SIZE - 2)


def add_original_wind_vectors(ax, lons, lats, u_wind, v_wind):
    """Overlay the original 1000-hPa GGW-minus-CTL wind vectors and key."""
    quiver = ax.quiver(
        lons[::WIND_SKIP],
        lats[::WIND_SKIP],
        u_wind[::WIND_SKIP, ::WIND_SKIP],
        v_wind[::WIND_SKIP, ::WIND_SKIP],
        transform=ccrs.PlateCarree(),
        scale=WIND_SCALE,
        width=WIND_WIDTH,
        zorder=102,
        minlength=0,
    )
    ax.quiverkey(
        quiver,
        WIND_KEY_X,
        WIND_KEY_Y,
        WIND_KEY_VALUE,
        WIND_KEY_LABEL,
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": FONT_SIZE - 4},
    )


def add_panel(fig, ax, lons, lats, panel):
    contour = ax.contourf(
        lons,
        lats,
        panel["data"],
        levels=panel["levels"],
        cmap=panel["cmap"],
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=CONTOUR_ZORDER,
    )

    significant = (
        (panel["pval"] < ALPHA)
        & np.isfinite(panel["pval"])
        & np.isfinite(panel["data"])
    )
    if np.any(significant):
        hatch = ax.contourf(
            lons,
            lats,
            significant.astype(int),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["///"],
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        for collection in hatch.collections:
            collection.set_edgecolor("white")
            collection.set_linewidth(0.5)

    ax.add_patch(
        Rectangle(
            (MDR_LON_MIN, MDR_LAT_MIN),
            MDR_LON_MAX - MDR_LON_MIN,
            MDR_LAT_MAX - MDR_LAT_MIN,
            facecolor="none",
            edgecolor="black",
            linewidth=MDR_BOX_LINEWIDTH,
            transform=ccrs.PlateCarree(),
            zorder=COAST_ZORDER + 5,
        )
    )

    if panel.get("show_wind_vectors", False):
        add_original_wind_vectors(
            ax,
            lons,
            lats,
            panel["u_wind"],
            panel["v_wind"],
        )
    apply_style(ax)

    ax.set_title(
        panel["title"],
        loc="left",
        fontsize=FONT_SIZE + 2,
        fontweight="bold",
    )
    colorbar_ax = ax.inset_axes([0.025, -0.27, 0.95, 0.05])
    colorbar = fig.colorbar(contour, cax=colorbar_ax, orientation="horizontal")
    style_colorbar(colorbar, panel["unit"])


def plot_all(dataset: xr.Dataset):
    lons = dataset["lon"].values
    lats = dataset["lat"].values

    fig = plt.figure(figsize=FIGSIZE)
    grid = fig.add_gridspec(3, 4)
    axes = [
        fig.add_subplot(grid[0, 0:2], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[0, 2:4], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[1, 0:2], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[1, 2:4], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[2, 0:2], projection=ccrs.PlateCarree()),
    ]
    plt.subplots_adjust(
        left=0.08,
        right=0.95,
        top=0.95,
        bottom=0.12,
        hspace=0.8,
        wspace=0.15,
    )

    panels = [
        {
            "title": "(a) SST",
            "data": dataset["sst_diff"].values,
            "pval": dataset["sst_pval"].values,
            "levels": np.linspace(-1, 1, 21),
            "cmap": "RdBu_r",
            "unit": "K",
        },
        {
            "title": "(b) SLP",
            "data": dataset["slp_diff"].values,
            "pval": dataset["slp_pval"].values,
            "levels": np.linspace(-100, 100, 21),
            "cmap": "RdBu_r",
            "unit": "Pa",
            "show_wind_vectors": True,
            "u_wind": dataset["wind1000_u_diff"].values,
            "v_wind": dataset["wind1000_v_diff"].values,
        },
        {
            "title": "(c) 1000 hPa Wind Speed",
            "data": dataset["wind1000_speed_diff"].values,
            "pval": dataset["wind1000_speed_pval"].values,
            "levels": np.linspace(-1, 1, 21),
            "cmap": "RdBu_r",
            "unit": r"m s$^{-1}$",
        },
        {
            "title": "(d) Latent Heat Flux",
            "data": dataset["lhflx_diff"].values,
            "pval": dataset["lhflx_pval"].values,
            "levels": np.linspace(-15, 15, 21),
            "cmap": "BrBG",
            "unit": r"W m$^{-2}$",
        },
        {
            "title": "(e) Total Cloud Cover",
            "data": dataset["cloud_cover_diff_pct"].values,
            "pval": dataset["cloud_cover_pval"].values,
            "levels": np.linspace(-5, 5, 21),
            "cmap": "RdBu_r",
            "unit": "%",
        },
    ]

    for ax, panel in zip(axes, panels):
        add_panel(fig, ax, lons, lats, panel)

    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(
        OUTPUT_PREFIX.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.show()


def main():
    if not INPUT_NC.exists():
        raise FileNotFoundError(
            f"Prepared input not found: {INPUT_NC}\n"
            "Run 05_prepare_data_with_significance_and_wind.py first."
        )
    with xr.open_dataset(INPUT_NC) as dataset:
        plot_all(dataset.load())

    for suffix in (".png", ".pdf", ".svg", ".tiff"):
        print(f"Wrote: {OUTPUT_PREFIX.with_suffix(suffix)}")


if __name__ == "__main__":
    main()
