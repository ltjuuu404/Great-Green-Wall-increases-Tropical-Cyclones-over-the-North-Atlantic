#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-panel precipitation figure with the corresponding wind vectors."""

from pathlib import Path
import math
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle


# DPI settings retained from the supplied plotting script.
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 600
warnings.filterwarnings("ignore")

INPUT_NC = Path(__file__).resolve().parent / "figure4_data.nc"
OUTPUT_PNG = Path(__file__).resolve().parent / "figure4.png"

LAT_SLICE = (-10, 50)
LON_SLICE = (-90, 50)
CENTRAL_LON = -10.0
# The width is unchanged.  The height is one third of the former 3-row layout.
FIGSIZE = (15, 6)

CMAP_CTL = "Blues"
CMAP_DIFF = "BrBG"
ALPHA = 0.05

FONT_NAME = "Times New Roman"
FONT_SIZE = 20
TITLE_FONT_SIZE = FONT_SIZE
mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times"],
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 18,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["font.size"] = FONT_SIZE

AXES_FRAME_LINEWIDTH = 1.8
CONTOUR_ZORDER = 1
SIG_ZORDER = 10
COAST_ZORDER = 50
SPINE_ZORDER = 100

# Main development region (MDR): 10-20 degrees N, 70-17.5 degrees W.
MDR_LON_MIN = -70.0
MDR_LON_MAX = -17.5
MDR_LAT_MIN = 10.0
MDR_LAT_MAX = 20.0
MDR_BOX_LINEWIDTH = 2.0

PRECIP_CTL_VMIN = 0
PRECIP_CTL_VMAX = 10
PRECIP_DIFF_VMIN = -1
PRECIP_DIFF_VMAX = 1

WIND_SCALE_CTL_KEY = 10
WIND_SCALE_DIFF_KEY = 1
WIND_VECTOR_SCALE_MULTIPLIER = 16.0


def format_lon_label(x):
    value = int(np.round(x))
    if value == 0:
        return "0°"
    return f"{abs(value)}°W" if value < 0 else f"{value}°E"


def format_lat_label(y):
    value = int(np.round(y))
    if value == 0:
        return "0°"
    return f"{abs(value)}°S" if value < 0 else f"{value}°N"


def normalize_lon_to_center(lon_values, center):
    lon = np.asarray(lon_values)
    return ((lon - center + 180) % 360) - 180 + center


def make_axes_bold(ax, linewidth=AXES_FRAME_LINEWIDTH):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_edgecolor("black")
        spine.set_zorder(SPINE_ZORDER)


def format_cb_value(value):
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.1f}"


def apply_cb_tick_format(colorbar):
    ticks = colorbar.get_ticks()
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([format_cb_value(tick) for tick in ticks])


def add_wind_vectors(ax, lons, lats, u, v, key_value):
    """Use the supplied script's vector thinning, scale, and key settings."""
    ny, nx = u.shape
    total = max(1, nx * ny)
    factor = int(np.ceil(np.sqrt(total / max(1, 200))))
    sx = max(3, factor)
    sy = max(3, factor)

    lon2, lat2 = np.meshgrid(lons[::sx], lats[::sy])
    u_sub = u[::sy, ::sx]
    v_sub = v[::sy, ::sx]
    magnitude = np.sqrt(u_sub**2 + v_sub**2)
    magnitude_95 = np.nanpercentile(magnitude, 95)
    scale_value = 1.0 if magnitude_95 < 1e-6 else magnitude_95 * WIND_VECTOR_SCALE_MULTIPLIER

    quiver = ax.quiver(
        lon2,
        lat2,
        u_sub,
        v_sub,
        transform=ccrs.PlateCarree(),
        width=0.005,
        pivot="middle",
        scale=scale_value,
        zorder=102,
        minlength=0,
    )
    ax.quiverkey(
        quiver,
        0.87,
        1.05,
        key_value,
        rf"{key_value} m s$^{{-1}}$",
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": FONT_SIZE - 4, "family": "Times New Roman"},
    )


def configure_map_axes(ax):
    ax.set_extent(
        [LON_SLICE[0], LON_SLICE[1], LAT_SLICE[0], LAT_SLICE[1]],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="110m", linewidth=0.5, zorder=CONTOUR_ZORDER + 0.1)
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.4,
        zorder=CONTOUR_ZORDER + 0.1,
    )

    major_xticks = np.arange(LON_SLICE[0], LON_SLICE[1] + 0.1, 30.0)
    major_yticks = np.arange(
        math.ceil(LAT_SLICE[0] / 10) * 10,
        math.floor(LAT_SLICE[1] / 10) * 10 + 1,
        20.0,
    )
    minor_xticks = (major_xticks[:-1] + major_xticks[1:]) / 2.0
    minor_yticks = (major_yticks[:-1] + major_yticks[1:]) / 2.0

    ax.set_xticks(major_xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(major_yticks, crs=ccrs.PlateCarree())
    ax.set_xticks(minor_xticks, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(minor_yticks, minor=True, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_lon_label(x)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: format_lat_label(y)))

    ax.tick_params(
        axis="x",
        which="major",
        direction="out",
        length=8,
        width=AXES_FRAME_LINEWIDTH,
        top=False,
        bottom=True,
        labelsize=FONT_SIZE - 4,
        labelbottom=True,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        direction="out",
        length=4,
        width=1.0,
        top=False,
        bottom=True,
        labelbottom=True,
    )
    ax.tick_params(
        axis="y",
        which="major",
        direction="out",
        length=8,
        width=AXES_FRAME_LINEWIDTH,
        left=True,
        right=False,
        labelsize=FONT_SIZE - 4,
        labelleft=True,
    )
    ax.tick_params(
        axis="y",
        which="minor",
        direction="out",
        length=4,
        width=1.0,
        left=True,
        right=False,
        labelleft=True,
    )
    ax.gridlines(draw_labels=False, linewidth=0.35, color="gray", linestyle="--", zorder=2)


def plot_1x2(dataset: xr.Dataset, outfile: Path) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    plt.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.22, wspace=0.13)

    lats = np.asarray(dataset["lat"].values)
    lons = normalize_lon_to_center(dataset["lon"].values, CENTRAL_LON)
    panels = [
        {
            "title": "(a) Precip (CTL)",
            "data": dataset["precip_ctl"].values,
            "u": dataset["u_ctl"].values,
            "v": dataset["v_ctl"].values,
            "levels": np.linspace(PRECIP_CTL_VMIN, PRECIP_CTL_VMAX, 21),
            "cmap": CMAP_CTL,
            "key": WIND_SCALE_CTL_KEY,
            "difference": False,
        },
        {
            "title": "(b) Precip (GGW-CTL)",
            "data": dataset["precip_diff"].values,
            "u": dataset["u_diff"].values,
            "v": dataset["v_diff"].values,
            "levels": np.linspace(PRECIP_DIFF_VMIN, PRECIP_DIFF_VMAX, 21),
            "cmap": CMAP_DIFF,
            "key": WIND_SCALE_DIFF_KEY,
            "difference": True,
        },
    ]

    contour_sets = []
    for ax, panel in zip(axes, panels):
        configure_map_axes(ax)
        ax.set_title(
            panel["title"],
            loc="left",
            fontsize=TITLE_FONT_SIZE+6,
            fontweight="bold",
        )
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
        contour_sets.append(contour)

        if panel["difference"]:
            significant = dataset["precip_pval"].values < ALPHA
            if np.any(significant):
                significance = ax.contourf(
                    lons,
                    lats,
                    significant.astype(int),
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=["///"],
                    transform=ccrs.PlateCarree(),
                    zorder=SIG_ZORDER,
                )
                for collection in significance.collections:
                    collection.set_edgecolor("white")
                    collection.set_linewidth(0.2)

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

        add_wind_vectors(
            ax,
            lons,
            lats,
            panel["u"],
            panel["v"],
            panel["key"],
        )
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1.2, zorder=COAST_ZORDER)
        make_axes_bold(ax)

    for ax, contour in zip(axes, contour_sets):
        position = ax.get_position()
        width = position.width * 0.9
        x0 = position.x0 + (position.width - width) / 2.0
        y0 = position.y0 - 0.075
        colorbar_ax = fig.add_axes([x0, y0, width, 0.025])
        colorbar = fig.colorbar(contour, cax=colorbar_ax, orientation="horizontal")
        colorbar.set_label(r"mm day$^{-1}$", fontsize=FONT_SIZE - 4, labelpad=-2)
        colorbar.ax.tick_params(labelsize=FONT_SIZE - 4)
        apply_cb_tick_format(colorbar)

    # The supplied script uses an interactive display.  The existing outfile
    # argument is also honored here so that the revised figure is reproducible.
    fig.savefig(outfile, dpi=600, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(
        outfile.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.show()


def main() -> None:
    if not INPUT_NC.exists():
        raise FileNotFoundError(
            f"Prepared input not found: {INPUT_NC}\nRun 01_prepare_data.py first."
        )
    with xr.open_dataset(INPUT_NC) as dataset:
        plot_1x2(dataset.load(), OUTPUT_PNG)
    print(f"Wrote: {OUTPUT_PNG}")
    print(f"Wrote: {OUTPUT_PNG.with_suffix('.pdf')}")
    print(f"Wrote: {OUTPUT_PNG.with_suffix('.svg')}")
    print(f"Wrote: {OUTPUT_PNG.with_suffix('.tiff')}")


if __name__ == "__main__":
    main()
