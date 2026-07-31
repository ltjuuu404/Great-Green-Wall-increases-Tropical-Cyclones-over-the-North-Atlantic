from __future__ import annotations

"""Reproduce Supplementary Figure S1 from figureS1_data.nc only.

This script is intentionally independent of the original data-processing and
plotting scripts.  It reads no raw model files and does not read or overwrite
the metrics CSV in the parent directory.
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import xarray as xr


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_FILE = PACKAGE_DIR / "figureS1_data.nc"
OUTPUT_DIR = PACKAGE_DIR

EXTENT = [-100.0, 45.0, 0.0, 40.0]
MDR_BOX = (-70.0, -17.5, 10.0, 20.0)

FONT_SIZE = 16
BORDER_WIDTH = 0.75
MAJOR_GRID_WIDTH = 0.5
CONTOUR_LINE_WIDTH = 0.35
CONTOUR_LINE_ALPHA = 0.55

# Tick customization knobs.
LON_TICK_MIN = -100.0
LON_TICK_MAX = 40.0
LON_TICK_INTERVAL = 20.0
LAT_TICK_MIN = 0.0
LAT_TICK_MAX = 40.0
LAT_TICK_INTERVAL = 10.0

COLORBAR_WIDTH = "80%"
COLORBAR_HEIGHT = "6%"
COLORBAR_Y_OFFSET = -0.30
COLORBAR_LABEL_PAD = 2


def make_ticks(start: float, stop: float, interval: float) -> np.ndarray:
    return np.arange(start, stop + interval * 0.5, interval)


LON_TICKS = make_ticks(LON_TICK_MIN, LON_TICK_MAX, LON_TICK_INTERVAL)
LAT_TICKS = make_ticks(LAT_TICK_MIN, LAT_TICK_MAX, LAT_TICK_INTERVAL)


def setup_style() -> None:
    """Use bold only for subplot titles; keep all other text normal."""
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE-4,
            "axes.titlesize": FONT_SIZE-4,
            "xtick.labelsize": FONT_SIZE-4,
            "ytick.labelsize": FONT_SIZE-4,
            "legend.fontsize": FONT_SIZE-4,
            "axes.linewidth": BORDER_WIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_mdr_box(ax) -> None:
    lon0, lon1, lat0, lat1 = MDR_BOX
    ax.add_patch(
        Rectangle(
            (lon0, lat0),
            lon1 - lon0,
            lat1 - lat0,
            fill=False,
            edgecolor="black",
            linewidth=BORDER_WIDTH,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )
    )


def format_map(ax) -> None:
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeee", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=BORDER_WIDTH, edgecolor="black", zorder=6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="black", zorder=6)
    ax.set_xticks(LON_TICKS, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_TICKS, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.0,
        width=BORDER_WIDTH,
        top=False,
        right=False,
    )
    ax.minorticks_off()
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=MAJOR_GRID_WIDTH,
        color="0.65",
        alpha=0.85,
        linestyle="--",
        xlocs=LON_TICKS,
        ylocs=LAT_TICKS,
    )
    try:
        ax.spines["geo"].set_linewidth(BORDER_WIDTH)
        ax.spines["geo"].set_edgecolor("black")
    except Exception:
        pass
    add_mdr_box(ax)


def add_panel_label(ax, label: str, text: str) -> None:
    ax.text(
        0.0,
        1.055,
        f"{label} {text}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE,
        fontweight="bold",
        clip_on=False,
        zorder=20,
    )


def add_horizontal_colorbar(
    fig,
    ax,
    mesh,
    ticks: list[float],
    unit_label: str,
    tick_labels: list[str] | None = None,
) -> None:
    cax = inset_axes(
        ax,
        width=COLORBAR_WIDTH,
        height=COLORBAR_HEIGHT,
        loc="lower center",
        bbox_to_anchor=(0.0, COLORBAR_Y_OFFSET, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    cbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
        extend="both",
        spacing="uniform",
        ticks=ticks,
    )
    cbar.outline.set_linewidth(BORDER_WIDTH)
    cbar.ax.tick_params(
        axis="x",
        which="major",
        direction="out",
        length=2.5,
        width=BORDER_WIDTH,
        top=False,
        bottom=True,
        labelsize=FONT_SIZE-4,
        pad=1.5,
    )
    cbar.ax.minorticks_off()
    if tick_labels is not None:
        cbar.ax.set_xticklabels(tick_labels)
    cbar.set_label(unit_label, fontsize=FONT_SIZE-4, fontweight="normal", labelpad=COLORBAR_LABEL_PAD)


def map_panel(
    fig,
    ax,
    da: xr.DataArray,
    levels: list[float],
    ticks: list[float],
    cmap_name: str,
    label: str,
    text: str,
    unit_label: str,
    tick_labels: list[str] | None = None,
) -> None:
    lon2, lat2 = np.meshgrid(da["lon"].values, da["lat"].values)
    cmap = plt.get_cmap(cmap_name, len(levels) - 1)
    mesh = ax.contourf(
        lon2,
        lat2,
        da.values,
        levels=levels,
        cmap=cmap,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.contour(
        lon2,
        lat2,
        da.values,
        levels=levels,
        colors="0.25",
        linewidths=CONTOUR_LINE_WIDTH,
        alpha=CONTOUR_LINE_ALPHA,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    format_map(ax)
    add_panel_label(ax, label, text)
    add_horizontal_colorbar(fig, ax, mesh, ticks, unit_label, tick_labels=tick_labels)


def load_plot_fields() -> xr.Dataset:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing extracted figure data: {DATA_FILE}")
    with xr.open_dataset(DATA_FILE) as source:
        required = ["gpcp_on_model", "ctl_prect", "ctl_minus_gpcp", "ggw_minus_ctl"]
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"Missing variables in {DATA_FILE.name}: {', '.join(missing)}")
        return source[required].load()


def main() -> None:
    setup_style()
    fields = load_plot_fields()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9.8, 5.25), dpi=300)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.985,
        top=0.88,
        bottom=0.14,
        wspace=0.18,
        hspace=0.25,
    )
    axes = [fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree()) for i in range(2) for j in range(2)]

    abs_levels = [0, 1, 2, 3, 4, 6, 8, 10, 12]
    bias_levels = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    response_levels = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

    map_panel(fig, axes[0], fields["gpcp_on_model"], abs_levels, abs_levels, "YlGnBu", "(a)", "GPCP", r"(mm day$^{-1}$)")
    map_panel(fig, axes[1], fields["ctl_prect"], abs_levels, abs_levels, "YlGnBu", "(b)", "CTL", r"(mm day$^{-1}$)")
    map_panel(
        fig,
        axes[2],
        fields["ctl_minus_gpcp"],
        bias_levels,
        bias_levels,
        "RdBu",
        "(c)",
        "CTL - GPCP",
        r"(mm day$^{-1}$)",
        ["-4", "-3", "-2", "-1", "0", "1", "2", "3", "4"],
    )
    map_panel(
        fig,
        axes[3],
        fields["ggw_minus_ctl"],
        response_levels,
        response_levels,
        "RdBu",
        "(d)",
        "GGW - CTL",
        r"(mm day$^{-1}$)",
        ["-2", "-1.5", "-1", "-0.5", "0", "0.5", "1", "1.5", "2"],
    )

    output_stem = OUTPUT_DIR / "figureS1"
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
