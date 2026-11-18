from __future__ import annotations

"""Plot revised three-panel vertical Figure R2 from local prepared data."""

# Package portability setup: local paths and interactive display; plot parameters retained.
from pathlib import Path as _PackagePath
import os as _package_os
import matplotlib as _package_mpl
# Keep an interactive IDE backend; recover if an earlier run selected Agg.
if _package_mpl.get_backend().lower() == "agg":
    _package_mpl.use("TkAgg")
_package_here = _PackagePath(__file__).resolve().parent
_package_os.chdir(_package_here)
_package_output = _package_here / "outputs"
_package_output.mkdir(exist_ok=True)
import cartopy as _package_cartopy
_package_map_dir = str(_package_here.parent / "map_data")
_package_cartopy.config["data_dir"] = _package_map_dir
_package_cartopy.config["pre_existing_data_dir"] = _package_map_dir


from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "FS01_gpcp_ctl_precipitation.nc"
OUTPUT_DIR = HERE / "outputs"
EXTENT = (-100.0, 45.0, 0.0, 40.0)
MDR_BOX = (-70.0, -17.5, 10.0, 20.0)
MDR_LINE_WIDTH = 1.5  # pt
FONT = "Times New Roman"
FONT_SIZE = 20
PANEL_TITLE_SIZE = 30
BORDER_WIDTH = 0.75
AXIS_FRAME_WIDTH = 0.75
MAJOR_TICK_WIDTH = 0.75
GRID_WIDTH = 0.5

# Geographic tick controls. Major ticks carry labels and grid lines; minor
# ticks are unlabeled. Change the ranges and intervals here when needed.
LON_TICK_START = -90.0
LON_TICK_STOP = 45.0
LON_MAJOR_INTERVAL = 30.0
LON_MINOR_INTERVAL = 15.0
LAT_TICK_START = 0.0
LAT_TICK_STOP = 40.0
LAT_MAJOR_INTERVAL = 20.0
LAT_MINOR_INTERVAL = 10.0

# Explicit per-map horizontal colourbar geometry. The rectangular scale is
# 80% of the map width; its two extensions are allocated additional space.
COLORBAR_WIDTH_RATIO = 0.80
COLORBAR_HEIGHT_RATIO = 0.06
COLORBAR_GAP_RATIO = 0.17
COLORBAR_EXTEND_FRACTION = 0.05


def build_ticks(start: float, stop: float, major_interval: float,
                minor_interval: float) -> tuple[np.ndarray, np.ndarray]:
    """Return labeled major ticks and unlabeled minor ticks."""
    major = np.arange(start, stop + 0.5 * major_interval, major_interval)
    minor_candidates = np.arange(start, stop + 0.5 * minor_interval, minor_interval)
    is_major = np.isclose(minor_candidates[:, None], major[None, :]).any(axis=1)
    return major, minor_candidates[~is_major]


LON_MAJOR_TICKS, LON_MINOR_TICKS = build_ticks(
    LON_TICK_START, LON_TICK_STOP, LON_MAJOR_INTERVAL, LON_MINOR_INTERVAL
)
LAT_MAJOR_TICKS, LAT_MINOR_TICKS = build_ticks(
    LAT_TICK_START, LAT_TICK_STOP, LAT_MAJOR_INTERVAL, LAT_MINOR_INTERVAL
)


def setup_style() -> None:
    mpl.rcParams.update({
        # Keep literal values here so automated publication preflight can audit them.
        "font.family": "Times New Roman", "font.size": 20, "font.weight": "normal",
        "axes.labelweight": "normal", "axes.linewidth": AXIS_FRAME_WIDTH,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def format_map(ax) -> None:
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeee", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=BORDER_WIDTH, edgecolor="black", zorder=6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="black", zorder=6)
    ax.set_xticks(LON_MAJOR_TICKS, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_MAJOR_TICKS, crs=ccrs.PlateCarree())
    ax.set_xticks(LON_MINOR_TICKS, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_MINOR_TICKS, minor=True, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_FRAME_WIDTH)
        spine.set_edgecolor("black")
    ax.tick_params(which="major", direction="out", length=5, width=MAJOR_TICK_WIDTH,
                   top=False, right=False, labeltop=False, labelright=False,
                   labelbottom=True, labelleft=True, labelsize=FONT_SIZE)
    ax.tick_params(which="minor", direction="out", length=0, width=BORDER_WIDTH,
                   top=False, right=False, labeltop=False, labelright=False,
                   labelbottom=False, labelleft=False)
    ax.gridlines(draw_labels=False, linewidth=GRID_WIDTH, color="0.65", alpha=0.85,
                 linestyle="--", xlocs=LON_MAJOR_TICKS, ylocs=LAT_MAJOR_TICKS)
    ax.add_patch(Rectangle((MDR_BOX[0], MDR_BOX[2]), MDR_BOX[1]-MDR_BOX[0], MDR_BOX[3]-MDR_BOX[2],
                           fill=False, edgecolor="black", linewidth=MDR_LINE_WIDTH,
                           transform=ccrs.PlateCarree(), zorder=7))


def panel(ax, field, levels, cmap, title):
    lon2, lat2 = np.meshgrid(field.lon, field.lat)
    mesh = ax.contourf(lon2, lat2, field, levels=levels,
                       cmap=plt.get_cmap(cmap, len(levels)-1), extend="both",
                       transform=ccrs.PlateCarree(), zorder=2)
    ax.contour(lon2, lat2, field, levels=levels, colors="0.25", linewidths=0.35,
               alpha=0.55, transform=ccrs.PlateCarree(), zorder=3)
    format_map(ax)
    ax.text(0, 1.055, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=PANEL_TITLE_SIZE, fontfamily=FONT,
            fontweight="bold", clip_on=False)
    return mesh


def add_horizontal_colorbar(fig, mesh, ax, levels, unit, label_every: int = 1) -> None:
    allocated_width = COLORBAR_WIDTH_RATIO * (1 + 2 * COLORBAR_EXTEND_FRACTION)
    cax = ax.inset_axes([
        (1 - allocated_width) / 2,
        -COLORBAR_GAP_RATIO - COLORBAR_HEIGHT_RATIO,
        allocated_width,
        COLORBAR_HEIGHT_RATIO,
    ])
    # Select ticks from the actual contour boundaries. The precipitation
    # levels are deliberately non-uniform, so never interpolate tick values.
    boundaries = np.asarray(levels, dtype=float)
    tick_indices = np.unique(np.r_[
        np.arange(0, len(boundaries), max(1, int(label_every))),
        len(boundaries) - 1,
        np.flatnonzero(np.isclose(boundaries, 0.0)),
    ]).astype(int)
    ticks = boundaries[tick_indices]
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both",
                      extendfrac=COLORBAR_EXTEND_FRACTION,
                      spacing="uniform", ticks=ticks)
    cb.set_ticks(ticks, labels=[f"{value:.0f}" for value in ticks])
    cb.ax.set_aspect("auto")
    cb.outline.set_linewidth(BORDER_WIDTH)
    cb.outline.set_edgecolor("black")
    cb.ax.tick_params(which="major", direction="out", length=3.5, width=BORDER_WIDTH,
                      top=False, bottom=True, labeltop=False, labelbottom=True,
                      labelsize=FONT_SIZE)
    cb.ax.minorticks_off()
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.xaxis.set_label_position("bottom")
    cb.set_label(unit, labelpad=3, fontsize=FONT_SIZE, fontweight="normal")


def main() -> None:
    setup_style()
    with xr.open_dataset(DATA_FILE) as source:
        ds = source.load()
    fig = plt.figure(figsize=(10.4, 13.0), dpi=300)
    gs = fig.add_gridspec(3, 1, left=0.075, right=0.89, top=0.95, bottom=0.12, hspace=0.90)
    axes = [fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree()) for i in range(3)]
    abs_levels = [0, 1, 2, 3, 4, 6, 8, 10, 12]
    bias_levels = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    abs_mesh = panel(axes[0], ds["gpcp_on_model"], abs_levels, "YlGnBu", "(a) GPCP")
    ctl_mesh = panel(axes[1], ds["ctl_prect"], abs_levels, "YlGnBu", "(b) CTL")
    bias_mesh = panel(axes[2], ds["ctl_minus_gpcp"], bias_levels, "RdBu", "(c) CTL - GPCP")

    add_horizontal_colorbar(fig, abs_mesh, axes[0], abs_levels, r"mm d$^{-1}$")
    add_horizontal_colorbar(fig, ctl_mesh, axes[1], abs_levels, r"mm d$^{-1}$")
    add_horizontal_colorbar(
        fig, bias_mesh, axes[2], bias_levels, r"mm d$^{-1}$", label_every=2
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = OUTPUT_DIR / "figure_r2_precip_validation_three_vertical"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    # plt.show(block=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
