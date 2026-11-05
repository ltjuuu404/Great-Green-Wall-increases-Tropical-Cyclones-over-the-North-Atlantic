from __future__ import annotations

"""Plot the single retained AODDUST response panel from prepared data."""

# Package portability setup: paths and batch export only; source plot parameters retained.
from pathlib import Path as _PackagePath
import os as _package_os
import matplotlib as _package_mpl
_package_mpl.use("Agg")
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "figure_r5_data.nc"
OUTPUT_DIR = HERE / "outputs"
FONT = "Times New Roman"
FONT_SIZE = 18
BORDER_WIDTH = 0.75
AXIS_BORDER_WIDTH = 1.5
MAJOR_TICK_WIDTH = 1.5
MINOR_TICK_WIDTH = 0.75
EXTENT = (-100.0, 45.0, 0.0, 40.0)
MDR_BOX = (-70.0, -17.5, 10.0, 20.0)
LON_TICKS = np.arange(-90.0, 31.0, 30.0)
LAT_TICKS = np.arange(0.0, 41.0, 20.0)
LON_MINOR_TICKS = (LON_TICKS[:-1] + LON_TICKS[1:]) / 2.0
LAT_MINOR_TICKS = (LAT_TICKS[:-1] + LAT_TICKS[1:]) / 2.0
LEVELS = [-0.20, -0.12, -0.08, -0.04, -0.005, -0.0001, 0.0001, 0.002, 0.004, 0.006]
TICKS = [-0.20, -0.12, -0.08, -0.04, 0.0, 0.002, 0.004, 0.006]
COLORS = ["#00483d", "#007465", "#23a393", "#78cfc4", "#d9f3ee", "#ffffff", "#fff0c7", "#d99b3d", "#704106"]


def main() -> None:
    plt.switch_backend("Agg")
    mpl.rcParams.update({"font.family": FONT, "font.size": FONT_SIZE,
                         "font.weight": "normal", "axes.labelweight": "normal",
                         "axes.linewidth": BORDER_WIDTH, "hatch.linewidth": 0.35,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    with xr.open_dataset(DATA_FILE) as source:
        ds = source.load()
    lon, lat = ds.lon.values, ds.lat.values
    lon2, lat2 = np.meshgrid(lon, lat)
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, len(COLORS))
    fig = plt.figure(figsize=(8.0, 3.25), dpi=300)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    mesh = ax.contourf(lon2, lat2, ds.aoddust_diff, levels=LEVELS, cmap=cmap,
                       norm=norm, extend="both", transform=ccrs.PlateCarree(), zorder=2)
    ax.contour(lon2, lat2, ds.aoddust_diff, levels=LEVELS, colors="0.25",
               linewidths=0.35, alpha=0.55, transform=ccrs.PlateCarree(), zorder=3)
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeee", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=BORDER_WIDTH, edgecolor="black", zorder=6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="black", zorder=6)
    ax.add_patch(Rectangle((MDR_BOX[0], MDR_BOX[2]), MDR_BOX[1]-MDR_BOX[0], MDR_BOX[3]-MDR_BOX[2],
                           fill=False, edgecolor="black", linewidth=BORDER_WIDTH,
                           transform=ccrs.PlateCarree(), zorder=7))
    ax.set_xticks(LON_TICKS, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_TICKS, crs=ccrs.PlateCarree())
    ax.set_xticks(LON_MINOR_TICKS, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_MINOR_TICKS, minor=True, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(which="major", direction="out", length=3, width=MAJOR_TICK_WIDTH, top=False, right=False)
    ax.tick_params(which="minor", direction="out", length=2, width=MINOR_TICK_WIDTH, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(AXIS_BORDER_WIDTH)
    ax.gridlines(draw_labels=False, linewidth=0.5, color="0.65", alpha=0.85,
                 linestyle="--", xlocs=LON_TICKS, ylocs=LAT_TICKS)
    ax.text(0, 1.055, "AODDUST response", transform=ax.transAxes, ha="left",
            va="bottom", fontweight="bold", fontsize=26, clip_on=False)
    cax = inset_axes(ax, width="80%", height="6%", loc="lower center",
                     bbox_to_anchor=(0, -0.30, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both",
                      spacing="uniform", ticks=TICKS)
    cb.ax.set_xticklabels(["-200", "-120", "-80", "-40", "0", "2", "4", "6"])
    cb.ax.tick_params(which="major", direction="out", length=2.5, width=BORDER_WIDTH, top=False)
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(BORDER_WIDTH)
    cb.set_label(r"AOD ($\times 10^{-3}$)", labelpad=2)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.83, bottom=0.18)
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = OUTPUT_DIR / "figure_r5_aoddust_response"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
