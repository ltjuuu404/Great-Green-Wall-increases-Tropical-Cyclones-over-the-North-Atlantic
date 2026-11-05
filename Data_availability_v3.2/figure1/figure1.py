from __future__ import annotations

"""Plot revised Figure 1 with only the southern 100-mm annual isohyet branch."""

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
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "figure1_data.nc"
OUTPUT_DIR = HERE / "outputs"

# Global style and user-adjustable map ticks.
FONT = "Times New Roman"
FONT_SIZE = 18
TITLE_SIZE = 21
BORDER_WIDTH = 0.75
EXTENT = (-20.0, 40.0, 0.0, 30.0)
LON_MAJOR = np.arange(-20.0, 41.0, 20.0)
LON_MINOR = np.arange(-10.0, 40.0, 20.0)
LAT_MAJOR = np.arange(0.0, 31.0, 10.0)
LAT_MINOR = np.arange(5.0, 30.0, 10.0)


def format_map(ax) -> None:
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#eeeeee", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="white", edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black", linewidth=BORDER_WIDTH, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="black", linewidth=0.35, zorder=5)
    ax.set_xticks(LON_MAJOR, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_MAJOR, crs=ccrs.PlateCarree())
    ax.set_xticks(LON_MINOR, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_MINOR, minor=True, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(which="major", direction="out", length=5, width=BORDER_WIDTH, top=False, right=False)
    ax.tick_params(which="minor", length=0, width=0, top=False, right=False)
    ax.gridlines(
        draw_labels=False,
        linewidth=0.5,
        color="0.65",
        alpha=0.85,
        linestyle="--",
        xlocs=LON_MAJOR,
        ylocs=LAT_MAJOR,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(BORDER_WIDTH)
        spine.set_edgecolor("black")


def main() -> None:
    plt.switch_backend("Agg")
    mpl.rcParams.update({
        "font.family": FONT,
        "font.size": FONT_SIZE,
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.linewidth": BORDER_WIDTH,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    with xr.open_dataset(DATA_FILE) as source:
        ds = source.load()

    veg_levels = np.linspace(0, 100, 11)
    greens = plt.cm.Greens(np.linspace(0.15, 1.0, len(veg_levels) - 2))
    veg_cmap = ListedColormap(np.vstack(([1, 1, 1, 1], greens)))
    veg_norm = BoundaryNorm(veg_levels, veg_cmap.N)

    diff = ds.vegetation_change_percent.values
    diff_levels = np.linspace(0, max(1.0, float(np.nanmax(diff))), 11)
    diff_cmap = LinearSegmentedColormap.from_list("white_green", ["#ffffff", "#2E8B57"], N=256)
    diff_norm = BoundaryNorm(diff_levels, diff_cmap.N)

    fig = plt.figure(figsize=(9.3, 12.8), dpi=300)
    gs = fig.add_gridspec(3, 1, left=0.10, right=0.86, top=0.96, bottom=0.055, hspace=0.30)
    axes = [fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree()) for i in range(3)]
    titles = ["(a) CTL", "(b) GGW", "(c) GGW - CTL"]
    fields = [ds.ctl_vegetation_percent, ds.ggw_vegetation_percent, ds.vegetation_change_percent]
    meshes = []

    for index, (ax, title, field) in enumerate(zip(axes, titles, fields)):
        mesh = ax.pcolormesh(
            field.lon,
            field.lat,
            field,
            shading="auto",
            cmap=veg_cmap if index < 2 else diff_cmap,
            norm=veg_norm if index < 2 else diff_norm,
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        format_map(ax)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", loc="left", pad=5)
        meshes.append(mesh)

    axes[2].plot(
        ds.southern_100mm_lon,
        ds.southern_100mm_lat,
        color="#2166ac",
        linestyle="--",
        linewidth=2.0,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )
    axes[2].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#2166ac",
                linestyle="--",
                linewidth=2.0,
                label=r"100 mm yr$^{-1}$",
            )
        ],
        loc="upper right",
        frameon=False,
        facecolor="white",
        prop={"family": FONT, "size": FONT_SIZE - 3, "weight": "normal"},
    )

    fig.canvas.draw()
    positions = [ax.get_position() for ax in axes]
    colorbar_x = max(position.x1 for position in positions) + 0.018
    cax1 = fig.add_axes([colorbar_x, positions[1].y0, 0.018, positions[0].y1 - positions[1].y0])
    cb1 = fig.colorbar(
        meshes[0],
        cax=cax1,
        orientation="vertical",
        boundaries=veg_levels,
        ticks=np.linspace(0, 100, 6),
        extend="both",
    )
    cb1.set_label("%", labelpad=8)
    cb1.outline.set_linewidth(BORDER_WIDTH)

    cax2 = fig.add_axes([colorbar_x, positions[2].y0, 0.018, positions[2].height])
    cb2 = fig.colorbar(
        meshes[2],
        cax=cax2,
        orientation="vertical",
        boundaries=diff_levels,
        ticks=np.linspace(diff_levels[0], diff_levels[-1], 6),
        extend="both",
    )
    cb2.set_label("%", labelpad=8)
    cb2.outline.set_linewidth(BORDER_WIDTH)

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = OUTPUT_DIR / "figure1_southern_100mm_isohyet_only"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
