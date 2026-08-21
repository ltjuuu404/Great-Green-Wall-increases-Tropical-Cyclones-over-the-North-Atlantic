from __future__ import annotations

"""Plot revised three-panel Figure R8 with zero-split discrete colors."""

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
DATA_FILE = HERE / "figureS3_data.nc"
OUTPUT_DIR = HERE / "outputs"
FONT = "Times New Roman"
FONT_SIZE = 16
BORDER_WIDTH = 0.75
EXTENT = (-100.0, 45.0, 0.0, 40.0)
MDR_BOX = (-70.0, -17.5, 10.0, 20.0)
LON_TICKS = np.arange(-100.0, 41.0, 20.0)
LAT_TICKS = np.arange(0.0, 41.0, 10.0)
LEVELS = [-2.0, -1.5, -1.0, -0.75, -0.5, -0.3, -0.1, -0.05,
          0.0, 0.05, 0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
NEGATIVE_COLORS = [
    "#6b3d0c", "#8c510a", "#a6611a", "#bf812d",
    "#dfc27d", "#ead89d", "#f2e5c4", "#f8f0dc",
]
POSITIVE_COLORS = [
    "#e5f5f3", "#d0ece8", "#b5e1dc", "#80cdc1",
    "#5ab4ac", "#359b92", "#16867d", "#01665e",
]
TITLES = [
    "(a) Uncoupled precipitation + 850 wind",
    "(b) Coupled: Precp + 850 wind",
    "(c) C - UC: precipitation + 850 wind",
]


def format_map(ax) -> None:
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeee", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=BORDER_WIDTH, edgecolor="black", zorder=6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="black", zorder=6)
    ax.add_patch(Rectangle((MDR_BOX[0], MDR_BOX[2]), MDR_BOX[1]-MDR_BOX[0], MDR_BOX[3]-MDR_BOX[2],
                           fill=False, edgecolor="black", linewidth=BORDER_WIDTH,
                           transform=ccrs.PlateCarree(), zorder=8))
    ax.set_xticks(LON_TICKS, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_TICKS, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(which="major", direction="out", length=3, width=BORDER_WIDTH, top=False, right=False)
    ax.tick_params(which="minor", length=0, width=0)
    ax.minorticks_off()
    ax.gridlines(draw_labels=False, linewidth=0.5, color="0.65", alpha=0.85,
                 linestyle="--", xlocs=LON_TICKS, ylocs=LAT_TICKS)


def draw_panel(fig, ax, lon, lat, field, p, u, v, title) -> None:
    lon2, lat2 = np.meshgrid(lon, lat)
    cmap = ListedColormap(NEGATIVE_COLORS + POSITIVE_COLORS, name="precip_zero_split")
    cmap.set_under(NEGATIVE_COLORS[0])
    cmap.set_over(POSITIVE_COLORS[-1])
    norm = BoundaryNorm(LEVELS, cmap.N)
    mesh = ax.contourf(lon2, lat2, field, levels=LEVELS, cmap=cmap, norm=norm,
                       extend="both", transform=ccrs.PlateCarree(), zorder=2)
    ax.contour(lon2, lat2, field, levels=LEVELS, colors="0.25", linewidths=0.30,
               alpha=0.45, transform=ccrs.PlateCarree(), zorder=3)
    if p is not None:
        sig = p < 0.05
        if np.any(sig):
            hatch = ax.contourf(lon2, lat2, np.where(sig, 1.0, np.nan), levels=[0.5, 1.5],
                                colors="none", hatches=["////"], transform=ccrs.PlateCarree(), zorder=5)
            hatch.set_edgecolor("white")
            hatch.set_linewidth(0)
    format_map(ax)
    step = max(2, int(np.ceil(np.sqrt(np.size(u) / 3000))))
    mag95 = float(np.nanpercentile(np.hypot(u, v), 95))
    q = ax.quiver(lon[::step], lat[::step], u[::step, ::step], v[::step, ::step],
                  transform=ccrs.PlateCarree(), color="black", pivot="middle",
                  scale=max(0.1, mag95 * 8.0) * 1.3, width=0.0025,
                  headwidth=12, headlength=8.5, headaxislength=2, zorder=5.8)
    ax.quiverkey(q, X=0.93, Y=1.06, U=1.0, label=r"1 m s$^{-1}$",
                 labelpos="N", labelsep=0.05, coordinates="axes",
                 fontproperties={"family": FONT, "weight": "normal", "size": FONT_SIZE-1})
    ax.text(0, 1.055, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FONT_SIZE+4, fontweight="bold", clip_on=False)
    cax = inset_axes(ax, width="90%", height="8.5%", loc="lower center",
                     bbox_to_anchor=(0, -0.24, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both",
                      spacing="uniform", ticks=LEVELS[::2])
    cb.outline.set_linewidth(BORDER_WIDTH)
    cb.ax.tick_params(which="major", direction="out", length=2.5, width=BORDER_WIDTH,
                      top=False, bottom=True)
    cb.ax.minorticks_off()
    cb.set_label(r"(mm day$^{-1}$)", labelpad=2)


def main() -> None:
    plt.switch_backend("Agg")
    mpl.rcParams.update({"font.family": FONT, "font.size": FONT_SIZE,
                         "font.weight": "normal", "axes.labelweight": "normal",
                         "axes.linewidth": BORDER_WIDTH, "hatch.linewidth": 0.35,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    with xr.open_dataset(DATA_FILE) as source:
        ds = source.load()
    lon, lat = ds.lon.values, ds.lat.values
    fig = plt.figure(figsize=(8.5, 10.0), dpi=300)
    gs = fig.add_gridspec(3, 1, left=0.065, right=0.88, top=0.955, bottom=0.035, hspace=0.70)
    prefixes = ("noncoupled", "coupled", "delta")
    for index, prefix in enumerate(prefixes):
        ax = fig.add_subplot(gs[index, 0], projection=ccrs.PlateCarree())
        p = None if prefix == "delta" else ds[f"{prefix}_prect_p"].values
        draw_panel(fig, ax, lon, lat, ds[f"{prefix}_prect_diff"].values, p,
                   ds[f"{prefix}_u850_diff"].values, ds[f"{prefix}_v850_diff"].values,
                   TITLES[index])
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = OUTPUT_DIR / "figureS3"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
