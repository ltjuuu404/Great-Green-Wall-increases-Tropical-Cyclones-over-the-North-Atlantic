from __future__ import annotations

"""Plot revised Figure R6 from prepared local data only."""

from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import xarray as xr


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "figureS4_data.nc"
STATS_FILE = HERE / "figureS4_statistics.csv"
OUTPUT_DIR = HERE / "outputs"
FONT = "Times New Roman"
FONT_SIZE = 8.0
BORDER_WIDTH = 0.75
GRID_COLOR = "#A7A7A7"
CTL_COLOR = "#2166AC"
GGW_COLOR = "#D73027"
NEUTRAL_COLOR = "#5A5A5A"
MDR_LON = (-70.0, -17.5)
MDR_LAT = (10.0, 20.0)
MAP_EXTENT = (-100.0, 40.0, 0.0, 50.0)
RNG_SEED = 20260814


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": FONT, "font.size": FONT_SIZE, "font.weight": "normal",
        "axes.labelweight": "normal", "axes.titleweight": "normal",
        "axes.linewidth": BORDER_WIDTH, "xtick.major.width": BORDER_WIDTH,
        "ytick.major.width": BORDER_WIDTH, "xtick.direction": "out",
        "ytick.direction": "out", "hatch.linewidth": 0.35,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def table_from_nc(ds: xr.Dataset, prefix: str) -> pd.DataFrame:
    column_key = f"{prefix}_columns"
    names = (
        ds.attrs[column_key].split("|")
        if column_key in ds.attrs
        else sorted(name.split("__", 1)[1] for name in ds.data_vars if name.startswith(f"{prefix}__"))
    )
    return pd.DataFrame({name: ds[f"{prefix}__{name}"].values for name in names})


def discrete_cmap(colors):
    cmap = ListedColormap(colors)
    cmap.set_under(colors[0]); cmap.set_over(colors[-1])
    return cmap


def map_axis(ax, title: str) -> None:
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F0F0F0", edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=BORDER_WIDTH, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black", linewidth=0.35, zorder=5)
    lon_ticks = np.arange(-100, 41, 40); lat_ticks = np.arange(0, 51, 10)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree()); ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels([f"{abs(int(x))}°W" if x < 0 else (f"{int(x)}°E" if x > 0 else "0°") for x in lon_ticks])
    ax.set_yticklabels([f"{int(y)}°N" if y > 0 else "0°" for y in lat_ticks])
    ax.tick_params(which="major", top=False, right=False, length=2.5, width=BORDER_WIDTH)
    ax.tick_params(which="minor", length=0, width=0)
    ax.gridlines(draw_labels=False, xlocs=lon_ticks, ylocs=lat_ticks,
                 linewidth=0.5, color=GRID_COLOR, linestyle="--", alpha=0.8)
    ax.add_patch(Rectangle((MDR_LON[0], MDR_LAT[0]), MDR_LON[1]-MDR_LON[0], MDR_LAT[1]-MDR_LAT[0],
                           facecolor="none", edgecolor="black", linewidth=BORDER_WIDTH,
                           transform=ccrs.PlateCarree(), zorder=7))
    ax.set_title(title, loc="left", pad=3, fontweight="bold")


def colorbar(fig, ax, mesh, ticks) -> None:
    cax = inset_axes(ax, width="80%", height="7.0%", loc="lower center",
                     bbox_to_anchor=(0, -0.45, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=ticks,
                      extend="both", spacing="uniform")
    cb.outline.set_linewidth(BORDER_WIDTH)
    cb.ax.tick_params(which="major", length=2, width=BORDER_WIDTH, pad=0)
    cb.ax.minorticks_off()
    cb.set_label(r"(systems JAS$^{-1}$ per 5° cell)", labelpad=2)


def style_axis(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(BORDER_WIDTH)
    ax.tick_params(which="major", top=False, right=False, length=2.5, width=BORDER_WIDTH)
    ax.tick_params(which="minor", length=0, width=0)
    ax.grid(True, which="major", color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.8)
    ax.grid(False, which="minor")


def violin_panel(ax, ctl, ggw, title, ylabel, p_value, seed) -> None:
    violin = ax.violinplot([ctl, ggw], positions=[0, 1], widths=0.72,
                           showmeans=False, showmedians=False, showextrema=False, bw_method=0.35)
    for body, color in zip(violin["bodies"], [CTL_COLOR, GGW_COLOR]):
        body.set_facecolor(color); body.set_edgecolor("black"); body.set_linewidth(BORDER_WIDTH); body.set_alpha(0.72)
    quartiles = np.percentile(np.vstack([ctl, ggw]), [25, 50, 75], axis=1)
    ax.vlines([0, 1], quartiles[0], quartiles[2], color="black", linewidth=3.2, zorder=3)
    ax.scatter([0, 1], quartiles[1], s=15, color="white", edgecolor="black", linewidth=0.55, zorder=4)
    rng = np.random.default_rng(seed)
    ax.scatter(rng.normal(0, 0.025, len(ctl)), ctl, s=9, color=CTL_COLOR,
               edgecolor="black", linewidth=0.3, alpha=0.82, zorder=2)
    ax.scatter(1+rng.normal(0, 0.025, len(ggw)), ggw, s=9, color=GGW_COLOR,
               edgecolor="black", linewidth=0.3, alpha=0.82, zorder=2)
    ax.scatter([0, 1], [np.mean(ctl), np.mean(ggw)], marker="D", s=26,
               color=[CTL_COLOR, GGW_COLOR], edgecolor="black", linewidth=0.65, zorder=5)
    ax.text(0.98, 0.98, f"CTL={np.mean(ctl):.1f}\nGGW={np.mean(ggw):.1f}\np={p_value:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5})
    ax.set_xticks([0, 1], ["CTL", "GGW"]); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=3, fontweight="bold")
    style_axis(ax)


def main() -> None:
    plt.switch_backend("Agg"); setup_style()
    with xr.open_dataset(DATA_FILE) as source:
        ds = source.load()
    stats = pd.read_csv(STATS_FILE).set_index("metric")
    fig = plt.figure(figsize=(183/25.4, 118/25.4), facecolor="white")
    outer_grid = fig.add_gridspec(2, 1, height_ratios=[0.62, 0.82], hspace=0.43)
    map_grid = outer_grid[0, 0].subgridspec(1, 3, wspace=0.2)
    lower_grid = outer_grid[1, 0].subgridspec(1, 2, wspace=0.55)
    map_axes = [fig.add_subplot(map_grid[0, i], projection=ccrs.PlateCarree()) for i in range(3)]
    sequential_levels = np.arange(0, 7.01, 1); sequential_ticks = sequential_levels
    sequential_cmap = discrete_cmap(["#FFF7BC", "#FEE391", "#FEC44F", "#FE9929", "#EC7014", "#CC4C02", "#8C2D04"])
    diff_levels = np.array([-1.5,-1,-0.6,-0.3,-0.1,0.1,0.3,0.6,1,1.5]); diff_ticks = [-1.5,-0.6,0,0.6,1.5]
    diff_cmap = discrete_cmap(["#053061", "#2166AC", "#67A9CF", "#D1E5F0", "#FFFFFF", "#FDDBC7", "#EF8A62", "#B2182B", "#67001F"])
    fields = [ds.ctl_mean_smoothed.values, ds.ggw_mean_smoothed.values, ds.difference_smoothed.values]
    titles = ["(a) CTL track density", "(b) GGW track density", "(c) GGW - CTL"]
    for i, (ax, field, title) in enumerate(zip(map_axes, fields, titles)):
        levels = sequential_levels if i < 2 else diff_levels
        cmap = sequential_cmap if i < 2 else diff_cmap
        mesh = ax.contourf(ds.longitude, ds.latitude, field, levels=levels, cmap=cmap,
                           norm=BoundaryNorm(levels, cmap.N), extend="both",
                           transform=ccrs.PlateCarree(), zorder=0)
        if i == 2:
            sig = ds.paired_t_p.values < 0.05
            if np.any(sig):
                hatch = ax.contourf(ds.longitude, ds.latitude, np.where(sig, 1.0, np.nan),
                                    levels=[0.5,1.5], colors="none", hatches=["////"],
                                    transform=ccrs.PlateCarree(), zorder=4)
                hatch.set_edgecolor("white"); hatch.set_linewidth(0)
        map_axis(ax, title); colorbar(fig, ax, mesh, sequential_ticks if i < 2 else diff_ticks)

    frequency_row = stats.loc["AEW frequency"]
    density_row = stats.loc["MDR AEW density"]
    violin_panel(fig.add_subplot(lower_grid[0,0]), ds.ctl_aew_frequency.values,
                 ds.ggw_aew_frequency.values, "(d) AEW frequency in MDR",
                 "AEWs per JAS", frequency_row.paired_permutation_p_two_sided, RNG_SEED)
    violin_panel(fig.add_subplot(lower_grid[0,1]), ds.ctl_mdr_aew_density.values,
                 ds.ggw_mdr_aew_density.values, "(e) AEW track density in MDR",
                 "AEW density in MDR per JAS", density_row.paired_permutation_p_two_sided, RNG_SEED+1)

    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.11, top=0.97)
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = OUTPUT_DIR / "figureS4"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
