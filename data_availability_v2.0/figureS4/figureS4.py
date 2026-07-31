#!/usr/bin/env python3
"""Reproduce Supplementary Figure S4 from one self-contained NetCDF file."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_FILE = PACKAGE_DIR / "figureS4_data.nc"
OUTPUT_STEM = PACKAGE_DIR / "figureS4"

MAP_EXTENT = (-100.0, 40.0, 0.0, 50.0)
MDR_LON_BOUNDS = (-70.0, -17.5)
MDR_LAT_BOUNDS = (10.0, 20.0)
CTL_COLOR = "#2166AC"
GGW_COLOR = "#D73027"
NEUTRAL_COLOR = "#5A5A5A"
GRID_COLOR = "#A7A7A7"
FONT_SIZE = 8.0
LINE_WIDTH = 0.75
FIG_WIDTH_IN = 183.0 / 25.4
FIG_HEIGHT_IN = 118.0 / 25.4
RNG_SEED = 20260721

TICK_CONFIG = {
    "map_lon_min": -100,
    "map_lon_max": 40,
    "map_lon_step": 40,
    "map_lat_min": 0,
    "map_lat_max": 50,
    "map_lat_step": 10,
    "count_y_min": 25,
    "count_y_step": 5,
    "response_x_min": -20,
    "response_x_max": 20,
    "response_x_step": 10,
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.weight": "normal",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.linewidth": LINE_WIDTH,
            "xtick.major.width": LINE_WIDTH,
            "ytick.major.width": LINE_WIDTH,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "hatch.linewidth": 0.35,
        }
    )


def table_from_nc(dataset: xr.Dataset, prefix: str) -> pd.DataFrame:
    columns = dataset.attrs.get(f"{prefix}_columns", "")
    names = columns.split("|") if columns else sorted(
        name.split("__", 1)[1]
        for name in dataset.data_vars
        if name.startswith(f"{prefix}__")
    )
    return pd.DataFrame({name: dataset[f"{prefix}__{name}"].values for name in names})


def apply_text_policy(figure: mpl.figure.Figure) -> None:
    """Use Times New Roman and bold only labels that begin with a panel letter."""
    for artist in figure.findobj(match=Text):
        text = artist.get_text() or ""
        artist.set_fontfamily("Times New Roman")
        artist.set_fontweight("bold" if text.lstrip().startswith(tuple(f"({x})" for x in "abcdef")) else "normal")


def discrete_cmap(colors: list[str]) -> ListedColormap:
    cmap = ListedColormap(colors)
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    return cmap


def map_axis(ax, label: str) -> None:
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F0F0F0", edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=LINE_WIDTH, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black", linewidth=0.35, zorder=5)
    lon_ticks = np.arange(TICK_CONFIG["map_lon_min"], TICK_CONFIG["map_lon_max"] + 1, TICK_CONFIG["map_lon_step"])
    lat_ticks = np.arange(TICK_CONFIG["map_lat_min"], TICK_CONFIG["map_lat_max"] + 1, TICK_CONFIG["map_lat_step"])
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels([f"{abs(int(x))}°W" if x < 0 else (f"{int(x)}°E" if x > 0 else "0°") for x in lon_ticks])
    ax.set_yticklabels([f"{int(y)}°N" if y > 0 else "0°" for y in lat_ticks])
    ax.tick_params(top=False, right=False, labeltop=False, labelright=False, length=2.5, width=LINE_WIDTH)
    ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False, xlocs=lon_ticks, ylocs=lat_ticks,
        linewidth=0.5, color=GRID_COLOR, linestyle="--", alpha=0.8, zorder=1,
    )
    ax.set_title(label, loc="left", pad=3, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_linewidth(LINE_WIDTH)
        spine.set_edgecolor("black")


def add_mdr_box(ax) -> None:
    ax.add_patch(
        Rectangle(
            (MDR_LON_BOUNDS[0], MDR_LAT_BOUNDS[0]),
            MDR_LON_BOUNDS[1] - MDR_LON_BOUNDS[0],
            MDR_LAT_BOUNDS[1] - MDR_LAT_BOUNDS[0],
            facecolor="none", edgecolor="black", linewidth=LINE_WIDTH,
            transform=ccrs.PlateCarree(), zorder=7,
        )
    )


def add_horizontal_colorbar(fig, ax, mappable, ticks, unit: str) -> None:
    cax = inset_axes(
        ax, width="80%", height="7.5%", loc="lower center",
        bbox_to_anchor=(0.0, -0.45, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0,
    )
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks, extend="both")
    cbar.outline.set_linewidth(LINE_WIDTH)
    cbar.ax.tick_params(length=2.0, width=LINE_WIDTH, pad=0)
    cbar.set_label(unit, labelpad=2)


def style_cartesian_axis(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LINE_WIDTH)
        spine.set_color("black")
    ax.tick_params(top=False, right=False, width=LINE_WIDTH, length=2.5)
    ax.grid(True, which="major", axis="both", color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.8)


def make_figure(dataset: xr.Dataset) -> mpl.figure.Figure:
    annual = table_from_nc(dataset, "annual")
    metric_stats = table_from_nc(dataset, "statistics")
    monthly_stats = table_from_nc(dataset, "monthly_statistics")

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.0], hspace=0.50, wspace=0.65)
    map_axes = [fig.add_subplot(grid[0, i], projection=ccrs.PlateCarree()) for i in range(3)]

    sequential_levels = np.arange(0.0, 7.01, 1.0)
    sequential_ticks = np.arange(0.0, 7.01, 1.0)
    sequential_cmap = discrete_cmap(["#FFF7BC", "#FEE391", "#FEC44F", "#FE9929", "#EC7014", "#CC4C02", "#8C2D04"])
    sequential_norm = BoundaryNorm(sequential_levels, sequential_cmap.N)
    diff_levels = np.array([-1.5, -1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0, 1.5])
    diff_ticks = np.array([-1.5, -0.6, 0.0, 0.6, 1.5])
    diff_cmap = discrete_cmap(["#053061", "#2166AC", "#67A9CF", "#D1E5F0", "#FFFFFF", "#FDDBC7", "#EF8A62", "#B2182B", "#67001F"])
    diff_norm = BoundaryNorm(diff_levels, diff_cmap.N)

    lon = dataset["longitude"].values
    lat = dataset["latitude"].values
    fields = [dataset["ctl_mean_smoothed"].values, dataset["ggw_mean_smoothed"].values, dataset["difference_smoothed"].values]
    titles = ["(a) CTL track density", "(b) GGW track density", "(c) GGW - CTL"]
    for i, (ax, field, title) in enumerate(zip(map_axes, fields, titles)):
        if i < 2:
            mesh = ax.contourf(lon, lat, field, levels=sequential_levels, cmap=sequential_cmap, norm=sequential_norm, extend="both", transform=ccrs.PlateCarree(), zorder=0)
            add_horizontal_colorbar(fig, ax, mesh, sequential_ticks, r"(systems JAS$^{-1}$ per 5° cell)")
        else:
            mesh = ax.contourf(lon, lat, field, levels=diff_levels, cmap=diff_cmap, norm=diff_norm, extend="both", transform=ccrs.PlateCarree(), zorder=0)
            significance = dataset["paired_t_p"].values < 0.05
            if np.any(significance):
                hatch = ax.contourf(lon, lat, significance.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"], transform=ccrs.PlateCarree(), zorder=4)
                for collection in hatch.collections:
                    collection.set_edgecolor("white")
                    collection.set_linewidth(0.0)
            add_horizontal_colorbar(fig, ax, mesh, diff_ticks, r"(systems JAS$^{-1}$ per 5° cell)")
        map_axis(ax, title)
        add_mdr_box(ax)

    ax_d = fig.add_subplot(grid[1, 0])
    ctl_count = annual[annual["experiment"] == "CTL"].sort_values("pair_index")["count_total"].to_numpy(float)
    ggw_count = annual[annual["experiment"] == "GGW"].sort_values("pair_index")["count_total"].to_numpy(float)
    rng = np.random.default_rng(RNG_SEED)
    positions = [rng.normal(0.0, 0.025, len(ctl_count)), 1 + rng.normal(0.0, 0.025, len(ggw_count))]
    violin = ax_d.violinplot([ctl_count, ggw_count], positions=[0, 1], widths=0.72, showmeans=False, showmedians=False, showextrema=False, bw_method=0.35)
    for body, color in zip(violin["bodies"], [CTL_COLOR, GGW_COLOR]):
        body.set_facecolor(color); body.set_edgecolor("black"); body.set_linewidth(0.75); body.set_alpha(0.72)
    quartiles = np.percentile(np.vstack([ctl_count, ggw_count]), [25, 50, 75], axis=1)
    ax_d.vlines([0, 1], quartiles[0], quartiles[2], color="black", linewidth=3.2, zorder=3)
    ax_d.scatter([0, 1], quartiles[1], s=15, color="white", edgecolor="black", linewidth=0.55, zorder=4)
    ax_d.scatter(positions[0], ctl_count, s=9, color=CTL_COLOR, edgecolor="black", linewidth=0.3, alpha=0.82)
    ax_d.scatter(positions[1], ggw_count, s=9, color=GGW_COLOR, edgecolor="black", linewidth=0.3, alpha=0.82)
    ax_d.scatter([0, 1], [np.mean(ctl_count), np.mean(ggw_count)], marker="D", s=26, color=[CTL_COLOR, GGW_COLOR], edgecolor="black", linewidth=0.65, zorder=5)
    count_stat = metric_stats.loc[metric_stats["metric"] == "count_total"].iloc[0]
    ax_d.text(0.98, 0.98, f"CTL={np.mean(ctl_count):.2f}\nGGW={np.mean(ggw_count):.2f}\np={count_stat['paired_permutation_p']:.3f}", transform=ax_d.transAxes, ha="right", va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5})
    ax_d.set_xticks([0, 1], ["CTL", "GGW"])
    ymax = max(np.max(ctl_count), np.max(ggw_count)) + 3
    ax_d.set_ylim(TICK_CONFIG["count_y_min"], ymax)
    ax_d.set_yticks(np.arange(TICK_CONFIG["count_y_min"], ymax + 0.1, TICK_CONFIG["count_y_step"]))
    ax_d.set_ylabel("AEWs per JAS")
    ax_d.set_title("(d) Annual AEW number", loc="left", pad=3, fontweight="bold")
    style_cartesian_axis(ax_d)

    ax_e = fig.add_subplot(grid[1, 1])
    x = np.arange(3); width = 0.34
    ax_e.bar(x - width / 2, monthly_stats["ctl_mean"], width, yerr=monthly_stats["ctl_sem"], color=CTL_COLOR, edgecolor="black", linewidth=0.55, capsize=2, label="CTL")
    ax_e.bar(x + width / 2, monthly_stats["ggw_mean"], width, yerr=monthly_stats["ggw_sem"], color=GGW_COLOR, edgecolor="black", linewidth=0.55, capsize=2, label="GGW")
    for idx, row in monthly_stats.iterrows():
        top = max(row["ctl_mean"] + row["ctl_sem"], row["ggw_mean"] + row["ggw_sem"])
        ax_e.text(idx, top + 0.25, f"p={row['paired_permutation_p']:.2f}", ha="center", va="bottom")
    ax_e.set_xticks(x, ["Jul", "Aug", "Sep"])
    ax_e.set_ylabel("AEWs per month")
    ax_e.set_ylim(0, max((monthly_stats["ctl_mean"] + monthly_stats["ctl_sem"]).max(), (monthly_stats["ggw_mean"] + monthly_stats["ggw_sem"]).max()) + 1.5)
    ax_e.set_title("(e) Monthly AEW number", loc="left", pad=3, fontweight="bold")
    ax_e.legend(frameon=False, facecolor="white", loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    style_cartesian_axis(ax_e)

    ax_f = fig.add_subplot(grid[1, 2])
    forest = metric_stats[metric_stats["metric"] != "count_total"].copy().iloc[::-1].reset_index(drop=True)
    y = np.arange(len(forest))
    ax_f.axvline(0.0, color="black", linewidth=0.75, zorder=1)
    xerr = np.vstack([forest["percent_change"] - forest["percent_ci_low"], forest["percent_ci_high"] - forest["percent_change"]])
    colors = [GGW_COLOR if value >= 0 else CTL_COLOR for value in forest["percent_change"]]
    ax_f.errorbar(forest["percent_change"], y, xerr=xerr, fmt="none", ecolor=NEUTRAL_COLOR, elinewidth=0.8, capsize=2, zorder=2)
    ax_f.scatter(forest["percent_change"], y, c=colors, s=20, edgecolor="black", linewidth=0.45, zorder=3)
    short_labels = {"Atlantic-reaching": "Atlantic reach", "MDR-reaching": "MDR reach", "Duration": "Duration", "Track length": "Length", "Westward speed": "Speed", "Mean strength": "Mean CV", "Peak strength": "Peak CV"}
    for yi, row in forest.iterrows():
        ax_f.text(TICK_CONFIG["response_x_max"] - 0.5, yi, f"p={row['paired_permutation_p']:.2f}", ha="right", va="center")
    ax_f.set_yticks(y, [short_labels[label] for label in forest["label"]])
    ax_f.set_xlim(TICK_CONFIG["response_x_min"], TICK_CONFIG["response_x_max"])
    ax_f.set_xticks(np.arange(TICK_CONFIG["response_x_min"], TICK_CONFIG["response_x_max"] + 0.1, TICK_CONFIG["response_x_step"]))
    ax_f.set_xlabel("Change relative to CTL (%)")
    ax_f.set_title("(f) Seed-property response", loc="left", pad=3, fontweight="bold")
    style_cartesian_axis(ax_f)

    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.11, top=0.97)
    apply_text_policy(fig)
    return fig


def main() -> None:
    configure_style()
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Missing extracted figure data: {DATA_FILE}")
    with xr.open_dataset(DATA_FILE) as source:
        dataset = source.load()
    figure = make_figure(dataset)
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT_STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
