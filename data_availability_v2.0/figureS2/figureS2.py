from __future__ import annotations

"""Draw the SST-current-precipitation mechanism figure from one NC file.

Data input
----------
Only ``figureS2_data.nc`` in this directory is opened. This script does not
import any project plotting/helper module and does not read CSV, NPY, NPZ, or
model-history files.
"""

import math
import re
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy import stats
from scipy.interpolate import griddata
import xarray as xr


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_FILE = PACKAGE_DIR / "figureS2_data.nc"
OUTPUT_DIR = PACKAGE_DIR

EXTENT = (-100.0, 45.0, 0.0, 40.0)
NORTH_SST_BOX = {"lon_min": -80.0, "lon_max": -20.0, "lat_min": 10.0, "lat_max": 20.0}
SOUTH_SST_BOX = {"lon_min": -80.0, "lon_max": -20.0, "lat_min": 0.0, "lat_max": 10.0}
SAHEL_BOX = {"lon_min": -20.0, "lon_max": 40.0, "lat_min": 10.0, "lat_max": 20.0}

NORTH_COLOR = "#2166ac"
SOUTH_COLOR = "#d95f02"
SAHEL_COLOR = "#7b3294"

FONT_SIZE = 19
BORDER_WIDTH = 0.75
FRAME_WIDTH = 1.20
GRID_WIDTH = 0.5
LON_MAJOR_TICKS = np.arange(-90.0, 31.0, 30.0)
LON_MINOR_TICKS = np.arange(-75.0, 46.0, 30.0)
LAT_MAJOR_TICKS = np.arange(0.0, 41.0, 20.0)
LAT_MINOR_TICKS = np.arange(10.0, 40.0, 20.0)
PANEL_TITLE_RE = re.compile(r"^\s*\([A-Za-z0-9]+\)(?:\s|$)")

SST_LEVELS = [-0.8, -0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
ADV_LEVELS = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]
PRECT_LEVELS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]


def setup_style() -> None:
    """Use Times New Roman and bold only for subplot titles."""
    plt.switch_backend("Agg")
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "normal",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE-4,
            "axes.labelweight": "normal",
            "axes.titlesize": FONT_SIZE-4,
            "axes.titleweight": "normal",
            "xtick.labelsize": FONT_SIZE-4,
            "ytick.labelsize": FONT_SIZE-4,
            "legend.fontsize": FONT_SIZE-4,
            "axes.linewidth": FRAME_WIDTH,
            "xtick.major.width": BORDER_WIDTH,
            "ytick.major.width": BORDER_WIDTH,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "hatch.color": "white",
            "hatch.linewidth": 0.35,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_figure_data() -> xr.Dataset:
    required = [
        "tlat",
        "tlon",
        "sst_diff",
        "sst_p",
        "adv_diff",
        "adv_p",
        "ulat",
        "ulon",
        "u_diff",
        "v_diff",
        "prect_diff",
        "prect_p",
        "delta_sst_gradient",
        "delta_sahel_precip",
    ]
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing extracted figure data: {DATA_FILE}")
    with xr.open_dataset(DATA_FILE) as source:
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"Missing variables in {DATA_FILE.name}: {', '.join(missing)}")
        return source.load()


def normalize_longitude(lon: np.ndarray) -> np.ndarray:
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def regrid_curvilinear(
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    field: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    lon = normalize_longitude(source_lon)
    valid = (
        np.isfinite(lon)
        & np.isfinite(source_lat)
        & np.isfinite(field)
        & (lon >= EXTENT[0] - 5.0)
        & (lon <= EXTENT[1] + 5.0)
        & (source_lat >= EXTENT[2] - 5.0)
        & (source_lat <= EXTENT[3] + 5.0)
    )
    if not np.any(valid):
        raise ValueError("No finite source points are available for map regridding.")
    target_lon_2d, target_lat_2d = np.meshgrid(target_lon, target_lat)
    return griddata(
        (lon[valid], np.asarray(source_lat)[valid]),
        np.asarray(field)[valid],
        (target_lon_2d, target_lat_2d),
        method=method,
    )


def add_panel_title(ax, label: str, title: str) -> None:
    ax.text(
        0.0,
        1.03,
        f"{label} {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE,
        fontweight="bold",
        clip_on=False,
        zorder=30,
    )


def add_region_box(
    ax,
    box: dict[str, float],
    color: str,
    label: str | None = None,
) -> None:
    ax.add_patch(
        Rectangle(
            (box["lon_min"], box["lat_min"]),
            box["lon_max"] - box["lon_min"],
            box["lat_max"] - box["lat_min"],
            fill=False,
            edgecolor=color,
            linewidth=1.15,
            transform=ccrs.PlateCarree(),
            zorder=22,
        )
    )
    if label:
        ax.text(
            box["lon_min"] + 1.0,
            box["lat_max"] + 0.7,
            label,
            color=color,
            fontsize=FONT_SIZE - 4,
            fontweight="normal",
            transform=ccrs.PlateCarree(),
            ha="left",
            va="bottom",
            zorder=23,
        )


def format_map(ax, fill_land: bool, box_mode: str) -> None:
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")
    if fill_land:
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#eeeeee", edgecolor="none", zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=BORDER_WIDTH, edgecolor="black", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.35, edgecolor="black", zorder=5)
    ax.set_xticks(LON_MAJOR_TICKS, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_MAJOR_TICKS, crs=ccrs.PlateCarree())
    ax.set_xticks(LON_MINOR_TICKS, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_MINOR_TICKS, minor=True, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=FONT_SIZE-4,
        width=BORDER_WIDTH,
        length=3.0,
        top=False,
        right=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=2.0,
        width=0.65,
        labelbottom=False,
        labelleft=False,
        top=False,
        right=False,
    )
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=LON_MAJOR_TICKS,
        ylocs=LAT_MAJOR_TICKS,
        linestyle="--",
        linewidth=GRID_WIDTH,
        color="#b8b8b8",
        alpha=0.9,
        zorder=1,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(FRAME_WIDTH)
        spine.set_edgecolor("black")

    if box_mode in {"sst", "all"}:
        add_region_box(ax, NORTH_SST_BOX, NORTH_COLOR)
        add_region_box(ax, SOUTH_SST_BOX, SOUTH_COLOR)
    if box_mode in {"precip", "all"}:
        add_region_box(ax, SAHEL_BOX, SAHEL_COLOR)


def add_horizontal_colorbar(fig, ax, mesh, ticks: list[float], unit: str) -> None:
    label_ticks = ticks[::2]
    cax = inset_axes(
        ax,
        width="78%",
        height="6%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.28, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    colorbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
        ticks=label_ticks,
        spacing="uniform",
        extend="both",
    )
    colorbar.outline.set_linewidth(BORDER_WIDTH)
    colorbar.ax.tick_params(
        axis="x",
        which="major",
        direction="out",
        labelsize=FONT_SIZE-4,
        width=BORDER_WIDTH,
        length=2.0,
        pad=2,
        top=False,
        bottom=True,
    )
    colorbar.ax.tick_params(axis="x", which="minor", length=0.0, width=0.0)
    colorbar.ax.minorticks_off()
    colorbar.set_label(unit, fontsize=FONT_SIZE-4, fontweight="normal", labelpad=2)


def draw_significance(ax, lon: np.ndarray, lat: np.ndarray, p_values: np.ndarray) -> None:
    significant = np.where(np.asarray(p_values) <= 0.05, 1.0, np.nan)
    ax.contourf(
        lon,
        lat,
        significant,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["///"],
        transform=ccrs.PlateCarree(),
        zorder=3.5,
    )


def draw_map_panel(
    fig,
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    p_values: np.ndarray | None,
    levels: list[float],
    cmap_name: str,
    panel_label: str,
    title: str,
    unit: str,
    *,
    fill_land: bool = True,
    box_mode: str = "none",
) -> None:
    cmap = plt.get_cmap(cmap_name, len(levels) - 1)
    norm = BoundaryNorm(levels, ncolors=len(levels) - 1, clip=False)
    mesh = ax.contourf(
        lon,
        lat,
        field,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.contour(
        lon,
        lat,
        field,
        levels=levels,
        colors="#3f3f3f",
        linewidths=0.25,
        alpha=0.45,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    format_map(ax, fill_land=fill_land, box_mode=box_mode)
    if p_values is not None:
        draw_significance(ax, lon, lat, p_values)
    add_panel_title(ax, panel_label, title)
    add_horizontal_colorbar(fig, ax, mesh, levels, unit)


def draw_current_vectors(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    u_ms: np.ndarray,
    v_ms: np.ndarray,
) -> None:
    mask = (
        (lon >= EXTENT[0])
        & (lon <= EXTENT[1])
        & (lat >= EXTENT[2])
        & (lat <= EXTENT[3])
        & np.isfinite(u_ms)
        & np.isfinite(v_ms)
    )
    valid_count = int(np.count_nonzero(mask))
    step = max(2, int(math.ceil(math.sqrt(max(valid_count, 1) / 450.0))))
    lon_q = lon[::step, ::step]
    lat_q = lat[::step, ::step]
    u_q = np.where(mask, u_ms * 100.0, np.nan)[::step, ::step]
    v_q = np.where(mask, v_ms * 100.0, np.nan)[::step, ::step]
    magnitude = np.hypot(u_q, v_q)
    magnitude_95 = float(np.nanpercentile(magnitude, 95)) if np.any(np.isfinite(magnitude)) else 1.0
    scale = max(0.1, magnitude_95 * 8.0) * 1.3 * 2.2
    quiver = ax.quiver(
        lon_q,
        lat_q,
        u_q,
        v_q,
        transform=ccrs.PlateCarree(),
        pivot="middle",
        color="black",
        scale=scale,
        width=0.0025,
        headwidth=12,
        headlength=8.5,
        headaxislength=2,
        zorder=3.8,
    )
    key = ax.quiverkey(
        quiver,
        0.90,
        1.05,
        5.0,
        r"5 cm s$^{-1}$",
        labelpos="N",
        labelsep=0.05,
        coordinates="axes",
        zorder=30,
        fontproperties={"size": FONT_SIZE - 4, "weight": "normal", "family": "Times New Roman"},
    )
    key.set_clip_on(False)
    key.text.set_clip_on(False)


def calculate_gradient_statistics(dataset: xr.Dataset) -> dict[str, float | np.ndarray]:
    x = np.asarray(dataset["delta_sst_gradient"].values, dtype=float)
    y = np.asarray(dataset["delta_sahel_precip"].values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        raise ValueError("At least three finite paired years are required for panel (d).")
    pearson_r, pearson_p = stats.pearsonr(x, y)
    regression = stats.linregress(x, y)
    return {
        "x": x,
        "y": y,
        "r": float(pearson_r),
        "p": float(pearson_p),
        "slope": float(regression.slope),
        "intercept": float(regression.intercept),
    }


def draw_scatter_panel(ax, result: dict[str, float | np.ndarray]) -> None:
    x = np.asarray(result["x"], dtype=float)
    y = np.asarray(result["y"], dtype=float)
    ax.scatter(
        x,
        y,
        s=32,
        facecolor="#ca3542",
        edgecolor="black",
        linewidth=0.55,
        alpha=0.9,
        zorder=3,
    )
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    y_line = float(result["slope"]) * x_line + float(result["intercept"])
    ax.plot(x_line, y_line, color="black", linewidth=1.2, zorder=4)
    ax.axhline(0.0, color="#666666", linewidth=BORDER_WIDTH, linestyle="--", zorder=1)
    ax.axvline(0.0, color="#666666", linewidth=BORDER_WIDTH, linestyle="--", zorder=1)
    ax.grid(True, which="major", linestyle="--", linewidth=GRID_WIDTH, color="#b8b8b8", alpha=0.9)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=FONT_SIZE-4,
        width=BORDER_WIDTH,
        length=3.0,
        top=False,
        right=False,
    )
    ax.tick_params(axis="both", which="minor", length=0.0, width=0.0, top=False, right=False)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_linewidth(FRAME_WIDTH)
        spine.set_edgecolor("black")
    ax.set_xlabel(r"$\Delta$[(10-20$^\circ$N) - (0-10$^\circ$N) SST] (K)")
    ax.set_ylabel(r"$\Delta$Sahel precip (mm day$^{-1}$)")
    p_value = float(result["p"])
    p_text = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
    ax.text(
        0.04,
        0.95,
        f"r={float(result['r']):.2f}\n{p_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE-4,
        fontweight="normal",
    )
    add_panel_title(ax, "(d)", "SST gradient-precipitation link")


def enforce_text_policy(fig) -> None:
    """Make all text normal except labels beginning with a panel identifier."""
    fig.canvas.draw()
    for artist in fig.findobj(match=Text):
        artist.set_fontfamily("Times New Roman")
        artist.set_fontweight("bold" if PANEL_TITLE_RE.match(artist.get_text() or "") else "normal")


def main() -> None:
    setup_style()
    dataset = load_figure_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_lon = np.arange(EXTENT[0], EXTENT[1] + 0.01, 1.0)
    plot_lat = np.arange(EXTENT[2], EXTENT[3] + 0.01, 1.0)
    plot_lon_2d, plot_lat_2d = np.meshgrid(plot_lon, plot_lat)

    sst = regrid_curvilinear(dataset["tlon"].values, dataset["tlat"].values, dataset["sst_diff"].values, plot_lon, plot_lat)
    sst_p = regrid_curvilinear(
        dataset["tlon"].values,
        dataset["tlat"].values,
        dataset["sst_p"].values,
        plot_lon,
        plot_lat,
        method="nearest",
    )
    advection = regrid_curvilinear(
        dataset["tlon"].values,
        dataset["tlat"].values,
        dataset["adv_diff"].values,
        plot_lon,
        plot_lat,
    )
    advection_p = regrid_curvilinear(
        dataset["tlon"].values,
        dataset["tlat"].values,
        dataset["adv_p"].values,
        plot_lon,
        plot_lat,
        method="nearest",
    )
    current_u = regrid_curvilinear(
        dataset["ulon"].values,
        dataset["ulat"].values,
        dataset["u_diff"].values,
        plot_lon,
        plot_lat,
    )
    current_v = regrid_curvilinear(
        dataset["ulon"].values,
        dataset["ulat"].values,
        dataset["v_diff"].values,
        plot_lon,
        plot_lat,
    )

    fig = plt.figure(figsize=(10.2, 5.6), dpi=300)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.985,
        top=0.875,
        bottom=0.11,
        wspace=0.18,
        hspace=0.55,
    )
    ax_a = fig.add_subplot(grid[0, 0], projection=ccrs.PlateCarree())
    ax_b = fig.add_subplot(grid[0, 1], projection=ccrs.PlateCarree())
    ax_c = fig.add_subplot(grid[1, 0], projection=ccrs.PlateCarree())
    ax_d = fig.add_subplot(grid[1, 1])

    draw_map_panel(
        fig,
        ax_a,
        plot_lon_2d,
        plot_lat_2d,
        sst,
        sst_p,
        SST_LEVELS,
        "RdBu_r",
        "(a)",
        "SST and current response",
        "K",
        box_mode="sst",
    )
    draw_current_vectors(ax_a, plot_lon_2d, plot_lat_2d, current_u, current_v)

    draw_map_panel(
        fig,
        ax_b,
        plot_lon_2d,
        plot_lat_2d,
        advection,
        advection_p,
        ADV_LEVELS,
        "BrBG_r",
        "(b)",
        "Surface temperature advection",
        r"K month$^{-1}$",
    )

    precip_lon = dataset["atmos_lon"].values
    precip_lat = dataset["atmos_lat"].values
    draw_map_panel(
        fig,
        ax_c,
        precip_lon,
        precip_lat,
        dataset["prect_diff"].values,
        dataset["prect_p"].values,
        PRECT_LEVELS,
        "BrBG",
        "(c)",
        "Precipitation with SST contours",
        r"mm day$^{-1}$",
        fill_land=False,
        box_mode="all",
    )
    sst_on_atmosphere = regrid_curvilinear(
        dataset["tlon"].values,
        dataset["tlat"].values,
        dataset["sst_diff"].values,
        precip_lon,
        precip_lat,
    )
    precip_lon_2d, precip_lat_2d = np.meshgrid(precip_lon, precip_lat)
    ax_c.contour(
        precip_lon_2d,
        precip_lat_2d,
        sst_on_atmosphere,
        levels=[0.1, 0.2, 0.4, 0.6],
        colors="#9b1d20",
        linewidths=0.8,
        transform=ccrs.PlateCarree(),
        zorder=3.7,
    )
    ax_c.contour(
        precip_lon_2d,
        precip_lat_2d,
        sst_on_atmosphere,
        levels=[-0.4, -0.2, -0.1],
        colors="#225ea8",
        linewidths=0.8,
        linestyles="--",
        transform=ccrs.PlateCarree(),
        zorder=3.7,
    )

    regression = calculate_gradient_statistics(dataset)
    draw_scatter_panel(ax_d, regression)
    enforce_text_policy(fig)

    output_stem = OUTPUT_DIR / "figureS2"
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved {output_stem.with_suffix('.png')}")
    print(f"Panel (d): r={float(regression['r']):.3f}, p={float(regression['p']):.3g}")


if __name__ == "__main__":
    main()
