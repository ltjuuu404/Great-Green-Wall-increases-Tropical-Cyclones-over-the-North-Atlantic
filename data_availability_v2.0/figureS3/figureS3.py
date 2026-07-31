from __future__ import annotations

"""Plot the coupled-versus-noncoupled comparison from the local NetCDF only."""

import re
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "figureS3_data.nc"
OUTPUT_DIR = HERE
OUTPUT_STEM = "figureS3"

# User-adjustable appearance and map settings.
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 10
FIGSIZE = (14.5, 7.6)
FIGURE_DPI = 300
EXPORT_DPI = 600
EXTENT = (-100.0, 45.0, 0.0, 40.0)
LON_TICKS = np.arange(-100.0, 41.0, 20.0)
LAT_TICKS = np.arange(0.0, 41.0, 10.0)
MDR_BOX = (-70.0, -17.5, 10.0, 20.0)
BORDER_WIDTH = 0.75
GRID_WIDTH = 0.5
MAX_QUIVER_ARROWS = 3000
MIN_QUIVER_STEP = 2
QUIVER_KEY = 1.0
SIGNIFICANCE_LEVEL = 0.05

PRECIP_LEVELS = [-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]
WIND850_LEVELS = [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]
WIND200_LEVELS = [-4.0, -2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 4.0]
PRECIP_COLORS = [
    "#6b3d0c", "#a6611a", "#dfc27d", "#f6e8c3",
    "#f5f5f5", "#c7eae5", "#5ab4ac", "#018571",
]
WIND_COLORS = [
    "#2166ac", "#4393c3", "#92c5de", "#f7f7f7",
    "#fddbc7", "#ef8a62", "#d6604d", "#b2182b",
]


def configure_style() -> None:
    """Use Times New Roman and keep all text normal except panel titles."""
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.weight": "normal",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.labelweight": "normal",
            "axes.titlesize": FONT_SIZE,
            "axes.titleweight": "normal",
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.linewidth": BORDER_WIDTH,
            "hatch.color": "white",
            "hatch.linewidth": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def validate_dataset(ds: xr.Dataset) -> None:
    required = {"lon", "lat"}
    fields = (
        "prect_diff", "u850_diff", "v850_diff", "wspd850_diff",
        "u200_diff", "v200_diff", "wspd200_diff",
    )
    required.update(f"{prefix}_{field}" for prefix in ("noncoupled", "coupled", "delta") for field in fields)
    required.update(
        f"{prefix}_{field}"
        for prefix in ("noncoupled", "coupled")
        for field in ("prect_p", "wspd850_p", "wspd200_p")
    )
    missing = sorted(required.difference(ds.variables))
    if missing:
        raise KeyError("Missing variables in figureS3_data.nc: " + ", ".join(missing))


def add_mdr_box(ax) -> None:
    lon0, lon1, lat0, lat1 = MDR_BOX
    ax.add_patch(
        Rectangle(
            (lon0, lat0), lon1 - lon0, lat1 - lat0,
            fill=False, edgecolor="black", linewidth=BORDER_WIDTH,
            transform=ccrs.PlateCarree(), zorder=8,
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
    ax.tick_params(axis="both", which="major", direction="out", length=3.0, width=BORDER_WIDTH, top=False, right=False)
    ax.tick_params(axis="both", which="minor", length=0, width=0, top=False, right=False)
    ax.minorticks_off()
    gridliner = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False, linewidth=GRID_WIDTH,
        color="0.65", alpha=0.85, linestyle="--",
        xlocs=LON_TICKS, ylocs=LAT_TICKS,
    )
    gridliner.n_steps = 90
    try:
        ax.spines["geo"].set_linewidth(BORDER_WIDTH)
        ax.spines["geo"].set_edgecolor("black")
    except (KeyError, AttributeError):
        pass
    add_mdr_box(ax)


def add_significance_hatching(ax, lon: np.ndarray, lat: np.ndarray, p: np.ndarray | None) -> None:
    if p is None:
        return
    significant = np.asarray(p, dtype=float) < SIGNIFICANCE_LEVEL
    if not np.any(significant):
        return
    lon2, lat2 = np.meshgrid(lon, lat)
    hatch = ax.contourf(
        lon2, lat2, np.where(significant, 1.0, np.nan),
        levels=[0.5, 1.5], colors="none", hatches=["////"],
        transform=ccrs.PlateCarree(), zorder=5, antialiased=False,
    )
    hatch.set_edgecolor("white")
    hatch.set_linewidth(0.0)


def add_wind_vectors(ax, lon: np.ndarray, lat: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
    step = max(MIN_QUIVER_STEP, int(np.ceil(np.sqrt(np.size(u) / MAX_QUIVER_ARROWS))))
    magnitude = np.hypot(u, v)
    finite = magnitude[np.isfinite(magnitude)]
    mag95 = float(np.nanpercentile(finite, 95.0)) if finite.size else 0.0
    scale = max(0.1, mag95 * 8.0) * 1.3
    q = ax.quiver(
        lon[::step], lat[::step], u[::step, ::step], v[::step, ::step],
        transform=ccrs.PlateCarree(), color="black", pivot="middle",
        scale=scale, width=0.0025, headwidth=12, headlength=8.5,
        headaxislength=2, zorder=5.8,
    )
    ax.quiverkey(
        q, X=0.90, Y=1.03, U=QUIVER_KEY,
        label=rf"{QUIVER_KEY:g} m s$^{{-1}}$", labelpos="N", labelsep=0.05,
        coordinates="axes",
        fontproperties={"family": FONT_FAMILY, "weight": "normal", "size": FONT_SIZE - 1},
    )


def add_colorbar(fig, ax, mesh, levels: list[float], unit: str, tick_labels: list[str]) -> None:
    cax = inset_axes(
        ax, width="80%", height="5.5%", loc="lower center",
        bbox_to_anchor=(0.0, -0.285, 1.0, 1.0),
        bbox_transform=ax.transAxes, borderpad=0.0,
    )
    cbar = fig.colorbar(
        mesh, cax=cax, orientation="horizontal", extend="both",
        spacing="uniform", ticks=levels,
    )
    cbar.ax.set_xticklabels(tick_labels, fontweight="normal")
    cbar.ax.tick_params(axis="x", which="major", direction="out", length=2.5, width=BORDER_WIDTH, top=False, bottom=True)
    cbar.ax.tick_params(axis="x", which="minor", length=0, width=0, top=False, bottom=False)
    cbar.ax.minorticks_off()
    cbar.outline.set_linewidth(BORDER_WIDTH)
    cbar.set_label(unit, fontweight="normal", fontsize=FONT_SIZE, labelpad=2)


def draw_panel(
    fig, ax, lon: np.ndarray, lat: np.ndarray, field: np.ndarray,
    p_value: np.ndarray | None, u: np.ndarray, v: np.ndarray,
    levels: list[float], colors: list[str], tick_labels: list[str],
    unit: str, title: str,
) -> None:
    lon2, lat2 = np.meshgrid(lon, lat)
    mesh = ax.contourf(
        lon2, lat2, field, levels=levels, cmap=ListedColormap(colors),
        extend="both", transform=ccrs.PlateCarree(), zorder=2,
    )
    ax.contour(
        lon2, lat2, field, levels=levels, colors="0.25", linewidths=0.30,
        alpha=0.45, transform=ccrs.PlateCarree(), zorder=3,
    )
    format_map(ax)
    add_significance_hatching(ax, lon, lat, p_value)
    add_wind_vectors(ax, lon, lat, u, v)
    ax.text(
        0.0, 1.055, title, transform=ax.transAxes, ha="left", va="bottom",
        fontsize=FONT_SIZE+4, fontweight="bold", clip_on=False,
    )
    add_colorbar(fig, ax, mesh, levels, unit, tick_labels)


def enforce_text_policy(fig) -> None:
    """Ensure that only subplot titles beginning with (a)-(i) are bold."""
    panel_pattern = re.compile(r"^\([a-i]\)\s")
    fig.canvas.draw()
    for artist in fig.findobj(match=Text):
        artist.set_fontfamily(FONT_FAMILY)
        artist.set_fontweight("bold" if panel_pattern.match(artist.get_text() or "") else "normal")


def main() -> None:
    configure_style()
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Required local input does not exist: {DATA_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(DATA_FILE) as source:
        ds = source.load()
    validate_dataset(ds)
    lon = np.asarray(ds["lon"].values)
    lat = np.asarray(ds["lat"].values)

    specs = [
        (
            "prect_diff", "u850_diff", "v850_diff", "prect_p",
            PRECIP_LEVELS, PRECIP_COLORS,
            ["-2", "-1", "-0.5", "-0.1", "0", "0.1", "0.5", "1", "2"],
            r"(mm day$^{-1}$)", "Precp + 850 wind",
        ),
        (
            "wspd850_diff", "u850_diff", "v850_diff", "wspd850_p",
            WIND850_LEVELS, WIND_COLORS,
            ["-2", "-1", "-0.5", "-0.25", "0", "0.25", "0.5", "1", "2"],
            r"(m s$^{-1}$)", "850 wind speed",
        ),
        (
            "wspd200_diff", "u200_diff", "v200_diff", "wspd200_p",
            WIND200_LEVELS, WIND_COLORS,
            ["-4", "-2", "-1", "-0.25", "0", "0.25", "1", "2", "4"],
            r"(m s$^{-1}$)", "200 wind speed",
        ),
    ]
    columns = (("noncoupled", "Non-coupled"), ("coupled", "Coupled"), ("delta", "C - NC"))

    fig = plt.figure(figsize=FIGSIZE, dpi=FIGURE_DPI)
    grid = fig.add_gridspec(
        3, 3, left=0.045, right=0.985, top=0.85, bottom=0.04,
        wspace=0.18, hspace=0.10,
    )
    axes = [[fig.add_subplot(grid[row, col], projection=ccrs.PlateCarree()) for col in range(3)] for row in range(3)]

    panel_number = 0
    for row, (field, u_name, v_name, p_name, levels, colors, ticks, unit, row_title) in enumerate(specs):
        for col, (prefix, column_title) in enumerate(columns):
            p_value = None if prefix == "delta" else np.asarray(ds[f"{prefix}_{p_name}"].values)
            draw_panel(
                fig, axes[row][col], lon, lat,
                np.asarray(ds[f"{prefix}_{field}"].values), p_value,
                np.asarray(ds[f"{prefix}_{u_name}"].values),
                np.asarray(ds[f"{prefix}_{v_name}"].values),
                levels, colors, ticks, unit,
                f"({chr(ord('a') + panel_number)}) {column_title}: {row_title}",
            )
            panel_number += 1

    enforce_text_policy(fig)
    output_paths = [OUTPUT_DIR / f"{OUTPUT_STEM}.{suffix}" for suffix in ("png", "pdf", "svg")]
    fig.savefig(output_paths[0], dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(output_paths[1], bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_paths[2], bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Input: {DATA_FILE}")
    for path in output_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
