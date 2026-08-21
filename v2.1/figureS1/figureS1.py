from __future__ import annotations

"""Plot revised three-panel Figure R2 with 850-hPa wind vectors."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "figureS1_data.nc"
OUTPUT_DIR = HERE / "outputs"
EXTENT = (-100.0, 45.0, 0.0, 40.0)
MDR_BOX = (-70.0, -17.5, 10.0, 20.0)
FONT = "Times New Roman"
FONT_SIZE = 20
PANEL_TITLE_SIZE = 30
BORDER_WIDTH = 0.75
AXIS_FRAME_WIDTH = 1.25
MAJOR_TICK_WIDTH = 1.25
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

# Wind-vector controls. Absolute climatologies use a 10 m s-1 key. The
# CTL-minus-ERA5 difference uses a 5 m s-1 key.
ABS_WIND_KEY = 10.0
DIFF_WIND_KEY = 5.0
QUIVER_KEY_X = 0.93
QUIVER_KEY_Y = 1.06
QUIVER_MAX_ARROWS = 3000
QUIVER_MIN_STEP = 2

COLORBAR_GAP = 0.018
COLORBAR_WIDTH = 0.020


def build_ticks(
    start: float, stop: float, major_interval: float, minor_interval: float
) -> tuple[np.ndarray, np.ndarray]:
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
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 20,
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.linewidth": AXIS_FRAME_WIDTH,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def format_map(ax) -> None:
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeee", edgecolor="none", zorder=0)
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"), linewidth=BORDER_WIDTH, edgecolor="black", zorder=6
    )
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
    ax.tick_params(
        which="major",
        direction="out",
        length=5,
        width=MAJOR_TICK_WIDTH,
        top=False,
        right=False,
        labeltop=False,
        labelright=False,
        labelbottom=True,
        labelleft=True,
        labelsize=FONT_SIZE,
    )
    ax.tick_params(
        which="minor",
        direction="out",
        length=2.5,
        width=BORDER_WIDTH,
        top=False,
        right=False,
        labeltop=False,
        labelright=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.gridlines(
        draw_labels=False,
        linewidth=GRID_WIDTH,
        color="0.65",
        alpha=0.85,
        linestyle="--",
        xlocs=LON_MAJOR_TICKS,
        ylocs=LAT_MAJOR_TICKS,
    )
    ax.add_patch(
        Rectangle(
            (MDR_BOX[0], MDR_BOX[2]),
            MDR_BOX[1] - MDR_BOX[0],
            MDR_BOX[3] - MDR_BOX[2],
            fill=False,
            edgecolor="black",
            linewidth=BORDER_WIDTH,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )
    )


def add_wind_vectors(ax, u: xr.DataArray, v: xr.DataArray, key_value: float) -> None:
    """Overlay automatically thinned vectors using the requested R8 style."""
    visible_u = u.sel(lon=slice(EXTENT[0], EXTENT[1]), lat=slice(EXTENT[2], EXTENT[3]))
    visible_v = v.sel(lon=slice(EXTENT[0], EXTENT[1]), lat=slice(EXTENT[2], EXTENT[3]))
    lon = visible_u["lon"].values
    lat = visible_u["lat"].values
    u_values = visible_u.values
    v_values = visible_v.values
    # Match the reference-script thinning rule using the full source grid.
    # For the present 96 x 144 grid this gives a step of three.
    step = max(QUIVER_MIN_STEP, int(np.ceil(np.sqrt(np.size(u) / QUIVER_MAX_ARROWS))))
    mag95 = float(np.nanpercentile(np.hypot(u_values, v_values), 95))
    quiver = ax.quiver(
        lon[::step],
        lat[::step],
        u_values[::step, ::step],
        v_values[::step, ::step],
        transform=ccrs.PlateCarree(),
        color="black",
        pivot="middle",
        scale=max(0.1, mag95 * 8.0) * 1.3,
        width=0.0025,
        headwidth=12,
        headlength=8.5,
        headaxislength=2,
        zorder=5.8,
    )
    ax.quiverkey(
        quiver,
        X=QUIVER_KEY_X,
        Y=QUIVER_KEY_Y,
        U=key_value,
        label=rf"{key_value:g} m s$^{{-1}}$",
        labelpos="N",
        labelsep=0.05,
        coordinates="axes",
        fontproperties={"family": FONT, "weight": "normal", "size": FONT_SIZE - 1},
    )


def panel(ax, field, u, v, levels, cmap, title, wind_key):
    lon2, lat2 = np.meshgrid(field.lon, field.lat)
    plot_cmap = plt.get_cmap(cmap, len(levels) - 1) if isinstance(cmap, str) else cmap
    mesh = ax.contourf(
        lon2,
        lat2,
        field,
        levels=levels,
        cmap=plot_cmap,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.contour(
        lon2,
        lat2,
        field,
        levels=levels,
        colors="0.25",
        linewidths=0.35,
        alpha=0.55,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    format_map(ax)
    add_wind_vectors(ax, u, v, wind_key)
    ax.text(
        0,
        1.055,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_TITLE_SIZE,
        fontfamily=FONT,
        fontweight="bold",
        clip_on=False,
    )
    return mesh


def add_vertical_colorbar(fig, mesh, cax, levels, unit, label_every: int = 1) -> None:
    cb = fig.colorbar(
        mesh, cax=cax, orientation="vertical", extend="both", spacing="uniform", ticks=levels
    )
    if label_every > 1:
        tick_labels = [
            f"{value:g}".replace("-", "\N{MINUS SIGN}") if index % label_every == 0 else ""
            for index, value in enumerate(levels)
        ]
        cb.set_ticks(levels, labels=tick_labels)
    cb.ax.set_aspect("auto")
    cb.outline.set_linewidth(BORDER_WIDTH)
    cb.ax.tick_params(
        which="major",
        direction="out",
        length=3.5,
        width=BORDER_WIDTH,
        left=False,
        right=True,
        labelleft=False,
        labelright=True,
        labelsize=FONT_SIZE,
    )
    cb.ax.minorticks_off()
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.yaxis.set_label_position("right")
    cb.set_label(unit, labelpad=10, fontsize=FONT_SIZE, fontweight="normal")


def main() -> None:
    plt.switch_backend("Agg")
    setup_style()
    with xr.open_dataset(DATA_FILE) as source:
        ds = source.load()

    fig = plt.figure(figsize=(10.4, 10.2), dpi=300)
    gs = fig.add_gridspec(3, 1, left=0.075, right=0.835, top=0.96, bottom=0.055, hspace=0.28)
    axes = [fig.add_subplot(gs[index, 0], projection=ccrs.PlateCarree()) for index in range(3)]
    abs_levels = [0, 1, 2, 3, 4, 6, 8, 10, 12]
    bias_levels = [-4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4]
    bias_ticks = [-4, -2, 0, 2, 4]
    base_bias_cmap = plt.get_cmap("RdBu", 8)
    bias_colors = [base_bias_cmap(index) for index in range(4)]
    bias_colors += ["#ffffff"]
    bias_colors += [base_bias_cmap(index) for index in range(4, 8)]
    bias_cmap = ListedColormap(bias_colors, name="RdBu_blank_pm0p5")

    abs_mesh = panel(
        axes[0],
        ds["gpcp_on_model"],
        ds["era5_u850"],
        ds["era5_v850"],
        abs_levels,
        "YlGnBu",
        "(a) OBS",
        ABS_WIND_KEY,
    )
    panel(
        axes[1],
        ds["ctl_prect"],
        ds["ctl_u850"],
        ds["ctl_v850"],
        abs_levels,
        "YlGnBu",
        "(b) CTL",
        ABS_WIND_KEY,
    )
    bias_mesh = panel(
        axes[2],
        ds["ctl_minus_gpcp"],
        ds["ctl_minus_era5_u850"],
        ds["ctl_minus_era5_v850"],
        bias_levels,
        bias_cmap,
        "(c) CTL - OBS",
        DIFF_WIND_KEY,
    )

    # Use the rendered geographic-axis positions so both vertical colorbars
    # align exactly with the map panels despite Cartopy's fixed aspect ratio.
    fig.canvas.draw()
    positions = [ax.get_position() for ax in axes]
    cbar_x = max(position.x1 for position in positions) + COLORBAR_GAP
    shared_cax = fig.add_axes(
        [cbar_x, positions[1].y0, COLORBAR_WIDTH, positions[0].y1 - positions[1].y0]
    )
    bias_cax = fig.add_axes([cbar_x, positions[2].y0, COLORBAR_WIDTH, positions[2].height])
    add_vertical_colorbar(fig, abs_mesh, shared_cax, abs_levels, r"(mm day$^{-1}$)")
    add_vertical_colorbar(fig, bias_mesh, bias_cax, bias_ticks, r"(mm day$^{-1}$)")

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = OUTPUT_DIR / "figureS1"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
