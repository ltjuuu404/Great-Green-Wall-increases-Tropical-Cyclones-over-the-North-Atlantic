#!/usr/bin/env python3
"""Plot the dust–radiation–TCGF figure from one self-contained NetCDF file.

Input
-----
figureS5_data.nc

The input file already contains the smoothed map fields, significance fields,
annual MDR TCGF counts, and the statistical summary used in the figure. This
script performs no extraction, smoothing, statistical testing, or raw-data
processing.
"""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# -----------------------------------------------------------------------------
# Files
# -----------------------------------------------------------------------------
WORK = Path(__file__).resolve().parent
INPUT_NC = WORK / "figureS5_data.nc"
OUTPUT_DIR = WORK
OUTPUT_STEM = "figureS5"


# -----------------------------------------------------------------------------
# Figure settings
# -----------------------------------------------------------------------------
EXTENT = (-100.0, 45.0, 0.0, 40.0)
MDR_BOX = (-70.0, -17.5, 10.0, 20.0)

TITLE_FONT_SIZE = 16
FONT_SIZE = TITLE_FONT_SIZE - 4
BORDER_WIDTH = 0.75
MAJOR_GRID_WIDTH = 0.5
CONTOUR_LINE_WIDTH = 0.35
CONTOUR_LINE_ALPHA = 0.55
SIGNIFICANCE_LEVEL = 0.05

LON_MAJOR_TICKS = np.arange(-90.0, 30.1, 30.0)
LON_MINOR_TICKS = np.arange(-75.0, 45.1, 30.0)
LAT_MAJOR_TICKS = np.arange(0.0, 40.1, 20.0)
LAT_MINOR_TICKS = np.arange(10.0, 30.1, 20.0)

COLORBAR_WIDTH = "80%"
COLORBAR_HEIGHT = "6%"
COLORBAR_Y_OFFSET = -0.30
COLORBAR_LABEL_PAD = 2

TCGF_Y_MIN = 0.0
TCGF_Y_MAJOR_INTERVAL = 5.0
TCGF_Y_MINOR_INTERVAL = 2.5
MAP_BOX_ASPECT = (EXTENT[3] - EXTENT[2]) / (EXTENT[1] - EXTENT[0])

REQUIRED_VARIABLES = {
    "aoddust_diff",
    "aoddust_p",
    "aoddust_ctl",
    "dustsf_diff_scaled",
    "dustsf_p",
    "fsns_diff",
    "fsns_p",
    "ts_ocean_diff",
    "ts_ocean_p",
    "tcgf_yearly__ctl_mdr_tcgf",
    "tcgf_yearly__ggw_mdr_tcgf",
    "tcgf_significance__ctl_mean",
    "tcgf_significance__ggw_mean",
    "tcgf_significance__mean_difference",
    "tcgf_significance__paired_permutation_p",
}


def setup_style() -> None:
    """Use Times New Roman and bold only the subplot titles."""
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
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
            "hatch.linewidth": 0.35,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_figure_data() -> xr.Dataset:
    """Load the only input file and verify its plotting variables."""
    if not INPUT_NC.is_file():
        raise FileNotFoundError(f"Missing figure input: {INPUT_NC}")

    with xr.open_dataset(INPUT_NC) as source:
        dataset = source.load()

    missing = sorted(REQUIRED_VARIABLES.difference(dataset.data_vars))
    if missing:
        raise KeyError(f"Missing required variables in {INPUT_NC.name}: {missing}")
    return dataset


def add_panel_title(ax, panel: str, title: str) -> None:
    """Place the only bold text element above a subplot."""
    ax.text(
        0.0,
        1.055,
        f"{panel} {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        clip_on=False,
        zorder=20,
    )


def add_mdr_box(ax) -> None:
    lon_min, lon_max, lat_min, lat_max = MDR_BOX
    ax.add_patch(
        Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            fill=False,
            edgecolor="black",
            linewidth=BORDER_WIDTH,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )
    )


def format_map(ax, draw_mdr_box: bool = True) -> None:
    """Apply the shared map extent, ticks, grid, coastlines, and border."""
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")
    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor="#eeeeee",
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=BORDER_WIDTH,
        edgecolor="black",
        zorder=6,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.35,
        edgecolor="black",
        zorder=6,
    )

    ax.set_xticks(LON_MAJOR_TICKS, crs=ccrs.PlateCarree())
    ax.set_xticks(LON_MINOR_TICKS, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_MAJOR_TICKS, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_MINOR_TICKS, minor=True, crs=ccrs.PlateCarree())
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
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=2.0,
        width=BORDER_WIDTH,
        top=False,
        right=False,
    )

    gridlines = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=MAJOR_GRID_WIDTH,
        color="0.65",
        alpha=0.85,
        linestyle="--",
        xlocs=LON_MAJOR_TICKS,
        ylocs=LAT_MAJOR_TICKS,
    )
    gridlines.n_steps = 90

    try:
        ax.spines["geo"].set_linewidth(BORDER_WIDTH)
        ax.spines["geo"].set_edgecolor("black")
    except (AttributeError, KeyError):
        pass

    if draw_mdr_box:
        add_mdr_box(ax)


def add_significance_hatching(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    p_values: np.ndarray,
) -> None:
    """Overlay white hatching where p is below 0.05."""
    significant = np.asarray(p_values, dtype=float) < SIGNIFICANCE_LEVEL
    if not np.any(significant):
        return

    lon_2d, lat_2d = np.meshgrid(lon, lat)
    hatch_field = np.where(significant, 1.0, np.nan)
    hatching = ax.contourf(
        lon_2d,
        lat_2d,
        hatch_field,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["////"],
        transform=ccrs.PlateCarree(),
        zorder=4,
        antialiased=False,
    )
    hatching.set_edgecolor("white")
    hatching.set_linewidth(0.0)


def add_horizontal_colorbar(
    fig,
    ax,
    mappable,
    ticks: list[float],
    unit_label: str,
    tick_labels: list[str] | None = None,
) -> None:
    """Add a discrete horizontal colorbar below one map panel."""
    colorbar_ax = inset_axes(
        ax,
        width=COLORBAR_WIDTH,
        height=COLORBAR_HEIGHT,
        loc="lower center",
        bbox_to_anchor=(0.0, COLORBAR_Y_OFFSET, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    colorbar = fig.colorbar(
        mappable,
        cax=colorbar_ax,
        orientation="horizontal",
        extend="both",
        spacing="uniform",
        ticks=ticks,
    )
    colorbar.outline.set_linewidth(BORDER_WIDTH)
    colorbar.ax.tick_params(
        axis="x",
        which="major",
        direction="out",
        length=2.5,
        width=BORDER_WIDTH,
        top=False,
        bottom=True,
        labelsize=FONT_SIZE,
        pad=1.5,
    )
    colorbar.ax.tick_params(
        axis="x",
        which="minor",
        length=0,
        width=0,
        top=False,
        bottom=False,
    )
    colorbar.ax.minorticks_off()
    if tick_labels is not None:
        sparse_labels = [
            label if index % 2 == 0 else ""
            for index, label in enumerate(tick_labels)
        ]
        colorbar.ax.set_xticklabels(sparse_labels)
    colorbar.set_label(
        unit_label,
        fontsize=FONT_SIZE,
        fontweight="normal",
        labelpad=COLORBAR_LABEL_PAD,
    )


def map_coordinates(data_array: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Return the longitude and latitude coordinates of a map field."""
    if "lon" in data_array.coords and "lat" in data_array.coords:
        return data_array["lon"].values, data_array["lat"].values
    if "tc_lon" in data_array.coords and "tc_lat" in data_array.coords:
        return data_array["tc_lon"].values, data_array["tc_lat"].values
    raise KeyError(f"No longitude/latitude coordinates found for {data_array.name}")


def plot_response_map(
    fig,
    ax,
    field: xr.DataArray,
    p_values: xr.DataArray,
    levels: list[float],
    ticks: list[float],
    panel: str,
    title: str,
    unit_label: str,
    tick_labels: list[str],
    cmap_name: str = "RdBu_r",
    colors: list[str] | None = None,
    draw_mdr_box: bool = True,
) -> None:
    """Draw one response field with significance and a discrete colorbar."""
    lon, lat = map_coordinates(field)
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    if colors is None:
        cmap = plt.get_cmap(cmap_name, len(levels) - 1)
    else:
        cmap = ListedColormap(colors)
        cmap.set_under(colors[0])
        cmap.set_over(colors[-1])

    norm = BoundaryNorm(levels, len(levels) - 1, clip=False)
    filled = ax.contourf(
        lon_2d,
        lat_2d,
        field.values,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.contour(
        lon_2d,
        lat_2d,
        field.values,
        levels=levels,
        colors="0.25",
        linewidths=CONTOUR_LINE_WIDTH,
        alpha=CONTOUR_LINE_ALPHA,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    format_map(ax, draw_mdr_box=draw_mdr_box)
    add_significance_hatching(ax, lon, lat, p_values.values)
    add_panel_title(ax, panel, title)
    add_horizontal_colorbar(
        fig,
        ax,
        filled,
        ticks,
        unit_label,
        tick_labels=tick_labels,
    )


def plot_ctl_aoddust(fig, ax, field: xr.DataArray) -> None:
    """Draw the CTL JAS AODDUST climatology."""
    levels = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.60]
    colors = [
        "#fff7bc",
        "#fee391",
        "#fec44f",
        "#fe9929",
        "#ec7014",
        "#cc4c02",
        "#993404",
        "#662506",
    ]
    cmap = ListedColormap(colors)
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    norm = BoundaryNorm(levels, len(colors), clip=False)

    lon, lat = map_coordinates(field)
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    filled = ax.contourf(
        lon_2d,
        lat_2d,
        field.values,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.contour(
        lon_2d,
        lat_2d,
        field.values,
        levels=levels,
        colors="0.25",
        linewidths=CONTOUR_LINE_WIDTH,
        alpha=CONTOUR_LINE_ALPHA,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    format_map(ax, draw_mdr_box=True)
    add_panel_title(ax, "(b)", "CTL AODDUST")
    add_horizontal_colorbar(
        fig,
        ax,
        filled,
        levels,
        "AOD",
        tick_labels=[f"{value:g}" for value in levels],
    )


def scalar_from_nc(dataset: xr.Dataset, variable: str) -> float:
    """Read one scalar value stored along the one-row summary dimension."""
    values = np.asarray(dataset[variable].values).squeeze()
    if values.size != 1:
        raise ValueError(f"Expected one value in {variable}, found {values.size}")
    return float(values)


def plot_mdr_tcgf(ax, dataset: xr.Dataset) -> None:
    """Draw paired annual MDR TCGF counts stored in the NetCDF file."""
    ctl = np.asarray(dataset["tcgf_yearly__ctl_mdr_tcgf"].values, dtype=float)
    ggw = np.asarray(dataset["tcgf_yearly__ggw_mdr_tcgf"].values, dtype=float)
    if ctl.shape != ggw.shape:
        raise ValueError(f"CTL and GGW TCGF shapes differ: {ctl.shape} and {ggw.shape}")

    rng = np.random.default_rng(20_260_717)
    jitter = rng.normal(0.0, 0.025, size=ctl.size)
    for index in range(ctl.size):
        ax.plot(
            [jitter[index], 1.0 + jitter[index]],
            [ctl[index], ggw[index]],
            color="0.72",
            linewidth=0.75,
            zorder=1,
        )

    ax.scatter(
        jitter,
        ctl,
        s=28,
        color="#2166ac",
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )
    ax.scatter(
        1.0 + jitter,
        ggw,
        s=28,
        color="#d73027",
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )
    ax.scatter(
        [0.0, 1.0],
        [ctl.mean(), ggw.mean()],
        s=82,
        color=["#053061", "#a50026"],
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
    )
    ax.plot(
        [0.0, 1.0],
        [ctl.mean(), ggw.mean()],
        color="black",
        linewidth=1.2,
        zorder=3,
    )

    y_max = max(float(np.nanmax(ctl)), float(np.nanmax(ggw))) + 3.0
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(TCGF_Y_MIN, y_max)
    ax.set_box_aspect(MAP_BOX_ASPECT)
    ax.set_xticks([0.0, 1.0], labels=["CTL", "GGW"])
    y_major_ticks = np.arange(
        TCGF_Y_MIN,
        y_max + TCGF_Y_MAJOR_INTERVAL * 0.5,
        TCGF_Y_MAJOR_INTERVAL,
    )
    y_minor_ticks = np.arange(
        TCGF_Y_MIN + TCGF_Y_MINOR_INTERVAL,
        y_max + TCGF_Y_MINOR_INTERVAL * 0.5,
        TCGF_Y_MAJOR_INTERVAL,
    )
    ax.set_yticks(y_major_ticks)
    ax.set_yticks(y_minor_ticks, minor=True)
    ax.set_ylabel("")
    ax.set_xlabel("(Events per JAS)", fontweight="normal", labelpad=4)
    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=MAJOR_GRID_WIDTH,
        color="0.75",
    )
    ax.xaxis.grid(False)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.0,
        width=BORDER_WIDTH,
        top=False,
        right=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=2.0,
        width=BORDER_WIDTH,
        top=False,
        right=False,
    )
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_linewidth(BORDER_WIDTH)
        ax.spines[side].set_color("black")

    add_panel_title(ax, "(f)", "MDR TCGF")
    ctl_mean = scalar_from_nc(dataset, "tcgf_significance__ctl_mean")
    ggw_mean = scalar_from_nc(dataset, "tcgf_significance__ggw_mean")
    difference = scalar_from_nc(dataset, "tcgf_significance__mean_difference")
    p_value = scalar_from_nc(
        dataset,
        "tcgf_significance__paired_permutation_p",
    )
    ax.annotate(
        f"{ctl_mean:.2f}",
        xy=(0.0, ctl_mean),
        xytext=(-9, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=FONT_SIZE,
        fontweight="normal",
        zorder=6,
    )
    ax.annotate(
        f"{ggw_mean:.2f}",
        xy=(1.0, ggw_mean),
        xytext=(9, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=FONT_SIZE,
        fontweight="normal",
        zorder=6,
    )
    statistics_text = (
        f"Diff={difference:.2f}\n"
        f"p={p_value:.3f}"
    )
    ax.text(
        0.97,
        0.95,
        statistics_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_SIZE,
        fontweight="normal",
        bbox={
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": BORDER_WIDTH,
            "alpha": 0.95,
            "pad": 3,
        },
    )


def create_figure(dataset: xr.Dataset):
    """Create the six-panel figure from the loaded NetCDF dataset."""
    fig = plt.figure(figsize=(14.5, 5.4), dpi=300)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.045,
        right=0.985,
        top=0.88,
        bottom=0.14,
        wspace=0.18,
        hspace=0.38,
    )
    axes = [
        fig.add_subplot(grid[0, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[0, 2], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[1, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[1, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[1, 2]),
    ]

    plot_response_map(
        fig,
        axes[0],
        dataset["aoddust_diff"],
        dataset["aoddust_p"],
        levels=[
            -0.20,
            -0.12,
            -0.08,
            -0.04,
            -0.005,
            -0.0001,
            0.0001,
            0.002,
            0.004,
            0.006,
        ],
        ticks=[-0.20, -0.12, -0.08, -0.04, 0.0, 0.002, 0.004, 0.006],
        panel="(a)",
        title="AODDUST response",
        unit_label=r"AOD ($\times 10^{-3}$)",
        tick_labels=["-200", "-120", "-80", "-40", "0", "2", "4", "6"],
        colors=[
            "#00483d",
            "#007465",
            "#23a393",
            "#78cfc4",
            "#d9f3ee",
            "#ffffff",
            "#fff0c7",
            "#d99b3d",
            "#704106",
        ],
    )
    plot_ctl_aoddust(fig, axes[1], dataset["aoddust_ctl"])
    plot_response_map(
        fig,
        axes[2],
        dataset["dustsf_diff_scaled"],
        dataset["dustsf_p"],
        levels=[-1200, -900, -600, -300, 0, 100, 300, 450, 600],
        ticks=[-1200, -900, -600, -300, 0, 100, 300, 450, 600],
        panel="(c)",
        title="Dust emission",
        unit_label=r"($\times 10^{-12}$ kg m$^{-2}$ s$^{-1}$)",
        tick_labels=["-1200", "-900", "-600", "-300", "0", "100", "300", "450", "600"],
        cmap_name="PuOr_r",
        draw_mdr_box=False,
    )
    plot_response_map(
        fig,
        axes[3],
        dataset["fsns_diff"],
        dataset["fsns_p"],
        levels=[-12, -8, -4, -1, 0, 1, 4, 8, 12],
        ticks=[-12, -8, -4, -1, 0, 1, 4, 8, 12],
        panel="(d)",
        title="FSNS",
        unit_label=r"(W m$^{-2}$)",
        tick_labels=["-12", "-8", "-4", "-1", "0", "1", "4", "8", "12"],
    )
    plot_response_map(
        fig,
        axes[4],
        dataset["ts_ocean_diff"],
        dataset["ts_ocean_p"],
        levels=[-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6],
        ticks=[-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6],
        panel="(e)",
        title="Ocean TS",
        unit_label=r"($\times 10^{-1}$ K)",
        tick_labels=["-4", "-3", "-2", "-1", "0", "1", "2", "4", "6"],
    )
    plot_mdr_tcgf(axes[5], dataset)
    return fig


def save_figure(fig) -> None:
    """Write editable vector outputs and a high-resolution PNG preview."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.png", dpi=300, **common)
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.pdf", **common)
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.svg", **common)


def main() -> None:
    plt.switch_backend("Agg")
    setup_style()
    dataset = load_figure_data()
    try:
        figure = create_figure(dataset)
        save_figure(figure)
        plt.close(figure)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
