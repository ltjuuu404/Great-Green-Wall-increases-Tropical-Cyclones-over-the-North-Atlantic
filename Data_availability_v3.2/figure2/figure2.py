from __future__ import annotations

"""Draw the seasonal cycle and IBTrACS/CTL spatial distributions from one NC."""

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


import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature


warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------------------------------------------------------
# Input and output
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "selected_data" / "seasonal_cycle_ibtracs_ctl_spatial_combined.nc"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_STEM = OUTPUT_DIR / "seasonal_cycle_with_ibtracs_ctl_spatial_revised_mdr_odd_month_labels"


# -----------------------------------------------------------------------------
# Global figure controls
# -----------------------------------------------------------------------------
FONT = "Times New Roman"
TEXT_SIZE = 32
TITLE_SIZE = 42
# Intended double-column placement width. The larger drawing canvas below is
# scaled to this physical width during manuscript layout.
FINAL_WIDTH_MM = 183.0
FIG_WIDTH = 12.0
FIG_HEIGHT = 12.2
FIG_DPI = 600
FRAME_WIDTH = 1.5
GRID_WIDTH = 0.5

LINE_COLORS = {"CTL": "#2C7FB8", "IBTrACS": "#252525"}
LINE_MARKERS = {"CTL": "o", "IBTrACS": "^"}
MONTH_LABELS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
MONTH_LABEL_MONTHS = (1, 3, 5, 7, 9, 11)  # Label these months; retain all 12 ticks.
SEASONAL_Y_MIN = 0.0
SEASONAL_Y_MAX = 7.0
SEASONAL_Y_MAJOR_STEP = 1.0
SEASONAL_Y_MINOR_STEP = 0.5

MAP_LON_MIN = -90.0
MAP_LON_MAX = 30.0
MAP_LAT_MIN = 0.0
MAP_LAT_MAX = 40.0
MAP_CENTRAL_LON = -30.0
MAP_LON_MAJOR_STEP = 30.0
MAP_LAT_MAJOR_STEP = 20.0
MAP_LON_MINOR_STEP = 15.0
MAP_LAT_MINOR_STEP = 10.0

# Main development region: longitude west/east, latitude south/north (degrees).
SHOW_MDR = True
MDR_BOX = (-70.0, -17.5, 10.0, 20.0)
MDR_COLOR = "black"
MDR_LINEWIDTH = 2.5

COLORBAR_MIN = 0.0
COLORBAR_MAX = 0.15
COLORBAR_BINS = 15
COLORBAR_TICKS = (0.00, 0.03, 0.06, 0.09, 0.12, 0.15)
COLORBAR_WIDTH = 0.014
COLORBAR_GAP = 0.04
PALETTE = ("white", "#fff7bc", "#fee391", "#fdae61", "#f46d43", "#d73027", "#a50026")


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Times New Roman", "Arial", "sans-serif"],
            "font.serif": [FONT],
            "font.size": 18,
            "font.weight": "normal",
            "axes.labelsize": 18,
            "axes.titlesize": 22,
            "axes.labelweight": "normal",
            "axes.titleweight": "bold",
            "axes.linewidth": FRAME_WIDTH,
            "legend.frameon": False,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 100,
            "savefig.dpi": FIG_DPI,
        }
    )


def format_latitude(value: float) -> str:
    if np.isclose(value, 0.0):
        return "0°"
    return f"{abs(int(round(value)))}°{'N' if value > 0 else 'S'}"


def format_longitude(value: float) -> str:
    normalized = float((value + 180.0) % 360.0 - 180.0)
    if np.isclose(normalized, 0.0):
        return "0°"
    return f"{abs(int(round(normalized)))}°{'E' if normalized > 0 else 'W'}"


def normalize_longitude_to_center(values: np.ndarray, center: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - center + 180.0) % 360.0 - 180.0 + center


def make_density_colormap() -> tuple[np.ndarray, ListedColormap, BoundaryNorm]:
    boundaries = np.linspace(COLORBAR_MIN, COLORBAR_MAX, COLORBAR_BINS + 1)
    linear = LinearSegmentedColormap.from_list(
        "white_warm_interpolated", PALETTE, N=COLORBAR_BINS
    )
    colors = linear(np.linspace(0.0, 1.0, COLORBAR_BINS))
    colors[0] = np.array([1.0, 1.0, 1.0, 1.0])
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=False)
    return boundaries, cmap, norm


def make_figure_axes() -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor="white")
    grid = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=(1.05, 1.0, 1.0),
        hspace=0.46,
    )
    seasonal_axis = fig.add_subplot(grid[0, 0])
    observation_axis = fig.add_subplot(
        grid[1, 0], projection=ccrs.PlateCarree(central_longitude=MAP_CENTRAL_LON)
    )
    control_axis = fig.add_subplot(
        grid[2, 0], projection=ccrs.PlateCarree(central_longitude=MAP_CENTRAL_LON)
    )
    return fig, seasonal_axis, observation_axis, control_axis


def make_axis_text_normal(axis: plt.Axes) -> None:
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontfamily(FONT)
        label.set_fontweight("normal")
        label.set_fontsize(TEXT_SIZE)


def style_full_frame(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(FRAME_WIDTH)


def plot_seasonal_cycle(axis: plt.Axes, dataset: xr.Dataset) -> None:
    months = np.asarray(dataset["month"].values, dtype=float)
    axis.axvspan(6.5, 9.5, color="#F4E7B2", alpha=0.45, zorder=-2)
    axis.text(
        8.0,
        SEASONAL_Y_MAX - 0.16,
        "JAS",
        ha="center",
        va="top",
        color="#705A13",
        fontsize=TEXT_SIZE,
        fontweight="normal",
    )

    for dataset_name in ("CTL", "IBTrACS"):
        mean = np.asarray(
            dataset["seasonal_mean"].sel(dataset=dataset_name).values, dtype=float
        )
        ci_low = np.asarray(
            dataset["seasonal_ci_low"].sel(dataset=dataset_name).values, dtype=float
        )
        ci_high = np.asarray(
            dataset["seasonal_ci_high"].sel(dataset=dataset_name).values, dtype=float
        )
        jas_mean = float(dataset["seasonal_jas_mean"].sel(dataset=dataset_name))
        color = LINE_COLORS[dataset_name]
        axis.fill_between(
            months,
            ci_low,
            ci_high,
            color=color,
            alpha=0.13,
            linewidth=0.0,
            zorder=1,
        )
        axis.plot(
            months,
            mean,
            color=color,
            linewidth=2.8,
            marker=LINE_MARKERS[dataset_name],
            markersize=8.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.8,
            label=f"{dataset_name} (JAS = {jas_mean:.1f})",
            zorder=3,
        )

    axis.set_title(
        "(a) Seasonal cycle",
        loc="left",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=8,
    )
    axis.set_xlim(0.65, 12.35)
    axis.set_ylim(SEASONAL_Y_MIN, SEASONAL_Y_MAX)
    month_tick_labels = [
        MONTH_LABELS[int(month) - 1] if int(month) in MONTH_LABEL_MONTHS else ""
        for month in months
    ]
    axis.set_xticks(months, month_tick_labels)
    seasonal_y_ticks = np.arange(
        SEASONAL_Y_MIN,
        SEASONAL_Y_MAX + SEASONAL_Y_MAJOR_STEP * 0.5,
        SEASONAL_Y_MAJOR_STEP,
    )
    axis.set_yticks(seasonal_y_ticks)
    axis.set_yticklabels(
        [
            f"{int(value)}"
            if (int(round(value)) % 2 == 0 or np.isclose(value, SEASONAL_Y_MAX))
            else ""
            for value in seasonal_y_ticks
        ]
    )
    axis.xaxis.set_minor_locator(MultipleLocator(0.5))
    axis.yaxis.set_minor_locator(MultipleLocator(SEASONAL_Y_MINOR_STEP))
    axis.set_xlabel("Month", fontsize=TEXT_SIZE)
    axis.set_ylabel(
        "TCGF",
        fontsize=TEXT_SIZE,
        labelpad=8,
    )
    axis.grid(
        which="major",
        color="0.70",
        linestyle="--",
        linewidth=GRID_WIDTH,
        zorder=-1,
    )
    axis.grid(False, which="minor")
    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=8,
        width=FRAME_WIDTH,
        top=False,
        right=False,
        labelsize=TEXT_SIZE,
    )
    axis.tick_params(axis="both", which="minor", length=0.0, width=0.0)
    style_full_frame(axis)
    make_axis_text_normal(axis)
    legend = axis.legend(
        loc="upper left",
        frameon=False,
        facecolor="white",
        fontsize=TEXT_SIZE-4,
        handlelength=2.4,
    )
    for text in legend.get_texts():
        text.set_fontfamily(FONT)
        text.set_fontweight("normal")


def configure_map_ticks(axis: plt.Axes) -> None:
    longitude_major_raw = np.arange(
        MAP_LON_MIN, MAP_LON_MAX + MAP_LON_MAJOR_STEP * 0.5, MAP_LON_MAJOR_STEP
    )
    longitude_major_plot = normalize_longitude_to_center(
        longitude_major_raw, MAP_CENTRAL_LON
    )
    order = np.argsort(longitude_major_plot)
    longitude_major_plot = longitude_major_plot[order]
    longitude_major_raw = longitude_major_raw[order]

    latitude_major = np.arange(
        MAP_LAT_MIN, MAP_LAT_MAX + MAP_LAT_MAJOR_STEP * 0.5, MAP_LAT_MAJOR_STEP
    )
    longitude_minor_raw = np.arange(
        MAP_LON_MIN + MAP_LON_MINOR_STEP,
        MAP_LON_MAX,
        MAP_LON_MAJOR_STEP,
    )
    longitude_minor_plot = normalize_longitude_to_center(
        longitude_minor_raw, MAP_CENTRAL_LON
    )
    latitude_minor = np.arange(
        MAP_LAT_MIN + MAP_LAT_MINOR_STEP,
        MAP_LAT_MAX,
        MAP_LAT_MAJOR_STEP,
    )

    axis.set_xticks(longitude_major_plot, crs=ccrs.PlateCarree())
    axis.set_yticks(latitude_major, crs=ccrs.PlateCarree())
    axis.set_xticks(longitude_minor_plot, minor=True, crs=ccrs.PlateCarree())
    axis.set_yticks(latitude_minor, minor=True, crs=ccrs.PlateCarree())

    longitude_labels = [
        format_longitude(value) if index % 2 == 0 else ""
        for index, value in enumerate(longitude_major_raw)
    ]
    axis.set_xticklabels(longitude_labels, fontsize=TEXT_SIZE)
    axis.set_yticklabels(
        [format_latitude(value) for value in latitude_major], fontsize=TEXT_SIZE
    )
    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=8,
        width=FRAME_WIDTH,
        top=False,
        right=False,
        pad=7,
    )
    axis.tick_params(axis="x", which="minor", length=0.0, width=0.0)
    axis.tick_params(
        axis="y",
        which="minor",
        direction="out",
        length=4,
        width=FRAME_WIDTH * 0.75,
        left=True,
        right=False,
        labelleft=False,
    )

    gridlines = axis.gridlines(
        draw_labels=False,
        linestyle="--",
        linewidth=GRID_WIDTH,
        color="gray",
        alpha=0.6,
    )
    gridlines.xlocator = mticker.FixedLocator(longitude_major_plot)
    gridlines.ylocator = mticker.FixedLocator(latitude_major)
    make_axis_text_normal(axis)


def plot_spatial_panel(
    axis: plt.Axes,
    dataset: xr.Dataset,
    dataset_name: str,
    title: str,
    boundaries: np.ndarray,
    cmap: ListedColormap,
    norm: BoundaryNorm,
):
    lon = np.asarray(dataset["lon"].values, dtype=float)
    lat = np.asarray(dataset["lat"].values, dtype=float)
    values = np.asarray(
        dataset["spatial_mean_smooth"].sel(dataset=dataset_name).values,
        dtype=float,
    )
    expected_shape = (lat.size, lon.size)
    if values.shape != expected_shape:
        raise ValueError(
            f"{dataset_name} spatial field has shape {values.shape}; "
            f"expected {expected_shape}"
        )
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    filled = axis.contourf(
        lon_2d,
        lat_2d,
        values,
        levels=boundaries,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )
    axis.contour(
        lon_2d,
        lat_2d,
        values,
        levels=boundaries,
        colors="gray",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
        zorder=1.5,
    )
    axis.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor="lightgray",
        edgecolor="none",
        zorder=2,
    )
    axis.coastlines(resolution="50m", linewidth=1.5, color="black", zorder=3)
    if SHOW_MDR:
        west, east, south, north = MDR_BOX
        axis.add_patch(
            Rectangle(
                (west, south), east - west, north - south,
                fill=False, edgecolor=MDR_COLOR, linewidth=MDR_LINEWIDTH,
                linestyle="-", transform=ccrs.PlateCarree(), zorder=5,
            )
        )
    axis.set_extent(
        [MAP_LON_MIN, MAP_LON_MAX, MAP_LAT_MIN, MAP_LAT_MAX],
        crs=ccrs.PlateCarree(),
    )
    axis.set_title(
        title,
        loc="left",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=8,
    )
    configure_map_ticks(axis)
    style_full_frame(axis)
    return filled


def add_shared_colorbar(
    fig: plt.Figure,
    observation_axis: plt.Axes,
    control_axis: plt.Axes,
    mappable,
    boundaries: np.ndarray,
) -> None:
    fig.canvas.draw()
    observation_position = observation_axis.get_position()
    control_position = control_axis.get_position()
    axes_right = max(observation_position.x1, control_position.x1)
    colorbar_bottom = control_position.y0
    colorbar_top = observation_position.y1
    colorbar_height = colorbar_top - colorbar_bottom
    colorbar_left = axes_right + COLORBAR_GAP
    cax = fig.add_axes(
        [colorbar_left, colorbar_bottom, COLORBAR_WIDTH, colorbar_height]
    )
    colorbar = fig.colorbar(
        mappable,
        cax=cax,
        orientation="vertical",
        extend="both",
        boundaries=boundaries,
    )
    colorbar.set_ticks(COLORBAR_TICKS)
    colorbar.set_ticklabels(
        ["0" if np.isclose(value, 0.0) else f"{value:.2f}" for value in COLORBAR_TICKS]
    )
    colorbar.ax.tick_params(
        which="major", labelsize=TEXT_SIZE, width=FRAME_WIDTH, length=7
    )
    for label in colorbar.ax.get_yticklabels():
        label.set_fontfamily(FONT)
        label.set_fontweight("normal")
    colorbar.outline.set_linewidth(FRAME_WIDTH)


def main() -> None:
    plt.switch_backend("Agg")
    setup_style()
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"{DATA_FILE} does not exist. Run 01_prepare_combined_data.py first."
        )
    with xr.open_dataset(DATA_FILE) as source:
        dataset = source.load()

    required = {
        "seasonal_mean",
        "seasonal_ci_low",
        "seasonal_ci_high",
        "seasonal_jas_mean",
        "spatial_mean_smooth",
    }
    missing = required.difference(dataset.data_vars)
    if missing:
        raise KeyError(f"Combined NetCDF is missing variables: {sorted(missing)}")

    boundaries, cmap, norm = make_density_colormap()
    fig, seasonal_axis, observation_axis, control_axis = make_figure_axes()
    plot_seasonal_cycle(seasonal_axis, dataset)
    observation_mappable = plot_spatial_panel(
        observation_axis,
        dataset,
        "IBTrACS",
        "(b) IBTrACS",
        boundaries,
        cmap,
        norm,
    )
    plot_spatial_panel(
        control_axis,
        dataset,
        "CTL",
        "(c) CTL",
        boundaries,
        cmap,
        norm,
    )

    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.055, top=0.960, hspace=0.46)
    fig.canvas.draw()

# 取得 b 子图最终实际位置
    map_pos = observation_axis.get_position()

# 取得 a 子图当前位置
    seasonal_pos = seasonal_axis.get_position()

# 让 a 的左右边界与 b 完全一致
    seasonal_axis.set_position([
    map_pos.x0,          # left
    seasonal_pos.y0,    # bottom 保持不变
    map_pos.width,       # width 与 b 相同
    seasonal_pos.height # height 保持不变
])
    add_shared_colorbar(
        fig, observation_axis, control_axis, observation_mappable, boundaries
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=FIG_DPI, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".tiff"), dpi=FIG_DPI, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(f"Saved combined figure: {OUTPUT_STEM}.png/.tiff/.pdf/.svg")


if __name__ == "__main__":
    main()
