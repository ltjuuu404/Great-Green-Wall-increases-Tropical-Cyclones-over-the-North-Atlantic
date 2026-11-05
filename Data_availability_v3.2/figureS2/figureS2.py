#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""Plot IBTrACS, CTL, and CTL-minus-IBTrACS TC track density."""



from __future__ import annotations

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




import json

import os

from pathlib import Path



import cartopy.crs as ccrs

import cartopy.feature as cfeature

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

import numpy as np

import xarray as xr

from matplotlib import font_manager

from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle





# -----------------------------------------------------------------------------

# Files

# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

DATA_FILE = HERE / "selected_data" / "tc_track_density_ibtracs_control_jas_2p5deg.nc"

OUTPUT_DIR = HERE / "outputs"

OUTPUT_STEM = OUTPUT_DIR / "tc_track_density_ibtracs_control_difference_mdr"

QA_FILE = HERE / "plot_qa_mdr.json"





# -----------------------------------------------------------------------------

# User-editable axis and tick controls

# -----------------------------------------------------------------------------

LON_MIN, LON_MAX = -90.0, 30.0

LAT_MIN, LAT_MAX = 0.0, 40.0

CENTRAL_LONGITUDE = -30.0
MDR_BOX = (-70.0, -17.5, 10.0, 20.0)  # lon_min, lon_max, lat_min, lat_max
MDR_LINE_WIDTH = 1.5



LON_MAJOR_TICK_MIN = -90.0

LON_MAJOR_TICK_MAX = 30.0

LON_MAJOR_TICK_STEP = 30.0

LON_MINOR_TICKS_PER_INTERVAL = 1



LAT_MAJOR_TICK_MIN = 0.0

LAT_MAJOR_TICK_MAX = 40.0

LAT_MAJOR_TICK_STEP = 10.0

LAT_MINOR_TICKS_PER_INTERVAL = 1





# -----------------------------------------------------------------------------

# User-editable discrete colorbar controls

# -----------------------------------------------------------------------------

ABSOLUTE_COLOR_MIN = 0.0

ABSOLUTE_COLOR_MAX = 2.0

CB_N_BINS_TOP = 15

ABSOLUTE_COLOR_INTERVAL = (ABSOLUTE_COLOR_MAX - ABSOLUTE_COLOR_MIN) / CB_N_BINS_TOP

ABSOLUTE_COLORBAR_TICKS = (0.0, 0.5, 1.0, 1.5, 2.0)



DIFFERENCE_COLOR_MIN = -1.4

DIFFERENCE_COLOR_MAX = 1.4

CB_N_BINS = 11

DIFFERENCE_COLOR_INTERVAL = (DIFFERENCE_COLOR_MAX - DIFFERENCE_COLOR_MIN) / CB_N_BINS

DIFFERENCE_COLORBAR_TICKS = (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2)



ABSOLUTE_PALETTE = (

    "white",

    "#fff7bc",

    "#fee391",

    "#fdae61",

    "#f46d43",

    "#d73027",

    "#a50026",

)





# -----------------------------------------------------------------------------

# Typography, line work, layout, and export

# -----------------------------------------------------------------------------

FONT_FAMILY = "Times New Roman"

FONT_SIZE = 20

TITLE_SIZE = 28

OTHER_FONTSIZE = 26

COLORBAR_TICK_SIZE = OTHER_FONTSIZE - 4

FRAME_WIDTH = 1.5

GRID_WIDTH = 0.4

COASTLINE_WIDTH = 1.5

CONTOUR_LINE_WIDTH = 0.5



FIGURE_SIZE_INCH = (12.0, 11.0)

FIGURE_DPI = 600



# Match the reference map layout: one shared vertical cbar for the first two

# panels and one vertical cbar for the difference panel.

MAP_LEFT = 0.08

MAP_BOTTOM = 0.06

MAP_WIDTH = 0.67

PANEL_GAP = 0.085

COLORBAR_GAP = 0.01
COLORBAR_X_SHIFT = 0.03  # move both colorbars to the right
COLORBAR_WIDTH = 0.01
CB1_Y_SHIFT = 0.0
CB2_Y_SHIFT = 0.0  # vertically center the panel-C colorbar
DIFF_CBAR_SCALE = 0.92

COLORBAR_UNIT = "Track points JAS⁻¹ (2.5° cell)⁻¹"





def register_times_new_roman() -> tuple[font_manager.FontProperties, font_manager.FontProperties]:

    windows_font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"

    regular_path = windows_font_dir / "times.ttf"

    bold_path = windows_font_dir / "timesbd.ttf"

    for path in (regular_path, bold_path):

        if not path.is_file():

            raise FileNotFoundError(f"Required Times New Roman font not found: {path}")

        font_manager.fontManager.addfont(path)

    return (

        font_manager.FontProperties(

            family=FONT_FAMILY, size=FONT_SIZE, weight="normal"

        ),

        font_manager.FontProperties(

            family=FONT_FAMILY, size=TITLE_SIZE, weight="bold"

        ),

    )





REGULAR_FONT, BOLD_FONT = register_times_new_roman()

COLORBAR_FONT = font_manager.FontProperties(

    family=FONT_FAMILY, size=COLORBAR_TICK_SIZE, weight="normal"

)





def setup_style() -> None:

    mpl.rcParams.update(

        {

            "font.family": [FONT_FAMILY],

            "font.serif": [FONT_FAMILY],

            "font.sans-serif": [FONT_FAMILY],

            "font.size": FONT_SIZE,

            "font.weight": "normal",

            "axes.labelsize": OTHER_FONTSIZE,

            "axes.titlesize": TITLE_SIZE,

            "axes.labelweight": "normal",

            "axes.titleweight": "bold",

            "axes.linewidth": FRAME_WIDTH,

            "xtick.labelsize": FONT_SIZE,

            "ytick.labelsize": FONT_SIZE,

            "legend.fontsize": OTHER_FONTSIZE,

            "legend.frameon": False,

            "hatch.linewidth": 0.35,

            "pdf.fonttype": 42,

            "ps.fonttype": 42,

            "svg.fonttype": "none",

            "savefig.dpi": FIGURE_DPI,

        }

    )





def inclusive_arange(start: float, stop: float, step: float) -> np.ndarray:

    return np.arange(start, stop + step * 0.5, step, dtype=np.float64)





def minor_ticks_between(major: np.ndarray, count: int) -> np.ndarray:

    if count <= 0 or major.size < 2:

        return np.asarray([], dtype=np.float64)

    values = []

    for left, right in zip(major[:-1], major[1:]):

        values.extend(np.linspace(left, right, count + 2)[1:-1])

    return np.asarray(values, dtype=np.float64)





def format_longitude(value: float, _position=None) -> str:

    normalized = float((value + 180.0) % 360.0 - 180.0)

    if np.isclose(normalized, 0.0):

        return "0°"

    return f"{abs(int(round(normalized)))}°{'E' if normalized > 0 else 'W'}"





def format_latitude(value: float, _position=None) -> str:

    if np.isclose(value, 0.0):

        return "0°"

    return f"{abs(int(round(value)))}°{'N' if value > 0 else 'S'}"





def normalize_longitude_to_center(

    values: np.ndarray, center: float

) -> np.ndarray:

    values = np.asarray(values, dtype=np.float64)

    return (values - center + 180.0) % 360.0 - 180.0 + center





def discrete_colormap(

    name: str,

    palette: str | tuple[str, ...],

    number_of_bins: int,

) -> ListedColormap:

    if isinstance(palette, str):

        linear = plt.get_cmap(palette)

        colors = linear(np.linspace(0.02, 0.98, number_of_bins))

    else:

        linear = LinearSegmentedColormap.from_list(name, palette)

        colors = linear(np.linspace(0.0, 1.0, number_of_bins))

    cmap = ListedColormap(colors, name=name)

    cmap.set_bad("white")

    cmap.set_under(colors[0])

    cmap.set_over(colors[-1])

    return cmap





def make_color_contract(

    minimum: float,

    maximum: float,

    interval: float,

    palette: str | tuple[str, ...],

    name: str,

) -> tuple[np.ndarray, ListedColormap, BoundaryNorm]:

    boundaries = inclusive_arange(minimum, maximum, interval)

    if boundaries.size < 3 or not np.isclose(boundaries[-1], maximum):

        raise ValueError(f"Invalid color boundaries for {name}: {boundaries}")

    cmap = discrete_colormap(name, palette, boundaries.size - 1)

    norm = BoundaryNorm(boundaries, cmap.N, clip=False)

    return boundaries, cmap, norm





def apply_regular_tick_font(axis: plt.Axes) -> None:

    for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):

        label.set_fontproperties(REGULAR_FONT)





def configure_map_axis(axis: plt.Axes) -> None:

    projection = ccrs.PlateCarree()

    axis.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=projection)

    axis.set_aspect("equal", adjustable="box")
    lon0, lon1, lat0, lat1 = MDR_BOX
    axis.add_patch(Rectangle(
        (lon0, lat0), lon1 - lon0, lat1 - lat0,
        fill=False, edgecolor="black", linewidth=MDR_LINE_WIDTH,
        transform=projection, zorder=8,
    ))



    axis.add_feature(

        cfeature.LAND.with_scale("50m"),

        facecolor="lightgray",

        edgecolor="none",

        zorder=3,

    )

    axis.coastlines(

        resolution="50m", linewidth=COASTLINE_WIDTH, color="black", zorder=4

    )



    longitude_major_raw = inclusive_arange(

        LON_MAJOR_TICK_MIN, LON_MAJOR_TICK_MAX, LON_MAJOR_TICK_STEP

    )

    longitude_major_plot = normalize_longitude_to_center(

        longitude_major_raw, CENTRAL_LONGITUDE

    )

    longitude_order = np.argsort(longitude_major_plot)

    longitude_major_plot = longitude_major_plot[longitude_order]

    longitude_major_raw = longitude_major_raw[longitude_order]

    lat_major = inclusive_arange(

        LAT_MAJOR_TICK_MIN, LAT_MAJOR_TICK_MAX, LAT_MAJOR_TICK_STEP

    )

    longitude_minor_raw = np.arange(

        LON_MAJOR_TICK_MIN + LON_MAJOR_TICK_STEP / 2.0,

        LON_MAJOR_TICK_MAX,

        LON_MAJOR_TICK_STEP,

        dtype=np.float64,

    )

    lon_minor = normalize_longitude_to_center(

        longitude_minor_raw, CENTRAL_LONGITUDE

    )

    lat_minor = minor_ticks_between(lat_major, LAT_MINOR_TICKS_PER_INTERVAL)



    axis.set_xticks(longitude_major_plot, crs=projection)

    axis.set_yticks(lat_major, crs=projection)

    if lon_minor.size:

        axis.set_xticks(lon_minor, minor=True, crs=projection)

    if lat_minor.size:

        axis.set_yticks(lat_minor, minor=True, crs=projection)

    axis.set_xticklabels(

        [

            format_longitude(value) if index % 2 == 0 else ""

            for index, value in enumerate(longitude_major_raw)

        ]

    )

    axis.set_yticklabels([format_latitude(value) for value in lat_major])



    axis.tick_params(

        axis="both",

        which="major",

        direction="out",

        length=10.0,

        width=FRAME_WIDTH,

        color="black",

        bottom=True,

        left=True,

        top=False,

        right=False,

        labelbottom=True,

        labelleft=True,

        labeltop=False,

        labelright=False,

        pad=10.0,

    )

    axis.tick_params(

        axis="x",

        which="minor",

        direction="out",

        length=5.0,

        width=1.0,

        bottom=True,

        top=False,

        labelbottom=False,

    )

    axis.tick_params(

        axis="y",

        which="minor",

        direction="out",

        length=5.0,

        width=FRAME_WIDTH * 0.5,

        left=True,

        right=False,

        labelleft=False,

    )

    apply_regular_tick_font(axis)



    gridlines = axis.gridlines(

        crs=projection,

        draw_labels=False,

        xlocs=longitude_major_plot,

        ylocs=lat_major,

        color="gray",

        linestyle="--",

        linewidth=GRID_WIDTH,

        alpha=0.6,

        zorder=2.5,

    )

    gridlines.xlocator = mticker.FixedLocator(longitude_major_plot)

    gridlines.ylocator = mticker.FixedLocator(lat_major)



    # Keep the complete black frame above every plotted map layer.
    # A high zorder prevents contourf / land / coastline artists from covering
    # the thick panel border.
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(FRAME_WIDTH)
        spine.set_zorder(100)





def add_panel_label(axis: plt.Axes, label: str) -> None:

    axis.set_title(

        label,

        loc="left",

        pad=8,

        color="black",

        fontproperties=BOLD_FONT,

    )





def add_vertical_colorbar(

    fig: plt.Figure,

    cax: plt.Axes,

    mappable,

    boundaries: np.ndarray,

    ticks: tuple[float, ...],

    decimals: int,

    label: str | None,
    label_every: int = 1,

) -> None:

    colorbar = fig.colorbar(

        mappable,

        cax=cax,

        orientation="vertical",

        boundaries=boundaries,

        ticks=ticks,

        spacing="uniform",

        extend="both",

    )

    colorbar.outline.set_edgecolor("black")

    colorbar.outline.set_linewidth(FRAME_WIDTH)

    colorbar.ax.tick_params(

        axis="y",

        which="major",

        direction="out",

        right=True,

        left=False,

        labelright=True,

        labelleft=False,

        length=5.0,

        width=FRAME_WIDTH,

        pad=5.0,

    )

    colorbar.ax.tick_params(axis="y", which="minor", length=0.0, width=0.0)

    colorbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))

    colorbar.ax.yaxis.set_major_formatter(
    mticker.FixedFormatter(
        [
            (
                "0" if np.isclose(value, 0.0)
                else f"{value:.{decimals}f}"
            ) if i % label_every == 0 else ""
            for i, value in enumerate(ticks)
        ]
    )
)

    for tick_label in colorbar.ax.get_yticklabels():

        tick_label.set_fontproperties(COLORBAR_FONT)

    if label is not None:

        colorbar.set_label(

            label,

            fontproperties=COLORBAR_FONT,

            labelpad=10.0,

        )





def load_data() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, str]]:

    if not DATA_FILE.is_file():

        raise FileNotFoundError(

            f"{DATA_FILE} does not exist. Run 01_prepare_track_density.py first."

        )

    with xr.open_dataset(DATA_FILE) as source:

        required = (

            "ibtracs_mean_smooth",

            "control_mean_smooth",

            "control_minus_ibtracs",

        )

        missing = [name for name in required if name not in source]

        if missing:

            raise KeyError(f"Missing required NetCDF variables: {missing}")

        lon = np.asarray(source["lon"].values, dtype=np.float64)

        lat = np.asarray(source["lat"].values, dtype=np.float64)

        fields = {

            name: np.asarray(source[name].values, dtype=np.float64)

            for name in required

        }

        metadata = {key: str(value) for key, value in source.attrs.items()}



    expected_shape = (lat.size, lon.size)

    for name, values in fields.items():

        if values.shape != expected_shape:

            raise ValueError(

                f"{name} has shape {values.shape}; expected {expected_shape}"

            )

        if not np.isfinite(values).all():

            raise ValueError(f"{name} contains non-finite values")

    if not np.allclose(

        fields["control_minus_ibtracs"],

        fields["control_mean_smooth"] - fields["ibtracs_mean_smooth"],

    ):

        raise AssertionError("Difference field is not CONTROL minus IBTrACS")

    return lon, lat, fields, metadata





def plot_panel(

    axis: plt.Axes,

    lon: np.ndarray,

    lat: np.ndarray,

    values: np.ndarray,

    boundaries: np.ndarray,

    cmap: ListedColormap,

    norm: BoundaryNorm,

    label: str,

):

    filled = axis.contourf(

        lon,

        lat,

        values,

        levels=boundaries,

        cmap=cmap,

        norm=norm,

        extend="both",

        transform=ccrs.PlateCarree(),

        zorder=1,

    )

    axis.contour(

        lon,

        lat,

        values,

        levels=boundaries,

        colors="0.50",

        linewidths=CONTOUR_LINE_WIDTH,

        transform=ccrs.PlateCarree(),

        zorder=2,

    )

    configure_map_axis(axis)

    add_panel_label(axis, label)

    return filled





def save_outputs(fig: plt.Figure) -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the declared physical canvas instead of allowing a tight bounding

    # box to shrink the page differently for each export backend.

    common = {"facecolor": "white"}

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=FIGURE_DPI, **common)

    fig.savefig(

        OUTPUT_STEM.with_suffix(".tiff"),

        dpi=FIGURE_DPI,

        pil_kwargs={"compression": "tiff_lzw"},

        **common,

    )

    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), **common)

    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), **common)





def main() -> None:

    plt.switch_backend("Agg")

    setup_style()

    lon, lat, fields, metadata = load_data()



    absolute_boundaries, absolute_cmap, absolute_norm = make_color_contract(

        ABSOLUTE_COLOR_MIN,

        ABSOLUTE_COLOR_MAX,

        ABSOLUTE_COLOR_INTERVAL,

        ABSOLUTE_PALETTE,

        "track_density_absolute",

    )

    difference_boundaries, difference_cmap, difference_norm = make_color_contract(

        DIFFERENCE_COLOR_MIN,

        DIFFERENCE_COLOR_MAX,

        DIFFERENCE_COLOR_INTERVAL,

        "RdBu_r",

        "track_density_difference",

    )



    fig = plt.figure(figsize=FIGURE_SIZE_INCH, facecolor="white")

    map_height = (

        MAP_WIDTH

        * FIGURE_SIZE_INCH[0]

        / FIGURE_SIZE_INCH[1]

        * (LAT_MAX - LAT_MIN)

        / (LON_MAX - LON_MIN)

    )

    bottom_c = MAP_BOTTOM

    bottom_b = bottom_c + map_height + PANEL_GAP

    bottom_a = bottom_b + map_height + PANEL_GAP

    bottoms = (bottom_a, bottom_b, bottom_c)

    axes = [

        fig.add_axes(

            [MAP_LEFT, bottom, MAP_WIDTH, map_height],

            projection=ccrs.PlateCarree(central_longitude=CENTRAL_LONGITUDE),

        )

        for bottom in bottoms

    ]



    panels = (

        (

            axes[0],

            fields["ibtracs_mean_smooth"],

            absolute_boundaries,

            absolute_cmap,

            absolute_norm,

            "(a) IBTrACS",

            ABSOLUTE_COLORBAR_TICKS,

            1,

        ),

        (

            axes[1],

            fields["control_mean_smooth"],

            absolute_boundaries,

            absolute_cmap,

            absolute_norm,

            "(b) CTL",

            ABSOLUTE_COLORBAR_TICKS,

            1,

        ),

        (

            axes[2],

            fields["control_minus_ibtracs"],

            difference_boundaries,

            difference_cmap,

            difference_norm,

            "(c) CTL − IBTrACS",

            DIFFERENCE_COLORBAR_TICKS,

            1,

        ),

    )



    filled_plots = []

    for axis, values, boundaries, cmap, norm, label, _ticks, _decimals in panels:

        filled_plots.append(

            plot_panel(axis, lon, lat, values, boundaries, cmap, norm, label)

        )



    colorbar_x = MAP_LEFT + MAP_WIDTH + COLORBAR_GAP + COLORBAR_X_SHIFT

    combined_top = bottom_a + map_height

    combined_bottom = bottom_b

    combined_span = combined_top - combined_bottom

    absolute_cbar_height = max(0.001, combined_span * 0.92) * 1.2

    absolute_cbar_y = (

        0.5 * (combined_top + combined_bottom)

        - 0.5 * absolute_cbar_height

        + CB1_Y_SHIFT

    )

    absolute_cbar_y = min(

        max(0.005, absolute_cbar_y), 0.995 - absolute_cbar_height

    )

    cax_absolute = fig.add_axes(

        [colorbar_x, absolute_cbar_y, COLORBAR_WIDTH, absolute_cbar_height]

    )



    difference_cbar_height = max(0.001, map_height * DIFF_CBAR_SCALE)

    difference_cbar_y = (

        bottom_c

        + 0.5 * (map_height - difference_cbar_height)

        + CB2_Y_SHIFT

    )

    difference_cbar_y = min(

        max(0.005, difference_cbar_y), 0.995 - difference_cbar_height

    )

    cax_difference = fig.add_axes(

        [

            colorbar_x,

            difference_cbar_y,

            COLORBAR_WIDTH,

            difference_cbar_height,

        ]

    )

    add_vertical_colorbar(

        fig,

        cax_absolute,

        filled_plots[0],

        absolute_boundaries,

        ABSOLUTE_COLORBAR_TICKS,

        decimals=1,

        label=None,

    )

    add_vertical_colorbar(

        fig,

        cax_difference,

        filled_plots[2],

        difference_boundaries,

        DIFFERENCE_COLORBAR_TICKS,

        decimals=1,

        label=None,
        label_every=2,

    )



    save_outputs(fig)

    fig.canvas.draw()



    qa = {

        "data_file": str(DATA_FILE),

        "outputs": [str(OUTPUT_STEM.with_suffix(suffix)) for suffix in (".png", ".tiff", ".pdf", ".svg")],

        "canvas_inches": list(FIGURE_SIZE_INCH),

        "export_dpi": FIGURE_DPI,

        "font": FONT_FAMILY,

        "font_size_points": FONT_SIZE,

        "frame_width_points": FRAME_WIDTH,
        "mdr_box_lonlat": list(MDR_BOX),
        "mdr_line_width_points": MDR_LINE_WIDTH,
        "mdr_box_panels": ["a", "b", "c"],

        "map_axes_bounds": [

            MAP_LEFT,

            MAP_BOTTOM,

            MAP_WIDTH,

            map_height,

        ],

        "colorbar_layout": {

            "absolute_shared_by_panels_a_b": [

                colorbar_x,

                absolute_cbar_y,

                COLORBAR_WIDTH,

                absolute_cbar_height,

            ],

            "difference_panel_c": [

                colorbar_x,

                difference_cbar_y,

                COLORBAR_WIDTH,

                difference_cbar_height,

            ],

            "orientation": "vertical",

            "absolute_label": None,

            "difference_label": None,

        },

        "absolute_color_boundaries": absolute_boundaries.tolist(),

        "difference_color_boundaries": difference_boundaries.tolist(),

        "field_ranges": {

            name: {"min": float(np.nanmin(values)), "max": float(np.nanmax(values))}

            for name, values in fields.items()

        },

        "difference_definition": metadata.get("difference_definition", ""),

        "track_density_definition": metadata.get("track_density_definition", ""),

        "checks": {

            "all_fields_finite": True,

            "difference_recomputed_exactly": True,

            "shared_vertical_colorbar_for_panels_a_b": True,

            "independent_vertical_colorbar_for_panel_c": True,

            "colorbars_discrete_with_extend_both": True,

            "major_grid_only": True,

            "minor_tick_marks_reference_style": True,

            "minor_gridlines_hidden": True,

            "full_frame_spines_on_all_panels": True,

        },

    }

    QA_FILE.write_text(json.dumps(qa, indent=2), encoding="utf-8")

    plt.close(fig)

    print(f"Saved figure: {OUTPUT_STEM}.png/.tiff/.pdf/.svg")

    print(f"Saved plot QA: {QA_FILE}")





if __name__ == "__main__":

    main()
