from __future__ import annotations


"""Plot the four-panel AEW response figure from prepared local data only.

Panel (a) is the original GGW-minus-CTL track-density response map.
Panels (b)--(d) show original annual AEW frequency, original MDR track density,
and the all-point JAS/MDR annual mean intensity, respectively. No genesis or
lysis restrictions are used for the new point-mean intensity.
"""


from pathlib import Path

from cartopy import config as cartopy_config
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


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
MAP_DATA_DIR = HERE.parent / "map_data"

DATA_FILE = HERE / "FS07_aew_frequency_density_intensity.nc"
STATS_FILE = HERE / "FS07_aew_statistics.csv"
OUTPUT_DIR = HERE / "outputs"


# ============================================================
# Font settings
# ============================================================

FONT = "Times New Roman"

# ------------------------------------------------------------
# Independent font-size controls
# ------------------------------------------------------------

# General fallback font size
FONT_SIZE = 12.0

# Panel (a): map
MAP_TITLE_SIZE = 12.0
MAP_TICK_SIZE = 10.0

# Panel (a): colorbar
CBAR_TICK_SIZE = 8.0
CBAR_LABEL_SIZE = 8.0

# Panels (b)-(d): violin plots
VIOLIN_TITLE_SIZE = 12.0
VIOLIN_TICK_SIZE = 8.0
VIOLIN_LABEL_SIZE = 8.0

# CTL / GGW / p-value text inside violin panels
STATS_TEXT_SIZE = 9.5
STATS_TEXT_POSITION = (0.92, 0.90)  # Axes fractions, measured from the lower left.


# ============================================================
# Other plotting settings
# ============================================================

BORDER_WIDTH = 0.75
GRID_COLOR = "#A7A7A7"

CTL_COLOR = "#2166AC"
GGW_COLOR = "#D73027"
NEUTRAL_COLOR = "#5A5A5A"

MDR_LON = (-70.0, -17.5)
MDR_LAT = (10.0, 20.0)

MAP_EXTENT = (-100.0, 0.0, 0.0, 50.0)
MAP_LON_TICK_STEP = 25.0
MAP_LAT_TICK_STEP = 20.0
FIGSIZE_MM = (165.0, 100.0)
VIOLIN_BOX_ASPECT = (MAP_EXTENT[3] - MAP_EXTENT[2]) / (MAP_EXTENT[1] - MAP_EXTENT[0])
VIOLIN_XLIM = (-0.48, 2.55)  # Leave room for the statistics at the right of the violins.

# D-panel limits and major tick interval; editable independently of panel C.
POINT_INTENSITY_YLIM = (5.0, 12.0)
POINT_INTENSITY_TICK_STEP = 1.0
POINT_INTENSITY_YTICKS = np.arange(
    POINT_INTENSITY_YLIM[0],
    POINT_INTENSITY_YLIM[1] + POINT_INTENSITY_TICK_STEP / 2,
    POINT_INTENSITY_TICK_STEP
)

FREQUENCY_YLIM = (25.0, 43.0)
FREQUENCY_YTICKS = np.arange(25.0, 45.0, 5.0)

RESPONSE_XLIM = (-20.0, 20.0)
RESPONSE_XTICKS = np.arange(-20.0, 21.0, 10.0)


# ============================================================
# Global style
# ============================================================

def setup_style() -> None:

    mpl.rcParams.update({

        "font.family": [FONT, "sans-serif"],
        "font.sans-serif": [
            FONT,
            "Arial",
            "Helvetica"
        ],

        "font.size": FONT_SIZE,
        "font.weight": "normal",

        "axes.labelweight": "normal",
        "axes.titleweight": "normal",

        "axes.linewidth": BORDER_WIDTH,

        "xtick.major.width": BORDER_WIDTH,
        "ytick.major.width": BORDER_WIDTH,

        "xtick.direction": "out",
        "ytick.direction": "out",

        "hatch.linewidth": 0.35,

        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


# ============================================================
# Utility functions
# ============================================================

def table_from_nc(
    ds: xr.Dataset,
    prefix: str
) -> pd.DataFrame:

    column_key = f"{prefix}_columns"

    names = (
        ds.attrs[column_key].split("|")
        if column_key in ds.attrs
        else sorted(
            name.split("__", 1)[1]
            for name in ds.data_vars
            if name.startswith(f"{prefix}__")
        )
    )

    return pd.DataFrame({
        name: ds[f"{prefix}__{name}"].values
        for name in names
    })


def discrete_cmap(colors):

    cmap = ListedColormap(colors)

    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])

    return cmap


# ============================================================
# Map panel
# ============================================================

def map_axis(
    ax,
    title: str
) -> None:

    ax.set_extent(
        MAP_EXTENT,
        crs=ccrs.PlateCarree()
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
        anchor="NW"
    )

    # Land
    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor="#F0F0F0",
        edgecolor="none",
        zorder=2
    )

    # Coastline
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        edgecolor="black",
        linewidth=BORDER_WIDTH,
        zorder=5
    )

    # Borders
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        edgecolor="black",
        linewidth=0.35,
        zorder=5
    )

    west, east, south, north = MAP_EXTENT
    lon_ticks = np.arange(west, east + 1e-8, MAP_LON_TICK_STEP)
    lat_ticks = np.arange(south, north + 1e-8, MAP_LAT_TICK_STEP)

    ax.set_xticks(
        lon_ticks,
        crs=ccrs.PlateCarree()
    )

    ax.set_yticks(
        lat_ticks,
        crs=ccrs.PlateCarree()
    )

    ax.set_xticklabels(
        [
            f"{abs(int(x))}°W"
            if x < 0
            else (
                f"{int(x)}°E"
                if x > 0
                else "0°"
            )
            for x in lon_ticks
        ],
        fontsize=MAP_TICK_SIZE
    )

    ax.set_yticklabels(
        [
            f"{int(y)}°N"
            if y > 0
            else "0°"
            for y in lat_ticks
        ],
        fontsize=MAP_TICK_SIZE
    )

    ax.tick_params(
        axis="both",
        which="major",
        top=False,
        right=False,
        length=2.5,
        width=BORDER_WIDTH,
        labelsize=MAP_TICK_SIZE
    )

    ax.tick_params(
        which="minor",
        length=0,
        width=0
    )

    # Grid lines follow the same tick spacing
    ax.gridlines(
        draw_labels=False,
        xlocs=lon_ticks,
        ylocs=lat_ticks,
        linewidth=0.5,
        color=GRID_COLOR,
        linestyle="--",
        alpha=0.8
    )

    # MDR box
    ax.add_patch(
        Rectangle(
            (
                MDR_LON[0],
                MDR_LAT[0]
            ),
            MDR_LON[1] - MDR_LON[0],
            MDR_LAT[1] - MDR_LAT[0],

            facecolor="none",
            edgecolor="black",

            linewidth=BORDER_WIDTH,

            transform=ccrs.PlateCarree(),
            zorder=7
        )
    )

    ax.set_title(
        title,
        loc="left",
        pad=3,
        fontweight="bold",
        fontsize=MAP_TITLE_SIZE
    )


# ============================================================
# Colorbar for panel (a)
# ============================================================

def colorbar(
    fig,
    ax,
    mesh,
    ticks
) -> None:

    # --------------------------------------------------------
    # Vertical position of colorbar
    #
    # Original:
    # bbox_to_anchor=(0, -0.45, 1, 1)
    #
    # Modified:
    # bbox_to_anchor=(0, -0.32, 1, 1)
    #
    # Less negative = closer to the map
    # e.g. -0.28 -> even closer
    #      -0.38 -> farther away
    # --------------------------------------------------------

    cax = inset_axes(
        ax,

        width="80%",
        height="7.0%",

        loc="lower center",

        bbox_to_anchor=(
            0,
            -0.32,
            1,
            1
        ),

        bbox_transform=ax.transAxes,

        borderpad=0
    )

    cb = fig.colorbar(
        mesh,
        cax=cax,

        orientation="horizontal",

        ticks=ticks,

        extend="both",

        spacing="uniform"
    )

    cb.outline.set_linewidth(
        BORDER_WIDTH
    )

    # Colorbar tick font
    cb.ax.tick_params(
        which="major",

        length=2,
        width=BORDER_WIDTH,

        pad=0,

        labelsize=CBAR_TICK_SIZE
    )

    cb.ax.minorticks_off()

    # Colorbar label font
    cb.set_label(
        r"",

        labelpad=2,

        fontsize=CBAR_LABEL_SIZE
    )


# ============================================================
# Violin plot axes
# ============================================================

def style_axis(ax) -> None:

    # The equal grid cells and matching aspect give each frame the map's dimensions.
    ax.set_box_aspect(VIOLIN_BOX_ASPECT)
    ax.set_anchor("NW")
    ax.set_xlim(*VIOLIN_XLIM)

    for spine in ax.spines.values():

        spine.set_visible(True)

        spine.set_color(
            "black"
        )

        spine.set_linewidth(
            BORDER_WIDTH
        )

    ax.tick_params(
        axis="both",

        which="major",

        top=False,
        right=False,

        length=2.5,

        width=BORDER_WIDTH,

        labelsize=VIOLIN_TICK_SIZE
    )

    ax.tick_params(
        which="minor",

        length=0,

        width=0
    )

    ax.grid(
        True,

        which="major",

        color=GRID_COLOR,

        linestyle="--",

        linewidth=0.5,

        alpha=0.8
    )

    ax.grid(
        False,

        which="minor"
    )


# ============================================================
# Violin panel
# ============================================================

def violin_panel(
    ax,
    ctl,
    ggw,
    title,
    ylabel,
    p_value,
    value_fmt=".1f",
    ylim=None,
    yticks=None
) -> None:

    violin = ax.violinplot(

        [ctl, ggw],

        positions=[0, 1],

        widths=0.72,

        showmeans=False,

        showmedians=False,

        showextrema=False,

        bw_method=0.35
    )

    # --------------------------------------------------------
    # Violin colors
    # --------------------------------------------------------

    for body, color in zip(
        violin["bodies"],
        [
            CTL_COLOR,
            GGW_COLOR
        ]
    ):

        body.set_facecolor(
            color
        )

        body.set_edgecolor(
            "black"
        )

        body.set_linewidth(
            BORDER_WIDTH
        )

        body.set_alpha(
            0.72
        )

    # --------------------------------------------------------
    # Quartiles
    # --------------------------------------------------------

    quartiles = np.percentile(

        np.vstack([
            ctl,
            ggw
        ]),

        [
            25,
            50,
            75
        ],

        axis=1
    )

    ax.vlines(

        [0, 1],

        quartiles[0],

        quartiles[2],

        color="black",

        linewidth=3.2,

        zorder=3
    )

    # Median points
    ax.scatter(

        [0, 1],

        quartiles[1],

        s=15,

        color="white",

        edgecolor="black",

        linewidth=0.55,

        zorder=4
    )

    # --------------------------------------------------------
    # Point jitter
    # --------------------------------------------------------

    jitter_ctl = (

        np.linspace(
            -0.025,
            0.025,
            len(ctl),
            endpoint=False
        )

        if len(ctl)

        else np.array([])
    )

    jitter_ggw = (

        np.linspace(
            -0.025,
            0.025,
            len(ggw),
            endpoint=False
        )

        if len(ggw)

        else np.array([])
    )

    # CTL individual points
    ax.scatter(

        jitter_ctl,

        ctl,

        s=9,

        color=CTL_COLOR,

        edgecolor="black",

        linewidth=0.3,

        alpha=0.82,

        zorder=2
    )

    # GGW individual points
    ax.scatter(

        1 + jitter_ggw,

        ggw,

        s=9,

        color=GGW_COLOR,

        edgecolor="black",

        linewidth=0.3,

        alpha=0.82,

        zorder=2
    )

    # --------------------------------------------------------
    # Mean values
    # --------------------------------------------------------

    ax.scatter(

        [0, 1],

        [
            np.mean(ctl),
            np.mean(ggw)
        ],

        marker="D",

        s=26,

        color=[
            CTL_COLOR,
            GGW_COLOR
        ],

        edgecolor="black",

        linewidth=0.65,

        zorder=5
    )

    # --------------------------------------------------------
    # CTL / GGW / p value text
    # --------------------------------------------------------

    ax.text(

        *STATS_TEXT_POSITION,

        (
            f"CTL={np.mean(ctl):{value_fmt}}\n"
            f"GGW={np.mean(ggw):{value_fmt}}\n"
            f"p={p_value:.2f}"
        ),

        transform=ax.transAxes,

        ha="right",

        va="top",

        fontsize=STATS_TEXT_SIZE,

        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "pad": 1.5
        }
    )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xticks(
        [0, 1],

        [
            "CTL",
            "GGW"
        ]
    )

    # --------------------------------------------------------
    # Y-axis label
    # --------------------------------------------------------

    ax.set_ylabel(
        ylabel,
        fontsize=VIOLIN_LABEL_SIZE
    )

    if ylim is not None:

        ax.set_ylim(
            *ylim
        )

    if yticks is not None:

        ax.set_yticks(
            yticks
        )

    # --------------------------------------------------------
    # Panel title
    # --------------------------------------------------------

    ax.set_title(

        title,

        loc="left",

        pad=3,

        fontweight="bold",

        fontsize=VIOLIN_TITLE_SIZE
    )

    style_axis(
        ax
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    plt.switch_backend(
        "Agg"
    )

    setup_style()
    if MAP_DATA_DIR.is_dir():
        cartopy_config["pre_existing_data_dir"] = str(MAP_DATA_DIR)

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    with xr.open_dataset(
        DATA_FILE
    ) as source:

        ds = source.load()

    stats = pd.read_csv(
        STATS_FILE
    ).set_index(
        "metric"
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = plt.figure(

        figsize=tuple(size_mm / 25.4 for size_mm in FIGSIZE_MM),

        facecolor="white"
    )

    grid = fig.add_gridspec(

        2,
        2,

        wspace=0.34,

        hspace=0.62
    )


    # ========================================================
    # (a) GGW minus CTL track-density response
    # ========================================================

    ax_a = fig.add_subplot(

        grid[0, 0],

        projection=ccrs.PlateCarree()
    )

    diff_levels = np.array([
        -1.5,
        -1.0,
        -0.6,
        -0.3,
        -0.1,
        0.1,
        0.3,
        0.6,
        1.0,
        1.5
    ])

    diff_ticks = [
        -1.5,
        -0.6,
        0.0,
        0.6,
        1.5
    ]

    diff_cmap = discrete_cmap([
        "#053061",
        "#2166AC",
        "#67A9CF",
        "#D1E5F0",
        "#FFFFFF",
        "#FDDBC7",
        "#EF8A62",
        "#B2182B",
        "#67001F"
    ])

    mesh = ax_a.contourf(

        ds.longitude,

        ds.latitude,

        ds.difference_smoothed.values,

        levels=diff_levels,

        cmap=diff_cmap,

        norm=BoundaryNorm(
            diff_levels,
            diff_cmap.N
        ),

        extend="both",

        transform=ccrs.PlateCarree(),

        zorder=0
    )


    # --------------------------------------------------------
    # Statistical significance
    # --------------------------------------------------------

    significance = (
        ds.paired_t_p.values < 0.05
    )

    if np.any(
        significance
    ):

        hatch = ax_a.contourf(

            ds.longitude,

            ds.latitude,

            np.where(
                significance,
                1.0,
                np.nan
            ),

            levels=[
                0.5,
                1.5
            ],

            colors="none",

            hatches=[
                "////"
            ],

            transform=ccrs.PlateCarree(),

            zorder=4
        )

        hatch.set_edgecolor(
            "white"
        )

        hatch.set_linewidth(
            0
        )


    map_axis(
        ax_a,
        "(a) GGW - CTL track density"
    )

    colorbar(
        fig,
        ax_a,
        mesh,
        diff_ticks
    )


    # (b) Original annual JAS AEW frequency, relocated.
    frequency_row = stats.loc[
        "AEW frequency"
    ]

    violin_panel(

        fig.add_subplot(
            grid[0, 1]
        ),

        ds.ctl_aew_frequency.values,

        ds.ggw_aew_frequency.values,

        "(b) Basin AEW frequency",

        "AEWs per JAS",

        frequency_row.paired_permutation_p_two_sided,

        value_fmt=".2f",

        ylim=FREQUENCY_YLIM,

        yticks=FREQUENCY_YTICKS
    )


    # (c) Original annual MDR/JAS track density, relocated from original panel D.
    density_row = stats.loc[
        "MDR AEW density"
    ]

    violin_panel(

        fig.add_subplot(
            grid[1, 0]
        ),

        ds.ctl_mdr_aew_density.values,

        ds.ggw_mdr_aew_density.values,

        "(c) MDR AEW track density",

        "AEW density in MDR per JAS",

        density_row.paired_permutation_p_two_sided,

        value_fmt=".1f",

        ylim=(
            230.0,
            540.0
        ),

        yticks=np.arange(
            250.0,
            551.0,
            50.0
        )
    )


    # (d) New annual point-weighted mean intensity inside the MDR during JAS.
    point_intensity_row = stats.loc[
        "MDR point-mean intensity"
    ]

    violin_panel(

        fig.add_subplot(
            grid[1, 1]
        ),

        ds.ctl_mdr_point_intensity.values,

        ds.ggw_mdr_point_intensity.values,

        "(d) MDR mean AEW intensity",

        r"MDR mean Vort ($10^{-6}$ s$^{-1}$)",

        point_intensity_row.paired_permutation_p_two_sided,

        value_fmt=".2f",

        ylim=POINT_INTENSITY_YLIM,

        yticks=POINT_INTENSITY_YTICKS
    )


    # ========================================================
    # Overall spacing
    # ========================================================

    fig.subplots_adjust(

        left=0.095,

        right=0.97,

        bottom=0.09,

        top=0.94
    )


    # ========================================================
    # Save
    # ========================================================

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    stem = (
        OUTPUT_DIR
        / "figure_aew_frequency_peak_point_intensity_4panel"
    )

    fig.savefig(

        stem.with_suffix(
            ".png"
        ),

        dpi=600,

        bbox_inches="tight",

        facecolor="white"
    )

    fig.savefig(

        stem.with_suffix(
            ".pdf"
        ),

        bbox_inches="tight",

        facecolor="white"
    )

    fig.savefig(

        stem.with_suffix(
            ".svg"
        ),

        bbox_inches="tight",

        facecolor="white"
    )

    fig.savefig(

        stem.with_suffix(
            ".tiff"
        ),

        dpi=600,

        bbox_inches="tight",

        facecolor="white"
    )

    plt.close(
        fig
    )


if __name__ == "__main__":
    main()
