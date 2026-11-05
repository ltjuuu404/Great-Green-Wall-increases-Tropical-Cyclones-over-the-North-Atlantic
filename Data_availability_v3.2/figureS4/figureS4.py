from __future__ import annotations

"""Plot precipitation with 1000-hPa wind and the original 200-hPa wind.

All panels omit statistical tests and significance overlays.
"""

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


from pathlib import Path
import json
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.quiver import Quiver, QuiverKey
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "figure_r8_data.nc"
OUTPUT_DIR = HERE / "outputs"
FONT = "Times New Roman"
FONT_SIZE = 22
BORDER_WIDTH = 0.75 
MAP_BORDER_WIDTH = 2  # Map frame and both major/minor tick line widths, in pt.
FIGSIZE = (17.5, 15.0)  # Keep map widths across the wider latitude range.
TITLE_Y = 1.00  # Initial drawing position; WIND_KEY_Y controls the automatic header placement.
TITLE_KEY_OFFSET_PT = 0.0  # Zero aligns the title bottom with the reference-arrow bottom.
WIND_KEY_REFERENCE = 1.0  # m s-1; retain the user's requested reference value.
WIND_SCALE_MULTIPLIER = 2.0  # Larger scale shortens all wind arrows; 2 gives about half length.
WIND_KEY_X_LEFT, WIND_KEY_X_RIGHT = 0.97, 0.97
WIND_KEY_Y, WIND_KEY_LABELSEP = 1.02, 0.14
STRICT_LAYOUT_CHECK = False  # True blocks export on overlaps; False permits manual styling.
MAJOR_TICK_LENGTH, MINOR_TICK_LENGTH = 6.0, 3.0
EXTENT = (-100.0, 45.0, -10.0, 40.0)
MDR_BOX = (-70.0, -17.5, 10.0, 20.0)
LON_TICK_START, LON_TICK_END, LON_TICK_STEP = -90.0, 30.0, 30.0
LAT_TICK_START, LAT_TICK_END, LAT_TICK_STEP = 0.0, 40.0, 20.0
LAT_MINOR_TICK_STEP = 10.0
MINOR_TICKS_PER_INTERVAL = 1
LON_TICKS = np.arange(LON_TICK_START, LON_TICK_END + 1e-8, LON_TICK_STEP)
LAT_TICKS = np.arange(LAT_TICK_START, LAT_TICK_END + 1e-8, LAT_TICK_STEP)
LON_MINOR_TICKS = np.array([left + (right-left)*j/(MINOR_TICKS_PER_INTERVAL+1)
                          for left, right in zip(LON_TICKS[:-1], LON_TICKS[1:])
                          for j in range(1, MINOR_TICKS_PER_INTERVAL+1)])
LAT_MINOR_TICKS = np.arange(np.ceil(EXTENT[2]/LAT_MINOR_TICK_STEP)*LAT_MINOR_TICK_STEP,
                            EXTENT[3] + 1e-8, LAT_MINOR_TICK_STEP)
LAT_MINOR_TICKS = LAT_MINOR_TICKS[
    ~np.isclose(LAT_MINOR_TICKS[:, None], LAT_TICKS[None, :]).any(axis=1)]
LEVELS = [-2.0, -1.5, -1.0, -0.75, -0.5, -0.3, -0.1, -0.05,
          0.0, 0.05, 0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
WIND200_LEVELS = [2.0 * value for value in LEVELS]
WIND200_COLORS = mpl.colormaps["RdBu_r"](np.linspace(0.03, 0.97, 16))
NEGATIVE_COLORS = [
    "#6b3d0c", "#8c510a", "#a6611a", "#bf812d",
    "#dfc27d", "#ead89d", "#f2e5c4", "#f8f0dc",
]
POSITIVE_COLORS = [
    "#e5f5f3", "#d0ece8", "#b5e1dc", "#80cdc1",
    "#5ab4ac", "#359b92", "#16867d", "#01665e",
]
TITLES = [
    "(a) UC: Precip + 1000 hPa wind", "(b) UC: 200 hPa wind",
    "(c) C: Precip + 1000 hPa wind", "(d) C: 200 hPa wind",
    "(e) C - UC: Precip + 1000 hPa wind", "(f) C - UC: 200 hPa wind",
]


def format_map(ax) -> None:
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_aspect("equal", adjustable="box")
    # No land fill or land/ocean data mask; the data colours cover the map.
    ax.set_facecolor("white")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=BORDER_WIDTH, edgecolor="black", zorder=6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="black", zorder=6)
    ax.add_patch(Rectangle((MDR_BOX[0], MDR_BOX[2]), MDR_BOX[1]-MDR_BOX[0], MDR_BOX[3]-MDR_BOX[2],
                           fill=False, edgecolor="black", linewidth=BORDER_WIDTH,
                           transform=ccrs.PlateCarree(), zorder=8))
    ax.set_xticks(LON_TICKS, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_TICKS, crs=ccrs.PlateCarree())
    ax.set_xticks(LON_MINOR_TICKS, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(LAT_MINOR_TICKS, minor=True, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="major", direction="out", length=MAJOR_TICK_LENGTH,
                   width=MAP_BORDER_WIDTH, bottom=True, left=True, top=False, right=False,
                   labelbottom=True, labelleft=True, labeltop=False, labelright=False)
    ax.tick_params(which="minor", direction="out", length=MINOR_TICK_LENGTH,
                   width=MAP_BORDER_WIDTH-0.5, bottom=True, left=True, top=False, right=False,
                   labelbottom=False, labelleft=False, labeltop=False, labelright=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(MAP_BORDER_WIDTH)
    ax.gridlines(draw_labels=False, linewidth=0.5, color="0.65", alpha=0.85,
                 linestyle="--", xlocs=LON_TICKS, ylocs=LAT_TICKS)


def draw_panel(fig, ax, lon, lat, field, u, v, title, levels, unit, colors=None) -> None:
    lon2, lat2 = np.meshgrid(lon, lat)
    palette = NEGATIVE_COLORS + POSITIVE_COLORS if colors is None else colors
    cmap = ListedColormap(palette, name="precip_zero_split" if colors is None else "wind200_redblue")
    cmap.set_under(palette[0])
    cmap.set_over(palette[-1])
    norm = BoundaryNorm(levels, cmap.N)
    mesh = ax.contourf(lon2, lat2, field, levels=levels, cmap=cmap, norm=norm,
                       extend="both", transform=ccrs.PlateCarree(), zorder=2)
    ax.contour(lon2, lat2, field, levels=levels, colors="0.25", linewidths=0.30,
               alpha=0.45, transform=ccrs.PlateCarree(), zorder=3)
    format_map(ax)
    step = max(2, int(np.ceil(np.sqrt(np.size(u) / 3000))))
    mag95 = float(np.nanpercentile(np.hypot(u, v), 95))
    q = ax.quiver(lon[::step], lat[::step], u[::step, ::step], v[::step, ::step],
                  transform=ccrs.PlateCarree(), color="black", pivot="middle",
                  scale=max(0.1, mag95 * 8.0) * 1.3 * WIND_SCALE_MULTIPLIER, width=0.0025,
                  headwidth=12, headlength=8.5, headaxislength=2, zorder=5.8)
    key_x = WIND_KEY_X_LEFT if colors is None else WIND_KEY_X_RIGHT
    ax.quiverkey(q, X=key_x, Y=WIND_KEY_Y, U=WIND_KEY_REFERENCE,
                 label=fr"{WIND_KEY_REFERENCE:g} m s$^{{-1}}$",
                 labelpos="N", labelsep=WIND_KEY_LABELSEP, coordinates="axes",
                 fontproperties={"family": FONT, "weight": "normal", "size": FONT_SIZE-1})
    ax.text(0, TITLE_Y, title, transform=ax.transAxes, ha="left", va="center",
            fontsize=FONT_SIZE+4, fontweight="bold", clip_on=False)
    cax = inset_axes(ax, width="90%", height="6%", loc="lower center",
                     bbox_to_anchor=(0, -0.24, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both",
                      spacing="uniform", ticks=levels[::2])
    cb.outline.set_linewidth(BORDER_WIDTH)
    cb.ax.tick_params(which="major", direction="out", length=2.5, width=BORDER_WIDTH,
                      top=False, bottom=True)
    cb.ax.minorticks_off()
    cb.set_label(unit, labelpad=2)


def align_titles_with_wind_keys(fig):
    """Keep TITLE_Y fixed and align the wind-key arrow bottom with the title bottom."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    offset = TITLE_KEY_OFFSET_PT * fig.dpi / 72.0

    maps = [axis for axis in fig.axes if hasattr(axis, "projection")]

    for ax in maps:
        key = next(
            artist for artist in ax.artists
            if isinstance(artist, QuiverKey)
        )

        # 使用 TITLE_Y 控制标题高度，不再自动覆盖
        ax.texts[0].set_y(TITLE_Y)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for ax in maps:
        key = next(
            artist for artist in ax.artists
            if isinstance(artist, QuiverKey)
        )

        title_box = ax.texts[0].get_window_extent(renderer)
        arrow_box = key_arrow_bounds(key)

        displacement = title_box.y0 - offset - arrow_box.y0

        key.Y += displacement / ax.bbox.height
        key.stale = True
    """Keep the current title placement, then align the key-arrow bottom to it."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    offset = TITLE_KEY_OFFSET_PT*fig.dpi/72.0
    maps = [axis for axis in fig.axes if hasattr(axis, "projection")]
    # Preserve the title height used by the previous label-centred layout.
    for ax in maps:
        key = next(artist for artist in ax.artists if isinstance(artist, QuiverKey))
        label_box = key.text.get_window_extent(renderer)
        label_center = (label_box.y0+label_box.y1)/2.0
        title_y = ax.transAxes.inverted().transform((ax.bbox.x0, label_center))[1]
        ax.texts[0].set_y(title_y)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Measure the full arrow polygon, including the lower edge of its head.
    for ax in maps:
        key = next(artist for artist in ax.artists if isinstance(artist, QuiverKey))
        title_box = ax.texts[0].get_window_extent(renderer)
        arrow_box = key_arrow_bounds(key)
        displacement = title_box.y0-offset-arrow_box.y0
        key.Y += displacement/ax.bbox.height
        key.stale = True


def key_arrow_bounds(key):
    """Bounds of the actual reference arrow, including its axes-coordinate offset."""
    vector = key.vector
    offset = vector.get_offset_transform().transform(vector.get_offsets()[0])
    vertices = np.concatenate([vector.get_transform().transform(path.vertices)+offset
                               for path in vector.get_paths()])
    return Bbox.from_extents(*vertices.min(axis=0), *vertices.max(axis=0))


def wind_key_labels(fig):
    # QuiverKey draws its label itself, so tight export needs these extra artists.
    return [artist.text for ax in fig.axes for artist in ax.artists
            if isinstance(artist, QuiverKey)]


def check_layout(fig, wind200_coverage, wind1000_coverage):
    """Check rendered labels, wind keys, map extents and absence of hatches."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    maps = [ax for ax in fig.axes if hasattr(ax, "projection")]
    texts, panels = [], []
    for index, ax in enumerate(fig.axes):
        artists = [*ax.texts, ax.xaxis.label, ax.yaxis.label,
                   *ax.get_xticklabels(), *ax.get_yticklabels()]
        artists += [artist.text for artist in ax.artists if isinstance(artist, QuiverKey)]
        for artist in artists:
            if artist.get_visible() and artist.get_text().strip():
                texts.append((index, artist.get_text(), artist.get_window_extent(renderer)))
    overlaps, intrusions, arrow_overlaps = [], [], []
    for index, (owner, label, box) in enumerate(texts):
        for _, other_label, other in texts[index+1:]:
            if (min(box.x1, other.x1)-max(box.x0, other.x0) > 0.5
                    and min(box.y1, other.y1)-max(box.y0, other.y0) > 0.5):
                overlaps.append([label, other_label])
        for ax in maps:
            if fig.axes[owner] is ax:
                continue
            other = ax.get_window_extent(renderer)
            if (min(box.x1, other.x1)-max(box.x0, other.x0) > 0.5
                    and min(box.y1, other.y1)-max(box.y0, other.y0) > 0.5):
                intrusions.append(label)
    for ax in maps:
        quivers = [artist for artist in ax.collections if isinstance(artist, Quiver)]
        keys = [artist for artist in ax.artists if isinstance(artist, QuiverKey)]
        np.testing.assert_allclose(ax.get_extent(ccrs.PlateCarree()), EXTENT, atol=1e-8)
        np.testing.assert_allclose(ax.get_yticks(), LAT_TICKS)
        np.testing.assert_allclose(ax.get_yticks(minor=True), LAT_MINOR_TICKS)
        assert all(not label.get_text() for label in ax.get_yticklabels(minor=True))
        assert all(tick.tick1line.get_visible() for tick in ax.yaxis.get_minor_ticks())
        assert len(quivers) == len(keys) == 1
        assert "\n" not in ax.texts[0].get_text()
        title_box = ax.texts[0].get_window_extent(renderer)
        arrow_box = key_arrow_bounds(keys[0])
        bottom_offset = title_box.y0-arrow_box.y0
        np.testing.assert_allclose(bottom_offset, TITLE_KEY_OFFSET_PT*fig.dpi/72.0, atol=0.5)
        assert keys[0].U == WIND_KEY_REFERENCE
        for _, label, other in texts:
            if (min(arrow_box.x1, other.x1)-max(arrow_box.x0, other.x0) > 0.5
                    and min(arrow_box.y1, other.y1)-max(arrow_box.y0, other.y0) > 0.5):
                arrow_overlaps.append([ax.texts[0].get_text(), label])
        assert not any(getattr(artist, "get_hatch", lambda: None)()
                       for artist in ax.get_children())
        colorbar = next(candidate._colorbar for candidate in fig.axes
                        if hasattr(candidate, "_colorbar") and candidate._colorbar.mappable.axes is ax)
        box, cb_box = ax.get_position(), colorbar.ax.get_position()
        panels.append({"title": ax.texts[0].get_text(), "extent": list(EXTENT),
                       "latitude_major_ticks": LAT_TICKS.tolist(),
                       "latitude_minor_ticks": LAT_MINOR_TICKS.tolist(),
                       "latitude_minor_labels": False,
                       "map_box": list(box.bounds), "colorbar_box": list(cb_box.bounds),
                       "colorbar_height_ratio": cb_box.height/box.height,
                       "quiver_arrow_count": int(quivers[0].N),
                       "finite_vector_count": int((~quivers[0].Umask).sum())
                       if np.ndim(quivers[0].Umask) else int(quivers[0].N),
                       "single_line_title": True, "land_background_fill": False,
                       "wind_key_reference_m_s": float(keys[0].U),
                       "wind_scale_multiplier": WIND_SCALE_MULTIPLIER,
                       "wind_quiver_scale": float(quivers[0].scale),
                       "wind_key_arrow_width_pt": float(arrow_box.width/fig.dpi*72.0),
                       "title_key_alignment": "title bottom to reference-arrow bottom",
                       "title_key_bottom_offset_pixels": float(bottom_offset),
                       "wind_key_effective_y": float(keys[0].Y),
                       "significance_overlay": False})
    report = {"text_count": len(texts), "text_overlaps": overlaps,
              "adjacent_map_intrusions": intrusions, "wind_key_arrow_text_overlaps": arrow_overlaps,
              "panels": panels,
              "wind200_coverage": wind200_coverage, "wind1000_coverage": wind1000_coverage,
              "wind1000_mask_policy": "none",
              "strict_layout_check": bool(STRICT_LAYOUT_CHECK),
              "passed": not (overlaps or intrusions or arrow_overlaps)}
    (HERE / "plot_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        message = (f"Layout warning: {len(overlaps)} text overlaps, "
                   f"{len(intrusions)} adjacent-map intrusions, "
                   f"{len(arrow_overlaps)} wind-key/text overlaps. "
                   f"Details: {HERE / 'plot_qa.json'}")
        if STRICT_LAYOUT_CHECK:
            raise ValueError(message + " Set STRICT_LAYOUT_CHECK=False to export while adjusting the layout.")
        print(message + " Export continues (STRICT_LAYOUT_CHECK=False).", flush=True)
        return
    print("Layout verified: six maps, 1000/200-hPa wind, no significance overlays.", flush=True)


def main() -> None:
    plt.switch_backend("Agg")
    mpl.rcParams.update({"font.family": FONT, "font.size": FONT_SIZE,
                         "font.weight": "normal", "axes.labelweight": "normal",
                         "axes.linewidth": BORDER_WIDTH,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    with xr.open_dataset(DATA_FILE) as source:
        ds = source.load()
    if float(ds.attrs.get("pressure1000_hpa", 0)) != 1000.0:
        raise ValueError("Run 01_prepare_data.py to extract exact 1000-hPa wind first.")
    if ds.attrs.get("wind1000_mask_policy") != "none":
        raise ValueError("Run 01_prepare_data.py to rebuild 1000-hPa winds without a mask.")
    visible = ds.sel(lon=slice(EXTENT[0], EXTENT[1]), lat=slice(EXTENT[2], EXTENT[3]))
    wind200_coverage = {}
    wind1000_coverage = {}
    for prefix in ("noncoupled", "coupled", "delta"):
        for variable in ("u1000_diff", "v1000_diff"):
            name = f"{prefix}_{variable}"
            missing = int(np.count_nonzero(~np.isfinite(visible[name].values)))
            wind1000_coverage[name] = {"missing_cells": missing, "total_cells": int(visible[name].size)}
            if missing:
                raise ValueError(f"Unexpected gaps in the unmasked 1000-hPa field: {name}")
        for variable in ("wspd200_diff", "u200_diff", "v200_diff"):
            name = f"{prefix}_{variable}"
            missing = int(np.count_nonzero(~np.isfinite(visible[name].values)))
            wind200_coverage[name] = {"missing_cells": missing, "total_cells": int(visible[name].size)}
            if missing:
                raise ValueError(f"Unexpected gaps in the unmasked 200-hPa field: {name}")
    lon, lat = ds.lon.values, ds.lat.values
    fig = plt.figure(figsize=FIGSIZE, dpi=300)
    gs = fig.add_gridspec(3, 2, left=0.045, right=0.985, top=0.935, bottom=0.075,
                         hspace=0.80, wspace=0.13)
    prefixes = ("noncoupled", "coupled", "delta")
    for index, prefix in enumerate(prefixes):
        ax = fig.add_subplot(gs[index, 0], projection=ccrs.PlateCarree())
        draw_panel(fig, ax, lon, lat, ds[f"{prefix}_prect_diff"].values,
                   ds[f"{prefix}_u1000_diff"].values, ds[f"{prefix}_v1000_diff"].values,
                   TITLES[2 * index], LEVELS, r"mm d$^{-1}$")
        ax200 = fig.add_subplot(gs[index, 1], projection=ccrs.PlateCarree())
        draw_panel(fig, ax200, lon, lat, ds[f"{prefix}_wspd200_diff"].values,
                   ds[f"{prefix}_u200_diff"].values, ds[f"{prefix}_v200_diff"].values,
                   TITLES[2 * index + 1], WIND200_LEVELS, r"m s$^{-1}$", colors=WIND200_COLORS)
    align_titles_with_wind_keys(fig)
    check_layout(fig, wind200_coverage, wind1000_coverage)
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = OUTPUT_DIR / "figureS4"
    extras = wind_key_labels(fig)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.06,
                bbox_extra_artists=extras)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04,
                bbox_extra_artists=extras)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04,
                bbox_extra_artists=extras)
    fig.savefig(OUTPUT_DIR / "figureS4_preview.png", dpi=150,
                bbox_inches="tight", pad_inches=0.06, bbox_extra_artists=extras)
    plt.close(fig)


if __name__ == "__main__":
    main()
