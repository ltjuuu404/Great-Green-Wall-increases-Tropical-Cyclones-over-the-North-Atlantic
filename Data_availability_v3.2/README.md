# Figure source-data package v3.2

This package provides processed figure inputs and standalone Python plotting scripts for *Great Green Wall increases Tropical Cyclones over the North Atlantic*. The 19 figure entries follow the **v3.2 sheet** of the author-supplied `data_availability.xlsx`, read on 6 November 2026. The original workbook and original source scripts are not modified.

## Contents

| Excel figure | Folder | Script | Input files | Description |
|---|---|---|---|---|
| Figure 1 | `figure1` | `figure1.py` | `figure1_data.nc` | Vegetation coverage and experimental design |
| Figure 2 | `figure2` | `figure2.py` | `selected_data/seasonal_cycle_ibtracs_ctl_spatial_combined.nc` | TC genesis seasonal cycle and spatial validation |
| Figure 3 | `figure3` | `figure3.py` | `JAS_MSE_smoothed_data.nc` | Moist static energy cross-sections |
| Figure 4 | `figure4` | `figure4.py` | `ggw_ctl_six_field_response_with_pvalues_and_wind_precip.nc` | SST, SLP, winds, heat flux, cloud and precipitation responses |
| Figure 5 | `figure5` | `figure5.py` | `row1_850_circulation.nc`; `row2_500_omega.nc`; `row3_700_rh.nc`; `row4_200_shear.nc` | Large-scale environmental conditions and circulation |
| Figure 6 | `figure6` | `figure6.py` | `JAS_TC_genesis_2p5deg_processed.nc` | JAS observed and simulated TC genesis |
| Figure 7 | `figure7` | `figure7.py` | `figure3_data_ocean_masked_new.nc` | DGPI and ENGPI climatology, response and decomposition |
| Figure 8 | `figure8` | `figure8.py` | `gpi_comparison_all30y_raw.nc`; `paper_figure9_rh700_reference.nc` | Coupled and uncoupled GPI comparison |
| Figure 9 | `figure9` | `figure9.py` | `figure8_data.nc` | Steering flow: Excel Figure9 entry, not the manuscript track-density panel |
| Figure 10 | `figure10` | `figure10.py` | `figure8_data.nc` | Steering flow from the legacy script explicitly listed in Excel |
| Figure 11 | `figure11` | `figure11.py` | `figure12_wind_data.nc` | 200, 500 and 850 hPa wind anomalies |
| Figure S1 | `figureS1` | `figureS1.py` | `figure_r2_data.nc` | GPCP and CTL precipitation validation |
| Figure S2 | `figureS2` | `figureS2.py` | `selected_data/tc_track_density_ibtracs_control_jas_2p5deg.nc` | IBTrACS and CTL TC track-density validation |
| Figure S3 | `figureS3` | `figureS3.py` | `figure2_albedo_ctl.nc`; `figure2_albedo_diff.nc`; `figure2_fsns_diff.nc`; `figure2_latent_heat_diff.nc`; `figure2_sensible_heat_diff.nc` | Surface albedo and surface energy fluxes |
| Figure S4 | `figureS4` | `figureS4.py` | `figure_r8_data.nc` | Coupled and uncoupled precipitation and 1000/200 hPa wind comparison |
| Figure S5 | `figureS5` | `figureS5.py` | `figure_data.nc`; `config.json` | Coupled and uncoupled vorticity, vertical motion and RH600 comparison |
| Figure S6 | `figureS6` | `figureS6.py` | `JAS_smoothed_data.nc` | Meridional circulation cross-section |
| Figure S7 | `figureS7` | `figureS7.py` | `figure_r6_data.nc`; `aew_frequency_mdr_density_statistics.csv` | AEW track density, intensity and frequency: selected current script version |
| Figure S8 | `figureS8` | `figureS8.py` | `figure_r5_data.nc` | Dust aerosol optical-depth response |

## Reproduction

Use Python 3.9 or later with the dependencies in `requirements.txt`. Install Times New Roman separately if it is not available; the selected scripts retain their original font settings. Keep the complete package directory, including `map_data`, together.

```bash
pip install -r requirements.txt
python figure1/figure1.py
python figureS2/figureS2.py
```

Each script resolves its inputs relative to its own directory and uses an Agg backend for batch rendering. The original plotting parameters and scientific calculations are retained. For source scripts that only displayed a figure, the packaged version saves that figure to `outputs/figureN.png` at 300 dpi before display. Figure S4 exports use the shorter figureS4 filename to avoid the Windows path-length limit. Other scripts retain their source export filenames and output directories. Run one script at a time; some legacy source layouts are unusually tall or dense.

The included NetCDF and CSV files are the processed inputs actually read by the selected plotting scripts. They do not include the complete raw model or observational archives. Raw-data extraction scripts are not required to plot these prepared inputs and are not distributed as standalone reproductions. Figure S5 retains its configuration file; the raw-data locations in its `cases` metadata are not read during plotting.

## Mapping and version notes

- **Figure 9 follows the Excel mapping exactly.** Its selected source draws a two-panel steering-flow figure, whereas Figure 9 of the previously audited manuscript is a TC track-density response. This package does not silently substitute a different script. The entry must be corrected in the workbook before claiming manuscript Figure 9 reproduction.
- Figure 10 uses the separate legacy script explicitly listed in Excel. Its absolute `E:/Data_Availability/figure8/figure8_data.nc` input was found and copied, and the packaged script reads that copy.
- Directory entries use the plotting script rather than data-preparation helpers. Figure 11 uses `plot_figure12_new_recut.py`, the recut plotting variant in the specified directory. The original selected script for every entry is recorded in `manifest.csv`.
- Figure 5 retains the source `row3_700_rh.nc` data and its source labels, including any pressure-label inconsistency. The source also uses a very tall `(6, 100)` inch canvas. These source choices have not been scientifically or stylistically corrected for this release.
- Figure 8 retains the source comparison of the original coupled paper fields with the current 30-year uncoupled fields, including its stated differing seasonal definitions. It is not a new harmonized analysis.
- Figure S7 follows the current `02_plot_new.py` version in Excel, including its panel order. That order differs from the previously audited Supplementary Information figure. Update the manuscript caption before substituting a newly generated image.
- Figure S6 retains the selected source's 70 degrees W to 17.5 degrees E averaging domain and vertical-vector scaling. Packaging does not change those definitions.

## Provenance and integrity

`manifest.csv` lists figures, plotting scripts, inputs and source worksheet cells. `source_excel_mapping.csv` preserves the exact v3.2 path entries. `provenance.csv` records original source paths and SHA-256 hashes; distributed scientific data are byte-identical copies. Only paths, batch-backend setup and missing file exports are adapted in the plotting scripts.

`checksums.sha256` records the SHA-256 digest of every distributed file except itself. The Release contains a ZIP archive and a separate SHA-256 checksum for that archive.
