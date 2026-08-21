# Figure source-data package v2.1

This directory contains the processed, figure-ready data and standalone Python plotting scripts for the manuscript *Great Green Wall increases Tropical Cyclones over the North Atlantic*. The contents follow the figure mapping supplied for version 2.1.

## Contents

| Figure | Folder | Script | Input file(s) | Description |
|---|---|---|---|---|
| Figure 1 | `figure1` | `figure1.py` | `figure1_data.nc` | Experimental design and prescribed vegetation modification |
| Figure 2 | `figure2` | `figure2.py` | `figure2_albedo_ctl.nc`; `figure2_albedo_diff.nc`; `figure2_fsns_diff.nc`; `figure2_latent_heat_diff.nc`; `figure2_sensible_heat_diff.nc` | Surface albedo and surface-energy-flux responses |
| Figure 3 | `figure3` | `figure3.py` | `figure3_data.nc` | Moist-static-energy cross-section |
| Figure 4 | `figure4` | `figure4.py` | `figure4_data.nc` | CTL precipitation and GGW-induced precipitation and 850-hPa wind responses |
| Figure 5 | `figure5` | `figure5.py` | `figure5_data.nc` | SST, sea-level pressure, near-surface wind, latent-heat-flux, and cloud responses |
| Figure 6 | `figure6` | `figure6.py` | `figure6_row1_850_circulation.nc`; `figure6_row2_500_omega.nc`; `figure6_row3_700_rh.nc`; `figure6_row4_200_shear.nc` | Large-scale circulation, ascent, midlevel humidity, and vertical-wind-shear responses |
| Figure 7 | `figure7` | `figure7.py` | `figure7_data.nc` | Meridional atmospheric circulation cross-section |
| Figure 8 | `figure8` | `figure8.py` | `figure8_data.nc` | Observed and simulated tropical-cyclone genesis frequency |
| Figure 9 | `figure9` | `figure9.py` | `figure9_data.nc` | Tropical-cyclone genesis indices and factor decomposition |
| Figure 10 | `figure10` | `figure10.py` | `figure10_data.nc` | Tropical-cyclone track-density response |
| Figure 11 | `figure11` | `figure11.py` | `figure11_data.nc` | Pressure-weighted tropical-cyclone steering flow |
| Figure 12 | `figure12` | `figure12.py` | `figure12_data.nc` | 200-, 500-, and 850-hPa circulation responses |
| Figure S1 | `figureS1` | `figureS1.py` | `figureS1_data.nc` | Precipitation and 850-hPa circulation validation |
| Figure S2 | `figureS2` | `figureS2.py` | `figureS2_data.nc` | IBTrACS and CTL seasonal-cycle and spatial-distribution comparison |
| Figure S3 | `figureS3` | `figureS3.py` | `figureS3_data.nc` | Uncoupled-versus-coupled precipitation and 850-hPa circulation responses |
| Figure S4 | `figureS4` | `figureS4.py` | `figureS4_data.nc`; `figureS4_statistics.csv` | African easterly-wave seed frequency and MDR density response |
| Figure S5 | `figureS5` | `figureS5.py` | `figureS5_data.nc` | Dust aerosol optical-depth response |

## Reproduction

Each plotting script reads only the data file or files stored in the same figure folder. Paths are resolved relative to the script location, so no author-specific local path is required.

Install the listed Python dependencies and run a script from any working directory, for example:

```bash
pip install -r requirements.txt
python figure1/figure1.py
python figureS2/figureS2.py
```

The NetCDF and CSV files are processed figure inputs. They are not the complete raw CESM, IBTrACS, or observational archives. Consult the manuscript Methods and Data Availability sections for the original data sources and processing definitions.

## File integrity

SHA-256 checksums for every distributed file are recorded in `checksums.sha256`.
