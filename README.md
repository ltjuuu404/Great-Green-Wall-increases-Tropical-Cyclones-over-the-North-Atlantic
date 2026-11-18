# Great Green Wall increases Tropical Cyclones over the North Atlantic

This repository provides processed figure-ready data and Python plotting scripts for the manuscript and its Supplementary Information.

The current figure source-data package is [`Data_availability_v3.5`](Data_availability_v3.5/README.md), covering Figures 1–11 and S1–S8 in the 17 November 2026 manuscript and supplement. The [figure manifest](Data_availability_v3.5/manifest.csv) lists the processed inputs, plotting script and archived image for each figure.

The archived [`Data_availability_v3.2`](Data_availability_v3.2/README.md) package follows the v3.2 sheet of the author-supplied figure mapping. It includes 19 figure entries, their processed input data, standalone plotting scripts, offline basemaps and SHA-256 checksums. See its README for the source-selection and figure-version notes.

Download the ZIP package and its checksum from [Data Availability v3.2](https://github.com/ltjuuu404/Great-Green-Wall-increases-Tropical-Cyclones-over-the-North-Atlantic/releases/tag/Data_availability_v3.2).

The earlier [`v2.1`](v2.1/README.md) release remains available. All package input paths are resolved relative to the plotting scripts.

The earlier [`data_availability_v2.0`](data_availability_v2.0/README.md) directory is retained as an archived manuscript version.

## Data sources

| Dataset | Use in the study | Access |
| --- | --- | --- |
| NOAA NCEI International Best Track Archive for Climate Stewardship (IBTrACS), version 4, North Atlantic | Observed tropical cyclone genesis and tracks for 1991–2020 (Figures 2, 6 and S2). | [Dataset archive](https://www.ncei.noaa.gov/products/international-best-track-archive) · [North Atlantic CSV (v04r01)](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv) |
| GPCP monthly precipitation, v2.3 (2.5° grid) | 1991–2020 annual precipitation isohyets and July–September precipitation climatology (Figures 1 and S1). | [NOAA PSL NetCDF catalogue](https://psl.noaa.gov/thredds/catalog/Datasets/gpcp/catalog.html?dataset=Datasets/gpcp/precip.mon.mean.nc) · [Direct NetCDF download](https://psl.noaa.gov/thredds/fileServer/Datasets/gpcp/precip.mon.mean.nc) |
| Merged Hadley–NOAA OI sea surface temperature and sea ice concentration (HadOIBl) | Prescribed monthly SST and sea ice in the atmosphere-only experiments. The year-2000 climatology is based on 1995–2005 (Supplementary Text S1). | [NSF NCAR dataset DOI](https://doi.org/10.5065/r33v-sv91) |
| CESM2 standard model inputs and CLM5 year-2000 control surface data | Standard forcings and baseline land cover for the experiments. Figure 1 inputs include processed baseline and GGW vegetation fields. | [CESM2 model input archive](https://www.cesm.ucar.edu/models/cesm2#model-input-data) · [CLM5 control surface NetCDF](https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/lnd/clm2/surfdata_map/release-clm5.0.18/surfdata_1.9x2.5_hist_78pfts_CMIP6_simyr2000_c190304.nc) |
| Study-specific CESM2 and CAM6–CLM5 simulations (CTL, GGW, CTL-Un and GGW-Un) | Processed climate fields, tropical cyclone and African easterly wave diagnostics, genesis potential indices and dust optical depth. | [Processed inputs by figure](Data_availability_v3.5/manifest.csv) |
| Natural Earth vector basemaps | Coastlines, land, ocean and national boundaries in the mapped figures. | [Natural Earth downloads](https://www.naturalearthdata.com/downloads/) · [Bundled shapefiles](Data_availability_v3.5/map_data/) |

The model files linked through the figure manifest are processed source data for reproducing the figures. Full simulation archives are not deposited in this repository.
