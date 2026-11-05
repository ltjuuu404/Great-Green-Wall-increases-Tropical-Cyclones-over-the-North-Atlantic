# Package validation

Validation date: 6 November 2026.

- All 19 selected plotting scripts were executed successfully in a separate relocated copy, from a working directory outside the figure folders, using Python 3.9.13 and the locally installed scientific dependencies.
- Every script produced at least one nonempty PNG, PDF or SVG output. All 27 NetCDF files were opened and loaded successfully with xarray.
- All 29 distributed input files, including the supplementary CSV and JSON configuration, match their source SHA-256 digests.
- Execution checks are recorded in `execution_checks.csv`. These checks establish data access and export, not scientific equivalence to the manuscript or compliance with every journal figure-style rule.
- The source Figure 5 `(6, 100)` inch canvas produces an unusually tall, sparse export. Its source layout is retained.
- The source Figure S4 layout checker reports a wind-key/latitude-label overlap involving the first panel and 40 degrees N, with `STRICT_LAYOUT_CHECK=False`. The source layout is retained. Figure S5's text-layout checker completed without overlaps.
- Figure 7 exports SVG only. Generated QA figures were inspected in a contact sheet where PNG outputs were available; that contact sheet does not constitute full-resolution visual validation of every output.
- The Figure 9 mapping and Figure S7 version differences described in the README remain author-side source-selection issues. They were not silently corrected during packaging.

No full raw-data reconstruction, statistical reanalysis, or manuscript caption revision is asserted by this release.
