# Portal masters

Display tables for a later frontend. `index.json` is the entry point. It lists
every file written here, with size and SHA-256. `dataset.csv` lists those
files plus the canonical tables they were built from.

Two ATM populations stay separate. `cash_files` is terminal ids 1–100.
`legacy_demo` is terminal ids 1001–1006. Do not join them.

Blank cells are unknown. Device type, cassette capacity, verified site type,
coordinates, uptime, failed withdrawals, footfall, and local events were not
collected for the 100 synthetic ATMs. `unavailable.csv` is the list to show
instead of filling those fields.

`location_label` is the text after ` - ` in the location string. It is not a
verified site classification. State and RBI office are a curated proxy.
Regional holidays come from the RBI office matrix with a blank `known_at`, so
they are retrospective. Payday is the assumed 1st, 15th, and last-weekday rule.

`cash_summary.json` rolls up `net_demand` from `data/processed/daily_cash_out.csv`.
The denomination value 100 is a rounding unit.

Weather coordinates cover the four Maharashtra city centroids used by
`simulated/maharashtra/`. That series is ERA5 reanalysis. The governed feature
registry still marks weather unavailable. The switch field list applies only
to the SYNATM0001 mock extract.

Cassette rows live only in `data/master/legacy_demo/dim_cassette_configuration.csv`.

Regenerate with `python3 scripts/prepare_portal_masters.py`.
