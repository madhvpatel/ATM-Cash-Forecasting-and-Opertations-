# Maharashtra simulated run

Eight Maharashtra ATMs, 1 Sep 2024 through 24 Sep 2026. Demand is the synthetic daily cash-out already in the extract. Exogenous series are public calendars and city weather. Footfall, demographics, festival scores, cassette capacity, and routes were not filled in.

Nothing in the repo imports these notes. `data/README.md` describes the raw catalog. This file describes only this run.

```bash
python3 simulated/maharashtra/run_simulated.py
```

Weather JSON for that command sits in `/tmp/mh-wx/` (Open-Meteo archive, retrieved 2026-10-21). The script copies a city-day table into `data/weather_city_daily.csv`. Assumptions are in `assumptions.json`. Results are in `output/`. The read of those tables is in `FINDINGS.md`.

A mock switch extract for SYNATM0001 is `data/switch_log_SYNATM0001.csv`, rebuilt with `python3 simulated/maharashtra/mock_switch_log.py`. Field meanings are in `data/switch_log_SYNATM0001_fields.csv`.
