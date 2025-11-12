# DFS Contest Analyzer

A Streamlit + DuckDB pipeline for exploring DraftKings NBA contest results. The app ingests a DraftKings contest standings export and matching salaries file, normalizes both datasets, computes exposures/combos/stacks, persists Parquet outputs, and exposes an interactive UI for slicing the contest by percentile or rank.

## Project layout

```
dfsanalyzer/
├── app/
│   ├── app.py                 # Streamlit entry point
│   └── components/            # UI helpers (filters, etc.)
├── core/
│   ├── aggregate.py           # Combo, exposure, stack builders
│   ├── ingest.py              # CSV ingestion + normalization
│   └── util.py                # Shared helpers
├── data/
│   ├── sample_salaries.csv    # Tiny development inputs
│   └── sample_standings.csv   # Tiny development inputs
├── requirements.txt
└── README.md
```

Each ingestion run writes Parquet artifacts to `data/<timestamp>/` so downstream analysis or future app sessions can reuse the derived tables.

## Getting started

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Streamlit:

   ```bash
   streamlit run app/app.py
   ```

4. Use the sidebar to either load the included sample files or upload real DraftKings contest CSVs from the same slate. The UI exposes four tabs:

   * **Overview** – high-level metrics, points histogram, top duplicate lineups, unmatched player diagnostics.
   * **Users** – select a username to inspect their lineups, exposures, and the combos they used within the active percentile/rank filter.
   * **Combos** – explore player-name combos, same-team stacks, or same-game stacks with top-N limits and CSV export.
   * **Field** – review field ownership with search + CSV download.

A shared percentile/rank filter (with quick Top 50/25/10/1% chips) drives all tabs so you can focus on specific contest buckets. CSV downloads respect the filter and only include entries currently in view.

## Notes

* The pipeline performs minimal name normalization before matching standings to salaries. Any remaining mismatches appear under the "Unmatched players" expander.
* Combos persist a list of contributing entry IDs so the app can recalc counts quickly when filters change without recomputing all combinations.
* Update `requirements.txt` or the components under `core/` as DraftKings schemas evolve.
