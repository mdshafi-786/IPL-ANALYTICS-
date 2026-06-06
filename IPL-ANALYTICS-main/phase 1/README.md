# From Dots to Damage — How IPL Batsmen Eliminated Dead Balls

An exploratory data analysis project examining how IPL batting has fundamentally transformed over 18 seasons (2007–2025), using ball-by-ball delivery data.

---

## Overview

The central question driving this analysis: **Has the way batsmen score fundamentally shifted, and did it come at a cost?**

Across 12 findings, this project traces the evolution from a dot-ball-heavy game to an era of relentless boundary hitting — and measures what, if anything, bowlers and batsmen gave up in the process.

---

## Key Findings

| # | Finding |
|---|---------|
| 1 | Sixes nearly doubled per match while fours stayed flat |
| 2 | The fours-to-sixes ratio fell from ~3:1 to ~2:1 over 18 seasons |
| 3 | Dot ball % dropped significantly, especially in the Powerplay |
| 4 | Powerplay transformation: dots down, sixes up season by season |
| 5 | All three phases (Powerplay, Middle, Death) now score more runs per over |
| 6 | Boundaries increased most in Powerplay and Death overs |
| 7 | 18-year boundary heatmap shows a clear, sustained upward shift |
| 8 | Dismissals per match stayed flat despite surging aggression |
| 9 | Sixes doubled; wickets did not — aggression came without a cost |
| 10 | Aggression Index (Boundary% + Six% − Dot%) has steadily climbed |
| 11 | Bowler economy rates rose from ~7.2 to ~9.2 runs per over |
| 12 | Dismissal types shifted — more caught, fewer bowled |

---

## Project Structure

```
.
├── IPL_Analysis.ipynb                        # Main analysis notebook
├── deliveries_updated_ipl_upto_2025.csv      # Ball-by-ball delivery data
├── matches_updated_ipl_upto_2025.csv         # Match-level metadata
├── assets/                                   # Output charts folder
│   ├── 01_sixes_vs_fours_per_match.png
│   ├── 02_ratio_fours_vs_sixes_per_season.png
│   ├── 03_powerplay_dot_ball_revolution.png
│   ├── 04_powerplay_dots_vs_sixes.png
│   ├── 05_runs_per_over_by_phase.png
│   ├── 06_boundary_pct_by_over.png
│   ├── 07_boundary_heatmap.png
│   ├── 08_dismissals_per_match.png
│   ├── 09_sixes_vs_dismissals.png
│   ├── 10_aggression_index.png
│   ├── 11_economy_rate_by_season.png
│   └── 12_dismissal_types_by_era.png
└── README.md
```

---

## Data

Two CSV files power this analysis:

- **`deliveries_updated_ipl_upto_2025.csv`** — Every ball bowled across all IPL seasons, including batsman runs, extras, dismissal kind, over number, and more.
- **`matches_updated_ipl_upto_2025.csv`** — Match-level data including season, venue, teams, and match result.

The two tables are merged on `matchId` so each delivery record carries its season context.

---

## Feature Engineering

Four helper columns are derived during analysis:

- **`total_runs`** — Batsman runs + extras per ball
- **`over_num`** — 1-indexed over number (raw data is 0-indexed)
- **`phase`** — Innings phase: Powerplay (1–6), Middle (7–15), or Death (16–20)
- **`era`** — Early (2007–2015) vs Recent (2016–2025) for era comparisons

---

## Aggression Index

A composite metric created to capture batting intent in a single number:

```
Aggression Index = Boundary% + Six% − Dot%
```

Higher values indicate more attacking batting. The index has risen steadily across all 18 IPL seasons.

---

## Requirements

```
python >= 3.8
pandas
numpy
matplotlib
seaborn
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn
```

---

## Usage

1. Clone or download the repository.
2. Place both CSV data files in the same directory as the notebook.
3. Open `IPL_Analysis.ipynb` in Jupyter and run all cells.

Output charts are saved as `.png` files (150 dpi) in the `assets/` directory.

---

## Conclusion

Over 18 IPL seasons, batting has been fundamentally reprogrammed. Batsmen converted dot balls into boundaries — especially in the Powerplay — without getting out more often. This wasn't reckless; it was a skill upgrade. Bowlers paid the price, with economy rates climbing from 7.2 to 9.2 runs per over.
