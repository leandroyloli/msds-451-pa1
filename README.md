# MSDS 451 Programming Assignment 1

Next-day direction of GLD (SPDR Gold Shares). Lagged price features, all-subsets selection by the Akaike information criterion, an XGBoost classifier, and a hold-out window that is not used for selection or tuning.

The written report is [report/msds451-programming-assignment-1.pdf](report/msds451-programming-assignment-1.pdf).

## Result

Development window: 9 January 2006 through 14 December 2023 (4,515 sessions). Ten sessions are dropped. Hold-out: 2 January 2024 through 9 October 2026 (696 sessions).

| Rule | Hold-out accuracy | Hold-out AUC |
| --- | ---: | ---: |
| Always call an up day | 0.569 |  |
| Repeat yesterday's direction | 0.458 | 0.448 |
| Price-level XGBoost | 0.432 | 0.532 |
| Stationary-feature XGBoost | 0.552 | 0.501 |

Calling every day an up day, the majority class on the development window, beats both models. Cross-validated AUC on the development window was already about 0.51. The report explains the price-level model's collapse: 654 of 696 hold-out closes are above every close in the development window.

## Layout

```
analysis.py          download, features, subset search, tuning, hold-out score
build_report.py      PDF from output/metrics.json and figures/
data/                GLD prices and the feature table
figures/             charts cited in the report
output/metrics.json  scores, selected features, hyperparameters
report/              the PDF
```

## Run

Python 3.12. From this directory:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python analysis.py
python build_report.py
```

`analysis.py` downloads GLD from Yahoo Finance, writes `data/`, and refits the search. The subset search scores 32,767 logistic regressions for each feature family and then runs a 30-draw randomized search. On the machine used for this submission that took about 14 minutes.

`python analysis.py --figures-only` redraws the hold-out ROC curves from the hyperparameters already stored in `output/metrics.json`. It does not retune.

The random seed is 2026. The development cut is 31 December 2023, with a 10-session gap before the hold-out.

## How to use AI assistants

Cursor, using Grok 4.7, wrote `analysis.py` and `build_report.py`, ran the search, and drafted the report and this file from `output/metrics.json`. The session started from the prompt "Assigment module 3." The hold-out rows were not used to choose features or the hyperparameters reported in the PDF. The tables in the report match the program output.
