"""MSDS 451 Programming Assignment 1.

Predict the direction of the next daily return on GLD from lagged price
features. Feature selection and hyperparameter search stay on a development
window. The hold-out window is scored once.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
OUTPUT = ROOT / "output"

SYMBOL = "GLD"
START = "2006-01-01"
END = "2026-10-11"
DEV_END = "2023-12-31"
GAP = 10
RANDOM_STATE = 2026
N_SEARCH = 30

PRICE_FEATURES = [
    "CloseLag1",
    "CloseLag2",
    "CloseLag3",
    "HMLLag1",
    "HMLLag2",
    "HMLLag3",
    "OMCLag1",
    "OMCLag2",
    "OMCLag3",
    "VolumeLag1",
    "VolumeLag2",
    "VolumeLag3",
    "CloseEMA2",
    "CloseEMA4",
    "CloseEMA8",
]

STATIONARY_FEATURES = [
    "RetLag1",
    "RetLag2",
    "RetLag3",
    "RelHMLLag1",
    "RelHMLLag2",
    "RelHMLLag3",
    "RelOMCLag1",
    "RelOMCLag2",
    "RelOMCLag3",
    "LogVolChgLag1",
    "LogVolChgLag2",
    "LogVolChgLag3",
    "RetEMA2",
    "RetEMA4",
    "RetEMA8",
]


def download_prices() -> pl.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "gld_prices.csv"
    frame = yf.Ticker(SYMBOL).history(start=START, end=END, auto_adjust=True)
    frame = frame.reset_index()
    frame.to_csv(path, index=False)
    prices = pl.read_csv(path, try_parse_dates=True)
    date_col = "Date" if "Date" in prices.columns else prices.columns[0]
    if date_col != "Date":
        prices = prices.rename({date_col: "Date"})
    drop_cols = [c for c in ("Dividends", "Stock Splits", "StockSplits", "Capital Gains") if c in prices.columns]
    prices = prices.drop(drop_cols)
    prices = prices.with_columns(pl.col("Date").cast(pl.Date)).sort("Date")
    prices.write_csv(path)
    return prices


def engineer(prices: pl.DataFrame) -> pl.DataFrame:
    frame = prices
    frame = frame.with_columns(pl.col("Close").shift(1).alias("CloseLag1"))
    frame = frame.with_columns(pl.col("Close").shift(2).alias("CloseLag2"))
    frame = frame.with_columns(pl.col("Close").shift(3).alias("CloseLag3"))
    frame = frame.with_columns(pl.col("Close").shift(4).alias("CloseLag4"))

    frame = frame.with_columns((pl.col("High") - pl.col("Low")).alias("HML"))
    frame = frame.with_columns(pl.col("HML").shift(1).alias("HMLLag1"))
    frame = frame.with_columns(pl.col("HML").shift(2).alias("HMLLag2"))
    frame = frame.with_columns(pl.col("HML").shift(3).alias("HMLLag3"))

    frame = frame.with_columns((pl.col("Open") - pl.col("Close")).alias("OMC"))
    frame = frame.with_columns(pl.col("OMC").shift(1).alias("OMCLag1"))
    frame = frame.with_columns(pl.col("OMC").shift(2).alias("OMCLag2"))
    frame = frame.with_columns(pl.col("OMC").shift(3).alias("OMCLag3"))

    frame = frame.with_columns(pl.col("Volume").shift(1).alias("VolumeLag1"))
    frame = frame.with_columns(pl.col("Volume").shift(2).alias("VolumeLag2"))
    frame = frame.with_columns(pl.col("Volume").shift(3).alias("VolumeLag3"))
    frame = frame.with_columns(pl.col("Volume").shift(4).alias("VolumeLag4"))

    frame = frame.with_columns(pl.col("CloseLag1").ewm_mean(half_life=1, ignore_nulls=True).alias("CloseEMA2"))
    frame = frame.with_columns(pl.col("CloseLag1").ewm_mean(half_life=2, ignore_nulls=True).alias("CloseEMA4"))
    frame = frame.with_columns(pl.col("CloseLag1").ewm_mean(half_life=4, ignore_nulls=True).alias("CloseEMA8"))

    frame = frame.with_columns((pl.col("Close") / pl.col("CloseLag1")).log().alias("LogReturn"))
    frame = frame.with_columns((pl.col("CloseLag1") / pl.col("CloseLag2")).log().alias("RetLag1"))
    frame = frame.with_columns((pl.col("CloseLag2") / pl.col("CloseLag3")).log().alias("RetLag2"))
    frame = frame.with_columns((pl.col("CloseLag3") / pl.col("CloseLag4")).log().alias("RetLag3"))

    frame = frame.with_columns((pl.col("HMLLag1") / pl.col("CloseLag1")).alias("RelHMLLag1"))
    frame = frame.with_columns((pl.col("HMLLag2") / pl.col("CloseLag2")).alias("RelHMLLag2"))
    frame = frame.with_columns((pl.col("HMLLag3") / pl.col("CloseLag3")).alias("RelHMLLag3"))
    frame = frame.with_columns((pl.col("OMCLag1") / pl.col("CloseLag1")).alias("RelOMCLag1"))
    frame = frame.with_columns((pl.col("OMCLag2") / pl.col("CloseLag2")).alias("RelOMCLag2"))
    frame = frame.with_columns((pl.col("OMCLag3") / pl.col("CloseLag3")).alias("RelOMCLag3"))

    frame = frame.with_columns((pl.col("VolumeLag1") / pl.col("VolumeLag2")).log().alias("LogVolChgLag1"))
    frame = frame.with_columns((pl.col("VolumeLag2") / pl.col("VolumeLag3")).log().alias("LogVolChgLag2"))
    frame = frame.with_columns((pl.col("VolumeLag3") / pl.col("VolumeLag4")).log().alias("LogVolChgLag3"))

    frame = frame.with_columns(pl.col("RetLag1").ewm_mean(half_life=1, ignore_nulls=True).alias("RetEMA2"))
    frame = frame.with_columns(pl.col("RetLag1").ewm_mean(half_life=2, ignore_nulls=True).alias("RetEMA4"))
    frame = frame.with_columns(pl.col("RetLag1").ewm_mean(half_life=4, ignore_nulls=True).alias("RetEMA8"))

    frame = frame.with_columns(
        pl.when(pl.col("LogReturn") > 0.0).then(pl.lit(1)).otherwise(pl.lit(0)).alias("Target")
    )
    needed = ["Date", "Open", "High", "Low", "Close", "Volume", "LogReturn", "Target", *PRICE_FEATURES, *STATIONARY_FEATURES]
    frame = frame.select(needed).drop_nulls()
    finite = [c for c in frame.columns if c != "Date"]
    frame = frame.filter(pl.all_horizontal([pl.col(c).is_finite() for c in finite]))
    frame.write_csv(DATA / "gld_features.csv")
    return frame


def split_windows(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    dev_end = np.datetime64(DEV_END)
    dates = frame["Date"].to_numpy()
    holdout_start = int(np.argmax(dates > dev_end))
    if holdout_start == 0 and dates[0] > dev_end:
        raise RuntimeError("Development window is empty.")
    dev_stop = holdout_start - GAP
    if dev_stop < 500 or holdout_start >= frame.height:
        raise RuntimeError("Not enough rows to form a development window and a hold-out.")
    development = frame[:dev_stop]
    holdout = frame[holdout_start:]
    meta = {
        "dev_start": str(development["Date"][0]),
        "dev_end": str(development["Date"][-1]),
        "n_dev": development.height,
        "gap_n": GAP,
        "gap_start": str(frame["Date"][dev_stop]),
        "gap_end": str(frame["Date"][holdout_start - 1]),
        "test_start": str(holdout["Date"][0]),
        "test_end": str(holdout["Date"][-1]),
        "n_test": holdout.height,
        "n_rows": frame.height,
    }
    return development, holdout, meta


def adf_report(development: pl.DataFrame) -> dict:
    out = {}
    for name in ("Close", "LogReturn"):
        series = development[name].to_numpy()
        stat, pvalue, *_ = adfuller(series, autolag="AIC")
        out[name] = {"adf_stat": float(stat), "pvalue": float(pvalue)}
    return out


def all_subset_indices(n_features: int) -> list[tuple[int, ...]]:
    subsets = []
    for size in range(1, n_features + 1):
        subsets.extend(combinations(range(n_features), size))
    return subsets


def aic_batch(columns: list[tuple[int, ...]], x: np.ndarray, y: np.ndarray) -> list[tuple[str, float, int]]:
    rows = []
    model = LogisticRegression(solver="lbfgs", max_iter=400)
    n = len(y)
    for cols in columns:
        model.fit(x[:, cols], y)
        loglik = -log_loss(y, model.predict_proba(x[:, cols]), labels=[0, 1]) * n
        k = len(cols) + 1
        rows.append((" ".join(str(i) for i in cols), float(2 * k - 2 * loglik), len(cols)))
    return rows


def search_aic(x: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    from joblib import Parallel, delayed

    subsets = all_subset_indices(x.shape[1])
    n_jobs = 8
    chunks = [subsets[i::n_jobs] for i in range(n_jobs)]
    parts = Parallel(n_jobs=n_jobs)(delayed(aic_batch)(chunk, x, y) for chunk in chunks)
    rows = [row for part in parts for row in part]
    table = pl.DataFrame(
        {
            "features": [row[0] for row in rows],
            "aic": [row[1] for row in rows],
            "k": [row[2] for row in rows],
        }
    ).sort("aic")
    top = table.head(10)
    top_lists = [tuple(int(i) for i in text.split()) for text in top["features"].to_list()]
    counts = Counter(index for subset in top_lists for index in subset)
    selected_idx = select_indices(counts, top_lists, len(names))
    return {
        "n_subsets": table.height,
        "best_aic": float(top["aic"][0]),
        "top10": [
            {
                "features": text,
                "names": [names[int(i)] for i in text.split()],
                "aic": float(aic),
                "k": int(k),
            }
            for text, aic, k in zip(top["features"].to_list(), top["aic"].to_list(), top["k"].to_list())
        ],
        "frequency": {names[i]: int(counts[i]) for i in range(len(names))},
        "selected": [names[i] for i in selected_idx],
        "selected_index": selected_idx,
    }


def select_indices(counts: Counter, top_lists: list[tuple[int, ...]], n_features: int) -> list[int]:
    """Keep a feature when it appears in at least half of the ten lowest-AIC subsets.

    If that rule keeps fewer than two features, use the single lowest-AIC subset.
    If it keeps more than eight, keep the eight with the highest counts.
    """
    chosen = [i for i in range(n_features) if counts[i] >= 5]
    if len(chosen) < 2:
        chosen = list(top_lists[0])
    if len(chosen) > 8:
        chosen = sorted(range(n_features), key=lambda i: (-counts[i], i))[:8]
    return chosen


def matrix(frame: pl.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return frame.select(names).to_numpy(), frame["Target"].to_numpy().astype(int)


def tune_and_score(
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    plot_name: str | None = None,
) -> dict:
    cv = TimeSeriesSplit(n_splits=5, gap=GAP)
    baseline = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
    )
    baseline_pipe = Pipeline([("scaler", StandardScaler()), ("clf", baseline)])
    cv_out = cross_validate(
        baseline_pipe,
        x_dev,
        y_dev,
        cv=cv,
        scoring=["accuracy", "roc_auc"],
        n_jobs=1,
    )
    search = RandomizedSearchCV(
        estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
        param_distributions={
            "clf__max_depth": randint(2, 8),
            "clf__min_child_weight": randint(1, 10),
            "clf__subsample": uniform(0.5, 0.5),
            "clf__learning_rate": uniform(0.01, 0.1),
            "clf__n_estimators": randint(80, 301),
        },
        n_iter=N_SEARCH,
        scoring="roc_auc",
        cv=TimeSeriesSplit(n_splits=5, gap=GAP),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_dev, y_dev)
    model = search.best_estimator_
    dev_proba = model.predict_proba(x_dev)[:, 1]
    test_proba = model.predict_proba(x_test)[:, 1]
    dev_pred = (dev_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)
    params = {key.replace("clf__", ""): _jsonable(value) for key, value in search.best_params_.items()}
    if plot_name is not None:
        save_holdout_plots(plot_name, y_test, test_proba, test_pred)
    return {
        "baseline_cv_accuracy_mean": float(cv_out["test_accuracy"].mean()),
        "baseline_cv_accuracy_std": float(cv_out["test_accuracy"].std()),
        "baseline_cv_auc_mean": float(cv_out["test_roc_auc"].mean()),
        "baseline_cv_auc_std": float(cv_out["test_roc_auc"].std()),
        "best_params": params,
        "best_cv_auc": float(search.best_score_),
        "development_insample": score_block(y_dev, dev_pred, dev_proba),
        "holdout": score_block(y_test, test_pred, test_proba),
        "mean_proba_dev": float(dev_proba.mean()),
        "mean_proba_holdout": float(test_proba.mean()),
    }


def score_block(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    matrix_counts = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "confusion": matrix_counts,
        "report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["negative return", "positive return"],
            digits=3,
        ),
    }


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def baselines(y_dev: np.ndarray, y_test: np.ndarray, ret_lag1_test: np.ndarray) -> dict:
    majority = int(np.round(y_dev.mean())) if y_dev.mean() != 0.5 else 0
    # majority class on the development window, applied unchanged to the hold-out
    majority = int(Counter(y_dev).most_common(1)[0][0])
    persist = (ret_lag1_test > 0).astype(int)
    return {
        "majority_class": majority,
        "majority_accuracy": float(accuracy_score(y_test, np.full_like(y_test, majority))),
        "persist_accuracy": float(accuracy_score(y_test, persist)),
        "persist_auc": float(roc_auc_score(y_test, persist)),
    }


def plot_price(frame: pl.DataFrame, windows: dict) -> None:
    dates = frame["Date"].to_numpy()
    close = frame["Close"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(dates, close, color="#4E2A84", linewidth=0.8)
    test_start = np.datetime64(windows["test_start"])
    ax.axvspan(test_start, dates[-1], color="#E4E0EE", alpha=0.9, label="Hold-out")
    ax.set_title("GLD adjusted close")
    ax.set_ylabel("Dollars per share")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "gld_close.png", dpi=150)
    plt.close(fig)


def plot_heatmap(development: pl.DataFrame, names: list[str], path: Path, title: str) -> None:
    cols = ["LogReturn", *names]
    corr = np.corrcoef(development.select(cols).to_numpy(), rowvar=False)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(
        corr,
        xticklabels=cols,
        yticklabels=cols,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_frequency(frequency: dict, selected: list[str], path: Path, title: str) -> None:
    names = list(frequency)
    counts = [frequency[name] for name in names]
    colors = ["#4E2A84" if name in selected else "#B6A8C9" for name in names]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh(names[::-1], counts[::-1], color=colors[::-1])
    ax.set_xlabel("Appearances in the 10 lowest-AIC subsets")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_holdout_plots(name: str, y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    RocCurveDisplay.from_predictions(
        y_true, y_proba, ax=ax, name=name, curve_kwargs={"color": "#4E2A84"}
    )
    ax.plot([0, 1], [0, 1], color="#888888", linewidth=0.8)
    ax.set_title(f"{name}: hold-out ROC")
    fig.tight_layout()
    fig.savefig(FIGURES / f"roc_{name}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Down", "Up"],
        cmap="Purples",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"{name}: hold-out calls at probability 0.5")
    fig.tight_layout()
    fig.savefig(FIGURES / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def fitted_model(params: dict) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    tree_method="hist",
                    max_depth=int(params["max_depth"]),
                    min_child_weight=int(params["min_child_weight"]),
                    subsample=float(params["subsample"]),
                    learning_rate=float(params["learning_rate"]),
                    n_estimators=int(params["n_estimators"]),
                ),
            ),
        ]
    )


def refit_figures() -> None:
    """Redraw hold-out charts from the saved hyperparameters. Does not retune."""
    from sklearn.metrics import roc_curve

    FIGURES.mkdir(parents=True, exist_ok=True)
    results = json.loads((OUTPUT / "metrics.json").read_text(encoding="utf-8"))
    frame = pl.read_csv(DATA / "gld_features.csv", try_parse_dates=True)
    development, holdout, _ = split_windows(frame)
    curves = {}
    labels = {"price": "Price levels", "stationary": "Stationary"}
    for name in ("price", "stationary"):
        selected = results[name]["selected"]
        x_dev, y_dev = matrix(development, selected)
        x_test, y_test = matrix(holdout, selected)
        model = fitted_model(results[name]["best_params"])
        model.fit(x_dev, y_dev)
        dev_proba = model.predict_proba(x_dev)[:, 1]
        test_proba = model.predict_proba(x_test)[:, 1]
        test_pred = (test_proba >= 0.5).astype(int)
        save_holdout_plots(name, y_test, test_proba, test_pred)
        results[name]["mean_proba_dev"] = float(dev_proba.mean())
        results[name]["mean_proba_holdout"] = float(test_proba.mean())
        curves[name] = (y_test, test_proba, test_pred)
        print(name, "mean proba", results[name]["mean_proba_dev"], results[name]["mean_proba_holdout"], flush=True)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    colors = {"price": "#4E2A84", "stationary": "#B8860B"}
    for name, (y_true, proba, _) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = results[name]["holdout"]["auc"]
        ax.plot(fpr, tpr, color=colors[name], label=f"{labels[name]} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#888888", linewidth=0.8, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Hold-out ROC")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_holdout.png", dpi=150)
    plt.close(fig)
    (OUTPUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def class_balance(y: np.ndarray) -> dict:
    zeros = int(np.sum(y == 0))
    ones = int(np.sum(y == 1))
    return {"n0": zeros, "n1": ones, "share1": ones / (zeros + ones)}


def run_family(name: str, names: list[str], development: pl.DataFrame, holdout: pl.DataFrame) -> dict:
    print(f"AIC search: {name}", flush=True)
    x_dev_all, y_dev = matrix(development, names)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_dev_all)
    selection = search_aic(x_scaled, y_dev, names)
    selected = selection["selected"]
    print(f"Selected {name}: {selected}", flush=True)
    x_dev, _ = matrix(development, selected)
    x_test, y_test = matrix(holdout, selected)
    print(f"Tuning {name}", flush=True)
    scores = tune_and_score(x_dev, y_dev, x_test, y_test, plot_name=name)
    plot_heatmap(
        development,
        selected,
        FIGURES / f"corr_{name}.png",
        f"{name}: correlation on the development window",
    )
    plot_frequency(
        selection["frequency"],
        selected,
        FIGURES / f"frequency_{name}.png",
        f"{name}: feature frequency in the 10 best AIC subsets",
    )
    return {"features": names, **selection, **scores}


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Downloading GLD", flush=True)
    prices = download_prices()
    frame = engineer(prices)
    development, holdout, windows = split_windows(frame)
    y_dev = development["Target"].to_numpy().astype(int)
    y_test = holdout["Target"].to_numpy().astype(int)
    plot_price(frame, windows)
    results = {
        "symbol": SYMBOL,
        "start": START,
        "end": END,
        "n_prices": prices.height,
        "windows": windows,
        "balance_dev": class_balance(y_dev),
        "balance_test": class_balance(y_test),
        "adf": adf_report(development),
        "baselines": baselines(y_dev, y_test, holdout["RetLag1"].to_numpy()),
        "search": {
            "n_iter": N_SEARCH,
            "scoring": "roc_auc",
            "cv": "TimeSeriesSplit n_splits=5 gap=10",
            "random_state": RANDOM_STATE,
        },
        "price": run_family("price", PRICE_FEATURES, development, holdout),
        "stationary": run_family("stationary", STATIONARY_FEATURES, development, holdout),
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUTPUT / "holdout_reports.txt").write_text(
        "PRICE LEVELS\n"
        + results["price"]["holdout"]["report"]
        + "\nSTATIONARY\n"
        + results["stationary"]["holdout"]["report"],
        encoding="utf-8",
    )
    print(json.dumps({k: results[k] for k in ("windows", "adf", "baselines")}, indent=2), flush=True)
    print("price hold-out", results["price"]["holdout"]["accuracy"], results["price"]["holdout"]["auc"], flush=True)
    print(
        "stationary hold-out",
        results["stationary"]["holdout"]["accuracy"],
        results["stationary"]["holdout"]["auc"],
        flush=True,
    )


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        refit_figures()
    else:
        main()
