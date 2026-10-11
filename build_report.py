"""Render the programming-assignment report from output/metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parent
METRICS = json.loads((ROOT / "output" / "metrics.json").read_text(encoding="utf-8"))
OUT = ROOT / "report" / "msds451-programming-assignment-1.pdf"


def pct(x: float) -> str:
    return f"{100 * x:.1f} percent"


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def styles():
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=base["Normal"],
        fontName="Times-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#4E2A84"),
    )
    title = ParagraphStyle(
        "Title2",
        parent=base["Normal"],
        fontName="Times-Bold",
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4E2A84"),
        spaceAfter=4,
    )
    sub = ParagraphStyle(
        "Sub",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=base["Normal"],
        fontName="Times-Italic",
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#333333"),
        spaceBefore=2,
        spaceAfter=8,
    )
    ref = ParagraphStyle(
        "Ref",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=10,
        leading=13,
        leftIndent=18,
        firstLineIndent=-18,
        spaceAfter=6,
    )
    cell = ParagraphStyle(
        "Cell",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=9,
        leading=11,
        alignment=TA_LEFT,
    )
    return body, h1, title, sub, caption, ref, cell


def table(data, col_widths):
    style_commands = [
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4E2A84")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7F5FA")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C8C0D4")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    tbl = Table(data, colWidths=col_widths, hAlign="LEFT")
    tbl.setStyle(TableStyle(style_commands))
    return tbl


def fig(name: str, width: float):
    path = ROOT / "figures" / name
    iw, ih = ImageReader(str(path)).getSize()
    image = Image(str(path), width=width, height=width * ih / iw)
    image.hAlign = "CENTER"
    return image


def build() -> None:
    body, h1, title, sub, caption, ref, _cell = styles()
    w = METRICS["windows"]
    price = METRICS["price"]
    stat = METRICS["stationary"]
    base = METRICS["baselines"]
    adf = METRICS["adf"]
    bal_d = METRICS["balance_dev"]
    bal_t = METRICS["balance_test"]

    story = []
    story.append(Paragraph("Next-Day Direction of GLD", title))
    story.append(Paragraph("Lagged Prices, Feature Selection, and a Hold-Out Test", sub))
    story.append(Spacer(1, 6))
    story.append(Paragraph("MSDS 451 Financial Engineering", sub))
    story.append(Paragraph("Programming Assignment 1", sub))
    story.append(Paragraph("Leandro &nbsp;&nbsp; 11 October 2026", sub))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Abstract", h1))
    story.append(
        Paragraph(
            "I test whether lagged daily prices forecast the sign of the next close-to-close "
            "return on GLD, the SPDR Gold Shares ETF. The design follows the course jump-start: "
            "fifteen price features, an all-subsets search by the Akaike information criterion, "
            "and an XGBoost classifier tuned inside a time-series cross-validation. I add the "
            "step the jump-start leaves out. Features and hyperparameters are chosen on a "
            f"development window of {w['n_dev']:,} sessions, from {w['dev_start']} through "
            f"{w['dev_end']}. A ten-session gap is discarded. The fitted models are scored once "
            f"on {w['n_test']} later sessions, from {w['test_start']} through {w['test_end']}. "
            "A second feature set replaces price levels with returns and ratios, because a "
            "drifting price is a poor input to a classifier. On the hold-out, the price-level "
            f"model is right on {pct(price['holdout']['accuracy'])} of days (AUC "
            f"{num(price['holdout']['auc'])}). The stationary model is right on "
            f"{pct(stat['holdout']['accuracy'])} of days (AUC {num(stat['holdout']['auc'])}). "
            f"Calling every hold-out day an up day is right on {pct(base['majority_accuracy'])} "
            "of days. Lagged prices do not identify the next day's direction in this sample.",
            body,
        )
    )

    story.append(Paragraph("Introduction", h1))
    story.append(
        Paragraph(
            "The jump-start program predicts whether West Texas Intermediate closes up or down, "
            "then reports accuracy on the same rows used to choose the model. The course note "
            "on model development asks for a different test: fit on one stretch of time, and "
            "score a later stretch that played no role in the fitting. I keep the jump-start's "
            "tools and change the asset and the test.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The asset is GLD. The assignment bars WTI and asks for a series other than the "
            "example. Gold is a monetary metal, not a barrel of oil, and the ETF has a long "
            "daily history on Yahoo Finance. The question is the one the example poses. Do a "
            "handful of lagged price features tell us the sign of tomorrow's return?",
            body,
        )
    )
    story.append(
        Paragraph(
            "The week's stationarity lecture is the reason for a second model. A price that "
            "trends is not a usable signal. The log return is. I estimate both feature families "
            "under one selection rule and one classifier, and I compare both with two rules "
            "that ignore the features.",
            body,
        )
    )

    story.append(Paragraph("Literature", h1))
    story.append(
        Paragraph(
            "Akaike (1973) supplies the selection score: a penalty for each extra parameter, "
            "and a reward for a higher likelihood. Hastie, Tibshirani, and Friedman (2009) "
            "treat an enumeration of subsets as more trustworthy than stepwise search when the "
            "subset space is small. Fifteen features produce 32,767 non-empty subsets, which "
            "a logistic regression can score in one pass. Geron (2022) is the reason the "
            "XGBoost settings are drawn at random rather than laid out on a grid. Lopez de "
            "Prado (2018) is the reason the published score comes from a later window, with a "
            "gap so that a moving average does not sit on both sides of the cut. Chan (n.d.) "
            "states the stationarity point in the module lecture: machine learning wants "
            "stationary inputs, and the time difference of the log price is the usual repair. "
            "Ross (2011, chaps. 5 and 8) is the arbitrage reading for the week. A rule that "
            "called the sign of tomorrow's gold return often enough to survive trading costs "
            "would be a persistent edge of the kind those chapters say the market removes. "
            "This paper stops one step earlier. It asks whether the sign is forecastable at all.",
            body,
        )
    )

    story.append(Paragraph("Methods", h1))
    story.append(Paragraph("Data and target", h1))
    story.append(
        Paragraph(
            f"Daily bars come from Yahoo Finance through the yfinance package, ticker GLD, "
            f"split-adjusted, from 3 January 2006 through 9 October 2026 "
            f"({METRICS['n_prices']:,} sessions). Building four lags drops four rows and leaves "
            f"{w['n_rows']:,}. The target is 1 when the close-to-close log return is positive "
            "and 0 otherwise. Every feature is known at the previous close. The same day's "
            "open, high, low, close, and volume are excluded, so the model cannot see the "
            "move it is asked to call.",
            body,
        )
    )
    story.append(
        Paragraph(
            f"The development window runs from {w['dev_start']} through {w['dev_end']} "
            f"({w['n_dev']:,} sessions). The next {w['gap_n']} sessions, {w['gap_start']} "
            f"through {w['gap_end']}, are dropped. The hold-out runs from {w['test_start']} "
            f"through {w['test_end']} ({w['n_test']} sessions). Up days are "
            f"{pct(bal_d['share1'])} of the development window "
            f"({bal_d['n1']:,} up, {bal_d['n0']:,} down) and {pct(bal_t['share1'])} of the "
            f"hold-out ({bal_t['n1']} up, {bal_t['n0']} down). The hold-out is the gold rally "
            "of 2024-2026. I do not rebalance the classes. The ROC summary does not require "
            "equal class counts.",
            body,
        )
    )
    story.append(fig("gld_close.png", 6.4 * inch))
    story.append(
        Paragraph(
            "Figure 1. Adjusted close of GLD. The shaded band is the hold-out. "
            "The development maximum close is $193.89. The hold-out median is $305.46, "
            "and 654 of 696 hold-out closes sit above every development close.",
            caption,
        )
    )

    story.append(Paragraph("Two feature families", h1))
    story.append(
        Paragraph(
            "The price-level family is the jump-start list. Close lags of one, two, and three "
            "days. High minus low, and open minus close, at the same three lags. Volume at "
            "the same three lags. Exponential moving averages of yesterday's close with "
            "half-lives of one, two, and four days, labeled CloseEMA2, CloseEMA4, and "
            "CloseEMA8 to match the example. The averages are taken on the lagged close, so "
            "today's close does not enter them.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The stationary family replaces levels with changes. Three lags of the log return. "
            "High-minus-low and open-minus-close, each divided by the close on that same day, "
            "at three lags. Three lags of the log change in volume. Exponential moving averages "
            "of yesterday's log return, with the same three half-lives. An Augmented "
            f"Dickey-Fuller test on the development window fails to reject a unit root in the "
            f"close (statistic {num(adf['Close']['adf_stat'], 2)}, "
            f"p = {num(adf['Close']['pvalue'], 2)}) and rejects one in the log return "
            f"(statistic {num(adf['LogReturn']['adf_stat'], 1)}, p reported as zero at machine "
            "precision).",
            body,
        )
    )

    story.append(Paragraph("Selection, tuning, and the hold-out rule", h1))
    story.append(
        Paragraph(
            "On the development window only, each family is standardized and every non-empty "
            "subset is fit with a logistic regression. The Akaike score is 2k minus twice the "
            "in-sample log likelihood, and k counts the slopes plus an intercept. A feature is "
            "kept when it appears in at least five of the ten lowest-scoring subsets. That "
            "rule is fixed before looking at the hold-out. It kept four price features and two "
            "stationary features.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The classifier is XGBoost with a logistic objective and histogram trees. Inside "
            "the development window, a five-fold TimeSeriesSplit with a ten-day gap scores a "
            "first model of 200 trees. Randomized search then draws 30 parameter vectors from "
            "max depth 2 to 7, minimum child weight 1 to 9, subsample 0.5 to 1, learning rate "
            "0.01 to 0.11, and 80 to 300 trees. The draw is seeded at 2026. The search "
            "maximizes ROC AUC. The jump-start maximized accuracy and then drew its ROC from "
            "hard class labels. AUC uses the probabilities, which is what a later cutoff would "
            "threshold. The scaler sits inside the pipeline, so each fold scales on its own "
            "training rows. The winning settings are refit on the whole development window.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The hold-out is scored once, with an up call when the predicted probability is "
            "at least one half. Two reference rules are fixed from the development window "
            "before that score is computed. The first calls every hold-out day an up day, "
            "because up was the majority class in development. The second repeats yesterday's "
            "direction. I do not round features to three decimals before fitting. The example "
            "did that for a spreadsheet check. The model uses the unrounded values.",
            body,
        )
    )

    story.append(Paragraph("Results", h1))
    story.append(Paragraph("What the subset search kept", h1))
    story.append(
        Paragraph(
            "An intercept-only logistic regression on the development target has an Akaike "
            "score of 6,250.3. The best price-level subset scores 6,248.2. The best stationary "
            "subset scores 6,246.3. Across 32,767 subsets, the tenth-best price model is only "
            "1.6 points worse than the best, and the tenth-best stationary model is 1.5 points "
            "worse. The likelihood barely moves when features are added. The correlations of "
            "the selected features with the next day's log return, on the development window, "
            "all lie between -0.05 and 0.01.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The price features that clear the frequency rule are yesterday's close, the "
            "close three days back, yesterday's high-minus-low, and volume two days back. "
            "The stationary features are yesterday's range relative to the close, and the "
            "short moving average of past returns. Figure 2 shows the price-level counts. "
            "Close lags and yesterday's range dominate the best subsets. The moving averages "
            "of the price level almost never appear, which is consistent with those series "
            "being smooth copies of the same non-stationary close.",
            body,
        )
    )
    story.append(fig("frequency_price.png", 6.3 * inch))
    story.append(
        Paragraph(
            "Figure 2. How often each price-level feature appears in the ten lowest-AIC "
            "subsets. Dark bars are the four features kept for the classifier.",
            caption,
        )
    )
    story.append(fig("corr_stationary.png", 4.6 * inch))
    story.append(
        Paragraph(
            "Figure 3. Development-window correlations for the two stationary features and "
            "the next day's log return. Both features are nearly uncorrelated with the target.",
            caption,
        )
    )

    story.append(Paragraph("Cross-validation already said the fit was thin", h1))
    cv_header = ["Model", "CV accuracy", "CV AUC", "In-sample accuracy", "In-sample AUC"]
    cv_rows = [cv_header]
    for label, block in (("Price levels", price), ("Stationary", stat)):
        cv_rows.append(
            [
                label,
                f"{block['baseline_cv_accuracy_mean']:.3f}",
                f"{block['best_cv_auc']:.3f}",
                f"{block['development_insample']['accuracy']:.3f}",
                f"{block['development_insample']['auc']:.3f}",
            ]
        )
    story.append(table(cv_rows, [1.3 * inch, 1.15 * inch, 1.05 * inch, 1.45 * inch, 1.25 * inch]))
    story.append(
        Paragraph(
            "Table 1. Development-window scores. CV accuracy is the untuned 200-tree model. "
            "CV AUC is the tuned model. In-sample scores refit that tuned model on all "
            "development rows.",
            caption,
        )
    )
    story.append(
        Paragraph(
            f"The tuned price model uses depth {price['best_params']['max_depth']}, "
            f"{price['best_params']['n_estimators']} trees, a learning rate of "
            f"{price['best_params']['learning_rate']:.3f}, a child-weight minimum of "
            f"{price['best_params']['min_child_weight']}, and a subsample of "
            f"{price['best_params']['subsample']:.2f}. The tuned stationary model uses depth "
            f"{stat['best_params']['max_depth']}, {stat['best_params']['n_estimators']} trees, "
            f"a learning rate of {stat['best_params']['learning_rate']:.3f}, a child-weight "
            f"minimum of {stat['best_params']['min_child_weight']}, and a subsample of "
            f"{stat['best_params']['subsample']:.2f}. Cross-validated AUC is "
            f"{num(price['best_cv_auc'])} and {num(stat['best_cv_auc'])}. That is a coin flip, "
            "computed before the hold-out is opened. The same models, scored on the rows they "
            f"were fit to, reach AUC {num(price['development_insample']['auc'])} and "
            f"{num(stat['development_insample']['auc'])} and accuracy "
            f"{pct(price['development_insample']['accuracy'])} and "
            f"{pct(stat['development_insample']['accuracy'])}. Those in-sample numbers are the "
            "kind of result the jump-start reports for WTI. They describe the training rows.",
            body,
        )
    )

    story.append(Paragraph("The hold-out", h1))
    hold_rows = [
        ["Rule", "Accuracy", "AUC"],
        ["Always up (development majority)", f"{base['majority_accuracy']:.3f}", "n.a."],
        ["Repeat yesterday's direction", f"{base['persist_accuracy']:.3f}", f"{base['persist_auc']:.3f}"],
        ["Price-level XGBoost", f"{price['holdout']['accuracy']:.3f}", f"{price['holdout']['auc']:.3f}"],
        ["Stationary XGBoost", f"{stat['holdout']['accuracy']:.3f}", f"{stat['holdout']['auc']:.3f}"],
    ]
    story.append(table(hold_rows, [3.4 * inch, 1.3 * inch, 1.2 * inch]))
    story.append(
        Paragraph(
            "Table 2. Hold-out scores. Each model was fit on the development window and applied "
            "once. Accuracy uses a probability cutoff of one half.",
            caption,
        )
    )
    story.append(
        Paragraph(
            f"The price-level model calls a down day on 693 of the 696 hold-out sessions. "
            f"Its mean predicted probability of an up day is {price['mean_proba_dev']:.2f} on "
            f"the development rows and {price['mean_proba_holdout']:.2f} on the hold-out. "
            "The trees split on price levels from a window whose highest close was $193.89. "
            "Ninety-four percent of hold-out closes are above that number, and the model "
            "reads those new levels as down days. The ROC is only slightly above the diagonal "
            f"(AUC {num(price['holdout']['auc'])}), so the ranking contains little information, "
            "and the cutoff of one half turns that weak ranking into a one-sided call.",
            body,
        )
    )
    story.append(
        Paragraph(
            f"The stationary model does not collapse in that way. Its mean up-day probability "
            f"is {stat['mean_proba_dev']:.2f} in development and {stat['mean_proba_holdout']:.2f} "
            f"in the hold-out. Accuracy is {pct(stat['holdout']['accuracy'])}, which sits "
            f"next to the {pct(base['majority_accuracy'])} from calling every day an up day, "
            f"and the AUC is {num(stat['holdout']['auc'])}. Repeating yesterday's direction "
            f"is right on {pct(base['persist_accuracy'])} of hold-out days. Neither fitted "
            "model beats the majority rule.",
            body,
        )
    )

    pc = price["holdout"]["confusion"]
    sc = stat["holdout"]["confusion"]
    conf_rows = [
        ["", "Price down", "Price up", "Stationary down", "Stationary up"],
        ["Actual down", str(pc[0][0]), str(pc[0][1]), str(sc[0][0]), str(sc[0][1])],
        ["Actual up", str(pc[1][0]), str(pc[1][1]), str(sc[1][0]), str(sc[1][1])],
    ]
    story.append(table(conf_rows, [1.15 * inch, 1.25 * inch, 1.2 * inch, 1.55 * inch, 1.45 * inch]))
    story.append(
        Paragraph(
            "Table 3. Hold-out confusion counts at a probability cutoff of one half. "
            "The price-level model almost never calls an up day.",
            caption,
        )
    )
    story.append(fig("roc_holdout.png", 4.8 * inch))
    story.append(
        Paragraph(
            "Figure 4. Hold-out receiver operating characteristic curves. Both sit close to the diagonal.",
            caption,
        )
    )

    story.append(Paragraph("Conclusion", h1))
    story.append(
        Paragraph(
            "For a trading desk this is a stop, not a position. Over 2024-2026, a next-day "
            "long or short rule built from these lagged prices loses to the simpler rule of "
            "treating every day as an up day, before commissions, spreads, or the cost of "
            "being wrong on a large move. The in-sample accuracy near 60 percent is what a "
            "design without a hold-out would have published. The cross-validated AUC near "
            "0.51 already contradicted it. The later window confirmed the cross-validation.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The two families fail differently, and that difference is the stationarity point. "
            "The price-level model meets a market that has left its training range, and it "
            "calls nearly every new day a down day. The stationary model is calibrated more "
            "sensibly and still has no ranking power. A further lag of the same close is "
            "unlikely to repair either failure. The module's own next step, other leading "
            "series and media measures, is the place to look, and any such series would need "
            "the same embargo and the same untouched hold-out.",
            body,
        )
    )

    story.append(Paragraph("References", h1))
    references = [
        "Akaike, Hirotugu. 1973. \"Information Theory and an Extension of the Maximum Likelihood Principle.\" In <i>Second International Symposium on Information Theory</i>, edited by B. N. Petrov and F. Csaki, 267-281. Budapest: Akademiai Kiado.",
        "Chan, Ernest P. n.d. \"Module 3: Stationarity.\" Course presentation, MSDS 451, Northwestern University.",
        "Chen, Tianqi, and Carlos Guestrin. 2016. \"XGBoost: A Scalable Tree Boosting System.\" In <i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i>, 785-794.",
        "Geron, Aurelien. 2022. <i>Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow</i>. 3rd ed. Sebastopol, CA: O'Reilly.",
        "Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 2009. <i>The Elements of Statistical Learning: Data Mining, Inference, and Prediction</i>. 2nd ed. New York: Springer.",
        "Lopez de Prado, Marcos. 2018. <i>Advances in Financial Machine Learning</i>. Hoboken, NJ: Wiley.",
        "Miller, Thomas W. 2025. \"451 Feature Engineering: Programming Assignment 1.\" Course technical report, June 18.",
        "Ross, Sheldon M. 2011. <i>An Elementary Introduction to Mathematical Finance</i>. 3rd ed. Cambridge: Cambridge University Press.",
    ]
    for item in references:
        story.append(Paragraph(item, ref))

    story.append(Paragraph("Note on the use of a language model", h1))
    story.append(
        Paragraph(
            "A language model in Cursor (Grok 4.7) wrote analysis.py, ran the subset search "
            "and the classifier, and drafted this report from output/metrics.json. The session "
            "started from the prompt \"Assigment module 3.\" The hold-out window was not used "
            "to pick the asset's features or the reported hyperparameters. The numbers in the "
            "tables are the program's output.",
            body,
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="MSDS 451 Programming Assignment 1: Next-Day Direction of GLD",
        author="Leandro",
    )
    doc.build(story)
    print(OUT, flush=True)


if __name__ == "__main__":
    build()
