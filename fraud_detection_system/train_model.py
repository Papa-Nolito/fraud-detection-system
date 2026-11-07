"""Trains, compares and evaluates the fraud models, then saves the best one.

Usage:  python train_model.py            (full comparison, ~1-2 minutes)
        python train_model.py --quick    (Random Forest only, a few seconds)

Output:
  models/fraud_model.joblib   - the deployed pipeline + decision threshold
  models/model_metrics.json   - accuracy / precision / recall / F1 etc. (used by Reports page)
  docs/figures/*.png          - confusion matrix, ROC curves, model comparison (for Chapter 5)
"""
import argparse
import json
import os
import time
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

import ml_engine as ml
from generate_dataset import generate

BASE = os.path.dirname(os.path.abspath(__file__))


def best_threshold(y_true, prob):
    """Pick the probability cut-off that maximises F1 (balances missed fraud vs false alarms)."""
    p, r, t = precision_recall_curve(y_true, prob)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])


def evaluate(y_true, prob, thr):
    """The exact metrics defined in section 5.5.1 of your proposal."""
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else 0.0
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": div(2 * precision * recall, precision + recall),
        "false_positive_rate": div(fp, fp + tn),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def train_and_save(df, model_dir, figures_dir=None, quick=False, seed=42):
    os.makedirs(model_dir, exist_ok=True)
    X, y = df[ml.RAW_COLUMNS], df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    Xtr, Xte = ml.prepare_frame(X_tr), ml.prepare_frame(X_te)

    candidates = {"Random Forest": RandomForestClassifier(
        n_estimators=60 if quick else 250, min_samples_leaf=3, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed)}
    if not quick:
        candidates["Logistic Regression"] = LogisticRegression(max_iter=2000, class_weight="balanced")
        candidates["Gradient Boosting"] = HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed)

    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    results, fitted, test_probs = {}, {}, {}
    for name, est in candidates.items():
        pipe = ml.build_pipeline(est)
        # Threshold is tuned on cross-validated predictions from the TRAINING data only,
        # so the test set stays untouched ("not used during training", section 5.5).
        oof = cross_val_predict(pipe, Xtr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr, cv_f1 = best_threshold(y_tr, oof)
        pipe.fit(Xtr, y_tr)
        prob = pipe.predict_proba(Xte)[:, 1]
        results[name] = {"threshold": thr, "cv_f1": cv_f1, **evaluate(y_te, prob, thr)}
        fitted[name], test_probs[name] = pipe, prob
        print(f"{name:20s} cv_f1={cv_f1:.3f} | test: P={results[name]['precision']:.3f} "
              f"R={results[name]['recall']:.3f} F1={results[name]['f1']:.3f}")

    best = max(results, key=lambda k: results[k]["cv_f1"])

    # Unsupervised anomaly detector, reported for comparison only (covers the 'anomaly
    # detection' element in your abstract). Not deployed: supervised models do better here.
    iso_info = None
    if not quick:
        prep = ml.build_preprocessor().fit(Xtr)
        iso = IsolationForest(n_estimators=200, contamination=float(y_tr.mean()),
                              random_state=seed).fit(prep.transform(Xtr))
        score = -iso.score_samples(prep.transform(Xte))     # higher = more anomalous
        iso_pred = (iso.predict(prep.transform(Xte)) == -1).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, iso_pred, labels=[0, 1]).ravel()
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        iso_info = {"precision": float(pr), "recall": float(rc),
                    "f1": float(2 * pr * rc / (pr + rc)) if pr + rc else 0.0,
                    "accuracy": float((tp + tn) / len(y_te)),
                    "false_positive_rate": float(fp / (fp + tn)),
                    "roc_auc": float(roc_auc_score(y_te, score)),
                    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
        results["Isolation Forest (unsupervised)"] = {"threshold": None, "cv_f1": None, **iso_info}
        test_probs["Isolation Forest (unsupervised)"] = score
        print(f"{'Isolation Forest':20s} test: P={pr:.3f} R={rc:.3f} F1={iso_info['f1']:.3f}")

    # Processing speed: average time to score ONE transaction (objective 5)
    pipe = fitted[best]
    sample = X_te.head(200)
    t0 = time.perf_counter()
    for i in range(len(sample)):
        pipe.predict_proba(ml.prepare_frame(sample.iloc[[i]]))
    avg_ms = (time.perf_counter() - t0) / len(sample) * 1000

    thr = results[best]["threshold"]
    joblib.dump({"pipeline": pipe, "threshold": thr, "model_name": best,
                 "features": ml.FEATURES}, os.path.join(model_dir, "fraud_model.joblib"))
    metrics = {
        "deployed_model": best, "threshold": thr,
        "trained_at": datetime.now().strftime(ml.TIME_FMT),
        "dataset_rows": int(len(df)), "fraud_rate": float(y.mean()),
        "train_rows": int(len(y_tr)), "test_rows": int(len(y_te)),
        "avg_prediction_ms": avg_ms, "models": results,
        "best": results[best],
    }
    with open(os.path.join(model_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if figures_dir:
        _save_figures(figures_dir, best, results, test_probs, y_te)
    print(f"\nDeployed: {best} (threshold {thr:.3f}) | avg scoring time {avg_ms:.1f} ms/transaction")
    return metrics


def _save_figures(fig_dir, best, results, probs, y_te):
    os.makedirs(fig_dir, exist_ok=True)
    r = results[best]
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"]); ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix - {best}")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "roc_curves.png"), dpi=150); plt.close(fig)

    names = list(results)
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, metric in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + (k - 1) * w, [results[n][metric] for n in names], w, label=metric.capitalize())
    ax.set_xticks(x, [n.replace(" (unsupervised)", "\n(unsup.)").replace(" ", "\n", 1) for n in names], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_title("Model comparison on the unseen test set"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "model_comparison.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default=os.path.join(BASE, "data", "transactions_synthetic.csv"))
    args = ap.parse_args()
    if os.path.exists(args.data):
        data = pd.read_csv(args.data)
    else:
        print("Dataset not found - generating one...")
        data = generate()
        os.makedirs(os.path.dirname(args.data), exist_ok=True)
        data.to_csv(args.data, index=False)
    train_and_save(data, os.path.join(BASE, "models"), os.path.join(BASE, "docs", "figures"),
                   quick=args.quick)
