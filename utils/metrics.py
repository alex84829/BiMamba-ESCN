# -*- coding: utf-8 -*-
"""Metrics and patient-level voting utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def binary_confusion_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


def compute_basic_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = binary_confusion_counts(y_true, y_pred)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = f1_score(y_true, y_pred) if (tp + fp + fn) > 0 else 0.0
    auc = roc_auc_score(y_true, y_prob) if (y_prob is not None and len(np.unique(y_true)) == 2) else np.nan
    return dict(acc=acc, prec=prec, rec=rec, spec=spec, f1=f1, auc=auc)


def select_threshold_youden_j(y_true, y_prob, n_grid=400, fallback=0.5):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)

    if len(np.unique(y)) < 2:
        return float(fallback), {"J": np.nan, "sens": np.nan, "spec": np.nan}

    lo = float(np.quantile(p, 0.001))
    hi = float(np.quantile(p, 0.999))
    if hi <= lo:
        lo, hi = float(p.min()), float(p.max())
        if hi <= lo:
            return float(fallback), {"J": np.nan, "sens": np.nan, "spec": np.nan}

    thr_grid = np.linspace(lo, hi, int(n_grid))
    best_thr, best_j = float(fallback), -1e9
    best_sens, best_spec = 0.0, 0.0

    for thr in thr_grid:
        yhat = (p >= thr).astype(int)
        tn, fp, fn, tp = binary_confusion_counts(y, yhat)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j_score = sens + spec - 1.0

        tie_better = (
            abs(j_score - best_j) < 1e-12
            and (
                abs(sens - spec) < abs(best_sens - best_spec) - 1e-12
                or (abs(sens - spec) <= abs(best_sens - best_spec) + 1e-12 and spec > best_spec)
            )
        )
        if j_score > best_j or tie_better:
            best_j = float(j_score)
            best_thr = float(thr)
            best_sens = float(sens)
            best_spec = float(spec)

    return best_thr, {"J": best_j, "sens": best_sens, "spec": best_spec}


def groupwise_majority_vote(y_true_unit, y_prob_unit, group_ids, threshold=0.5):
    """Patient-level majority voting with mean-probability tie-breaking."""
    y = np.asarray(y_true_unit).astype(int)
    p = np.asarray(y_prob_unit).astype(float)
    g = np.asarray(group_ids)

    yhat_unit = (p >= threshold).astype(int)
    df = pd.DataFrame({"gid": g, "y": y, "yhat": yhat_unit, "p": p})

    nunq = df.groupby("gid")["y"].nunique()
    if (nunq > 1).any():
        bad = nunq[nunq > 1].index.tolist()[:10]
        raise ValueError(f"Mixed labels within patient ID. Examples: {bad}")

    y_true_g, y_pred_g, n_units, p_means = [], [], [], []
    for gid, sub in df.groupby("gid"):
        yy = int(sub["y"].iloc[0])
        votes = sub["yhat"].to_numpy()
        p_mean = float(sub["p"].mean())
        pos_votes = int(votes.sum())
        n = int(len(votes))

        if pos_votes > n / 2:
            pred = 1
        elif pos_votes < n / 2:
            pred = 0
        else:
            pred = int(p_mean >= threshold)

        y_true_g.append(yy)
        y_pred_g.append(pred)
        n_units.append(n)
        p_means.append(p_mean)

    y_true_g = np.asarray(y_true_g, dtype=int)
    y_pred_g = np.asarray(y_pred_g, dtype=int)
    metrics = compute_basic_metrics(y_true_g, y_pred_g, y_prob=None)

    agg = pd.DataFrame(
        {
            "y_true": y_true_g,
            "y_pred_vote": y_pred_g,
            "n_units": n_units,
            "p_mean": p_means,
            "is_correct": (y_true_g == y_pred_g).astype(int),
        },
        index=list(df.groupby("gid").groups.keys()),
    )
    return metrics, agg


def groupwise_vote_k_report(y_true_unit, y_prob_unit, group_ids, k=10, threshold=0.5, n_repeat=50, seed=0):
    y = np.asarray(y_true_unit).astype(int)
    p = np.asarray(y_prob_unit).astype(float)
    g = np.asarray(group_ids)

    df = pd.DataFrame({"gid": g, "y": y, "p": p})
    nunq = df.groupby("gid")["y"].nunique()
    if (nunq > 1).any():
        bad = nunq[nunq > 1].index.tolist()[:10]
        raise ValueError(f"Mixed labels within patient ID. Examples: {bad}")

    gids = df["gid"].unique().tolist()
    rng = np.random.default_rng(seed)
    accs, precs, recs, specs = [], [], [], []

    for _ in range(n_repeat):
        y_true_g, y_pred_g = [], []
        for gid in gids:
            sub = df[df["gid"] == gid]
            yy = int(sub["y"].iloc[0])
            pp = sub["p"].to_numpy()
            n = len(pp)
            if n == 0:
                continue

            k_eff = min(int(k), int(n))
            idx = rng.choice(n, size=k_eff, replace=False)
            probs = pp[idx]
            votes = (probs >= threshold).astype(int)
            pos_votes = int(votes.sum())

            if pos_votes > k_eff / 2:
                pred = 1
            elif pos_votes < k_eff / 2:
                pred = 0
            else:
                pred = int(float(probs.mean()) >= threshold)

            y_true_g.append(yy)
            y_pred_g.append(pred)

        metrics = compute_basic_metrics(np.asarray(y_true_g), np.asarray(y_pred_g), y_prob=None)
        accs.append(metrics["acc"])
        precs.append(metrics["prec"])
        recs.append(metrics["rec"])
        specs.append(metrics["spec"])

    def mean_std(x):
        x = np.asarray(x, dtype=float)
        return float(x.mean()), float(x.std(ddof=1) if len(x) > 1 else 0.0)

    return {
        "acc": mean_std(accs),
        "prec": mean_std(precs),
        "rec": mean_std(recs),
        "spec": mean_std(specs),
        "n_groups": len(gids),
        "k": int(k),
        "n_repeat": int(n_repeat),
    }


def mean_std(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) if len(x) > 1 else 0.0)
