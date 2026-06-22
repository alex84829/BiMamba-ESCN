import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
import yaml

from dataset import (
    collect_files,
    expand_files,
    FoldNormalizer,
    pack_sequences_by_file,
    estimate_max_sequences_per_pid,
)
from models import BiMambaESCN
from utils.seed import set_seed
from utils.inference import predict_logits_batched
from utils.metrics import (
    compute_basic_metrics,
    select_threshold_youden_j,
    groupwise_majority_vote,
    groupwise_vote_k_report,
    mean_std,
)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train BiMamba-ESCN with subject-wise GroupKFold.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--mcs-dir", type=str, default=None, help="Override MCS feature directory.")
    parser.add_argument("--uws-dir", type=str, default=None, help="Override UWS feature directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    seq_cfg = cfg["sequence"]
    thr_cfg = cfg["threshold"]
    vote_cfg = cfg["vote"]
    model_cfg = cfg["model"]

    if args.mcs_dir is not None:
        data_cfg["mcs_dir"] = args.mcs_dir
    if args.uws_dir is not None:
        data_cfg["uws_dir"] = args.uws_dir

    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(train_cfg["seed"]))

    files, labels_file, patients, _ = collect_files(data_cfg["mcs_dir"], data_cfg["uws_dir"])
    print(f"[INFO] #MCS files={int((labels_file==1).sum())}, #UWS files={int((labels_file==0).sum())}, total files={len(files)}")
    print(f"[INFO] unique patients={len(np.unique(patients))}")

    k_max_pid, stats_k, _ = estimate_max_sequences_per_pid(
        files=files,
        seq_len=int(seq_cfg["seq_len"]),
        stride=int(seq_cfg["stride"]),
        pad_last=bool(seq_cfg["pad_last"]),
    )
    vote_k_base = list(vote_cfg.get("k_base", []))
    vote_k_list = sorted(set([k for k in vote_k_base if k <= k_max_pid] + ([k_max_pid] if k_max_pid > 0 else [])))
    print(f"[AUTO-K] pid seq-count stats: min={stats_k['pid_min']} med={stats_k['pid_med']} max={stats_k['pid_max']} (n_pid={stats_k['n_pid']})")
    print(f"[AUTO-K] VOTE_K_LIST={vote_k_list}")

    gkf = GroupKFold(n_splits=int(train_cfg["n_splits"]))

    seq_metrics_folds = []
    patientvote_metrics_folds = []
    thr_folds = []
    voteK_metrics_by_k = {k: [] for k in vote_k_list}

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(files, labels_file, groups=patients), 1):
        print(f"\n==== Fold {fold}/{int(train_cfg['n_splits'])} ====")
        tr_files, va_files = files[tr_idx], files[va_idx]

        tr_psd, tr_de, tr_plv, tr_wpli, y_tr, tr_pid, tr_fid = expand_files(
            tr_files, chans=int(model_cfg["chans"]), bands=int(model_cfg["bands"])
        )
        va_psd, va_de, va_plv, va_wpli, y_va, va_pid, va_fid = expand_files(
            va_files, chans=int(model_cfg["chans"]), bands=int(model_cfg["bands"])
        )

        normalizer = FoldNormalizer.fit(tr_psd, tr_de, tr_plv, tr_wpli)
        tr_psd, tr_de, tr_plv, tr_wpli = normalizer.transform(tr_psd, tr_de, tr_plv, tr_wpli)
        va_psd, va_de, va_plv, va_wpli = normalizer.transform(va_psd, va_de, va_plv, va_wpli)

        tr_psd_s, tr_de_s, tr_plv_s, tr_wpli_s, y_tr_s, tr_pid_s, tr_fid_s = pack_sequences_by_file(
            tr_psd, tr_de, tr_plv, tr_wpli, y_tr, tr_pid, tr_fid,
            seq_len=int(seq_cfg["seq_len"]), stride=int(seq_cfg["stride"]), pad_last=bool(seq_cfg["pad_last"]),
        )
        va_psd_s, va_de_s, va_plv_s, va_wpli_s, y_va_s, va_pid_s, va_fid_s = pack_sequences_by_file(
            va_psd, va_de, va_plv, va_wpli, y_va, va_pid, va_fid,
            seq_len=int(seq_cfg["seq_len"]), stride=int(seq_cfg["stride"]), pad_last=bool(seq_cfg["pad_last"]),
        )

        pid_counts = pd.Series(va_pid_s).value_counts()
        cnt = dict(zip(*np.unique(y_va_s, return_counts=True)))
        print(f"[Fold {fold}] val files={len(va_files)}, val patients={pid_counts.shape[0]}, #seq={len(y_va_s)}, seq label counts={cnt}")
        print(f"[Fold {fold}] seq/patient: min={int(pid_counts.min())}, med={int(pid_counts.median())}, max={int(pid_counts.max())}")

        tr_psd_t = torch.from_numpy(tr_psd_s).float()
        tr_de_t = torch.from_numpy(tr_de_s).float()
        tr_plv_t = torch.from_numpy(tr_plv_s).float()
        tr_wpli_t = torch.from_numpy(tr_wpli_s).float()
        y_tr_t = torch.from_numpy(y_tr_s).long()

        va_psd_t = torch.from_numpy(va_psd_s).float()
        va_de_t = torch.from_numpy(va_de_s).float()
        va_plv_t = torch.from_numpy(va_plv_s).float()
        va_wpli_t = torch.from_numpy(va_wpli_s).float()

        train_ds = TensorDataset(tr_psd_t, tr_de_t, tr_plv_t, tr_wpli_t, y_tr_t)
        train_loader = DataLoader(train_ds, batch_size=int(train_cfg["batch_size"]), shuffle=True, drop_last=False)

        model = BiMambaESCN(
            chans=int(model_cfg["chans"]),
            bands=int(model_cfg["bands"]),
            d_model=int(model_cfg["d_model"]),
            d_state=int(model_cfg["d_state"]),
            fusion=str(model_cfg["fusion"]),
            num_classes=int(model_cfg.get("num_classes", 2)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))

        for ep in range(1, int(train_cfg["epochs"]) + 1):
            model.train()
            for xb_psd, xb_de, xb_plv, xb_wpli, yb in train_loader:
                xb_psd = xb_psd.to(device)
                xb_de = xb_de.to(device)
                xb_plv = xb_plv.to(device)
                xb_wpli = xb_wpli.to(device)
                yb = yb.to(device)

                opt.zero_grad()
                logits = model(xb_psd, xb_de, xb_plv, xb_wpli)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()

            if ep == 1 or ep % 10 == 0 or ep == int(train_cfg["epochs"]):
                with torch.no_grad():
                    v_logits = predict_logits_batched(model, va_psd_t, va_de_t, va_plv_t, va_wpli_t, batch_size=int(train_cfg["val_batch_size"]), device=device)
                    v_pred = v_logits.argmax(1).numpy()
                    v_acc = (v_pred == y_va_s).mean()
                print(f"  Ep{ep:03d}  val_seq_acc(argmax)={v_acc:.3f}")

        tr_logits = predict_logits_batched(model, tr_psd_t, tr_de_t, tr_plv_t, tr_wpli_t, batch_size=int(train_cfg["val_batch_size"]), device=device)
        p_tr = tr_logits.softmax(1)[:, 1].numpy()
        best_thr, best_stat = select_threshold_youden_j(y_tr_s, p_tr, n_grid=int(thr_cfg["grid_n"]), fallback=float(thr_cfg["fallback"]))
        thr_folds.append(best_thr)
        print(f"[Fold {fold}] Youden-thr={best_thr:.4f} (J={best_stat['J']:.4f}, sens={best_stat['sens']:.4f}, spec={best_stat['spec']:.4f})")

        va_logits = predict_logits_batched(model, va_psd_t, va_de_t, va_plv_t, va_wpli_t, batch_size=int(train_cfg["val_batch_size"]), device=device)
        prob_np = va_logits.softmax(1)[:, 1].numpy()
        pred_argmax = va_logits.argmax(1).numpy()
        y_va_np = y_va_s.astype(int)

        m_seq = compute_basic_metrics(y_va_np, pred_argmax, y_prob=prob_np)
        print(f"Fold {fold} seq(argmax): Acc={m_seq['acc']:.3f} F1={m_seq['f1']:.3f} AUC={m_seq['auc']:.3f} Prec={m_seq['prec']:.3f} Rec={m_seq['rec']:.3f} Spec={m_seq['spec']:.3f}")
        seq_metrics_folds.append([m_seq["acc"], m_seq["f1"], m_seq["auc"], m_seq["prec"], m_seq["rec"], m_seq["spec"]])

        m_vote, agg_vote = groupwise_majority_vote(y_true_unit=y_va_np, y_prob_unit=prob_np, group_ids=va_pid_s, threshold=float(best_thr))
        print(f"         patient-wise(VOTE all seq @Youden): Acc={m_vote['acc']:.3f} Prec={m_vote['prec']:.3f} Rec={m_vote['rec']:.3f} Spec={m_vote['spec']:.3f} (val patients={len(agg_vote)})")
        patientvote_metrics_folds.append([m_vote["acc"], m_vote["prec"], m_vote["rec"], m_vote["spec"]])

        for k in vote_k_list:
            rep_vk = groupwise_vote_k_report(
                y_true_unit=y_va_np,
                y_prob_unit=prob_np,
                group_ids=va_pid_s,
                k=int(k),
                threshold=float(best_thr),
                n_repeat=int(vote_cfg["repeat"]),
                seed=2025 + fold * 1000 + int(k),
            )
            print(f"         patient-wise(VOTE@{k} seq @Youden): Acc={rep_vk['acc'][0]:.3f}±{rep_vk['acc'][1]:.3f} Prec={rep_vk['prec'][0]:.3f}±{rep_vk['prec'][1]:.3f} Rec={rep_vk['rec'][0]:.3f}±{rep_vk['rec'][1]:.3f} Spec={rep_vk['spec'][0]:.3f}±{rep_vk['spec'][1]:.3f} (val patients={rep_vk['n_groups']}, repeat={rep_vk['n_repeat']})")
            voteK_metrics_by_k[k].append([rep_vk["acc"][0], rep_vk["prec"][0], rep_vk["rec"][0], rep_vk["spec"][0]])

        pos = agg_vote[agg_vote["y_true"] == 1]["p_mean"].values
        neg = agg_vote[agg_vote["y_true"] == 0]["p_mean"].values
        if len(pos) > 0 and len(neg) > 0:
            print(f"[Check-A] patient mean-p: pos_min={float(pos.min()):.4f}, neg_max={float(neg.max()):.4f}, margin={float(pos.min()-neg.max()):.4f}")

    seq_metrics_folds = np.asarray(seq_metrics_folds, dtype=float)
    patientvote_metrics_folds = np.asarray(patientvote_metrics_folds, dtype=float)
    thr_folds = np.asarray(thr_folds, dtype=float)

    print("\n===== 10-fold Summary (T=5, stride=1, subject-wise GroupKFold) =====")
    m, s = mean_std(thr_folds); print(f"Selected threshold (Youden): {m:.4f} ± {s:.4f}")
    labels = ["Seq Accuracy    (argmax)", "Seq F1-score    (argmax)", "Seq AUC", "Seq Precision   (argmax)", "Seq Sensitivity (argmax)", "Seq Specificity (argmax)"]
    for i, label in enumerate(labels):
        m, s = mean_std(seq_metrics_folds[:, i]); print(f"{label}: {m:.3f} ± {s:.3f}")

    labels_p = ["Patient Accuracy (VOTE all seq @Youden)", "Patient Precision(VOTE all seq @Youden)", "Patient Sensitivity(VOTE all seq @Youden)", "Patient Specificity(VOTE all seq @Youden)"]
    for i, label in enumerate(labels_p):
        m, s = mean_std(patientvote_metrics_folds[:, i]); print(f"{label}: {m:.3f} ± {s:.3f}")

    if vote_k_list:
        print("\n----- Optional patient-wise VOTE@K seq @Youden: mean±SD over 10 folds -----")
        for k in vote_k_list:
            arr = np.asarray(voteK_metrics_by_k[k], dtype=float)
            m, s = mean_std(arr[:, 0]); print(f"K={k:<3d}  Acc : {m:.3f} ± {s:.3f}")
            m, s = mean_std(arr[:, 1]); print(f"      Prec: {m:.3f} ± {s:.3f}")
            m, s = mean_std(arr[:, 2]); print(f"      Sens: {m:.3f} ± {s:.3f}")
            m, s = mean_std(arr[:, 3]); print(f"      Spec: {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
