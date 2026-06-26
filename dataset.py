import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


DEFAULT_CHANS = 10
DEFAULT_BANDS = 5


def parse_label_pid_fid(fp: str) -> Tuple[int, str, str]:
    base = os.path.basename(fp)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Bad filename; expected at least two parts like MCS_001_1: {base}")

    tag = parts[0].upper()
    if tag == "MCS":
        lab = 1
    elif tag == "UWS":
        lab = 0
    else:
        raise ValueError(f"Unknown class prefix '{parts[0]}' in filename: {base}")

    pid = f"{parts[0]}_{parts[1]}"
    fid = stem
    return lab, pid, fid


def list_npz_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".npz"))


def collect_files(mcs_dir: str, uws_dir: str):
    mcs_files = list_npz_files(mcs_dir)
    uws_files = list_npz_files(uws_dir)
    files = np.array(mcs_files + uws_files, dtype=object)

    labels_file, patients, file_ids = [], [], []
    for fp in files:
        lab, pid, fid = parse_label_pid_fid(fp)
        labels_file.append(lab)
        patients.append(pid)
        file_ids.append(fid)

    return (
        files,
        np.asarray(labels_file, dtype=int),
        np.asarray(patients, dtype=object),
        np.asarray(file_ids, dtype=object),
    )


def pool_pair_to_nodal_strength(x: np.ndarray, chans: int = DEFAULT_CHANS, bands: int = DEFAULT_BANDS) -> np.ndarray:
    if x.ndim != 3 or x.shape[1] != chans * chans or x.shape[2] != bands:
        raise ValueError(f"Expected connectivity shape [n_epochs,{chans*chans},{bands}], got {x.shape}")
    x = x.reshape(-1, chans, chans, bands)
    return x.mean(axis=2).astype(np.float32)


def load_frames_raw(fp: str, chans: int = DEFAULT_CHANS, bands: int = DEFAULT_BANDS):
    lab, pid, fid = parse_label_pid_fid(fp)

    with np.load(fp) as d:
        psd = d["psd"].astype(np.float32)
        de = d["de_bands"].astype(np.float32)
        plv = pool_pair_to_nodal_strength(d["plv"].astype(np.float32), chans=chans, bands=bands)
        wpli = pool_pair_to_nodal_strength(d["wpli"].astype(np.float32), chans=chans, bands=bands)

    for name, arr in [("psd", psd), ("de", de), ("plv", plv), ("wpli", wpli)]:
        if arr.ndim != 3 or arr.shape[1:] != (chans, bands):
            raise ValueError(f"{name} in {fp} has shape {arr.shape}, expected [n_epochs,{chans},{bands}]")

    n = psd.shape[0]
    if not (de.shape[0] == plv.shape[0] == wpli.shape[0] == n):
        raise ValueError(f"Feature epoch counts differ in {fp}")

    y = np.full(n, lab, dtype=np.int64)
    pid_arr = np.array([pid] * n, dtype=object)
    fid_arr = np.array([fid] * n, dtype=object)
    return psd, de, plv, wpli, y, pid_arr, fid_arr


def expand_files(file_subset: np.ndarray, chans: int = DEFAULT_CHANS, bands: int = DEFAULT_BANDS):
    """Expand file-level samples to epoch-level arrays."""
    lists = [[] for _ in range(7)]
    for fp in file_subset:
        items = load_frames_raw(fp, chans=chans, bands=bands)
        for k, value in enumerate(items):
            lists[k].append(value)
    return [np.concatenate(x, axis=0) for x in lists]


@dataclass
class FoldNormalizer:

    mean_: Dict[str, np.ndarray]
    std_: Dict[str, np.ndarray]

    @staticmethod
    def fit(psd, de, plv, wpli) -> "FoldNormalizer":
        mean_, std_ = {}, {}
        for name, arr in [("psd", psd), ("de", de), ("plv", plv), ("wpli", wpli)]:
            mean_[name] = arr.mean(axis=0, keepdims=True).astype(np.float32)
            std_[name] = (arr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        return FoldNormalizer(mean_=mean_, std_=std_)

    def transform_one(self, arr: np.ndarray, name: str) -> np.ndarray:
        return ((arr - self.mean_[name]) / self.std_[name]).astype(np.float32)

    def transform(self, psd, de, plv, wpli):
        return (
            self.transform_one(psd, "psd"),
            self.transform_one(de, "de"),
            self.transform_one(plv, "plv"),
            self.transform_one(wpli, "wpli"),
        )


def pack_sequences_by_file(psd, de, plv, wpli, y, pid, fid, seq_len=5, stride=1, pad_last=False):
    psd = np.asarray(psd)
    de = np.asarray(de)
    plv = np.asarray(plv)
    wpli = np.asarray(wpli)
    y = np.asarray(y).astype(int)
    pid = np.asarray(pid, dtype=object)
    fid = np.asarray(fid, dtype=object)

    out_psd, out_de, out_plv, out_wpli = [], [], [], []
    out_y, out_pid, out_fid = [], [], []

    for file_id in pd.unique(fid):
        idx = np.where(fid == file_id)[0]
        if idx.size == 0:
            continue

        psd_f, de_f, plv_f, wpli_f = psd[idx], de[idx], plv[idx], wpli[idx]
        y_f, pid_f = y[idx], pid[idx]

        if np.unique(y_f).size != 1:
            raise ValueError(f"Mixed labels within file {file_id}")

        n = psd_f.shape[0]
        if n < seq_len and not pad_last:
            continue

        start = 0
        while start < n:
            end = start + seq_len
            if end <= n:
                sl = slice(start, end)
                out_psd.append(psd_f[sl])
                out_de.append(de_f[sl])
                out_plv.append(plv_f[sl])
                out_wpli.append(wpli_f[sl])
                out_y.append(int(y_f[0]))
                out_pid.append(pid_f[0])
                out_fid.append(file_id)
            else:
                if not pad_last:
                    break
                need = end - n
                out_psd.append(np.concatenate([psd_f[start:n], np.repeat(psd_f[-1][None, ...], need, axis=0)], axis=0))
                out_de.append(np.concatenate([de_f[start:n], np.repeat(de_f[-1][None, ...], need, axis=0)], axis=0))
                out_plv.append(np.concatenate([plv_f[start:n], np.repeat(plv_f[-1][None, ...], need, axis=0)], axis=0))
                out_wpli.append(np.concatenate([wpli_f[start:n], np.repeat(wpli_f[-1][None, ...], need, axis=0)], axis=0))
                out_y.append(int(y_f[0]))
                out_pid.append(pid_f[0])
                out_fid.append(file_id)
                break
            start += stride

    if len(out_y) == 0:
        raise RuntimeError("No sequences were created. Check SEQ_LEN, SEQ_STRIDE, and file lengths.")

    return (
        np.stack(out_psd, axis=0).astype(np.float32),
        np.stack(out_de, axis=0).astype(np.float32),
        np.stack(out_plv, axis=0).astype(np.float32),
        np.stack(out_wpli, axis=0).astype(np.float32),
        np.asarray(out_y, dtype=np.int64),
        np.asarray(out_pid, dtype=object),
        np.asarray(out_fid, dtype=object),
    )


def count_sequences_from_n_epochs(n_epochs: int, seq_len: int, stride: int, pad_last: bool) -> int:
    n = int(n_epochs)
    if n <= 0:
        return 0
    cnt = 0
    start = 0
    while start < n:
        end = start + seq_len
        if end <= n:
            cnt += 1
        else:
            if pad_last:
                cnt += 1
            break
        start += stride
    return cnt


def estimate_max_sequences_per_pid(files, seq_len, stride, pad_last):
    pid2count = {}
    fid2count = {}
    for fp in files:
        _, pid, fid = parse_label_pid_fid(fp)
        with np.load(fp) as d:
            n_epochs = int(d["psd"].shape[0])
        n_seq = count_sequences_from_n_epochs(n_epochs, seq_len, stride, pad_last)
        pid2count[pid] = pid2count.get(pid, 0) + n_seq
        fid2count[fid] = n_seq

    counts = np.array(list(pid2count.values()), dtype=int)
    stats = {
        "pid_min": int(counts.min()) if len(counts) else 0,
        "pid_med": int(np.median(counts)) if len(counts) else 0,
        "pid_max": int(counts.max()) if len(counts) else 0,
        "n_pid": int(len(counts)),
        "fid_min": int(min(fid2count.values())) if len(fid2count) else 0,
        "fid_med": int(np.median(list(fid2count.values()))) if len(fid2count) else 0,
        "fid_max": int(max(fid2count.values())) if len(fid2count) else 0,
        "n_fid": int(len(fid2count)),
    }
    return stats["pid_max"], stats, pid2count
