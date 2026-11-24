"""Metrik evaluasi kompetisi dan alat bantu untuk prediksi OOF.

Metrik yang dipakai diatur lewat cfg["METRIC"] di configs/config.yaml.
Metrik label (butuh threshold): "f1", "f1_macro", "accuracy".
Metrik probabilitas (tanpa threshold): "auc", "logloss".

Format OOF yang diharapkan (satu baris = satu potongan 30 frame):
    kolom "fold"  : nomor fold validasi
    kolom "label" : label asli 0/1
    kolom "prob"  : probabilitas cheat hasil model
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

METRIK_LABEL = {"f1", "f1_macro", "accuracy"}
METRIK_PROBA = {"auc", "logloss"}
LEBIH_KECIL_LEBIH_BAIK = {"logloss"}


def _cek_metrik(metrik):
    if metrik not in METRIK_LABEL | METRIK_PROBA:
        raise ValueError(f"Metrik '{metrik}' tidak dikenal. Pilih: {sorted(METRIK_LABEL | METRIK_PROBA)}")


def skor_label(y_true, y_pred, metrik="f1"):
    """Skor dari label prediksi 0/1."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    if metrik == "f1":
        return f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    if metrik == "f1_macro":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    if metrik == "accuracy":
        return accuracy_score(y_true, y_pred)
    raise ValueError(f"'{metrik}' bukan metrik label")


def skor(y_true, prob, metrik="f1", threshold=0.5):
    """Skor dari probabilitas. Metrik label memakai threshold, metrik probabilitas tidak."""
    _cek_metrik(metrik)
    y_true, prob = np.asarray(y_true).astype(int), np.asarray(prob, dtype=float)
    if metrik == "auc":
        if len(np.unique(y_true)) < 2:
            return np.nan  # AUC tidak terdefinisi kalau hanya ada satu kelas
        return roc_auc_score(y_true, prob)
    if metrik == "logloss":
        return log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1])
    return skor_label(y_true, (prob >= threshold).astype(int), metrik)


def cari_threshold(y_true, prob, metrik="f1", grid=None):
    """Cari threshold terbaik untuk metrik label.
    Pakai OOF gabungan semua fold, bukan per fold, supaya tidak overfit ke satu fold.
    Return: (threshold, skor). Untuk metrik probabilitas, threshold = 0.5 (tidak berpengaruh)."""
    _cek_metrik(metrik)
    if metrik in METRIK_PROBA:
        return 0.5, skor(y_true, prob, metrik)
    if grid is None:
        grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    hasil = [(t, skor(y_true, prob, metrik, t)) for t in grid]
    terbaik = max(s for _, s in hasil)
    # Kalau ada beberapa threshold dengan skor sama, ambil yang paling dekat 0.5 (lebih stabil)
    kandidat = [t for t, s in hasil if s == terbaik]
    t_pilih = min(kandidat, key=lambda t: abs(t - 0.5))
    return float(t_pilih), float(terbaik)


def skor_per_fold(oof, metrik="f1", threshold=0.5, kolom_label="label", kolom_prob="prob"):
    """Skor tiap fold + rata-rata, simpangan baku, dan skor OOF gabungan.
    Return: (tabel_per_fold, ringkasan_dict)."""
    oof = oof.dropna(subset=[kolom_label, kolom_prob])
    baris = []
    for fold, g in oof.groupby("fold"):
        baris.append({
            "fold": fold,
            "n": len(g),
            "rasio_cheat": g[kolom_label].mean(),
            "skor": skor(g[kolom_label], g[kolom_prob], metrik, threshold),
        })
    tabel = pd.DataFrame(baris)
    ringkasan = {
        "metrik": metrik,
        "threshold": threshold,
        "mean": tabel["skor"].mean(),
        "std": tabel["skor"].std(),
        "oof": skor(oof[kolom_label], oof[kolom_prob], metrik, threshold),
    }
    return tabel, ringkasan


def skor_per_grup(oof, kolom_grup, metrik="f1", threshold=0.5, kolom_label="label", kolom_prob="prob"):
    """Skor per grup (misal "view", "behavior", "subject_id") untuk mencari bagian yang lemah.
    Catatan: di grup yang isinya satu kelas saja (misal behavior normal), f1 bisa 0 dan auc NaN,
    jadi lihat juga kolom akurasi."""
    oof = oof.dropna(subset=[kolom_label, kolom_prob])
    baris = []
    for nilai, g in oof.groupby(kolom_grup):
        pred = (g[kolom_prob] >= threshold).astype(int)
        baris.append({
            kolom_grup: nilai,
            "n": len(g),
            "rasio_cheat": g[kolom_label].mean(),
            "rasio_pred_cheat": pred.mean(),
            "akurasi": accuracy_score(g[kolom_label].astype(int), pred),
            "skor": skor(g[kolom_label], g[kolom_prob], metrik, threshold),
        })
    return pd.DataFrame(baris).sort_values("akurasi")


def ringkas_teks(ringkasan):
    """Satu baris teks untuk print atau kolom notes di logs/experiments.csv."""
    return (f"{ringkasan['metrik']}: oof={ringkasan['oof']:.4f} "
            f"mean={ringkasan['mean']:.4f}±{ringkasan['std']:.4f} thr={ringkasan['threshold']:.2f}")
