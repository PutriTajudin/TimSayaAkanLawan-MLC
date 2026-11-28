"""Membangun index potongan 30 frame dari video + anotasi manual.

Logika ini dipakai dua tempat:
  - scripts/buat_index.py (laptop, menyimpan outputs/index_potongan.csv)
  - notebook 04/05 di Kaggle, yang tidak kebagian folder scripts/

Fungsi di sini TIDAK menulis apa pun ke annotations/ (di Kaggle folder itu hanya-baca).
"""
import os
from pathlib import Path

import pandas as pd

from .data import ON_KAGGLE, ROOT, get_path
from .labeling import frame_dalam_interval, gabung_interval, label_potongan

KOLOM = ["video", "mulai_frame", "label", "dipakai", "frame_cheat", "frame_ragu",
         "subject", "fold", "kategori", "sudut", "perilaku", "grid30"]


def baca_anotasi(anotasi_dir):
    """Baca semua annotations_<nama>.csv (kecuali annotations_all.csv). File kosong dilewati.
    File format lama (tanpa kolom jenis) dianggap semua barisnya cheat.
    Return: (interval_cheat, interval_ragu, tabel_gabungan) dengan dict {video: [(mulai, selesai), ...]}."""
    tabel = []
    for p in sorted(Path(anotasi_dir).glob("annotations_*.csv")):
        if p.name == "annotations_all.csv" or p.stat().st_size == 0:
            continue
        df = pd.read_csv(p)
        if "jenis" not in df.columns:
            df["jenis"] = "cheat"
        df["jenis"] = df["jenis"].fillna("cheat")
        tabel.append(df.assign(anotator=p.stem.replace("annotations_", "")))
    if not tabel:
        return {}, {}, None
    df = pd.concat(tabel, ignore_index=True)
    ganda = df.groupby("video")["anotator"].nunique()
    ganda = ganda[ganda > 1]
    if len(ganda):
        raise SystemExit(f"Video ini dianotasi oleh lebih dari satu orang: {list(ganda.index[:5])}")
    cheat, ragu = {}, {}
    for video, g in df.groupby("video"):
        c, r = g[g["jenis"] == "cheat"], g[g["jenis"] == "ragu"]
        if len(c):
            cheat[video] = [(int(a), int(b)) for a, b in zip(c["frame_mulai"], c["frame_selesai"])]
        if len(r):
            ragu[video] = [(int(a), int(b)) for a, b in zip(r["frame_mulai"], r["frame_selesai"])]
    return cheat, ragu, df


def _jumlah_frame(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def bangun_index(cfg, loncat=10, sementara=False):
    """Satu baris = satu potongan `SEQ_LEN` frame. Return DataFrame dengan kolom KOLOM.

    sementara=True hanya untuk uji kode: video cheat yang belum dianotasi diberi label
    video (semua potongan = 1). JANGAN dipakai untuk training akhir.

    df.attrs berisi: belum (daftar video cheat tanpa anotasi), n_rentang_ragu, n_video_ragu.
    """
    seq_len = cfg["SEQ_LEN"]
    train_dir = get_path(cfg, "raw") / "train"
    anotasi_dir = get_path(cfg, "annotations")

    folds = pd.read_csv(get_path(cfg, "folds"))
    if folds.empty:
        raise SystemExit("annotations/folds.csv masih kosong. Jalankan: python scripts/buat_folds.py")
    folds = folds.set_index("subject")["fold"].to_dict()

    anotasi, anotasi_ragu, _ = baca_anotasi(anotasi_dir)

    baris, belum = [], []
    for kategori in ("normal", "cheat"):
        for subjek in sorted(os.listdir(train_dir / kategori), key=lambda s: int(s.split("_")[1])):
            for nama in sorted(os.listdir(train_dir / kategori / subjek)):
                rel = f"train/{kategori}/{subjek}/{nama}"
                n_frame = _jumlah_frame(train_dir / kategori / subjek / nama)

                if kategori == "normal":
                    interval = []
                elif rel in anotasi:
                    interval = gabung_interval(anotasi[rel])
                    if any(b > n_frame for _, b in interval):
                        print(f"PERINGATAN: interval melewati akhir video: {rel}")
                else:
                    belum.append(rel)
                    interval = [(0, n_frame)]      # hanya dipakai kalau sementara=True

                interval_ragu = gabung_interval(anotasi_ragu.get(rel, []))
                sudut, perilaku = os.path.splitext(nama)[0].split("_", 1)
                for mulai in range(0, n_frame - seq_len + 1, loncat):
                    label, n_cheat = label_potongan(mulai, interval, seq_len)
                    baris.append({
                        "video": rel, "mulai_frame": mulai,
                        "label": -1 if label is None else label,
                        "dipakai": label is not None, "frame_cheat": n_cheat,
                        "frame_ragu": frame_dalam_interval(mulai, interval_ragu, seq_len),
                        "subject": subjek, "fold": folds[subjek], "kategori": kategori,
                        "sudut": sudut, "perilaku": perilaku,
                        "grid30": mulai % seq_len == 0,   # True = potongan tanpa geser (bentuk test)
                    })

    if belum and not sementara:
        raise SystemExit(f"Masih {len(belum)} video cheat belum dianotasi. Contoh: {belum[:3]}\n"
                         "Selesaikan anotasi, atau pakai --sementara hanya untuk uji kode.")
    if belum:
        print(f"PERINGATAN: {len(belum)} video cheat belum dianotasi (dipakai label video).")

    df = pd.DataFrame(baris, columns=KOLOM)
    df.attrs["belum"] = belum
    df.attrs["n_rentang_ragu"] = sum(len(v) for v in anotasi_ragu.values())
    df.attrs["n_video_ragu"] = len(anotasi_ragu)
    return df


def cari_index(cfg, nama="index_potongan.csv"):
    """Cari index_potongan.csv yang sudah jadi: di repo, di /kaggle/working, atau di output
    kernel lain yang dipasang lewat kernel_sources. Return Path atau None."""
    kandidat = [ROOT / "outputs" / nama]
    if ON_KAGGLE:
        kandidat.insert(0, Path("/kaggle/working/outputs") / nama)
        kandidat += sorted(Path("/kaggle/input").rglob(nama))
    for p in kandidat:
        if p.exists():
            return p
    return None


def muat_index(cfg, loncat=10, sementara=False):
    """Pakai index_potongan.csv kalau ada, kalau tidak bangun dari video + anotasi."""
    p = cari_index(cfg)
    if p is not None:
        print("Index dibaca dari:", p)
        return pd.read_csv(p)
    print("Index belum ada, dibangun dari video + anotasi.")
    return bangun_index(cfg, loncat=loncat, sementara=sementara)
