"""Membuat outputs/index_potongan.csv: satu baris = satu potongan 30 frame, dengan label dari anotasi manual.
Juga menggabungkan annotations_<nama>.csv menjadi annotations/annotations_all.csv.

Kolom frame_ragu = berapa frame dari potongan itu masuk bagian yang ditandai ragu (tombol r di alat anotasi).
Tanda ragu TIDAK mengubah label. Untuk uji "tanpa bagian ragu", buang baris yang frame_ragu > 0.

  python scripts/buat_index.py              # wajib: semua video cheat sudah dianotasi
  python scripts/buat_index.py --sementara  # hanya untuk uji kode: video cheat yang belum dianotasi
                                            # diberi label video (semua potongan = 1). JANGAN dipakai untuk training akhir.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import pandas as pd

from src.data import ROOT, get_path, load_config
from src.labeling import frame_dalam_interval, gabung_interval, label_potongan


def baca_anotasi(anotasi_dir):
    """Baca semua annotations_<nama>.csv (kecuali annotations_all.csv). File kosong dilewati.
    File format lama (tanpa kolom jenis) dianggap semua barisnya cheat.
    Return: (interval_cheat, interval_ragu, tabel_gabungan) dengan dict {video: [(mulai, selesai), ...]}."""
    tabel = []
    for p in sorted(anotasi_dir.glob("annotations_*.csv")):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementara", action="store_true")
    ap.add_argument("--loncat", type=int, default=10, help="geseran antar potongan (frame)")
    ap.add_argument("--out", default="outputs/index_potongan.csv")
    args = ap.parse_args()

    cfg = load_config()
    seq_len = cfg["SEQ_LEN"]
    raw = get_path(cfg, "raw")
    train_dir = raw / "train"
    anotasi_dir = get_path(cfg, "annotations")

    folds = pd.read_csv(get_path(cfg, "folds"))
    if folds.empty:
        raise SystemExit("annotations/folds.csv masih kosong. Jalankan: python scripts/buat_folds.py")
    folds = folds.set_index("subject")["fold"].to_dict()

    anotasi, anotasi_ragu, df_anotasi = baca_anotasi(anotasi_dir)
    if df_anotasi is not None:
        df_anotasi.to_csv(anotasi_dir / "annotations_all.csv", index=False, lineterminator="\n")

    baris, belum = [], []
    for kategori in ("normal", "cheat"):
        for subjek in sorted(os.listdir(train_dir / kategori), key=lambda s: int(s.split("_")[1])):
            for nama in sorted(os.listdir(train_dir / kategori / subjek)):
                rel = f"train/{kategori}/{subjek}/{nama}"
                cap = cv2.VideoCapture(str(train_dir / kategori / subjek / nama))
                n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if kategori == "normal":
                    interval = []
                elif rel in anotasi:
                    interval = gabung_interval(anotasi[rel])
                    if any(b > n_frame for _, b in interval):
                        print(f"PERINGATAN: interval melewati akhir video: {rel}")
                else:
                    belum.append(rel)
                    interval = [(0, n_frame)]      # hanya dipakai kalau --sementara

                interval_ragu = gabung_interval(anotasi_ragu.get(rel, []))
                sudut, perilaku = os.path.splitext(nama)[0].split("_", 1)
                for mulai in range(0, n_frame - seq_len + 1, args.loncat):
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

    if belum and not args.sementara:
        raise SystemExit(f"Masih {len(belum)} video cheat belum dianotasi. Contoh: {belum[:3]}\n"
                         "Selesaikan anotasi, atau pakai --sementara hanya untuk uji kode.")
    if belum:
        print(f"PERINGATAN: {len(belum)} video cheat belum dianotasi (dipakai label video).")

    df = pd.DataFrame(baris)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pakai = df[df["dipakai"]]
    print("Total potongan:", len(df), "| dipakai:", len(pakai), "| dibuang (tepat setengah):", len(df) - len(pakai))
    n_rentang = sum(len(v) for v in anotasi_ragu.values())
    print(f"Bagian ragu: {n_rentang} rentang di {len(anotasi_ragu)} video | potongan dipakai yang menyentuh bagian ragu: {int((pakai['frame_ragu'] > 0).sum())}")
    print("Label (potongan dipakai):\n", pakai.groupby("label").size().to_string())
    print("Potongan per fold:\n", pakai.groupby("fold").size().to_string())
    print("Tersimpan:", out)


if __name__ == "__main__":
    main()
