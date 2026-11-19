"""Mengisi annotations/folds.csv: 26 subjek dibagi ke N_FOLDS kelompok (default 5).
Aturan tetap, tanpa angka acak: fold = (nomor_subjek - 1) % N_FOLDS.
Contoh: subject_1 -> fold 0, subject_2 -> fold 1, subject_6 -> fold 0.
Hasilnya sudah ada di repo, jadi tidak perlu dijalankan ulang kecuali file ini kosong."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import get_path, load_config

JUMLAH_SUBJEK = 26

cfg = load_config()
file_folds = get_path(cfg, "folds")
with open(file_folds, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["subject", "fold"])
    for n in range(1, JUMLAH_SUBJEK + 1):
        w.writerow([f"subject_{n}", (n - 1) % cfg["N_FOLDS"]])
print("Tersimpan:", file_folds)
