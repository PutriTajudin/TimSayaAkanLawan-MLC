"""Cek data sudah lengkap. Hasil yang benar: 416 video, 1515 sequence x 30 frame, 45867 file."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import get_path, load_config

cfg = load_config()
raw = get_path(cfg, "raw")
train, test = raw / "train", raw / "test"

total = sum(1 for _, _, fs in os.walk(raw) for f in fs if not f.lower().endswith(".zip"))
video = {k: sum(len(os.listdir(train / k / s)) for s in os.listdir(train / k)) for k in ("normal", "cheat")}
seq = sorted(os.listdir(test))
salah = [s for s in seq if len(os.listdir(test / s)) != cfg["SEQ_LEN"]]

print("Lokasi data:", raw)
print("Video train:", video, "| total", sum(video.values()), "(harus 416)")
print("Sequence test:", len(seq), "(harus 1515) | sequence yang frame-nya bukan 30:", len(salah))
print("sample_submission.csv ada:", (raw / "sample_submission.csv").exists())
print("Total file:", total, "(harus 45867)")
