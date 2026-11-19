"""Path dan config yang otomatis menyesuaikan: jalan di laptop maupun di Kaggle.

Di laptop  : semua path relatif ke root repo (data/raw, outputs/, ...).
Di Kaggle  : data kompetisi dibaca dari /kaggle/input, hasil ditulis ke /kaggle/working.
"""
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
ON_KAGGLE = os.path.exists("/kaggle/input")
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# Path yang ikut di repo (dibawa ke Kaggle lewat dataset src) -> baca dari ROOT
REPO_KEYS = {"annotations", "folds", "logs"}


def load_config(path: str | Path = ROOT / "configs" / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_competition_dir(slug: str) -> Path:
    for p in (KAGGLE_INPUT / slug, KAGGLE_INPUT / "competitions" / slug):
        if p.exists():
            return p
    hits = sorted(KAGGLE_INPUT.rglob("sample_submission.csv"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError(
        f"Data kompetisi '{slug}' tidak ditemukan di /kaggle/input. "
        "Cek competition_sources di kernel-metadata.json."
    )


def get_path(cfg: dict, key: str, read: bool = False) -> Path:
    """Path untuk `key` di configs/config.yaml -> paths.

    read=True dipakai untuk membaca hasil notebook sebelumnya (mis. features dari 03
    di notebook 05). Di Kaggle, kalau belum ada di /kaggle/working, dicari di output
    kernel lain yang dipasang lewat kernel_sources.
    """
    rel = cfg["paths"][key]
    if not ON_KAGGLE:
        return ROOT / rel
    if key == "raw":
        return _find_competition_dir(cfg["kaggle"]["competition"])
    if key in REPO_KEYS:
        return ROOT / rel

    out = KAGGLE_WORKING / rel
    if read and not out.exists():
        name = Path(rel).name
        hits = [p for p in KAGGLE_INPUT.rglob(name) if p.is_dir()]
        if hits:
            return hits[0]
    if not read:
        out.mkdir(parents=True, exist_ok=True)
    return out


def load_folds(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(get_path(cfg, "folds"))
