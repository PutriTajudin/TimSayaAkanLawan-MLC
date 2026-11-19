"""Jalankan notebook di GPU Kaggle dari VS Code.

Pemakaian (dari root repo):
    python scripts/kaggle_run.py upload-src        # kirim src/, configs/, annotations/ ke Kaggle
    python scripts/kaggle_run.py push 03           # jalankan notebooks/03_*.ipynb di GPU Kaggle
    python scripts/kaggle_run.py push 01 --cpu     # tanpa GPU (hemat kuota)
    python scripts/kaggle_run.py status 03         # cek running / complete / error
    python scripts/kaggle_run.py output 03         # unduh hasil ke outputs/kaggle/<nama-kernel>/

Jalankan `upload-src` lagi setiap kali isi src/, configs/, atau annotations/ berubah,
sebelum `push`.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / ".kaggle_build"
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))["kaggle"]
USER = CFG["username"]

# Notebook yang butuh output notebook lain (dipasang lewat kernel_sources)
DEPENDS = {
    "04": ["03"],
    "05": ["03"],
    "06": ["03", "05"],
}


def run(cmd: list[str], check: bool = True) -> int:
    print(">", " ".join(cmd))
    code = subprocess.call(cmd)
    if check and code != 0:
        sys.exit(code)
    return code


def find_notebook(num: str) -> Path:
    hits = sorted((ROOT / "notebooks").glob(f"{num}_*.ipynb"))
    if not hits:
        sys.exit(f"Tidak ada notebook notebooks/{num}_*.ipynb")
    return hits[0]


def kernel_slug(num: str) -> str:
    stem = find_notebook(num).stem  # 03_feature_extraction
    return "mlc-" + stem.replace("_", "-")


def kernel_id(num: str) -> str:
    return f"{USER}/{kernel_slug(num)}"


def upload_src(message: str) -> None:
    slug = CFG["src_dataset"]
    stage = BUILD / slug
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for d in ("src", "configs", "annotations"):
        shutil.copytree(ROOT / d, stage / d, ignore=shutil.ignore_patterns("__pycache__"))
    (stage / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": slug, "id": f"{USER}/{slug}", "licenses": [{"name": "CC0-1.0"}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    # Dataset sudah ada -> versi baru; belum ada -> buat (private secara default)
    exists = subprocess.call(
        ["kaggle", "datasets", "status", f"{USER}/{slug}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0
    if exists:
        run(["kaggle", "datasets", "version", "-p", str(stage), "-m", message, "--dir-mode", "zip"])
    else:
        run(["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "zip"])
    print("Tunggu ~1 menit sampai dataset selesai diproses sebelum push notebook.")


def push(num: str, cpu: bool) -> None:
    nb = find_notebook(num)
    slug = kernel_slug(num)
    stage = BUILD / "kernels" / slug
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    shutil.copy2(nb, stage / nb.name)

    meta = {
        "id": kernel_id(num),
        "title": slug,
        "code_file": nb.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": not cpu,
        "enable_internet": True,
        "competition_sources": [CFG["competition"]],
        "dataset_sources": [f"{USER}/{CFG['src_dataset']}"],
        "kernel_sources": [kernel_id(d) for d in DEPENDS.get(num, [])],
    }
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cmd = ["kaggle", "kernels", "push", "-p", str(stage)]
    if not cpu:
        cmd += ["--accelerator", CFG["accelerator"]]
    run(cmd)
    print(f"Lihat di browser: https://www.kaggle.com/code/{kernel_id(num)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("upload-src")
    p.add_argument("-m", "--message", default="update src")
    p = sub.add_parser("push")
    p.add_argument("num")
    p.add_argument("--cpu", action="store_true")
    for name in ("status", "output"):
        sub.add_parser(name).add_argument("num")
    a = ap.parse_args()

    if a.cmd == "upload-src":
        upload_src(a.message)
    elif a.cmd == "push":
        push(a.num, a.cpu)
    elif a.cmd == "status":
        run(["kaggle", "kernels", "status", kernel_id(a.num)])
    elif a.cmd == "output":
        dest = ROOT / "outputs" / "kaggle" / kernel_slug(a.num)
        dest.mkdir(parents=True, exist_ok=True)
        run(["kaggle", "kernels", "output", kernel_id(a.num), "-p", str(dest), "-o"])
        print(f"Hasil tersimpan di {dest}")


if __name__ == "__main__":
    main()
