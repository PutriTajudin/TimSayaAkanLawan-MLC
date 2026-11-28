"""Jalankan notebook di GPU Kaggle dari VS Code.

Pemakaian (dari root repo):
    python scripts/kaggle_run.py upload-src        # kirim src/, configs/, annotations/ ke Kaggle
    python scripts/kaggle_run.py push 03           # kirim + tunggu sampai selesai + unduh hasil
    python scripts/kaggle_run.py push 03 --no-wait # kirim saja, jangan ditunggu
    python scripts/kaggle_run.py push 01 --cpu     # tanpa GPU (hemat kuota)
    python scripts/kaggle_run.py status 03         # cek running / complete / error
    python scripts/kaggle_run.py output 03         # unduh hasil ke outputs/kaggle/<nama-kernel>/

Jalankan `upload-src` lagi setiap kali isi src/, configs/, atau annotations/ berubah,
sebelum `push`.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / ".kaggle_build"
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))["kaggle"]


def _cari_username() -> str:
    """Akun Kaggle milik orang yang menjalankan skrip ini (bukan akun Putri secara paksa).

    Urutan: variabel KAGGLE_USERNAME -> ~/.kaggle/kaggle.json -> config.yaml.
    """
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    kj = Path.home() / ".kaggle" / "kaggle.json"
    if kj.exists():
        try:
            nama = json.loads(kj.read_text(encoding="utf-8")).get("username")
            if nama:
                return nama
        except (OSError, ValueError):
            pass
    return CFG["username"]


USER = _cari_username()

# Notebook yang butuh output notebook lain (dipasang lewat kernel_sources)
DEPENDS = {
    "04": ["03"],
    "05": ["03"],
    "06": ["03", "05"],
    "07": ["03"],
    "08": [],      # baca langsung dari data kompetisi (raw), tidak butuh output notebook 03
    "09": ["08"],
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


def get_status(num: str) -> str:
    """Kembalikan RUNNING / QUEUED / COMPLETE / ERROR / CANCEL / UNKNOWN: ... (huruf besar).

    Hanya membaca kata status dari kalimat 'has status "..."'. Pesan error jaringan sementara
    TIDAK dianggap status ERROR (sebelumnya kata 'error' di mana pun dianggap notebook gagal).
    """
    r = subprocess.run(
        ["kaggle", "kernels", "status", kernel_id(num)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    teks = (r.stdout + r.stderr).strip()
    m = re.search(r'has status\s+"?(?:KernelWorkerStatus\.)?([A-Za-z_]+)"?', teks)
    if not m:
        return "UNKNOWN: " + teks[:200]
    kata = m.group(1).upper()
    if kata.startswith("CANCEL"):
        return "CANCEL"
    return kata


def wait_done(num: str, tiap: int = 30, batas_menit: int = 600) -> str:
    """Tunggu sampai notebook selesai. Aman dihentikan dengan Ctrl+C (notebook tetap jalan di Kaggle)."""
    mulai = time.time()
    terakhir = None
    while True:
        st = get_status(num)
        if st != terakhir:
            print(f"[{time.strftime('%H:%M:%S')}] status: {st}")
            terakhir = st
        if st in ("COMPLETE", "ERROR", "CANCEL"):
            return st
        if st.startswith("UNKNOWN"):
            print("Status tidak terbaca. Cek manual dengan: python scripts/kaggle_run.py status", num)
        if time.time() - mulai > batas_menit * 60:
            print("Batas tunggu habis; notebook mungkin masih jalan di Kaggle.")
            return st
        time.sleep(tiap)


def download_output(num: str) -> Path:
    dest = ROOT / "outputs" / "kaggle" / kernel_slug(num)
    dest.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", kernel_id(num), "-p", str(dest), "-o"])
    print(f"Hasil tersimpan di {dest}")
    return dest


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
    p.add_argument("--no-wait", action="store_true", help="kirim saja, tidak menunggu selesai")
    for name in ("status", "output"):
        sub.add_parser(name).add_argument("num")
    a = ap.parse_args()

    if a.cmd == "upload-src":
        upload_src(a.message)
    elif a.cmd == "push":
        push(a.num, a.cpu)
        if a.no_wait:
            print("Tidak menunggu. Cek nanti: python scripts/kaggle_run.py status", a.num)
        else:
            st = wait_done(a.num)
            if st == "COMPLETE":
                download_output(a.num)  # sekalian membawa log
            else:
                print(f"Notebook berakhir dengan status {st}. Buka log di browser untuk melihat penyebabnya.")
                sys.exit(1)
    elif a.cmd == "status":
        print(get_status(a.num))
    elif a.cmd == "output":
        download_output(a.num)


if __name__ == "__main__":
    main()
