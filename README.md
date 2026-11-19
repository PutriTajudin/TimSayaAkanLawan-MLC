# TimSayaAkanLawan-MLC

## Struktur

```
configs/config.yaml     SEED, path, FPS, SEQ_LEN, N_FOLDS
data/raw/               train, test, sample_submission.csv (tidak di-commit)
data/frames/            hasil potong video, opsional (tidak di-commit)
annotations/            guideline, anotasi per orang, gabungan, folds.csv
notebooks/              01_eda -> 06_inference_submission
src/                    kode yang dipakai ulang oleh banyak notebook
features/               fitur hasil ekstraksi (tidak di-commit)
models/                 bobot model (tidak di-commit, dikumpulkan lewat ZIP)
outputs/                prediksi OOF per fold dan file submission (tidak di-commit)
logs/experiments.csv    catatan eksperimen (di-commit)
```

## Setup

```bash
pip install -r requirements.txt
```

Taruh data kompetisi di `data/raw/`.

## Urutan notebook

1. `01_eda.ipynb`
2. `02_make_sequences.ipynb`
3. `03_feature_extraction.ipynb`
4. `04_train_baseline.ipynb`
5. `05_train_temporal.ipynb`
6. `06_inference_submission.ipynb`

Setiap eksperimen dicatat di `logs/experiments.csv`.

## Menjalankan di GPU Kaggle

Kode ditulis di VS Code, lalu dijalankan di server Kaggle. Syarat:

- `kaggle.json` ada di `C:\Users\<nama>\.kaggle\` (jangan taruh di repo).
- Nomor HP akun Kaggle sudah diverifikasi (syarat GPU).
- Akun sudah join kompetisi `data-slayer-4`.

```bash
python scripts/kaggle_run.py upload-src   # setiap src/, configs/, annotations/ berubah
python scripts/kaggle_run.py push 03      # jalankan notebook 03 di GPU T4
python scripts/kaggle_run.py status 03
python scripts/kaggle_run.py output 03    # hasil -> outputs/kaggle/mlc-03-feature-extraction/
```

`push 01 --cpu` untuk notebook yang tidak butuh GPU. Notebook 04/05 otomatis membaca
output 03, dan 06 membaca output 03 + 05 (lihat `DEPENDS` di `scripts/kaggle_run.py`).
Di dalam notebook, selalu pakai `get_path(cfg, "...")` supaya path benar di laptop maupun Kaggle.
