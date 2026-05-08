## Context

Repo saat ini memiliki dua skrip Python untuk mengunduh dataset Dota 2 dari Kaggle:
- [import.py](../../../import.py) — 130 file (10 tahun × 12 file + 3 kuartal 2026 × 12 file)
- [import_constants.py](../../../import_constants.py) — 14 file Constants

Keduanya dijalankan sebagai CLI. Pengguna (mahasiswa pre-thesis) butuh alur kerja notebook untuk:
1. Inspeksi interaktif progres unduhan tanpa menunggu script selesai
2. Memperbarui dataset (mis. menambah folder kuartal baru) dengan hanya menjalankan ulang sel tertentu
3. Menghindari re-install / re-download dari awal saat eksperimen

File `.py` tidak boleh dihapus — pengguna ingin keduanya tetap bisa dijalankan sebagai script otomasi.

## Goals / Non-Goals

**Goals:**
- Menghasilkan dua notebook `.ipynb` yang secara fungsional ekuivalen dengan `.py`-nya.
- Struktur sel yang memudahkan re-run parsial (ubah konfigurasi → jalankan ulang sel unduhan saja).
- Menjaga idempotency (skip-if-exists) sehingga update inkremental aman.

**Non-Goals:**
- Tidak menulis ulang logika unduhan atau mengganti `kagglehub`.
- Tidak menyentuh struktur `./dota2_dataset_bersih/` atau format output.
- Tidak menambah Jupyter sebagai dependency terinstal otomatis — pengguna sudah punya kernel via VS Code.
- Tidak membuat abstraksi bersama (shared helper module) antar kedua notebook — duplikasi yang minor lebih jelas dibaca di notebook.

## Decisions

### 1. Format notebook: JSON `.ipynb` v4 ditulis langsung (bukan via `jupytext`)

Kita menulis file `.ipynb` sebagai JSON notebook v4 valid secara langsung (via `Write` tool). Alternatif yang dipertimbangkan:
- **`jupytext` convert**: butuh dependency tambahan hanya untuk one-shot conversion.
- **`nbformat` API Python**: butuh eksekusi script; over-engineering untuk 5 sel × 2 file.

JSON manual cukup sederhana karena strukturnya kecil dan stabil.

### 2. Pembagian sel (cell granularity)

Setiap notebook dibagi menjadi 5 sel kode + 1 sel markdown judul:
1. **Markdown** — judul + instruksi singkat.
2. **Imports** — `kagglehub`, `os`, `shutil`, `dotenv`.
3. **Kredensial** — `load_dotenv()` + validasi `KAGGLE_USERNAME`/`KAGGLE_KEY` (raise jika kurang).
4. **Konfigurasi** — `dataset_slug`, `target_folders`/`target_files`, `output_dir`, `os.makedirs`. Sel ini yang diedit pengguna saat menambah folder/file baru.
5. **Unduhan** — loop download dengan skip-if-exists. Sel ini yang di-rerun saat update.
6. **Ringkasan** — cetak `Proses selesai...`.

Rasional: sel 4 (konfigurasi) terpisah dari sel 5 (eksekusi) supaya pengguna bisa ubah `target_folders` lalu re-run hanya sel 5 — ini inti "update tanpa install semua dari awal lagi" yang diminta pengguna.

### 3. Pertahankan logika `.py` verbatim

Isi tiap sel = salinan blok yang bersesuaian dari `.py`. Tidak ada refactor. Alasannya: (a) script sudah bekerja, (b) perilaku identik memudahkan reasoning, (c) diff minimal.

### 4. Tidak menghapus `.py`

Eksplisit: `import.py` dan `import_constants.py` dibiarkan persis seperti saat ini. Notebook adalah *tambahan*, bukan pengganti.

## Risks / Trade-offs

- **[Drift antara `.py` dan `.ipynb`]** → Mitigasi: terima drift. Pengguna akan pakai `.ipynb` untuk eksplorasi; `.py` untuk otomasi. Jika logika harus berubah di masa depan, update keduanya manual (volume kecil — 2 file).
- **[JSON notebook ditulis tangan rawan typo]** → Mitigasi: setelah dibuat, buka di VS Code/Jupyter untuk memverifikasi parseable; jalankan sel pertama (imports) untuk sanity check.
- **[`kagglehub.dataset_download` mengunduh ke cache sebelum di-copy]** → Tidak di-mitigasi: ini perilaku bawaan `kagglehub` dan sama di versi `.py`. Tidak ada duplikasi disk yang signifikan karena cache di-share Kaggle API.

## Migration Plan

Tidak ada migrasi — hanya menambah file baru. Rollback = hapus dua file `.ipynb`.
