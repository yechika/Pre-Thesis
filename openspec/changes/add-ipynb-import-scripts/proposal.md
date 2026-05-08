## Why

Skrip unduhan dataset Kaggle saat ini hanya tersedia sebagai `.py` standalone (`import.py` dan `import_constants.py`). Untuk iterasi eksplorasi data dan pembaruan dataset berkala, menjalankan ulang script dari awal kurang praktis — pengguna butuh eksekusi per-sel, inspeksi variabel, dan progres inline yang khas notebook, tanpa harus re-install dependencies atau menghapus dataset lama.

## What Changes

- Tambahkan `import.ipynb` sebagai versi notebook dari `import.py` (unduhan per-folder tahun + file utama).
- Tambahkan `import_constants.ipynb` sebagai versi notebook dari `import_constants.py` (unduhan folder Constants).
- Pertahankan kedua file `.py` asli tanpa modifikasi — keduanya tetap sebagai entry point CLI/otomasi.
- Notebook harus idempotent: sel-nya aman dijalankan ulang untuk "update data baru" tanpa mengunduh ulang file yang sudah ada (logika skip-if-exists dipertahankan).
- Struktur sel notebook dipisah secara logis: (1) imports, (2) load & validasi `.env`, (3) konfigurasi (dataset slug, target folders/files, output dir), (4) loop unduhan, (5) ringkasan hasil.

## Capabilities

### New Capabilities
- `dataset-import-notebooks`: Versi interaktif (Jupyter) dari skrip unduhan dataset Dota 2 Kaggle, mendukung eksekusi per-sel dan pembaruan inkremental.

### Modified Capabilities
<!-- Tidak ada — skrip `.py` lama tidak dimodifikasi. -->

## Impact

- **File baru**: `import.ipynb`, `import_constants.ipynb` di root repo.
- **File tidak disentuh**: `import.py`, `import_constants.py` (tetap berfungsi).
- **Dependencies**: tidak ada dependency baru — `kagglehub`, `python-dotenv`, dan `shutil` sudah dipakai. Jupyter/IPython kernel diasumsikan tersedia di environment pengguna (VS Code / Jupyter).
- **Data**: output dir sama (`./dota2_dataset_bersih/...`) — notebook dan script berbagi cache sehingga bisa dipakai bergantian.
- **Kredensial**: keduanya tetap membaca `KAGGLE_USERNAME` / `KAGGLE_KEY` dari `.env`.
