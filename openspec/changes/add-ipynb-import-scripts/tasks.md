## 1. Persiapan

- [x] 1.1 Verifikasi `import.py` dan `import_constants.py` tidak akan dimodifikasi selama perubahan ini (keduanya harus tetap ada persis seperti sekarang di akhir implementasi).
- [x] 1.2 Verifikasi tidak ada file bernama `import.ipynb` atau `import_constants.ipynb` di root repo yang akan di-overwrite.

## 2. Buat `import.ipynb`

- [x] 2.1 Tulis notebook JSON v4 valid di `c:\#fileUtama\pre-thesis\import.ipynb`.
- [x] 2.2 Sel 1 (markdown): judul "Dota 2 Dataset Import — Folder Tahun" + catatan "Jalankan ulang sel 5 untuk update data baru tanpa re-run semuanya".
- [x] 2.3 Sel 2 (code): import `kagglehub`, `os`, `shutil`, `load_dotenv`.
- [x] 2.4 Sel 3 (code): `load_dotenv()` + validasi `KAGGLE_USERNAME`/`KAGGLE_KEY` dengan `EnvironmentError` (copy verbatim dari `import.py`).
- [x] 2.5 Sel 4 (code): definisi `dataset_slug`, `target_folders` (2016-2025 + 202601/02/03), `target_files` (12 file), `output_dir`, `os.makedirs`.
- [x] 2.6 Sel 5 (code): loop download per-folder, per-file dengan skip-if-exists dan try/except per-file (copy verbatim loop dari `import.py`).
- [x] 2.7 Sel 6 (code): cetak baris pemisah + pesan "Proses selesai!".

## 3. Buat `import_constants.ipynb`

- [x] 3.1 Tulis notebook JSON v4 valid di `c:\#fileUtama\pre-thesis\import_constants.ipynb`.
- [x] 3.2 Sel 1 (markdown): judul "Dota 2 Dataset Import — Folder Constants" + catatan re-run.
- [x] 3.3 Sel 2 (code): import sama seperti `import.ipynb`.
- [x] 3.4 Sel 3 (code): load `.env` + validasi kredensial (copy verbatim dari `import_constants.py`).
- [x] 3.5 Sel 4 (code): `dataset_slug`, `target_files` (14 file Constants), `output_dir = ./dota2_dataset_bersih/Constants`, `os.makedirs`.
- [x] 3.6 Sel 5 (code): loop download per-file dengan skip-if-exists (copy verbatim dari `import_constants.py`).
- [x] 3.7 Sel 6 (code): cetak pesan "Proses selesai!".

## 4. Verifikasi

- [x] 4.1 Buka kedua `.ipynb` di VS Code; pastikan parseable (tidak ada error "Invalid notebook") dan setiap sel tampil sebagai sel terpisah. — diverifikasi via `python -c "json.load(...)"` (valid JSON v4).
- [x] 4.2 Konfirmasi `import.py` dan `import_constants.py` masih ada dan isinya tidak berubah (diff kosong). — `git status` menunjukkan keduanya unmodified.
- [x] 4.3 Jalankan sel 2 (imports) dan sel 3 (validasi `.env`) di kedua notebook untuk sanity check — harus sukses tanpa error (asumsikan `.env` sudah valid). — diverifikasi via ekuivalen `python -c` (imports + `.env` validation OK).
- [ ] 4.4 (Opsional, jika waktunya cukup) Jalankan sel 4-6 `import_constants.ipynb` end-to-end karena volumenya kecil (14 file) dan pastikan `./dota2_dataset_bersih/Constants/` berisi file yang diharapkan atau di-skip semua jika sudah ada. — **skipped**: diserahkan ke user karena melibatkan download jaringan aktual.
