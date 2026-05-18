# dataset-import-notebooks

## Purpose

Menyediakan versi notebook Jupyter (`import.ipynb` dan `import_constants.ipynb`) dari skrip `import.py` dan `import_constants.py` untuk mengunduh dataset Dota 2 dari Kaggle, sehingga pengguna dapat menjalankan ulang sel unduhan secara inkremental tanpa re-run seluruh skrip.

## Requirements

### Requirement: Notebook versi dari import.py

Sistem SHALL menyediakan `import.ipynb` di root repo yang mereplikasi fungsionalitas `import.py` — mengunduh 12 file utama untuk setiap folder tahun (2016-2025) dan kuartal 2026 (202601, 202602, 202603) dari dataset Kaggle `bwandowando/dota-2-pro-league-matches-2023` ke `./dota2_dataset_bersih/<folder>/`.

Notebook MUST dipecah menjadi sel-sel logis terpisah: (1) imports, (2) load & validasi kredensial `.env`, (3) konfigurasi (dataset_slug, target_folders, target_files, output_dir), (4) loop unduhan per-file, (5) ringkasan hasil.

File `.py` asli (`import.py`) MUST TIDAK dimodifikasi atau dihapus.

#### Scenario: Notebook berhasil dieksekusi end-to-end
- **WHEN** pengguna menjalankan semua sel `import.ipynb` secara berurutan dengan `.env` valid
- **THEN** file-file target terunduh ke `./dota2_dataset_bersih/<folder>/` dan setiap sel menampilkan progres per-file (`[OK]` atau `[GAGAL]`) seperti script aslinya

#### Scenario: File .py asli tetap ada
- **WHEN** perubahan selesai di-apply
- **THEN** `import.py` masih ada di root repo dengan konten identik seperti sebelum perubahan

#### Scenario: Kredensial Kaggle hilang
- **WHEN** `.env` tidak berisi `KAGGLE_USERNAME` atau `KAGGLE_KEY` dan sel validasi dieksekusi
- **THEN** sel raise `EnvironmentError` dengan pesan yang menjelaskan variabel mana yang kurang

### Requirement: Notebook versi dari import_constants.py

Sistem SHALL menyediakan `import_constants.ipynb` di root repo yang mereplikasi fungsionalitas `import_constants.py` — mengunduh 14 file Constants (Abilities, Heroes, Items, dll.) dari dataset Kaggle yang sama ke `./dota2_dataset_bersih/Constants/`.

Notebook MUST dipecah menjadi sel-sel logis terpisah dengan struktur yang sama seperti `import.ipynb`.

File `.py` asli (`import_constants.py`) MUST TIDAK dimodifikasi atau dihapus.

#### Scenario: Notebook berhasil mengunduh folder Constants
- **WHEN** pengguna menjalankan semua sel `import_constants.ipynb` dengan `.env` valid
- **THEN** 14 file Constants target terunduh ke `./dota2_dataset_bersih/Constants/`

#### Scenario: File .py asli tetap ada
- **WHEN** perubahan selesai di-apply
- **THEN** `import_constants.py` masih ada di root repo dengan konten identik seperti sebelum perubahan

### Requirement: Idempoten untuk pembaruan inkremental

Kedua notebook SHALL melewati file yang sudah ada di `output_dir` (cek `os.path.exists` sebelum download) sehingga eksekusi ulang hanya mengunduh file baru/yang hilang.

#### Scenario: Re-run pada dataset yang sudah sebagian terunduh
- **WHEN** pengguna menjalankan ulang notebook sementara sebagian file target sudah ada di `output_dir`
- **THEN** sel loop unduhan MUST skip file yang sudah ada tanpa memanggil `kagglehub.dataset_download` untuk file tersebut, dan hanya mengunduh file yang belum ada

#### Scenario: Update dataset baru (tambah folder/kuartal baru)
- **WHEN** pengguna menambahkan entry folder baru ke daftar `target_folders` di sel konfigurasi dan menjalankan sel unduhan
- **THEN** hanya file pada folder baru yang diunduh; folder lama yang sudah lengkap tidak diunduh ulang

### Requirement: Error handling yang tidak menghentikan eksekusi

Loop unduhan SHALL menangkap exception per-file sehingga kegagalan satu file (mis. 404) tidak menghentikan pemrosesan file lainnya, sama seperti perilaku `.py` asli.

#### Scenario: Satu file gagal diunduh
- **WHEN** `kagglehub.dataset_download` melempar exception untuk satu file target
- **THEN** notebook mencetak penanda gagal untuk file tersebut lalu LANJUT ke file berikutnya tanpa raise keluar sel
