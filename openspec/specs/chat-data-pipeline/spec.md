# chat-data-pipeline

## Purpose

Pipeline ingest, clean, language filter, dan join metadata untuk chat in-game Dota 2 dari dataset bersih (2016-2026).

## Requirements

### Requirement: Ingest chat lintas dekade dari dataset bersih

Sistem SHALL menyediakan pipeline ingest yang membaca seluruh `chat.csv` di `dota2_dataset_bersih/<folder>/` untuk folder tahunan 2016-2025 dan folder kuartal 2026 (`202601`, `202602`, `202603`, `202604`), serta menggabungkannya dengan `main_metadata.csv` per-folder berdasarkan `match_id`.

Pipeline MUST membaca dari `dota2_dataset_bersih/` sebagai sumber read-only — tidak boleh memodifikasi, memindahkan, atau menghapus file di direktori tersebut.

#### Scenario: Ingest seluruh folder berhasil
- **WHEN** pengguna menjalankan notebook ingest dengan dataset lengkap di `dota2_dataset_bersih/`
- **THEN** seluruh folder tahunan 2016-2025 dan folder 2026 yang tersedia diproses tanpa error, dan output parquet per-folder ditulis ke `data/processed/<folder>.parquet`

#### Scenario: Folder kuartal 2026 belum lengkap
- **WHEN** sebagian folder kuartal 2026 (mis. `202604`) belum tersedia di `dota2_dataset_bersih/`
- **THEN** pipeline melewati folder yang tidak ada dengan log peringatan `[SKIP] folder=202604 (tidak ditemukan)` dan TIDAK menggagalkan eksekusi keseluruhan

#### Scenario: File raw tidak dimodifikasi
- **WHEN** pipeline ingest selesai dijalankan
- **THEN** seluruh file di `dota2_dataset_bersih/` (termasuk `chat.csv`, `main_metadata.csv`, dan `Constants/`) memiliki konten dan timestamp modifikasi yang identik dengan sebelum ingest

### Requirement: Eksklusi chatwheel dan emote non-tekstual

Pipeline SHALL mengeksklusi semua baris `chat.csv` dengan `type == "chatwheel"` dari output processed, karena mereka adalah perintah suara/visual non-tekstual yang berada di luar ruang lingkup penelitian (proposal §1.5).

Pipeline MUST mempertahankan baris `chat.csv` dengan `type` lainnya (mis. `chat`, `phrases`) untuk diproses lebih lanjut.

#### Scenario: Chatwheel dieksklusi
- **WHEN** baris dengan `type == "chatwheel"` ditemukan dalam `chat.csv`
- **THEN** baris tersebut TIDAK ada di output `data/processed/<folder>.parquet`, dan jumlah baris yang dieksklusi dilaporkan di log ingest

#### Scenario: Tipe chat lain dipertahankan
- **WHEN** baris dengan `type == "chat"` (atau tipe non-chatwheel lain) ditemukan
- **THEN** baris tersebut diproses lebih lanjut dan masuk ke output processed

### Requirement: Filter bahasa Inggris

Pipeline SHALL mendeteksi bahasa setiap pesan `key` (kolom teks) dan menyimpan hanya pesan berbahasa Inggris di output processed utama (`data/processed/<folder>.parquet`).

Pipeline MUST menyimpan pesan non-Inggris secara terpisah di `data/processed/<folder>_non_english.parquet` agar dapat dianalisis terpisah atau diaudit, sesuai proposal §1.5.

Pesan sangat pendek (≤ 5 token, seluruhnya ASCII alfanumerik) MAY di-shortcut sebagai bahasa Inggris untuk menghindari false negative dari language detector pada pesan seperti "gg", "wp", "ez".

#### Scenario: Pesan Inggris masuk ke output utama
- **WHEN** pipeline memproses pesan "Good game well played"
- **THEN** pesan tersimpan di `data/processed/<folder>.parquet` dengan kolom `lang == "en"`

#### Scenario: Pesan non-Inggris dipisah
- **WHEN** pipeline memproses pesan dalam bahasa Rusia atau Tionghoa
- **THEN** pesan TIDAK ada di output utama dan ada di `data/processed/<folder>_non_english.parquet` dengan kolom `lang_detected` yang menunjukkan bahasa terdeteksi

#### Scenario: Pesan pendek di-shortcut
- **WHEN** pipeline memproses pesan "gg" atau "ez wp"
- **THEN** pesan diklasifikasikan sebagai Inggris dan masuk ke output utama

### Requirement: Penggabungan fitur kontekstual dari main_metadata

Pipeline SHALL melakukan join antara `chat.csv` dan `main_metadata.csv` per-folder pada `match_id` dan menambahkan minimal kolom kontekstual berikut ke output processed: `start_date_time`, `duration`, `radiant_win`, `leagueid`, `patch`, `region`, `lobby_type`, `game_mode`, `radiant_team_id`, `dire_team_id`.

Pipeline MUST menambahkan kolom turunan: `match_outcome_for_player` (menang/kalah dari sudut pandang `player_slot` pengirim chat), `match_minute` (waktu chat dari awal pertandingan, dalam menit, dari kolom `time`), `year`, `month`, `quarter`.

#### Scenario: Join berhasil
- **WHEN** baris chat dengan `match_id == 7515635423` di-join dengan metadata
- **THEN** output processed berisi kolom kontekstual termasuk `start_date_time`, `radiant_win`, `leagueid`, dan kolom turunan `year=2024`, `month=1`

#### Scenario: Outcome dari sudut pandang pengirim
- **WHEN** chat dikirim oleh `player_slot < 128` (Radiant) di pertandingan dengan `radiant_win == True`
- **THEN** kolom `match_outcome_for_player` bernilai `"win"`; jika `radiant_win == False` maka `"loss"`

#### Scenario: Match_id tanpa metadata
- **WHEN** baris chat memiliki `match_id` yang tidak ditemukan di `main_metadata.csv` folder yang sama
- **THEN** baris tersebut dipertahankan dengan kolom kontekstual ber-NaN dan dilaporkan di log dengan jumlah agregat

### Requirement: Output parquet per-folder dengan schema stabil

Pipeline SHALL menulis output ke `data/processed/<folder>.parquet` dengan kompresi snappy dan schema yang stabil lintas folder, sehingga seluruh output dapat di-concat tanpa friction schema.

Schema MUST berisi minimal kolom: `match_id` (int64), `time` (float64), `type` (string), `key` (string), `slot` (float64), `player_slot` (float64), `lang` (string), `match_outcome_for_player` (string), `start_date_time` (timestamp), `duration` (int64), `radiant_win` (bool), `leagueid` (int64), `patch` (int64), `region` (float64), `year` (int16), `month` (int8), `quarter` (int8).

#### Scenario: Schema konsisten lintas folder
- **WHEN** seluruh output `data/processed/*.parquet` dibaca dan di-concat
- **THEN** operasi concat berhasil tanpa error schema mismatch

#### Scenario: Idempoten — re-run skip
- **WHEN** pipeline dijalankan ulang dan output `data/processed/<folder>.parquet` sudah ada DENGAN hash sumber yang tidak berubah
- **THEN** pipeline melewati folder tersebut dengan log `[SKIP] folder=2024 (cached)` dan TIDAK menulis ulang
