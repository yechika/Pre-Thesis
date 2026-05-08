## ADDED Requirements

### Requirement: Empat model deep learning yang dievaluasi

Sistem SHALL fine-tune dan/atau menerapkan empat model deep learning berikut sebagai kandidat:

1. **BERT-base-uncased** — fine-tune untuk sentimen 3-class DAN toksisitas multi-label.
2. **RoBERTa-base** — fine-tune untuk sentimen 3-class DAN toksisitas multi-label.
3. **DistilBERT-base-uncased** — fine-tune untuk sentimen 3-class DAN toksisitas multi-label.
4. **Detoxify (`unitary/toxic-bert`)** — pakai checkpoint yang sudah dilatih untuk toksisitas multi-label sebagai specialist; pada tugas sentimen, dipakai sebagai zero-shot reference dengan mapping `toxic_score → negative` (tidak fine-tune ulang untuk sentimen).

Setiap checkpoint MUST di-pin ke revision SHA Hugging Face (bukan branch `main`) supaya reproducible.

#### Scenario: Empat model tersedia
- **WHEN** notebook training selesai dijalankan
- **THEN** `models/` berisi minimal: `models/bert-sentiment/`, `models/bert-toxicity/`, `models/roberta-sentiment/`, `models/roberta-toxicity/`, `models/distilbert-sentiment/`, `models/distilbert-toxicity/`, dan `models/detoxify/manifest.json` (referensi ke checkpoint Hugging Face)

#### Scenario: Detoxify zero-shot untuk sentimen
- **WHEN** Detoxify dievaluasi di gold-test sentimen
- **THEN** sentimen diprediksi via mapping `toxic_score >= 0.5 → negative`, sisanya `neutral`/`positive` (tanpa fine-tune); rasional dicatat di laporan

### Requirement: Transfer learning dari dataset publik

Sistem SHALL fine-tune BERT-base, RoBERTa-base, dan DistilBERT di dataset publik:
- **Sentimen 3-class**: Sentiment140 (binary, dipetakan ke 3-class via threshold confidence) ATAU SST-2 + dataset auxiliary untuk netral. Pilihan didokumentasikan di `configs/experiment.yaml` dengan rasional.
- **Toksisitas multi-label**: Jigsaw Toxic Comment Classification + HateEval. Label gabungan dimapping ke 6-label skema Detoxify.

Dataset publik MUST di-cache lokal dengan SHA256 file sumber dicatat di manifest.

#### Scenario: Fine-tune sentimen berhasil
- **WHEN** training sentimen dijalankan untuk satu model (mis. BERT)
- **THEN** validation F1-macro pada split publik ≥ 0.70 dilaporkan dan checkpoint disimpan ke `models/bert-sentiment/`

#### Scenario: Fine-tune toksisitas berhasil
- **WHEN** training toksisitas multi-label dijalankan untuk satu model
- **THEN** validation F1-micro pada split Jigsaw ≥ 0.85 dilaporkan dan checkpoint disimpan ke `models/<model>-toxicity/`

#### Scenario: Dataset publik di-cache
- **WHEN** training dijalankan ulang
- **THEN** dataset publik dibaca dari cache lokal tanpa download ulang, dan SHA256 dicocokkan dengan manifest

### Requirement: Konfigurasi training yang dapat di-reproduce

Sistem SHALL menetapkan dan mendokumentasikan hyperparameter di `configs/experiment.yaml`, minimal: `learning_rate`, `batch_size`, `num_epochs`, `weight_decay`, `warmup_ratio`, `max_seq_length`, `gradient_accumulation_steps`, `seed`, `precision` (fp16/bf16/fp32), `optimizer`.

Sistem MUST menggunakan seed yang sama di seluruh komponen stochastic (data shuffling, dropout, weight init) sehingga eksekusi ulang dengan konfig identik menghasilkan checkpoint dengan loss kurva yang konsisten (toleransi numerik kecil dari non-determinisme CUDA).

#### Scenario: Konfig terdokumentasi
- **WHEN** training dijalankan
- **THEN** notebook training mencatat seluruh hyperparameter ke `reports/training_<model>_<task>.log`

#### Scenario: Re-run reproducible
- **WHEN** training dijalankan dua kali dengan konfig dan seed identik di hardware sama
- **THEN** loss kurva validation berimpit dalam toleransi ±1% di setiap epoch

### Requirement: Inferensi batched ke seluruh chat Dota 2

Sistem SHALL menjalankan inferensi setiap model fine-tuned (sentimen + toksisitas) ke seluruh `data/processed/*.parquet` dengan batching GPU.

Output inferensi MUST disimpan ke `data/inference/<model>_<task>/<folder>.parquet` dengan kolom: `match_id`, `time`, `player_slot`, kolom probabilitas per-class atau per-label, dan `predicted_label`.

Inferensi MUST checkpointable per-folder — gagal di tengah folder 2024 tidak menyebabkan rerun folder 2016-2023.

#### Scenario: Inferensi seluruh dekade
- **WHEN** inferensi BERT-sentimen dijalankan untuk seluruh folder
- **THEN** `data/inference/bert_sentiment/<folder>.parquet` dibuat untuk setiap folder yang ada di `data/processed/`

#### Scenario: Resume setelah crash
- **WHEN** inferensi dihentikan setelah folder 2020 selesai dan dijalankan ulang
- **THEN** folder 2016-2020 di-skip dan inferensi melanjutkan dari folder 2021

#### Scenario: Output kompatibel join
- **WHEN** output inferensi di-merge dengan `data/processed/`
- **THEN** join pada `(match_id, time, player_slot)` berhasil tanpa duplikasi atau missing
