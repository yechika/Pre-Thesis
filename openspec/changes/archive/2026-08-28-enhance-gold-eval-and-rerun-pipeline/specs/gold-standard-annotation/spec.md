## ADDED Requirements

### Requirement: Konsolidasi sampel manual dan probe evaluasi sekunder
Sistem SHALL mendukung konsolidasi sampel berlabel manual oleh peneliti (`annotator_id` selain `AI`) ke dalam dataset gold standard dan menghasilkan file probe sekunder berimbang (`data/gold/manual_labeled_probe.csv` / `data/gold/sample_for_labeling.csv`).

Proses review anotasi pada `notebooks/03_annotation_review.ipynb` TIDAK BOLEH memfilter keluar baris hanya karena annotator_id bukan AI, melainkan menyertakan semua baris yang memiliki label sentimen dan toksisitas valid.

#### Scenario: Anotasi manual disertakan dalam gold standard
- **WHEN** `03_annotation_review.ipynb` dijalankan
- **THEN** baris dari `data/gold/sample_for_labeling.csv` yang terisi lengkap disertakan ke dalam dataset konsolidasi tanpa tereliminasi oleh filter single-AI
