import kagglehub
import os
import shutil
from dotenv import load_dotenv

# Muat kredensial Kaggle dari file .env
load_dotenv()

# Validasi: pastikan KAGGLE_USERNAME dan KAGGLE_KEY sudah terisi
if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
    raise EnvironmentError(
        "KAGGLE_USERNAME atau KAGGLE_KEY tidak ditemukan di file .env!\n"
        "Pastikan .env berisi:\n"
        "  KAGGLE_USERNAME=username_kaggle_kamu\n"
        "  KAGGLE_KEY=api_key_kamu"
    )

dataset_slug = "bwandowando/dota-2-pro-league-matches-2023"

# Folder per tahun (Tahun dan Kuartal 2026)
# Catatan: Constants tidak diikutsertakan karena struktur isinya berbeda dari folder tahun
target_folders = [str(year) for year in range(2016, 2026)] + ["202601", "202602", "202603"]

# 12 File utama yang sama di setiap folder tahun
target_files = [
    "all_word_counts.csv",
    "chat.csv",
    "cosmetics.csv",
    "draft_timings.csv",
    "main_metadata.csv",
    "objectives.csv",
    "picks_bans.csv",
    "players.csv",
    "radiant_exp_adv.csv",
    "radiant_gold_adv.csv",
    "teamfights.csv",
    "teams.csv"
]

output_dir = "./dota2_dataset_bersih"
os.makedirs(output_dir, exist_ok=True)

print(f"Memulai unduhan {len(target_folders)} folder x {len(target_files)} file...")
print("(Metode per-file untuk menghindari /Images dan zip 8.3GB)\n" + "="*50)

for folder in target_folders:
    print(f"\nMengunduh isi folder: {folder}...")
    folder_path = os.path.join(output_dir, folder)
    os.makedirs(folder_path, exist_ok=True)
    
    for file in target_files:
        kaggle_path = f"{folder}/{file}"
        target_file_path = os.path.join(folder_path, file)
        
        # Skip jika sudah ada
        if os.path.exists(target_file_path):
            continue
            
        try:
            print(f"  -> {file}...", end="", flush=True)
            # Mengunduh spesifik file ini tanpa menyentuh file lain
            cache_path = kagglehub.dataset_download(dataset_slug, path=kaggle_path)
            shutil.copy2(cache_path, target_file_path)
            print(" [OK]")
        except Exception as e:
            # Gunakan penanganan error yang hening jika file memang tidak ada di API
            print(f" [GAGAL / 404 Tidak Ditemukan]")

print("="*50)
print(f"Proses selesai! Silakan cek folder '{output_dir}'.")