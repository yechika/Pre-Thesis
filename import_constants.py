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

# Daftar lengkap file di dalam folder Constants (bisa ditambah jika ada yang kurang)
target_files = [
    "Constants.Abilities.csv",
    "Constants.AbilitiesId.csv",
    "Constants.AghanimHeroUpgrades.csv",
    "Constants.ChatWheel.csv",
    "Constants.GameModes.csv",
    "Constants.HeroAbilitiesAndTalents.csv",
    "Constants.HeroLore.csv",
    "Constants.Heroes.csv",
    "Constants.ItemIDs.csv",
    "Constants.Items.csv",
    "Constants.Leagues.csv",
    "Constants.Patch.csv",
    "Constants.PatchNotes.csv",
    "Constants.Regions.csv"
]

output_dir = "./dota2_dataset_bersih/Constants"
os.makedirs(output_dir, exist_ok=True)

print(f"Memulai unduhan khusus folder Constants ({len(target_files)} file)...\n" + "="*50)

for file in target_files:
    kaggle_path = f"Constants/{file}"
    target_file_path = os.path.join(output_dir, file)
    
    # Skip jika sudah ada
    if os.path.exists(target_file_path):
        continue
        
    try:
        print(f"  -> {file}...", end="", flush=True)
        # Mengunduh spesifik file ini
        cache_path = kagglehub.dataset_download(dataset_slug, path=kaggle_path)
        shutil.copy2(cache_path, target_file_path)
        print(" [OK]")
    except Exception as e:
        # Menangani nama file yang rentan terpotong (typo) di screenshot dengan anggun
        print(f" [GAGAL / Harap pastikan nama file tepat]")

print("="*50)
print(f"Proses selesai! Silakan cek folder '{output_dir}'.")
