"""Generate visualizations for Chapter 4.1 (Dataset Characteristics)."""
import glob, os, sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'figure.dpi': 120, 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True})

PROC = Path("data/processed")
GOLD = Path("data/gold")
PLOTS = Path("reports/plots")
PLOTS.mkdir(parents=True, exist_ok=True)

C_BLUE, C_RED, C_GREY, C_ORANGE = '#3a7ca5', '#d1495b', '#b0b0b0', '#e3902f'

en_files = sorted(p for p in glob.glob(str(PROC / '[0-9]*.parquet')) if 'non_english' not in p)
ne_files = sorted(glob.glob(str(PROC / '*_non_english.parquet')))
TIER_MAP = {'Lainnya': 'Other', 'TI': 'The International', 'Major': 'Major',
            'DPC Tour': 'DPC Tour', 'DPC League': 'DPC Tour', 'Tour': 'DPC Tour'}

en_dfs = [pd.read_parquet(p, columns=['year', 'tier', 'match_id', 'key']) for p in en_files]
en_all = pd.concat(en_dfs, ignore_index=True)
en_all['tier_clean'] = en_all['tier'].map(lambda t: TIER_MAP.get(t, 'Other'))

ne_dfs = [pd.read_parquet(p, columns=['year']) for p in ne_files]
ne_all = pd.concat(ne_dfs, ignore_index=True) if ne_dfs else pd.DataFrame()

# Annual volume
ann_en = en_all.groupby('year').size()
ann_ne = ne_all.groupby('year').size() if not ne_all.empty else pd.Series(0, index=ann_en.index)

fig, ax = plt.subplots(figsize=(10, 4.5))
yrs = [int(y) for y in ann_en.index]
bar_en = ax.bar(yrs, ann_en.values, color=C_BLUE, label=f'English ({len(en_all):,} msgs)')
ax.set_title('Annual In-Game Chat Volume in Professional Dota 2 (2016–2026)', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Messages')
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / 'dataset_annual_volume.png')
plt.close()

# Tier distribution
tier_counts = en_all['tier_clean'].value_counts()
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(tier_counts.index[::-1], tier_counts.values[::-1], color=C_BLUE)
ax.set_title('Tournament Tier Distribution (English Corpus)', fontweight='bold')
ax.set_xlabel('Messages')
plt.tight_layout()
plt.savefig(PLOTS / 'dataset_tier_distribution.png')
plt.close()

# Gold distribution
gold_df = pd.read_csv(GOLD / 'gold_final.csv')
sent_counts = gold_df['sentiment'].value_counts()
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(sent_counts.index, sent_counts.values, color=[C_BLUE, C_GREY, C_RED])
ax.set_title(f'Gold-Standard Sentiment Distribution (n={len(gold_df):,})', fontweight='bold')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(PLOTS / 'dataset_sentiment_distribution.png')
plt.close()

print(f"Refreshed Chapter 4.1 figures in {PLOTS}")
print("=== CHARACTERISTICS VISUALIZATION COMPLETE ===")
