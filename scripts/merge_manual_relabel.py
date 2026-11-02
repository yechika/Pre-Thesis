"""Merge the three returned manual-annotation workbooks into the gold standard.

- Each annotation unit (row_id) maps to one or more gold rows via _groups.csv.
- Overlap units (labelled by all three annotators) get a majority vote; three-way sentiment ties are listed in
  _sentiment_ties.csv and resolved to 'neutral' with confidence 'adjudicated' unless ADJUDICATED overrides them.
- The previous train/test assignment is kept (same rows), only the labels are replaced.
- Previous (pilot) label files are moved to data/gold/archive_pilot/ before the new ones are written.

Writes data/gold/{gold_final,train,test}.csv and data/gold/manual_relabel/agreement.json.
"""
import json
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

GOLD = Path('data/gold')
REL = GOLD / 'manual_relabel'
ANNOTATORS = ['daniel', 'dhitan', 'aldiaz']
TOX = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
BIN = TOX + ['is_dota_jargon', 'is_ambiguous']
KEY = ['match_id', 'time', 'player_slot', 'key']
ADJUDICATED = {}   # row_id -> sentiment, filled after the authors discuss the tied units

def row_keys(df):
    return df[KEY].astype(object).where(df[KEY].notna(), '').astype(str).agg('|'.join, axis=1)


sheets = {a: pd.read_excel(REL / 'returned' / f'annotate_{a}.xlsx').set_index('row_id') for a in ANNOTATORS}
assign = pd.read_csv(REL / '_assignment.csv')
overlap = sorted(assign.loc[assign.in_overlap == 1, 'row_id'].unique())


def fleiss_kappa(cols):
    cats = sorted(set(np.concatenate(cols)))
    tab = np.array([[sum(c[i] == k for c in cols) for k in cats] for i in range(len(cols[0]))])
    n = len(cols)
    p_obs = ((tab * (tab - 1)).sum(1) / (n * (n - 1))).mean()
    p_exp = ((tab.sum(0) / tab.sum()) ** 2).sum()
    return (p_obs - p_exp) / (1 - p_exp)


agreement = {'n_units_overlap': len(overlap)}
for lab in ['sentiment', 'toxic', 'obscene', 'insult']:
    cols = [sheets[a].loc[overlap, lab].astype(str).to_numpy() for a in ANNOTATORS]
    agreement[lab] = {
        'fleiss_kappa': round(float(fleiss_kappa(cols)), 3),
        'pairwise_cohen': {f'{x}-{y}': round(float(cohen_kappa_score(sheets[x].loc[overlap, lab].astype(str),
                                                                     sheets[y].loc[overlap, lab].astype(str))), 3)
                           for x, y in combinations(ANNOTATORS, 2)},
    }

# ---- unit-level consensus ----
units, ties = [], []
for a in ANNOTATORS:
    own = assign[(assign.annotator == a) & (assign.in_overlap == 0)].row_id
    s = sheets[a].loc[own]
    for rid, r in s.iterrows():
        units.append({'row_id': rid, 'sentiment': r.sentiment, **{c: int(r[c]) for c in BIN},
                      'notes': r.notes if pd.notna(r.notes) else '', 'annotator_id': a,
                      'confidence_sentiment': 'single', 'n_annotators': 1})
for rid in overlap:
    rows = [sheets[a].loc[rid] for a in ANNOTATORS]
    votes = pd.Series([r.sentiment for r in rows]).value_counts()
    if votes.iloc[0] == 3:
        sent, conf = votes.index[0], 'unanimous'
    elif votes.iloc[0] == 2:
        sent, conf = votes.index[0], 'majority'
    else:
        sent, conf = ADJUDICATED.get(rid, 'neutral'), 'adjudicated'
        ties.append({'row_id': rid, 'message': rows[0].message, **{a: r.sentiment for a, r in zip(ANNOTATORS, rows)}})
    units.append({'row_id': rid, 'sentiment': sent, **{c: int(sum(int(r[c]) for r in rows) >= 2) for c in BIN},
                  'notes': ' | '.join(str(r.notes) for r in rows if pd.notna(r.notes)), 'annotator_id': '+'.join(ANNOTATORS),
                  'confidence_sentiment': conf, 'n_annotators': 3})
units = pd.DataFrame(units)
# label hierarchy from the codebook
units.loc[units[['severe_toxic', 'threat', 'identity_hate']].max(axis=1) == 1, 'toxic'] = 1
pd.DataFrame(ties).to_csv(REL / '_sentiment_ties.csv', index=False)

# ---- propagate to gold rows, keep old split ----
old = pd.read_csv(GOLD / 'gold_final.csv')
groups = pd.read_csv(REL / '_groups.csv')
assert len(groups) == len(old) and units.row_id.nunique() == groups.row_id.nunique()
meta = [c for c in old.columns if c not in
        ['annotator_id', 'sentiment', 'is_dota_jargon', 'is_ambiguous', 'notes', 'confidence_sentiment', 'n_annotators']
        and not c.startswith(('tox_', 'conf_tox_'))]
new = old[meta].copy()
new['row_id'] = groups.sort_values('gold_row').row_id.to_numpy()
new = new.merge(units.rename(columns={c: f'tox_{c}' for c in TOX}), on='row_id', how='left').drop(columns='row_id')
assert new.sentiment.notna().all() and len(new) == len(old)

archive = GOLD / 'archive_pilot'
archive.mkdir(exist_ok=True)
split = {}
for name in ['train', 'test']:
    part = pd.read_csv(GOLD / f'{name}.csv')
    split[name] = row_keys(part)
for f in ['gold_final.csv', 'train.csv', 'test.csv', 'agreement_report.md']:
    if (GOLD / f).exists():
        shutil.move(GOLD / f, archive / f)

new_key = row_keys(new)
new.to_csv(GOLD / 'gold_final.csv', index=False)
for name, keys in split.items():
    part = new[new_key.isin(set(keys))]
    assert len(part) == len(keys), (name, len(part), len(keys))
    part.to_csv(GOLD / f'{name}.csv', index=False)

agreement.update({
    'n_messages': len(new), 'n_units': int(units.row_id.nunique()), 'n_sentiment_ties': len(ties),
    'pct_ambiguous_messages': round(100 * new.is_ambiguous.mean(), 1),
    'sentiment_counts': new.sentiment.value_counts().to_dict(),
    'toxicity_counts': {c: int(new[f'tox_{c}'].sum()) for c in TOX},
    'jargon_messages': int(new.is_dota_jargon.sum()),
})
json.dump(agreement, open(REL / 'agreement.json', 'w'), indent=2)
print(json.dumps(agreement, indent=2))
