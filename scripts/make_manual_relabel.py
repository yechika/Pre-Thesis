"""Build blind manual-annotation workbooks (no prior labels shown) for the 3 author-annotators.

Identical messages in the same context (normalised text x end-of-match phase x sender-team result) are
labelled once; the label is later propagated to every gold row in that group (see _groups.csv).
Every group goes to exactly one annotator, plus a shared overlap set of N_OVERLAP groups labelled by
all three (for inter-annotator Cohen's/Fleiss' kappa). Row order is shuffled per sheet.
"""
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.styles import Font

SRC = 'data/gold/gold_final.csv'
OUT = 'data/gold/manual_relabel'
ANNOTATORS = ['daniel', 'dhitan', 'aldiaz']
N_OVERLAP = 300
SEED = 2026
LATE_FRAC = 0.8   # message in the last 20% of the match = end-of-match

g = pd.read_csv(SRC).reset_index().rename(columns={'index': 'gold_row'})
g['norm'] = g.key.astype(str).str.strip().str.lower()
g['late'] = (g.match_minute * 60 / g.duration >= LATE_FRAC).astype(int)
gkeys = ['norm', 'late', 'match_outcome_for_player']
g['row_id'] = g.groupby(gkeys, sort=False).ngroup()
g[['gold_row', 'row_id']].to_csv(f'{OUT}/_groups.csv', index=False)
# one representative per group (median-time occurrence), with group size as context
g = g.sort_values(['row_id', 'match_minute'])
size = g.groupby('row_id').row_id.transform('size')
pick = g.groupby('row_id').cumcount() == size // 2
d = g[pick].assign(n_occurrences=size[pick]).reset_index(drop=True)
rng = np.random.default_rng(SEED)
overlap = d.groupby('year', group_keys=False).apply(
    lambda g: g.sample(frac=N_OVERLAP / len(d), random_state=SEED)).row_id.tolist()
rest = rng.permutation(d.loc[~d.row_id.isin(overlap), 'row_id'].to_numpy())
parts = np.array_split(rest, len(ANNOTATORS))

ctx = pd.DataFrame({
    'row_id': d.row_id,
    'message': d.key.astype(str),
    'end_of_match': d.late.map({1: 'yes (last 20%)', 0: 'no'}),
    'match_minute': d.match_minute.round(1),
    'match_length_min': (d.duration / 60).round(1),
    'player_team_result': d.match_outcome_for_player,
    'year': d.year,
    'tier': d.tier,
    'n_occurrences': d.n_occurrences,
})
LABELS = ['sentiment', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'is_dota_jargon', 'is_ambiguous',
          'notes']

assign = []
for name, part in zip(ANNOTATORS, parts):
    ids = rng.permutation(np.concatenate([part, overlap]))
    sheet = ctx.set_index('row_id').loc[ids].reset_index()
    for c in LABELS:
        sheet[c] = ''
    wb = Workbook()
    ws = wb.active
    ws.title = 'annotate'
    ws.append(list(sheet.columns))
    for r in sheet.itertuples(index=False):
        ws.append(list(r))
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = 'C2'
    n = len(sheet) + 1
    col = {c: ws.cell(1, i + 1).column_letter for i, c in enumerate(sheet.columns)}
    dv_s = DataValidation(type='list', formula1='"positive,neutral,negative"', allow_blank=False)
    dv_b = DataValidation(type='list', formula1='"0,1"', allow_blank=False)
    ws.add_data_validation(dv_s)
    ws.add_data_validation(dv_b)
    dv_s.add(f"{col['sentiment']}2:{col['sentiment']}{n}")
    for c in LABELS[1:9]:
        dv_b.add(f'{col[c]}2:{col[c]}{n}')
    ws.column_dimensions[col['message']].width = 40
    ws.column_dimensions[col['notes']].width = 30
    wb.save(f'{OUT}/annotate_{name}.xlsx')
    assign += [(int(i), name, int(i in overlap)) for i in ids]
    print(name, len(sheet))

# kept by the coordinator only, not shared with annotators
pd.DataFrame(assign, columns=['row_id', 'annotator', 'in_overlap']).to_csv(f'{OUT}/_assignment.csv', index=False)
