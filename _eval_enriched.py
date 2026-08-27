"""Evaluate existing models on the inline-labeled enriched toxic-candidate probe set.

My labels are encoded below as LAB[idx] = [sentiment, toxic, severe_toxic, obscene,
threat, insult, identity_hate]. Joins each message to the stored model toxicity
predictions (no re-inference) and reports F1 + the ID-profanity blind-spot.
"""
import glob, os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

TOX = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SUP = ["toxic", "obscene", "insult", "identity_hate"]   # labels with positive support

# sentiment: 0=negative 1=neutral 2=positive ; then 6 binary tox flags
LAB = {
0:["neutral",1,0,1,0,0,0],1:["negative",1,0,0,0,1,0],2:["negative",1,0,1,0,0,0],3:["neutral",0,0,0,0,0,0],
4:["negative",1,0,1,0,0,0],5:["neutral",0,0,1,0,0,0],6:["neutral",0,0,0,0,0,0],7:["neutral",0,0,0,0,0,0],
8:["neutral",0,0,0,0,0,0],9:["negative",1,0,1,0,0,0],10:["neutral",0,0,0,0,0,0],11:["neutral",0,0,0,0,0,0],
12:["positive",0,0,0,0,0,0],13:["neutral",0,0,0,0,0,0],14:["neutral",0,0,0,0,0,0],15:["negative",1,0,0,0,0,0],
16:["neutral",0,0,1,0,0,0],17:["neutral",0,0,0,0,0,0],18:["negative",1,0,0,0,1,0],19:["neutral",0,0,0,0,0,0],
20:["neutral",0,0,0,0,0,0],21:["neutral",0,0,0,0,0,0],22:["positive",0,0,0,0,0,0],23:["neutral",0,0,1,0,0,0],
24:["positive",0,0,0,0,0,0],25:["negative",1,0,0,0,1,0],26:["negative",1,0,1,0,0,0],27:["neutral",0,0,0,0,0,0],
28:["neutral",1,0,1,0,0,0],29:["neutral",1,0,1,0,1,0],30:["neutral",0,0,0,0,0,0],31:["negative",1,0,0,0,1,0],
32:["neutral",0,0,0,0,0,0],33:["negative",1,0,0,0,1,0],34:["neutral",0,0,0,0,0,0],35:["negative",1,0,1,0,0,0],
36:["neutral",0,0,0,0,0,0],37:["neutral",0,0,0,0,0,0],38:["negative",1,0,1,0,0,0],39:["neutral",0,0,1,0,0,0],
40:["negative",1,0,0,0,1,0],41:["negative",1,0,1,0,0,0],42:["neutral",0,0,0,0,0,0],43:["neutral",0,0,0,0,0,0],
44:["neutral",0,0,0,0,0,0],45:["negative",1,0,1,0,1,0],46:["neutral",0,0,0,0,0,0],47:["positive",0,0,0,0,0,0],
48:["negative",1,0,1,0,0,0],49:["negative",1,0,1,0,0,0],50:["neutral",0,0,1,0,0,0],51:["neutral",0,0,0,0,0,0],
52:["negative",1,0,0,0,1,0],53:["neutral",0,0,1,0,0,0],54:["neutral",0,0,0,0,0,0],55:["neutral",0,0,0,0,0,0],
56:["negative",1,0,1,0,1,0],57:["negative",1,0,1,0,1,0],58:["neutral",0,0,0,0,0,0],59:["neutral",0,0,0,0,0,0],
60:["negative",1,0,1,0,1,0],61:["negative",1,0,0,0,1,0],62:["negative",1,0,1,0,0,0],63:["neutral",0,0,1,0,0,0],
64:["negative",1,0,1,0,0,0],65:["neutral",0,0,0,0,0,0],66:["neutral",0,0,0,0,0,0],67:["negative",1,0,1,0,1,0],
68:["negative",1,0,0,0,1,0],69:["neutral",0,0,0,0,0,0],70:["negative",1,0,0,0,1,0],71:["positive",0,0,0,0,0,0],
72:["neutral",0,0,1,0,0,0],73:["neutral",0,0,0,0,0,0],74:["negative",1,0,1,0,1,0],75:["neutral",0,0,0,0,0,0],
76:["neutral",0,0,1,0,0,0],77:["neutral",0,0,0,0,0,0],78:["negative",1,0,1,0,1,0],79:["positive",0,0,0,0,0,0],
80:["neutral",1,0,0,0,0,0],81:["neutral",0,0,0,0,0,0],82:["neutral",0,0,0,0,0,0],83:["negative",1,0,1,0,0,0],
84:["neutral",0,0,0,0,0,0],85:["negative",1,0,1,0,1,0],86:["neutral",0,0,1,0,0,0],87:["neutral",1,0,0,0,0,0],
88:["neutral",1,0,0,0,0,0],89:["neutral",0,0,1,0,0,0],90:["neutral",0,0,0,0,0,0],91:["negative",1,0,1,0,1,0],
92:["neutral",0,0,0,0,0,0],93:["neutral",0,0,0,0,0,0],94:["negative",1,0,1,0,0,0],95:["neutral",0,0,0,0,0,0],
96:["neutral",0,0,1,0,0,0],97:["neutral",0,0,0,0,0,0],98:["neutral",0,0,0,0,0,0],99:["negative",1,0,1,0,0,0],
100:["neutral",1,0,0,0,1,0],101:["neutral",0,0,1,0,0,0],102:["negative",1,0,0,0,1,0],103:["neutral",0,0,0,0,0,0],
104:["negative",1,0,0,0,1,0],105:["positive",0,0,1,0,0,0],106:["negative",1,0,0,0,0,0],107:["negative",1,0,1,0,1,0],
108:["negative",1,0,1,0,0,0],109:["neutral",0,0,0,0,0,0],110:["neutral",1,0,1,0,0,0],111:["neutral",0,0,0,0,0,0],
112:["neutral",0,0,1,0,0,0],113:["positive",0,0,0,0,0,0],114:["neutral",1,0,1,0,0,0],115:["neutral",0,0,0,0,0,0],
116:["neutral",0,0,0,0,0,0],117:["positive",0,0,0,0,0,0],118:["neutral",0,0,1,0,0,0],119:["neutral",0,0,0,0,0,0],
120:["negative",1,0,0,0,1,0],121:["negative",1,0,0,0,1,0],122:["neutral",0,0,0,0,0,0],123:["neutral",0,0,1,0,0,0],
124:["negative",1,0,1,0,1,0],125:["neutral",0,0,1,0,0,0],126:["neutral",0,0,1,0,0,0],127:["neutral",0,0,1,0,0,0],
128:["negative",1,0,1,0,1,0],129:["neutral",0,0,1,0,0,0],130:["negative",1,0,1,0,0,0],131:["negative",1,0,0,0,1,0],
132:["positive",0,0,0,0,0,0],133:["neutral",0,0,0,0,0,0],134:["neutral",1,0,1,0,0,0],135:["neutral",0,0,1,0,0,0],
136:["neutral",0,0,0,0,0,0],137:["neutral",0,0,1,0,0,0],138:["neutral",0,0,0,0,0,0],139:["positive",0,0,0,0,0,0],
140:["neutral",0,0,0,0,0,0],141:["neutral",0,0,1,0,0,0],142:["negative",1,0,1,0,1,1],143:["neutral",0,0,1,0,0,0],
144:["neutral",1,0,1,0,0,0],145:["neutral",0,0,0,0,0,0],146:["neutral",0,0,0,0,0,0],147:["negative",1,0,0,0,1,0],
148:["neutral",0,0,1,0,0,0],149:["neutral",0,0,0,0,0,0],150:["neutral",0,0,1,0,0,0],151:["negative",1,0,0,0,1,0],
152:["neutral",0,0,0,0,0,0],153:["positive",0,0,0,0,0,0],154:["negative",1,0,0,0,1,0],155:["negative",1,0,0,0,1,0],
156:["negative",1,0,1,0,1,0],157:["negative",1,0,0,0,0,0],158:["positive",0,0,0,0,0,0],159:["negative",1,0,1,0,1,0],
160:["negative",1,0,0,0,0,0],161:["neutral",0,0,0,0,0,0],162:["neutral",0,0,0,0,0,0],163:["neutral",0,0,0,0,0,0],
164:["neutral",0,0,0,0,0,0],165:["neutral",0,0,0,0,0,0],166:["neutral",0,0,1,0,0,0],167:["negative",1,0,0,0,1,0],
168:["neutral",0,0,1,0,0,0],169:["negative",1,0,0,0,1,0],170:["neutral",0,0,0,0,0,0],171:["neutral",0,0,1,0,0,0],
172:["negative",1,0,1,0,0,0],173:["negative",1,0,0,0,1,0],174:["neutral",0,0,0,0,0,0],175:["negative",1,0,0,0,1,0],
176:["negative",1,0,1,0,1,0],177:["negative",1,0,1,0,1,0],178:["neutral",0,0,1,0,0,0],179:["negative",1,0,0,0,1,0],
180:["neutral",0,0,0,0,0,0],181:["negative",1,0,1,0,1,0],182:["neutral",0,0,0,0,0,0],183:["negative",1,0,1,0,0,0],
184:["neutral",0,0,0,0,0,0],185:["negative",1,0,0,0,1,0],186:["neutral",0,0,1,0,0,0],187:["neutral",0,0,0,0,0,0],
188:["neutral",0,0,1,0,0,0],189:["neutral",0,0,0,0,0,0],190:["neutral",0,0,0,0,0,0],191:["negative",1,0,1,0,1,0],
192:["neutral",0,0,0,0,0,0],193:["negative",1,0,0,0,1,0],194:["neutral",0,0,1,0,0,0],195:["neutral",0,0,0,0,0,0],
196:["negative",1,0,1,0,1,0],197:["neutral",1,0,1,0,0,0],198:["neutral",0,0,1,0,0,0],199:["positive",0,0,0,0,0,0],
200:["neutral",0,0,1,0,0,0],201:["negative",1,0,0,0,1,0],202:["neutral",0,0,0,0,0,0],203:["neutral",0,0,1,0,0,0],
204:["neutral",0,0,0,0,0,0],205:["negative",1,0,1,0,1,0],206:["negative",1,0,0,0,1,0],207:["neutral",0,0,0,0,0,0],
208:["negative",1,0,1,0,1,0],209:["negative",1,0,0,0,1,0],210:["neutral",0,0,0,0,0,0],211:["neutral",0,0,0,0,0,0],
212:["neutral",0,0,0,0,0,0],213:["negative",1,0,0,0,1,0],214:["neutral",0,0,0,0,0,0],215:["neutral",0,0,0,0,0,0],
216:["negative",1,0,0,0,1,0],217:["neutral",0,0,1,0,0,0],218:["neutral",0,0,0,0,0,0],219:["negative",1,0,0,0,1,0],
220:["negative",1,0,1,0,0,0],221:["neutral",0,0,1,0,0,0],222:["negative",1,0,1,0,0,0],223:["neutral",0,0,0,0,0,0],
224:["neutral",0,0,0,0,0,0],225:["neutral",0,0,1,0,0,0],226:["negative",1,0,1,0,0,0],227:["neutral",0,0,1,0,0,0],
228:["neutral",0,0,0,0,0,0],229:["neutral",0,0,1,0,0,0],230:["neutral",0,0,0,0,0,0],231:["neutral",0,0,0,0,0,0],
232:["negative",1,0,1,0,1,0],233:["neutral",0,0,0,0,0,0],234:["negative",1,0,0,0,1,0],235:["negative",1,0,1,0,0,0],
236:["neutral",0,0,0,0,0,0],237:["neutral",0,0,1,0,0,0],238:["negative",1,0,0,0,1,0],239:["negative",1,0,0,0,1,0],
240:["negative",1,0,1,0,1,0],241:["neutral",0,0,0,0,0,0],
}

keyed = pd.read_parquet("reports/_enriched_sample_keyed.parquet").reset_index(drop=True)
keyed["k"] = list(zip(keyed.match_id, keyed.time, keyed.player_slot))
keyed["idx"] = range(len(keyed))
assert len(keyed) == len(LAB), f"{len(keyed)} rows vs {len(LAB)} labels"

lab = pd.DataFrame([[i] + LAB[i] for i in range(len(LAB))],
                   columns=["idx", "sentiment"] + [f"y_{t}" for t in TOX])
df = keyed.merge(lab, on="idx")
y = df[[f"y_{t}" for t in TOX]].astype(int).values

print("=== inline-labeled enriched probe ===  n =", len(df))
print("label positives:", dict(zip(TOX, y.sum(0).tolist())))
print("sentiment:", df.sentiment.value_counts().to_dict())

def model_pred(folder):
    inf = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"data/inference/{folder}/*.parquet"))],
                    ignore_index=True)
    inf["k"] = list(zip(inf.match_id, inf.time, inf.player_slot))
    inf = inf.drop_duplicates("k")
    m = df.merge(inf, on="k", how="left")
    pred = m[[f"pred_{t}" for t in TOX]].fillna(0).astype(int).values
    prob = m[[f"prob_{t}" for t in TOX]].fillna(0.0).values
    return pred, prob

rows = []
results = {}
for mk in ["bert", "roberta", "distilbert", "detoxify"]:
    pred, prob = model_pred(f"{mk}_toxicity")
    results[mk] = (pred, prob)
    fmi = f1_score(y, pred, average="micro", zero_division=0)
    sup_idx = [TOX.index(s) for s in SUP]
    fma = f1_score(y[:, sup_idx], pred[:, sup_idx], average="macro", zero_division=0)
    rec = recall_score(y, pred, average="micro", zero_division=0)
    prec = precision_score(y, pred, average="micro", zero_division=0)
    row = {"model": mk, "f1_micro": fmi, "f1_macro_sup": fma,
           "precision_micro": prec, "recall_micro": rec}
    for t in SUP:
        i = TOX.index(t)
        row[f"f1_{t}"] = f1_score(y[:, i], pred[:, i], zero_division=0)
    rows.append(row)

res = pd.DataFrame(rows)
res.to_csv("reports/eval_enriched_toxicity.csv", index=False)
print("\n", res.round(3).to_string(index=False))

# ---- ID-profanity blind-spot: of MY toxic positives that are ID-lexicon hits,
#       how many does Detoxify miss (prob_toxic < 0.5)? ----
det_pred, det_prob = results["detoxify"]
y_toxic = y[:, TOX.index("toxic")]
is_lex = df["lex"].astype(bool).values
det_toxic_prob = det_prob[:, TOX.index("toxic")]
pos = y_toxic == 1
lex_pos = pos & is_lex
det_caught = det_toxic_prob >= 0.5
print(f"\n=== blind-spot ===")
print(f"my toxic positives: {pos.sum()}  (lexicon-hit: {lex_pos.sum()})")
print(f"Detoxify recall on ALL toxic-pos:     {det_caught[pos].mean():.3f}")
print(f"Detoxify recall on LEXICON toxic-pos: {det_caught[lex_pos].mean():.3f}  "
      f"(miss {(~det_caught[lex_pos]).sum()}/{lex_pos.sum()})")
# split ID-profanity (lexicon hit, detox score was low at sampling) recall
id_prof = lex_pos & (df["max_toxicity_prob"].values < 0.5)
print(f"ID-profanity-style toxic-pos (lex & detox<0.5 at sample): {id_prof.sum()}, "
      f"Detoxify catches {det_caught[id_prof].sum()}")

# ---- export full labeled set for appendix ----
out = df[["idx", "year", "key", "lex", "max_toxicity_prob", "sentiment"]].copy()
for t in TOX:
    out[f"y_{t}"] = df[f"y_{t}"]
out["detox_pred_toxic"] = det_pred[:, TOX.index("toxic")]
out.to_csv("reports/enriched_gold_labeled.csv", index=False)
print("\nwrote reports/eval_enriched_toxicity.csv + reports/enriched_gold_labeled.csv")
