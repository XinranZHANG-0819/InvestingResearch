"""Compute the current snapshot and yearly historical snapshots with forward outcomes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from metrics import load_panel, snapshot, WORK

P = load_panel()
cur = snapshot(P, '2026-09')
cur.to_pickle(os.path.join(WORK, 'snap_current.pkl'))
print('current', cur.shape, cur.cat.value_counts().to_dict())
hist = []
for y in range(2010, 2024):
    s = snapshot(P, f'{y}-01', fwd=(36, 60))
    hist.append(s)
    print(y, len(s), s.tier.value_counts().to_dict(), flush=True)
H = pd.concat(hist, ignore_index=True)
H.to_pickle(os.path.join(WORK, 'snap_hist.pkl'))
print('hist', H.shape)
