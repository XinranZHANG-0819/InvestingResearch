"""Multi-factor regression used as a classification method (not as a skill measure).

For each fund-year (January snapshots 2010-2023), regress the last 36 monthly returns on the old
report's factors: market, 中证1000-沪深300, 中证800价值-成长, 5 industry groups minus market.
Category from the loadings:
  largest industry loading >= TH  -> that industry (科技/医药/消费/金融地产/周期)
  else 大小盘 loading >= TH_S      -> 小盘;  <= -TH_S -> 中大盘;  else 全部
Then everything else as the report: stability t vs that category's index (新锐 3y, 资深 5y),
ranked within (year, category, tier). Compare with tracking-error classification.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import load  # noqa: E402
from metrics import WORK, load_panel  # noqa: E402

RF_M = 0.015 / 12
IND = ['科技', '医药', '消费', '金融地产', '周期']
P = load_panel()
R, IDX = P['R'], P['IDX']
b = load('T02')
b['code'] = b.code.str.replace('.CSI', '', regex=False)
s = load('T33')
s = s[(s.indicator == 'close') & s.code.isin(['H30357', 'H30358'])][['date', 'code', 'value']].rename(columns={'value': 'close'})
px = pd.concat([b[['date', 'code', 'close']], s])
px['close'] = pd.to_numeric(px.close, errors='coerce')
px['date'] = pd.to_datetime(px.date)
px = px[px.close > 0].pivot_table(index='date', columns='code', values='close').sort_index()
ft = px.groupby(px.index.to_period('M')).head(1)
ft.index = ft.index.to_period('M')
fr = ft.pct_change()
FAC = pd.DataFrame({'mkt': IDX['全部'] - RF_M, 'smb': fr['H00852'] - fr['H00300'], 'hml': fr['H30358'] - fr['H30357'],
                    **{f'ind_{c}': IDX[c] - IDX['全部'] for c in IND}}).dropna()


def loadings(y):
    d = pd.concat([y.rename('y') - RF_M, FAC], axis=1, join='inner').dropna()
    if len(d) < 30:
        return None
    X = np.column_stack([np.ones(len(d)), d[FAC.columns].values])
    beta, *_ = np.linalg.lstsq(X, d.y.values, rcond=None)
    return dict(zip(FAC.columns, beta[1:]))


H = pd.read_pickle(os.path.join(WORK, 'snap_hist.pkl'))
H['year'] = H['T'].str[:4].astype(int)
L = []
for r in H[['code', 'T']].itertuples():
    iT = R.index.get_loc(pd.Period(r.T, 'M'))
    L.append(loadings(R[r.code].iloc[iT - 35:iT + 1]) or {})
L = pd.DataFrame(L, index=H.index)
H = pd.concat([H, L], axis=1)


def classify(row, th, th_s):
    ind = {c: row[f'ind_{c}'] for c in IND}
    c = max(ind, key=ind.get)
    if ind[c] >= th:
        return c
    if row.smb >= th_s:
        return '小盘'
    if row.smb <= -th_s:
        return '中大盘'
    return '全部'


def excess_t(code, T, cat, n):
    iT = R.index.get_loc(pd.Period(T, 'M'))
    rr = R[code].iloc[iT - n + 1:iT + 1]
    bb = IDX[cat].reindex(rr.index)
    m = rr.notna() & bb.notna()
    if m.sum() < n * 0.9:
        return np.nan
    ex = (rr[m] - bb[m]).values
    return ex.mean() / ex.std(ddof=1) * np.sqrt(len(ex))


def spread(sig, cell, out):
    n = H.groupby(cell)[sig].transform('count')
    p = H[sig].groupby(H[cell]).rank(pct=True).where(n >= 5)
    d = pd.DataFrame({'p': p, 'y': H.year, 'o': H[out]}).dropna()
    sp = (d[d.p > 0.8].groupby('y').o.mean() - d[d.p <= 0.2].groupby('y').o.mean()).dropna()
    return f'{sp.mean() * 100:+.2f}（{(sp > 0).sum()}/{len(sp)}）'


H['sig_te'] = np.where((H.tier == '资深') & H.t_5y.notna(), H.t_5y, H.t_3y)
H['cell_te'] = H.year.astype(str) + '|' + H.cat + '|' + H.tier
print('TE classification:', spread('sig_te', 'cell_te', 'fwd_ret_36'), '| 5y', spread('sig_te', 'cell_te', 'fwd_ret_60'))
res = {}
ok = H.smb.notna()
for th, th_s in [(0.2, 0.2), (0.3, 0.3), (0.4, 0.3), (0.5, 0.5)]:
    cat = H[ok].apply(classify, axis=1, args=(th, th_s)).reindex(H.index)
    k = f'mf_{th}_{th_s}'
    H[f'cat_{k}'] = cat
    t3 = [excess_t(c, T, ct, 36) if isinstance(ct, str) else np.nan for c, T, ct in zip(H.code, H['T'], cat)]
    t5 = [excess_t(c, T, ct, 60) if isinstance(ct, str) and tr == '资深' and pd.notna(t5o) else np.nan
          for c, T, ct, tr, t5o in zip(H.code, H['T'], cat, H.tier, H.t_5y)]
    H[f'sig_{k}'] = np.where(pd.notna(t5), t5, t3)
    H[f'cell_{k}'] = H.year.astype(str) + '|' + cat.fillna('na') + '|' + H.tier
    agree = (cat == H.cat).mean()
    print(f'\nthreshold industry {th} size {th_s}: agree with TE {agree:.0%}')
    print('  3y', spread(f'sig_{k}', f'cell_{k}', 'fwd_ret_36'), '| 5y', spread(f'sig_{k}', f'cell_{k}', 'fwd_ret_60'))
    print('  counts', cat.value_counts().to_dict())
    print('  TE counts', H.cat.value_counts().to_dict())
pd.to_pickle(H, os.path.join(WORK, 'mf_classify.pkl'))
