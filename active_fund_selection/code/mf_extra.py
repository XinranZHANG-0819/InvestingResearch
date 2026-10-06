"""Extra cuts for the multi-factor comparison and the holdings under each method."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from metrics import WORK, load_panel
import mf_compare as M   # re-runs the comparison (cheap) and exposes alpha_t, spread, data
H, MG = M.H, M.MG
H['sig_win'] = np.where((H.tier == '资深') & H.win_5y.notna(), H.win_5y, H.win_3y)
H['sig_t'] = np.where((H.tier == '资深') & H.t_5y.notna(), H.t_5y, H.t_3y)
print('\n== extra cuts (top20-bottom20, pp) ==')
for name, sig, cell in [('A2 类别指数t·类内', 'sig_t', 'cell'), ('C2 类别指数t·不分类', 'sig_t', 'tiercell'),
                        ('胜率·类内', 'sig_win', 'cell'), ('胜率·不分类', 'sig_win', 'tiercell'),
                        ('B 多因子·类内', 'mf_t', 'cell'), ('C 多因子·不分类', 'mf_t', 'tiercell')]:
    a = M.spread(H, sig, cell); b5 = M.spread(H, sig, cell, out='fwd_ret_60')
    print(f'{name:22s} 3y {a[0]:+.2f} {a[1]}/{a[2]} | 5y {b5[0]:+.2f} {b5[1]}/{b5[2]}')
# top group vs year average and bottom
for name, sig, cell in [('C 多因子·不分类', 'mf_t', 'tiercell'), ('C2 类别指数t·不分类', 'sig_t', 'tiercell')]:
    n = H.groupby(cell)[sig].transform('count'); p = H[sig].groupby(H[cell]).rank(pct=True).where(n >= 5)
    rel = H.fwd_ret_36 - H.groupby('year').fwd_ret_36.transform('mean')
    d = pd.DataFrame({'p': p, 'y': H.year, 'o': rel}).dropna()
    top = d[d.p > 0.8].groupby('y').o.mean(); bot = d[d.p <= 0.2].groupby('y').o.mean()
    print(name, 'top vs all avg', f'{top.mean()*100:+.2f} ({(top>0).sum()}/{len(top)})', 'bottom', f'{bot.mean()*100:+.2f} ({(bot<0).sum()}/{len(bot)})')
# category concentration of pooled top 20% by multi-factor
n = H.groupby('tiercell').mf_t.transform('count'); p = H.mf_t.groupby(H.tiercell).rank(pct=True).where(n >= 5)
print('pooled MF top20% category mix:', H[p > 0.8].cat.value_counts(normalize=True).round(2).to_dict())
print('all funds category mix:     ', H.cat.value_counts(normalize=True).round(2).to_dict())
# ---- current snapshot: holdings under each method ----
C = pd.read_pickle(os.path.join(WORK, 'current_enriched.pkl'))
R = M.R; iT = R.index.get_loc(pd.Period('2026-09', 'M'))
C['mf_t'] = [M.alpha_t(R[c].iloc[iT - 35:iT + 1])[1] for c in C.code]
C['mf_alpha'] = [M.alpha_t(R[c].iloc[iT - 35:iT + 1])[0] * 12 for c in C.code]
C['rank_mf_pool'] = C.groupby('tier').mf_t.rank(ascending=False, method='first')
C['n_tier'] = C.groupby('tier').code.transform('size')
C['rank_t_pool'] = C.groupby('tier').t.rank(ascending=False, method='first')
HOLD = ['110023', '002910', '007449', '004206', '001048', '001480', '005851', '006567', '001678']
cols = ['code', 'name', 'cat', 'tier', 'rank_t', 'n_cell', 't', 'rank_t_pool', 'mf_t', 'mf_alpha', 'rank_mf_pool', 'n_tier']
print(C[C.code.isin(HOLD)][cols].round(2).to_string())
C.to_pickle(os.path.join(WORK, 'current_mf.pkl'))
print(C[['t', 'mf_t']].corr().round(2).to_string())
print('rank corr within tier:', C.groupby('tier').apply(lambda g: g.t.rank().corr(g.mf_t.rank()), include_groups=False).round(2).to_dict())
