"""Side checks: TE vs R² vs no classification; top/bottom group vs category average; size by year."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from metrics import WORK
A = pd.read_pickle(os.path.join(WORK, 'analysis.pkl')); H = A['H']
def tw(base):
    v5, v3 = H.get(f'{base}_5y'), H.get(f'{base}_3y')
    return np.where((H.tier == '资深') & v5.notna(), v5, v3)
for b in ['tr2', 'winr2', 'winall']:
    H['sig_' + b] = tw(b)
H['cell_r2'] = H.year.astype(str) + '|' + H.cat_r2 + '|' + H.tier
H['grpall'] = H.year.astype(str) + '|' + H.tier
def pct(sig, cell):
    n = H.groupby(cell).code.transform('size')
    return H[sig].groupby(H[cell]).rank(pct=True).where(n >= 5)
def spread(p, out):
    d = pd.DataFrame({'p': p, 'y': H.year, 'o': H[out]}).dropna()
    top = d[d.p > 0.8].groupby('y').o.mean(); bot = d[d.p <= 0.2].groupby('y').o.mean(); sp = (top - bot).dropna()
    return f'{sp.mean()*100:+.2f} ({(sp>0).sum()}/{len(sp)})'
print('classification comparison (top20-bottom20, fwd 3y | 5y)')
for name, sig, cell in [('TE·稳定程度', 'sig_t', 'cell'), ('R²·稳定程度', 'sig_tr2', 'cell_r2'), ('不分类·稳定程度', 'sig_tall', 'grpall'),
                        ('TE·月度胜率', 'sig_win', 'cell'), ('R²·月度胜率', 'sig_winr2', 'cell_r2'), ('不分类·月度胜率', 'sig_winall', 'grpall')]:
    p = pct(sig, cell); print(name, spread(p, 'fwd_ret_36'), '|', spread(p, 'fwd_ret_60'))
print('top/bottom 20% relative to category-cell average')
for sig in ['sig_t', 'sig_win', 'sig_ex']:
    p = pct(sig, 'cell')
    for out in ['rel_ret_36', 'rel_ret_60']:
        d = pd.DataFrame({'p': p, 'y': H.year, 'o': H[out]}).dropna()
        top = d[d.p > 0.8].groupby('y').o.mean(); bot = d[d.p <= 0.2].groupby('y').o.mean()
        print(sig, out, f'top {top.mean()*100:+.2f} ({(top>0).sum()}/{len(top)})  bottom {bot.mean()*100:+.2f} ({(bot<0).sum()}/{len(bot)})')
H['size_pct_cat'] = H.groupby(['year', 'cat'])['size'].rank(pct=True)
d = H.dropna(subset=['rel_ret_36'])
g = d.groupby(['year', d.size_pct_cat > 0.8]).rel_ret_36.mean().unstack(); diff = (g[True] - g[False]) * 100
print('size top20% minus rest: mean', round(diff.mean(), 2), 'negative years', (diff < 0).sum(), '/', diff.notna().sum())
g5 = H.dropna(subset=['rel_ret_60']).groupby(['year', H.size_pct_cat > 0.8]).rel_ret_60.mean().unstack(); diff5 = (g5[True] - g5[False]) * 100
print('size 5y: mean', round(diff5.mean(), 2), 'negative', (diff5 < 0).sum(), '/', diff5.notna().sum())
print('size thresholds (median across categories of 80th pct, 亿):', H.groupby('year').apply(lambda x: x.groupby('cat')['size'].quantile(0.8).median(), include_groups=False).round(0).to_dict())
for c in ['全部', '中大盘', '小盘', '科技', '医药', '消费', '周期']:
    gg = H[H.cat == c]
    print(c, 'top20 minus rest', round((gg[gg.size_pct_cat > 0.8].rel_ret_36.mean() - gg[gg.size_pct_cat <= 0.8].rel_ret_36.mean()) * 100, 2), 'n_top', (gg.size_pct_cat > 0.8).sum())
s = H[H.surge] if 'surge' in H else H[(H['size'] / H.size_1y >= 3) & (H['size'] >= 10)]
sv = s.groupby('year').rel_ret_36.mean() * 100
print('surge by year', sv.round(1).to_dict(), 'neg years', (sv < 0).sum(), '/', len(sv))
