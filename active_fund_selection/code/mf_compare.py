"""Compare the category-index method with the old report's multi-factor method.

Steps change one thing at a time (forward 3y return, top 20% minus bottom 20%, 2010-2023):
  A  fund level, category-index stability t, ranked within category x tier (current method)
  B  fund level, multi-factor alpha t (36m), ranked within category x tier
  C  fund level, multi-factor alpha t (36m), ranked within tier only (old-style ranking)
  D  manager level (equal-weight curve of all funds managed), multi-factor alpha t, ranked
     within tier (资深 career >= 5y, 新锐 3-5y), forward return of a representative fund
  E  manager level, category-index stability t of the manager curve, ranked within category x tier
Multi-factor model (monthly, 36 months): market excess, 中证1000-沪深300, 中证800价值-成长,
5 industry groups minus market. Weekly data are unavailable, so frequency cannot be compared.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import load  # noqa: E402
from metrics import CATS, MIN_SIZE, MIN_STOCK, WORK, load_panel  # noqa: E402

RF_M = 0.015 / 12
IND = ['科技', '医药', '消费', '金融地产', '周期']
P = load_panel()
R, IDX, F, stints, size = P['R'], P['IDX'], P['funds'], P['stints'], P['size']

# ---------- factor returns on the same month keys ----------
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
FAC = pd.DataFrame({
    'mkt': IDX['全部'] - RF_M,
    'smb': fr['H00852'] - fr['H00300'],
    'hml': fr['H30358'] - fr['H30357'],
    **{f'ind_{c}': IDX[c] - IDX['全部'] for c in IND},
}).dropna()


def alpha_t(y):
    """y: monthly fund returns (Series on month keys). Returns (alpha_monthly, t) or (nan, nan)."""
    d = pd.concat([y.rename('y') - RF_M, FAC], axis=1, join='inner').dropna()
    if len(d) < 30:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(d)), d[FAC.columns].values])
    yy = d.y.values
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    dof = len(yy) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta[0], beta[0] / np.sqrt(cov[0, 0])


def spread(df, sig, cell, out='fwd_ret_36', min_cell=5):
    n = df.groupby(cell)[sig].transform('count')
    p = df[sig].groupby(df[cell]).rank(pct=True).where(n >= min_cell)
    d = pd.DataFrame({'p': p, 'y': df.year, 'o': df[out]}).dropna()
    top = d[d.p > 0.8].groupby('y').o.mean()
    bot = d[d.p <= 0.2].groupby('y').o.mean()
    sp = (top - bot).dropna()
    return sp.mean() * 100, int((sp > 0).sum()), len(sp), int((d.p > 0.8).sum())


# ---------- A-C: fund level ----------
H = pd.read_pickle(os.path.join(WORK, 'snap_hist.pkl'))
H['year'] = H['T'].str[:4].astype(int)
mf = []
for (code, T), _ in H.groupby(['code', 'T']):
    iT = R.index.get_loc(pd.Period(T, 'M'))
    a, t = alpha_t(R[code].iloc[iT - 35:iT + 1])
    mf.append((code, T, a, t))
H = H.merge(pd.DataFrame(mf, columns=['code', 'T', 'mf_alpha', 'mf_t']), on=['code', 'T'])
H['cell'] = H.year.astype(str) + '|' + H.cat + '|' + H.tier
H['tiercell'] = H.year.astype(str) + '|' + H.tier
res = {}
res['A 基金·类别指数·类内(稳定程度36月)'] = spread(H, 't_3y', 'cell')
res['A2 基金·类别指数·类内(本报告窗口)'] = spread(H.assign(sig=np.where((H.tier == '资深') & H.t_5y.notna(), H.t_5y, H.t_3y)), 'sig', 'cell')
res['B 基金·多因子·类内'] = spread(H, 'mf_t', 'cell')
res['C 基金·多因子·不分类'] = spread(H, 'mf_t', 'tiercell')
res['C2 基金·类别指数·不分类'] = spread(H, 't_3y', 'tiercell')

# ---------- D-E: manager level ----------
months = R.index
st = stints.copy()
st['m0'] = st.start.dt.to_period('M') + 2                     # first fully managed return month
st['m1'] = st.end.fillna(pd.Timestamp('2100-01-01')).dt.to_period('M')
st['m1'] = st.m1.where(st.m1 <= months[-1], months[-1])
rows = []
for r in st.itertuples():
    if r.m0 > r.m1:
        continue
    rng = pd.period_range(r.m0, r.m1, freq='M')
    rows.append(pd.DataFrame({'mgr_id': r.mgr_id, 'code': r.code, 'm': rng}))
MM = pd.concat(rows, ignore_index=True)
Rl = R.stack().rename('r').reset_index()
Rl.columns = ['m', 'code', 'r']
MM = MM.merge(Rl, on=['m', 'code'])
MC = MM.groupby(['mgr_id', 'm']).r.mean().unstack('mgr_id').sort_index()   # manager curves
career0 = stints.groupby('mgr_id').start.min()
mgr_name = stints.sort_values('start').groupby('mgr_id').mgr.last()


def rep_fund(mid, d, iT):
    """Representative fund at date d: alive, managed >= 1y, size >= 2亿, stock >= 60%, longest tenure."""
    g = ST_BY_MGR.get(mid)
    if g is None:
        return None
    cur = g[(g.start <= d - pd.DateOffset(years=1)) & (g.end.isna() | (g.end >= d))]
    best = None
    lag = d - pd.Timedelta(days=30)
    for r in cur.sort_values('start').itertuples():
        if r.code not in R.columns or pd.isna(R[r.code].iloc[iT]):
            continue
        sz = SIZE_BY_CODE.get(r.code)
        if sz is None:
            continue
        sz = sz[sz.date <= lag]
        last = sz.net_asset.dropna()
        if not len(last) or last.iloc[-1] < MIN_SIZE:
            continue
        srw = sz[sz.date > lag - pd.DateOffset(years=3)].stock_ratio.dropna()
        sr_v = srw.mean() if len(srw) else sz.stock_ratio.dropna().iloc[-4:].mean() if sz.stock_ratio.notna().any() else np.nan
        if pd.notna(sr_v) and sr_v < MIN_STOCK:
            continue
        best = r.code
        break
    return best


mrows = []
ST_BY_MGR = {m: g for m, g in stints.groupby('mgr_id')}
SIZE_BY_CODE = {c: g.sort_values('date') for c, g in size.groupby('code')}
for y in range(2010, 2024):
    T = pd.Period(f'{y}-01', 'M')
    d = T.start_time
    iT = months.get_loc(T)
    win = MC.iloc[iT - 35:iT + 1]
    ok = win.notna().sum() >= 30
    for mid in win.columns[ok]:
        car = (d - career0.get(mid, d)).days / 365.25
        if car < 3:
            continue
        code = rep_fund(mid, d, iT)
        if code is None:
            continue
        y_ = win[mid].dropna()
        a, t = alpha_t(y_)
        bb = IDX.reindex(y_.index)
        okb = bb.notna().all(axis=1)
        te = {c: (y_[okb] - bb.loc[okb, c]).std() for c in CATS}
        cat = min(te, key=te.get)
        ex = (y_[okb] - bb.loc[okb, cat]).values
        tc = ex.mean() / ex.std(ddof=1) * np.sqrt(len(ex))
        fwd = R[code].iloc[iT + 1:iT + 37]
        if len(fwd) < 36:
            continue
        fwd = fwd.fillna(IDX['全部'].reindex(fwd.index))
        mrows.append(dict(mgr_id=mid, year=y, tier='资深' if car >= 5 else '新锐', code=code, mf_t=t, cat=cat, cat_t=tc,
                          fwd_ret_36=np.prod(1 + fwd.values) ** (12 / 36) - 1))
    print('managers', y, sum(1 for r in mrows if r['year'] == y), flush=True)
MG = pd.DataFrame(mrows)
MG['tiercell'] = MG.year.astype(str) + '|' + MG.tier
MG['cell'] = MG.year.astype(str) + '|' + MG.cat + '|' + MG.tier
res['D 经理·多因子·不分类(原报告方式)'] = spread(MG, 'mf_t', 'tiercell')
res['E 经理·类别指数·类内'] = spread(MG, 'cat_t', 'cell')
res['E2 经理·类别指数·不分类'] = spread(MG, 'cat_t', 'tiercell')
for tier in ['资深', '新锐']:
    res[f'D-{tier}'] = spread(MG[MG.tier == tier], 'mf_t', 'tiercell')
    res[f'E-{tier}'] = spread(MG[MG.tier == tier], 'cat_t', 'cell')
    res[f'A-{tier}'] = spread(H[H.tier == tier], 't_3y', 'cell')
    res[f'C-{tier}'] = spread(H[H.tier == tier], 'mf_t', 'tiercell')

print('\n== forward 3y annualized return: top20% minus bottom20% (pp), positive years, n_top ==')
for k, (sp, pos, n, nt) in res.items():
    print(f'{k:32s} {sp:+.2f}  {pos}/{n}  n_top={nt}')
print('managers per year:', MG.groupby('year').size().to_dict())
pd.to_pickle(dict(H=H, MG=MG, res=res, FAC=FAC, MC=MC, career0=career0), os.path.join(WORK, 'mf_compare.pkl'))
