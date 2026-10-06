"""Per-fund snapshot at a month key T: eligibility, category, metrics, optional forward outcomes.

Month key m = month of the first trading day whose NAV closes the return; the return labelled m
covers the previous calendar month. A snapshot "at T" uses returns labelled <= T only.
"""
import os
import pickle

import numpy as np
import pandas as pd

WORK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work')
RF = 0.015          # annual risk-free rate for Sharpe (descriptive only)
MIN_SIZE = 2.0      # 亿元
MIN_STOCK = 60.0    # % average stock ratio over the last 3 years
CATS = ['全部', '中大盘', '小盘', '科技', '医药', '消费', '金融地产', '周期']


def load_panel():
    P = pickle.load(open(os.path.join(WORK, 'panel.pkl'), 'rb'))
    R = P['R'].copy()
    med = R.median(axis=1)
    bad = R.sub(med, axis=0).abs() > 0.6          # data errors (see notes)
    P['n_bad'] = int(bad.values.sum())
    P['R'] = R.mask(bad)
    return P


def _ann(r):
    return np.prod(1 + r) ** (12 / len(r)) - 1


def _dd(r):
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return nav / peak - 1


def window_stats(f, b):
    """f, b: aligned monthly returns of fund and its category index (no NaN)."""
    n = len(f)
    ex = f - b
    ar, br = _ann(f), _ann(b)
    vol = f.std(ddof=1) * np.sqrt(12)
    dd = _dd(f)
    mdd = dd.min()
    ulcer = np.sqrt(np.mean(dd ** 2))
    sd_ex = ex.std(ddof=1)
    down, up = b < 0, b > 0
    var_b = b.var(ddof=1)
    return dict(
        ret=ar, ex=ar - br, t=ex.mean() / sd_ex * np.sqrt(n) if sd_ex > 0 else np.nan,
        win=(ex > 0).mean(), vol=vol, mdd=mdd, ulcer=ulcer,
        martin=ar / ulcer if ulcer > 0 else np.nan,
        calmar=ar / -mdd if mdd < 0 else np.nan,
        sharpe=(ar - RF) / vol if vol > 0 else np.nan,
        beta=np.cov(f, b, ddof=1)[0, 1] / var_b if var_b > 0 else np.nan,
        dcap=f[down].mean() / b[down].mean() if down.sum() >= 3 else np.nan,
        ucap=f[up].mean() / b[up].mean() if up.sum() >= 3 else np.nan,
        n=n,
    )


def snapshot(P, T, fwd=(), require_size=True):
    """Eligible funds at month key T with category and metrics for 36m, 60m and tenure windows.

    fwd: tuple of horizons in months (e.g. (36, 60)) for forward outcomes.
    """
    R, IDX, F, stints, size = P['R'], P['IDX'], P['funds'], P['stints'], P['size']
    T = pd.Period(T, 'M')
    d = T.start_time                                   # ~ first trading day of month T
    keys = R.index
    if T not in keys:
        raise ValueError(T)
    iT = keys.get_loc(T)
    alive = R.loc[T].notna() | R.iloc[max(iT - 1, 0)].notna()

    cur = stints[(stints.start <= d) & (stints.end.isna() | (stints.end >= d))]
    ten_start = cur.groupby('code').start.min()
    mgrs = cur.groupby('code').apply(lambda g: '、'.join(g.sort_values('start').mgr), include_groups=False)
    mgr_ids = cur.groupby('code').mgr_id.apply(frozenset)

    lag = d - pd.Timedelta(days=30)                    # quarterly data published ~1 month later
    sz = size[size.date <= lag]
    last_size = sz.dropna(subset=['net_asset']).sort_values('date').groupby('code').net_asset.last()
    sz1 = size[size.date <= lag - pd.DateOffset(years=1)]
    size_1y = sz1.dropna(subset=['net_asset']).sort_values('date').groupby('code').net_asset.last()
    sr = sz[sz.date > lag - pd.DateOffset(years=3)].groupby('code').stock_ratio.mean()
    # stale coverage: no stock ratio in the last 3 years -> average of the last 4 reported quarters
    sr_last4 = (sz.dropna(subset=['stock_ratio']).sort_values('date').groupby('code')
                .stock_ratio.apply(lambda x: x.iloc[-4:].mean()))
    size_date = sz.dropna(subset=['net_asset']).groupby('code').date.max()

    rows = []
    for code in alive.index[alive]:
        if code not in ten_start.index:
            continue
        ts = ten_start[code]
        tenure = (d - ts).days / 365.25
        if tenure < 3:
            continue
        s_now = last_size.get(code, np.nan)
        if require_size and not (s_now >= MIN_SIZE):
            continue
        s_ratio = sr.get(code, np.nan)
        if pd.isna(s_ratio):
            s_ratio = sr_last4.get(code, np.nan)
        if pd.notna(s_ratio) and s_ratio < MIN_STOCK:
            continue
        r36 = R[code].iloc[iT - 35:iT + 1]
        if r36.isna().sum() > 2:
            continue
        b36 = IDX.reindex(r36.index)
        ok = r36.notna() & b36.notna().all(axis=1)
        f = r36[ok].values
        # category = index whose gap with the fund is least volatile (min tracking error);
        # R² is kept for reference but ignores scale, so it pulls funds toward volatile indices
        te = {c: (f - b36.loc[ok, c].values).std(ddof=1) * np.sqrt(12) for c in CATS}
        r2 = {c: np.corrcoef(f, b36.loc[ok, c].values)[0, 1] ** 2 for c in CATS}
        cat = min(te, key=te.get)
        row = dict(code=code, T=str(T), tenure=tenure, tier='资深' if tenure >= 5 else '新锐',
                   ten_start=ts.date(), mgr=mgrs.get(code, ''), mgr_ids=mgr_ids.get(code, frozenset()),
                   size=s_now, size_1y=size_1y.get(code, np.nan), stock_ratio=s_ratio,
                   size_stale=bool(size_date.get(code, d) < d - pd.DateOffset(years=2)),
                   cat=cat, te=te[cat], cat_r2=max(r2, key=r2.get),
                   **{f'te_{c}': v for c, v in te.items()}, **{f'r2_{c}': v for c, v in r2.items()})
        # baselines: everyone vs the all-market index; and category chosen by R² instead of TE
        for tag, bcat in (('all', '全部'), ('r2', max(r2, key=r2.get))):
            for w, n in (('3y', 36), ('5y', 60)):
                if w == '5y' and tenure < 5:
                    continue
                rr = R[code].iloc[iT - n + 1:iT + 1]
                bb = IDX[bcat].reindex(rr.index)
                m = rr.notna() & bb.notna()
                if m.sum() >= n * 0.9:
                    ex = (rr[m] - bb[m]).values
                    row[f't{tag}_{w}'] = ex.mean() / ex.std(ddof=1) * np.sqrt(len(ex))
                    row[f'win{tag}_{w}'] = (ex > 0).mean()
                    row[f'ex{tag}_{w}'] = _ann(rr[m].values) - _ann(bb[m].values)
        # windows: 36m, 60m (only inside tenure), tenure
        first_full = pd.Period(ts, 'M') + 2
        n_ten = iT - keys.get_loc(first_full) + 1 if first_full in keys else 0
        for w, n in (('3y', 36), ('5y', 60), ('ten', n_ten)):
            if n < 36 or (w == '5y' and tenure < 5) or iT - n + 1 < 0:
                continue
            rr = R[code].iloc[iT - n + 1:iT + 1]
            bb = IDX[cat].reindex(rr.index)
            m = rr.notna() & bb.notna()
            if m.sum() < n * 0.9:
                continue
            for k, v in window_stats(rr[m].values, bb[m].values).items():
                row[f'{k}_{w}'] = v
        for h in fwd:
            if iT + h >= len(keys):
                continue
            rr = R[code].iloc[iT + 1:iT + h + 1]
            bb = IDX[cat].reindex(rr.index)
            filled = rr.fillna(bb)                     # after liquidation: hold the category index
            if filled.isna().any():
                continue
            row[f'fwd_ret_{h}'] = _ann(filled.values)
            row[f'fwd_ex_{h}'] = _ann(filled.values) - _ann(bb.values)
            row[f'fwd_mdd_{h}'] = _dd(filled.values).min()
            row[f'fwd_alive_{h}'] = rr.notna().mean()
        if pd.isna(s_ratio) and not (row.get('beta_3y', 1.0) >= 0.5):
            continue
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.join(F[['name', 'type', 'company', 'status', 'inception']], on='code')
        # some share classes are listed as separate main codes in T34: same fund name (minus the
        # class letter), same size and same managers -> keep the oldest code
        norm = (out.name.str.replace('证券投资基金', '', regex=False).str.replace('型', '', regex=False)
                .str.replace(r'[（(]LOF[)）]', '', regex=True).str.replace(r'(A/B|[ABCE])$', '', regex=True))
        out = (out.assign(_k=norm + '|' + out['size'].round(2).astype(str) + '|' + out.mgr)
               .sort_values(['ten_start', 'code'])
               .drop_duplicates('_k').drop(columns='_k').reset_index(drop=True))
    return out
