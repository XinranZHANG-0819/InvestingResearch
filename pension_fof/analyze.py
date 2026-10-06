"""个人养老金 Y 份额(养老目标 FOF)筛选分析。

用法: python3 analyze.py <数据目录> <输出目录>

数据目录需包含(均来自 Google Drive 选基数据包):
  T39.csv            养老 FOF 与 Y 份额名单(已去除 markdown 转义)
  T35_m.csv.gz       月度单位净值(日期为每月首个交易日)
  T35b.csv           分红与拆分事件(仅覆盖主动权益基金)
  T34.csv            主动权益基金清单(用于构建主动权益等权指数)
  bond511010.csv     国债 ETF 日线(债券基准)
  hs300.csv          沪深300 日线(价格指数,参考用)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

END = pd.Period("2026-08", "M")  # 最新完整月份(由 2026-09-01 快照得到)
RF = 0.015  # 无风险利率假设(年化)


# ---------- 1. 名单:Y 份额 -> 同一产品的净值序列 ----------
t39 = pd.read_csv(DATA / "T39.csv", dtype={"code": str})
t39["cls"] = t39.note.str.extract(r"份额类别=(\w)")
t39["base"] = t39.name.str.strip().str.replace(r"(Y|A|C)$", "", regex=True)

nav = pd.read_csv(DATA / "T35_m.csv.gz", dtype={"code": str}, low_memory=False)
nav = nav[nav.indicator == "unit_nav"].copy()
nav["value"] = pd.to_numeric(nav.value, errors="coerce")
nav["date"] = pd.to_datetime(nav.date)
nav["m"] = nav.date.dt.to_period("M")
nav = nav.dropna(subset=["value"]).sort_values("date").drop_duplicates(["code", "m"], keep="last")
navcodes = set(nav.code)

COMPANY = re.compile(
    r"^(景顺长城|景顺|国泰海通|国泰|国寿安保|国投瑞银|国富|国联|财通资管|财通|华泰柏瑞|华泰紫金|汇丰晋信|"
    r"民生加银|前海开源|中信建投|东方红|易方达|汇添富|华夏|南方|嘉实|广发|工银|华安|兴全|中欧|富国|博时|"
    r"招商|鹏华|交银|银华|天弘|万家|平安|建信|泰康|长信|农银|中银|宏利|海富通|浦银|上银|申万|华商|大成|"
    r"兴业|摩根|中加|银河|长城|安信|永赢|信澳|英大|太平|鑫元|鹏扬|东方|中泰|光大|富达|华宝)")

rows = []
for _, r in t39[t39.cls == "Y"].iterrows():
    a = t39[(t39.base == r.base) & (t39.cls == "A")].code.tolist()
    src = a[0] if a and a[0] in navcodes else (r.code if r.code in navcodes else None)
    year = re.search(r"(20\d\d)", r["name"])
    hold = re.search(r"(一|两|三|五)年", r["name"])
    comp = COMPANY.match(r["name"])
    rows.append(dict(y=r.code, name=r["name"], tier=r.value, src=src,
                     src_is_y=(src == r.code), target_year=int(year.group(1)) if year else None,
                     hold=hold.group(1) + "年" if hold else "",
                     company=comp.group(1) if comp else ""))
uni = pd.DataFrame(rows)
assert uni.src.notna().all(), uni[uni.src.isna()]


# ---------- 2. 月度总收益(含分红、拆分调整) ----------
ev = pd.read_csv(DATA / "T35b.csv", dtype={"code": str})
ev["date"] = pd.to_datetime(ev.date)


def monthly_returns(codes):
    sub = nav[nav.code.isin(codes)]
    piv = sub.pivot(index="m", columns="code", values="value").sort_index()
    dates = sub.pivot(index="m", columns="code", values="date").sort_index()
    ret = piv / piv.shift(1) - 1
    e = ev[ev.code.isin(codes)]
    for _, x in e.iterrows():
        col = dates[x.code]
        after = col[col >= x.date]
        if after.empty:
            continue
        m = after.index[0]
        prev = piv[x.code].shift(1).loc[m]
        if x.indicator == "dividend_per_unit":
            ret.loc[m, x.code] = (piv.loc[m, x.code] + float(x.value)) / prev - 1
        elif x.indicator == "split_ratio":
            k = float(str(x.value).split(":")[1])
            ret.loc[m, x.code] = piv.loc[m, x.code] * k / prev - 1
    # 快照在每月首个交易日:m 期收益实际覆盖 m-1 月,按覆盖月份标记
    ret.index = ret.index - 1
    return ret


fof_ret = monthly_returns(set(uni.src))
fof_ret = fof_ret[fof_ret.index <= END]

# 主动权益等权指数(全部主动权益基金,含已清盘,剔除极端值)
t34 = pd.read_csv(DATA / "T34.csv", dtype={"code": str}, low_memory=False)
eq_codes = set(t34.code) & navcodes
eq_ret = monthly_returns(eq_codes).clip(-0.5, 1.0)
eq_idx = eq_ret.mean(axis=1, skipna=True)
eq_idx = eq_idx[(eq_idx.index <= END)]


def daily_to_monthly(path, datecol, pxcol):
    d = pd.read_csv(path, encoding="utf-8-sig")
    d["date"] = pd.to_datetime(d[datecol])
    d["m"] = d.date.dt.to_period("M")
    first = d.sort_values("date").groupby("m")[pxcol].first()  # 每月首个交易日,与基金快照对齐
    ret = first / first.shift(1) - 1
    ret.index = ret.index - 1  # 同基金净值口径,按覆盖月份标记
    return ret


bond = daily_to_monthly(DATA / "bond511010.csv", "日期", "收盘")
hs300 = daily_to_monthly(DATA / "hs300.csv", "date", "close")
bench = pd.DataFrame({"equity": eq_idx, "bond": bond, "hs300": hs300}).dropna(subset=["equity"])
bench = bench[bench.index <= END]
bench.to_csv(OUT / "benchmarks_monthly.csv")

# FOF 无分红记录:市场未明显下跌而单月大跌 >10%,视为未记录的分红,该月剔除
eqm = bench["equity"].reindex(fof_ret.index)
bad = (fof_ret.lt(-0.10)).mul(eqm.gt(-0.02), axis=0)
for m, c in zip(*np.where(bad.values)):
    print("剔除疑似分红月:", fof_ret.columns[c], fof_ret.index[m], round(fof_ret.iat[m, c], 3))
fof_ret = fof_ret.mask(bad)


# ---------- 3. 指标 ----------
def maxdd(r):
    w = (1 + r).cumprod()
    w = pd.concat([pd.Series([1.0]), w.reset_index(drop=True)])
    return float((w / w.cummax() - 1).min())


def metrics(r, b):
    r = r.dropna()
    n = len(r)
    if n < 12:
        return {}
    ann = (1 + r).prod() ** (12 / n) - 1
    vol = r.std() * np.sqrt(12)
    dd = maxdd(r)
    x = b.loc[r.index, ["equity", "bond"]].dropna()
    y = r.loc[x.index]
    X = np.column_stack([np.ones(len(x)), x["equity"].values, x.bond.values])
    coef, res, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    fitted = X @ coef
    resid = y.values - fitted
    te = resid.std(ddof=3) * np.sqrt(12)
    alpha = (1 + coef[0]) ** 12 - 1
    t_alpha = coef[0] / (resid.std(ddof=3) / np.sqrt(len(y))) if len(y) > 3 else np.nan
    down = x["equity"] < 0
    dcap = y[down].mean() / x["equity"][down].mean() if down.sum() >= 3 else np.nan
    up = x["equity"] > 0
    ucap = y[up].mean() / x["equity"][up].mean() if up.sum() >= 3 else np.nan
    return dict(n=n, ann_ret=ann, vol=vol, maxdd=dd,
                sharpe=(ann - RF) / vol if vol > 0 else np.nan,
                calmar=ann / abs(dd) if dd < 0 else np.nan,
                beta_eq=coef[1], beta_bond=coef[2], alpha=alpha, t_alpha=t_alpha,
                info_ratio=(alpha / te) if te > 0 else np.nan,
                up_cap=ucap, down_cap=dcap)


windows = {"1y": 12, "3y": 36, "5y": 60}
out = []
for _, u in uni.iterrows():
    r = fof_ret[u.src].dropna()
    rec = dict(y=u.y, name=u["name"], tier=u.tier, src=u.src, src_is_y=u.src_is_y,
               target_year=u.target_year, hold=u.hold, company=u.company,
               first_month=str(r.index.min()) if len(r) else None, months=len(r))
    for k, w in windows.items():
        start = END - w
        rr = r[(r.index > start) & (r.index <= END)]
        # 要求窗口起点前已成立;允许 1 个月因疑似分红被剔除
        if r.index.min() <= start + 1 and len(rr) >= w - 1:
            for mk, mv in metrics(rr, bench).items():
                rec[f"{k}_{mk}"] = mv
    if len(r) >= 12:
        for mk, mv in metrics(r, bench).items():
            rec[f"all_{mk}"] = mv
    # 自然年收益
    for yr in range(2019, 2027):
        rr = r[(r.index.year == yr)]
        if len(rr) == 12 or (yr == 2026 and len(rr) == END.month):
            rec[f"y{yr}"] = (1 + rr).prod() - 1
    out.append(rec)
res = pd.DataFrame(out)

# 基准自然年收益
bench_years = {}
for col in ["equity", "bond", "hs300"]:
    s = bench[col].dropna()
    bench_years[col] = {yr: (1 + s[s.index.year == yr]).prod() - 1 for yr in range(2019, 2027)}
pd.DataFrame(bench_years).to_csv(OUT / "benchmarks_yearly.csv")

res.to_csv(OUT / "fof_metrics.csv", index=False, encoding="utf-8-sig")
fof_ret.to_csv(OUT / "fof_monthly_returns.csv")
print("universe", len(res), "with 3y", res["3y_n"].notna().sum(), "with 5y", res["5y_n"].notna().sum())
