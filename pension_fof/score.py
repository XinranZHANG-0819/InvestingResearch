"""在偏股养老 FOF 中综合打分。

用法: python3 score.py <analyze.py 的输出目录>

偏股定义:近 3 年对主动权益等权指数的 beta >= 0.6。
打分(组内 z 分数加权):
  3 年 alpha 30% · 3 年信息比率 15% · 3 年夏普 20% · 3 年最大回撤 15% · 逐年 alpha 稳定性 20%
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1])
res = pd.read_csv(OUT / "fof_metrics.csv", dtype={"y": str, "src": str})
ret = pd.read_csv(OUT / "fof_monthly_returns.csv", index_col=0)
ret.index = pd.PeriodIndex(ret.index, freq="M")
bench = pd.read_csv(OUT / "benchmarks_monthly.csv", index_col=0)
bench.index = pd.PeriodIndex(bench.index, freq="M")

BETA_MIN = 0.6
pool = res[res["3y_beta_eq"] >= BETA_MIN].copy()


def yearly_alpha(src, b_eq, b_bd):
    """用 3 年回归的 beta,算每个自然年相对同 beta 股债组合的超额收益。"""
    r = ret[src].dropna()
    out = {}
    for yr in (2023, 2024, 2025, 2026):
        rr = r[r.index.year == yr]
        bb = bench.loc[rr.index]
        if len(rr) < 6:
            continue
        port = (1 + rr).prod() - 1
        ref = (1 + b_eq * bb["equity"] + b_bd * bb["bond"] + (1 - b_eq - b_bd) * 0.0).prod() - 1
        out[yr] = port - ref
    return out


ya = pool.apply(lambda u: yearly_alpha(u.src, u["3y_beta_eq"], u["3y_beta_bond"]), axis=1)
for yr in (2023, 2024, 2025, 2026):
    pool[f"xa{yr}"] = ya.map(lambda d: d.get(yr, np.nan))
xa = pool[[f"xa{y}" for y in (2023, 2024, 2025, 2026)]]
pool["xa_pos_share"] = (xa > 0).sum(axis=1) / xa.notna().sum(axis=1)
pool["xa_worst"] = xa.min(axis=1)


def z(s):
    return (s - s.mean()) / s.std()


pool["score"] = (0.30 * z(pool["3y_alpha"]) + 0.15 * z(pool["3y_info_ratio"])
                 + 0.20 * z(pool["3y_sharpe"]) + 0.15 * z(pool["3y_maxdd"])
                 + 0.10 * z(pool["xa_pos_share"]) + 0.10 * z(pool["xa_worst"]))
pool = pool.sort_values("score", ascending=False)
pool["rank"] = range(1, len(pool) + 1)
pool.to_csv(OUT / "equity_pool_scored.csv", index=False, encoding="utf-8-sig")

# 不足 3 年但偏股的新产品(只做观察,不参与排名)
young = res[res["3y_beta_eq"].isna() & (res["all_beta_eq"] >= BETA_MIN) & (res["all_n"] >= 18)]
young.sort_values("all_alpha", ascending=False).to_csv(OUT / "equity_young.csv", index=False, encoding="utf-8-sig")
print("pool", len(pool), "young", len(young))
