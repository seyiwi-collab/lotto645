"""로또 6/45 분석 + 생성기 빌드.
사용: python3 build.py          # 새 회차만 받아서 갱신 후 index.html 재생성
      python3 build.py --offline # 다운로드 없이 재생성
"""
import json, sys, time, urllib.request, datetime, warnings
from math import comb
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).parent
RAW = HERE / "draws_raw.json"
OUT = HERE / "index.html"
T = comb(45, 6)
warnings.filterwarnings("ignore", category=RuntimeWarning)  # macOS Accelerate matmul 오경고

# ---------- 1. 데이터 수집 (증분) ----------
API = "https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do?srchDir=center&srchLtEpsd=%d"
HDR = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest",
       "Referer": "https://www.dhlottery.co.kr/lt645/result"}


def fetch_around(n):
    req = urllib.request.Request(API % n, headers=HDR)
    return json.load(urllib.request.urlopen(req, timeout=20))["data"]["list"]


def update_raw():
    draws = {x["ltEpsd"]: x for x in json.load(open(RAW))} if RAW.exists() else {}
    n = max(draws) if draws else 1
    while True:
        new = [x for x in fetch_around(n) if x["ltEpsd"] not in draws]
        for x in new:
            draws[x["ltEpsd"]] = x
        if not new:
            break
        n = max(draws) + 3
        time.sleep(0.3)
    eps = sorted(draws)
    assert eps == list(range(1, eps[-1] + 1)), "누락 회차 존재"
    json.dump([draws[k] for k in eps], open(RAW, "w"), ensure_ascii=False)
    return [draws[k] for k in eps]


d = json.load(open(RAW)) if "--offline" in sys.argv else update_raw()
N = len(d)
nums = np.array([[x[f"tm{i}WnNo"] for i in range(1, 7)] for x in d])
X = np.zeros((N, 45))
for t in range(N):
    X[t, nums[t] - 1] = 1
tickets = np.array([x["wholEpsdSumNtslAmt"] for x in d], float) / 1000
w1 = np.array([x["rnk1WnNope"] for x in d], float)
a1 = np.array([x["rnk1WnAmt"] for x in d], float)
w4 = np.array([x["rnk4WnNope"] for x in d], float)
p1 = 1 / T
p4 = comb(6, 4) * comb(39, 2) / T

# ---------- 2. 번호 자체에 패턴이 있는가 ----------
freq = X.sum(0)
exp_freq = N * 6 / 45
chi = float(((freq - exp_freq) ** 2 / exp_freq).sum())
chi_p = float(1 - stats.chi2.cdf(chi, 44))

rng = np.random.default_rng(7)
backtest = []
for W, label in [(10, "최근 10회"), (30, "최근 30회"), (100, "최근 100회"), (None, "전체 누적")]:
    hot, cold = [], []
    for t in range(100, N):
        f = (X[:t] if W is None else X[t - W:t]).sum(0) + rng.random(45) * 1e-6
        hot.append(X[t, np.argsort(-f)[:6]].sum())
        cold.append(X[t, np.argsort(f)[:6]].sum())
    backtest.append({"label": label, "hot": round(float(np.mean(hot)), 3),
                     "cold": round(float(np.mean(cold)), 3), "n": len(hot)})
repeat_prev = round(float(np.mean([X[t] @ X[t - 1] for t in range(1, N)])), 3)

# ---------- 3. 사람들이 고르는 번호 (당첨자 수 기반 인기도) ----------
# 4등(4개 일치) 당첨자 수는 회차당 십만 명 단위라 '그 번호를 고른 사람이 얼마나 많은지'를 잘 드러냄.
# 판매량 대비 기대 당첨자 수와의 비율(log)을 45개 번호 더미로 회귀 → 번호별 인기도.
s = np.arange(99, N)  # 100회 이후 (판매가 1,000원 전환 이후)
y4 = np.log(w4 / (tickets * p4))


def fit_beta(idx):
    b, *_ = np.linalg.lstsq(X[idx], y4[idx], rcond=None)
    return b - b.mean()


def runlen(c):
    c = sorted(c); mr = run = 1
    for a, b in zip(c, c[1:]):
        run = run + 1 if b - a == 1 else 1
        mr = max(mr, run)
    return mr


run3 = np.array([runlen(c) >= 3 for c in nums], float)


def fit_k(beta, idx):
    """1등 당첨자 수 ~ NegBin(offset=기대 당첨자, 인기도점수, 3연속 여부)."""
    import statsmodels.api as sm
    Z = np.column_stack([np.ones(len(idx)), X[idx] @ beta, run3[idx]])
    m = sm.GLM(w1[idx], Z, family=sm.families.NegativeBinomial(alpha=0.1),
               offset=np.log(tickets[idx] * p1)).fit()
    return m.params, m.pvalues


# 표본 외 검증: 짝수 회차로 인기도 학습 → 홀수 회차 1등 당첨자 수로 효과 측정
ev, od = s[s % 2 == 0], s[s % 2 == 1]
b_even = fit_beta(ev)
oos_rho, oos_p = stats.spearmanr(X[od] @ b_even, np.log((w1[od] + 0.5) / (tickets[od] * p1)))
(k0, k_oos, _), _ = fit_k(b_even, od)

beta = fit_beta(s)
params, pvals = fit_k(beta, s)
k_full, run_pen = float(params[1]), float(params[2])
k = float(k_oos)  # 생성기는 보수적으로 표본 외 계수를 사용

R = np.array([rng.choice(45, 6, replace=False) for _ in range(200_000)])
mult = np.exp(k * beta[R].sum(1))
mult /= mult.mean()
gain = {q: round(float(1 / mult[mult <= np.quantile(mult, q)].mean()), 3) for q in (0.1, 0.2)}

# ---------- 4. 1등 당첨 사례 · 금액 ----------
au = [x["winType1"] for x in d]; ma = [x["winType2"] for x in d]; se = [x["winType3"] for x in d]


def row(i):
    x = d[i]
    return {"ep": x["ltEpsd"], "date": x["ltRflYmd"], "nums": nums[i].tolist(), "bonus": x["bnsWnNo"],
            "w": int(w1[i]), "amt": int(a1[i]), "auto": au[i], "manual": ma[i], "semi": se[i]}


many = [row(i) for i in np.argsort(-w1, kind="stable")[:8]]
modern = np.where(np.array([int(x["ltRflYmd"][:4]) for x in d]) >= 2005)[0]
pos = modern[w1[modern] > 0]
top_amt = [row(i) for i in pos[np.argsort(-a1[pos])][:3]]
low_amt = [row(i) for i in pos[np.argsort(a1[pos])][:3]]
tracked = [i for i in range(N) if au[i] + ma[i] + se[i] > 0]
mix = {"auto": sum(au[i] for i in tracked), "manual": sum(ma[i] for i in tracked),
       "semi": sum(se[i] for i in tracked), "from": d[tracked[0]]["ltEpsd"]}

years = {}
for i, x in enumerate(d):
    years.setdefault(x["ltRflYmd"][:4], []).append(i)
yearly = []
for y, idx in sorted(years.items()):
    if len(idx) < 40:
        continue
    idx = np.array(idx); wi = idx[w1[idx] > 0]
    yearly.append({"y": int(y), "w": round(float(w1[idx].mean()), 1),
                   "amt": round(float(a1[wi].mean()) / 1e8, 1),
                   "sales": round(float(tickets[idx].mean() * 1000) / 1e8)})

last = d[-1]
last_date = datetime.datetime.strptime(last["ltRflYmd"], "%Y%m%d").date()
data = {
    "N": N, "last": row(N - 1), "next_ep": N + 1,
    "next_date": (last_date + datetime.timedelta(days=7)).isoformat(),
    "freq": freq.astype(int).tolist(), "exp_freq": round(exp_freq, 1),
    "chi": round(chi, 1), "chi_p": round(chi_p, 3),
    "backtest": backtest, "repeat_prev": repeat_prev,
    "beta": [round(float(b), 4) for b in beta], "k": round(k, 4), "k_full": round(k_full, 4),
    "k_p": float(pvals[1]), "run_pen": round(run_pen, 4), "run_p": round(float(pvals[2]), 3),
    "oos_rho": round(float(oos_rho), 3), "oos_p": round(float(oos_p), 4), "gain": gain,
    "many": many, "top_amt": top_amt, "low_amt": low_amt, "mix": mix, "yearly": yearly,
    "avg_amt_recent": round(float(a1[pos[-260:]].mean()) / 1e8, 1),
    "past": [",".join(map(str, sorted(c))) for c in nums.tolist()],
}

html = (HERE / "template.html").read_text(encoding="utf-8")
OUT.write_text(html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
print(f"{N}회 ({last['ltRflYmd']})까지 반영 → {OUT.name}")
print(f"빈도 χ²={chi:.1f} p={chi_p:.3f} | 표본외 ρ={oos_rho:.3f} p={oos_p:.4f} | k={k:.2f} | 하위20% 수령액 ×{gain[0.2]}")
