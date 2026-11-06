"""How much operator feedback a fitted risk curve needs (synthetic labels from a known curve).

    python bench/risk_fit.py   -> results/risk_fit.json
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xray_align.risk import fit  # noqa: E402

TRUE = (4.5, 1.2)        # a made-up "truth" for the test, unrelated to any clinical value
rng = np.random.default_rng(11)
out = {"true_theta50": TRUE[0], "true_spread": TRUE[1], "deviation_range_deg": [0, 12], "trials": 200, "by_n": []}
for n in (20, 50, 100, 200, 500):
    t50, sp, fails = [], [], 0
    for _ in range(200):
        x = rng.uniform(0, 12, n)
        y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-(x - TRUE[0]) / TRUE[1]))).astype(int)
        m = fit(x, y)
        if m is None: fails += 1; continue
        t50.append(m.theta50); sp.append(m.spread)
    t50, sp = np.array(t50), np.array(sp)
    out["by_n"].append({"n": n, "theta50_p5": float(np.percentile(t50, 5)), "theta50_p50": float(np.median(t50)), "theta50_p95": float(np.percentile(t50, 95)),
                        "spread_p5": float(np.percentile(sp, 5)), "spread_p95": float(np.percentile(sp, 95)), "no_fit": fails})
Path(__file__).resolve().parents[1].joinpath("results", "risk_fit.json").write_text(json.dumps(out, indent=1))
for r in out["by_n"]: print(f"n={r['n']:4}: theta50 90% range {r['theta50_p5']:.2f}-{r['theta50_p95']:.2f} deg (true {TRUE[0]}), spread {r['spread_p5']:.2f}-{r['spread_p95']:.2f}, no fit {r['no_fit']}")
