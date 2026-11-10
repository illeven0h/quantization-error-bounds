"""
sanity_checks.py  --  run:  python sanity_checks.py
Each check prints PASS / FAIL. If everything passes, the building blocks of the
project behave the way the maths says they must.
"""
import numpy as np
from qbounds import (quantize_uniform, Scaler, InputStats, fit_ridge, MLP,
                     mlp_error_and_bounds)

results = []
def check(name, ok):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name)

rng = np.random.default_rng(0)

# 1. Quantizer: error never exceeds delta/2, and uses at most 2^bits levels
ok = True
for bits in range(2, 9):
    w = rng.normal(size=500)
    wq, delta = quantize_uniform(w, bits)
    ok &= np.abs(wq - w).max() <= delta / 2 + 1e-12
    ok &= len(np.unique(np.round(wq, 12))) <= 2 ** bits
check("quantizer: |wq - w| <= delta/2 and at most 2^bits levels (bits 2..8)", ok)

# 2. Step size follows delta = range / (2^bits - 1), so one extra bit gives
#    delta_{b+1}/delta_b = (2^b - 1)/(2^(b+1) - 1)  (0.43 at 2->3 bits, tending to 0.5)
w = rng.normal(size=500)
ratios = [quantize_uniform(w, b + 1)[1] / quantize_uniform(w, b)[1] for b in range(2, 8)]
expected = [(2 ** b - 1) / (2 ** (b + 1) - 1) for b in range(2, 8)]
check("quantizer: delta ratio per extra bit matches (2^b-1)/(2^(b+1)-1)", np.allclose(ratios, expected))

# 3. Scalers do what their names say (on the training set)
X = rng.normal(5, 3, size=(300, 4)) * np.array([1, 10, 0.1, 100])
Z = Scaler("zscore").fit(X).transform(X)
M = Scaler("minmax").fit(X).transform(X)
check("zscore: mean ~0 and std ~1", np.allclose(Z.mean(0), 0, atol=1e-9) and np.allclose(Z.std(0), 1))
check("minmax: every training feature lies in [0,1]", M.min() >= -1e-12 and M.max() <= 1 + 1e-12)

# 4. Closed-form fit agrees with numpy least squares (with intercept)
Xf = rng.normal(size=(200, 5)); yf = Xf @ rng.normal(size=5) + 3 + 0.1 * rng.normal(size=200)
w, b = fit_ridge(Xf, yf)
ref = np.linalg.lstsq(np.hstack([Xf, np.ones((200, 1))]), yf, rcond=None)[0]
check("fit_ridge matches numpy lstsq", np.allclose(np.append(w, b), ref, atol=1e-4))

# 5. THE KEY TEST: on the data used to compute the constants (max ||x||_1 and the
#    Mahalanobis radius R), a valid bound can NEVER be violated.
ok1 = ok2 = True
for trial in range(20):
    d = int(rng.integers(3, 30))
    Xt = rng.standard_t(3, size=(500, d)) * rng.uniform(.1, 10, d) + rng.normal(0, 5, d)
    wt = rng.normal(size=d)
    st = InputStats(Xt)
    for bits in range(2, 9):
        wq, _ = quantize_uniform(wt, bits)
        dw = wq - wt
        e = np.abs(Xt @ dw)
        ok1 &= e.max() <= st.bound1(dw)[0] * (1 + 1e-9)
        ok2 &= e.max() <= st.bound2(dw)[0] * (1 + 1e-9)
check("B1 >= measured error on its own calibration data (20 random problems x 7 bit-widths)", ok1)
check("B2 >= measured error on its own calibration data (20 random problems x 7 bit-widths)", ok2)

# 6. Same key test for the composed MLP bound
Xm = rng.normal(size=(400, 6)) * rng.uniform(.5, 5, 6) + rng.normal(0, 3, 6)
ym = np.tanh(Xm[:, 0]) + 0.1 * rng.normal(size=400)
Zs = Scaler("zscore").fit(Xm).transform(Xm)
net = MLP(6, 8, "reg", seed=0); net.fit(Zs, ym, Zs, ym, epochs=200)
ok = True
for bits in range(2, 9):
    _, _, e, B1, B2 = mlp_error_and_bounds(net, Zs, Zs, bits)
    ok &= np.abs(e).max() <= B1 * (1 + 1e-9) and np.abs(e).max() <= B2 * (1 + 1e-9)
check("MLP composed B1 and B2 >= measured error on calibration data", ok)

print("\n%d / %d checks passed" % (sum(results), len(results)))
