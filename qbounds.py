"""
qbounds.py
==========
Core library for the project

  "Input-Statistics-Dependent Quantization Error Bounds for Linear Models,
   with Feature Normalization as a Provable Error-Reduction Mechanism"

Contents
--------
1. quantize_uniform   : plain round-to-nearest uniform weight quantizer
2. Scaler             : raw / min-max / z-score feature scaling (fit on train only)
3. InputStats         : mean, covariance spectrum, Mahalanobis radius, l1 radius
                        and the two error bounds
                          B1  distribution-free (Anguita et al. baseline)
                          B2  second-moment / covariance-spectrum bound (ours)
4. Linear model       : closed-form ridge fit and gradient-descent training curves
5. NumPy MLP          : 1 hidden layer (ReLU), Adam, + composed layer-wise bounds
"""
import numpy as np

# ----------------------------------------------------------------------------
# 1. Uniform round-to-nearest quantizer
# ----------------------------------------------------------------------------
def quantize_uniform(w, bits):
    """Uniform quantization of a tensor with 2**bits levels spanning [min, max].

    Step  delta = (max - min) / (2**bits - 1);  levels = min + k*delta.
    Round-to-nearest, no clipping  =>  |wq - w| <= delta/2 elementwise.
    Returns (wq, delta).
    """
    w = np.asarray(w, dtype=float)
    lo, hi = w.min(), w.max()
    if hi == lo:
        return w.copy(), 0.0
    delta = (hi - lo) / (2 ** bits - 1)
    wq = lo + np.round((w - lo) / delta) * delta
    return wq, delta


# ----------------------------------------------------------------------------
# 2. Feature scaling (statistics are fitted on the TRAINING set only)
# ----------------------------------------------------------------------------
class Scaler:
    def __init__(self, kind):
        assert kind in ("raw", "minmax", "zscore")
        self.kind = kind

    def fit(self, X):
        if self.kind == "raw":
            self.shift, self.scale = np.zeros(X.shape[1]), np.ones(X.shape[1])
        elif self.kind == "minmax":
            self.shift = X.min(0)
            self.scale = X.max(0) - X.min(0)
        else:
            self.shift = X.mean(0)
            self.scale = X.std(0)
        self.scale = np.where(self.scale < 1e-12, 1.0, self.scale)
        return self

    def transform(self, X):
        return (X - self.shift) / self.scale


# ----------------------------------------------------------------------------
# 3. Input statistics and bounds
# ----------------------------------------------------------------------------
class InputStats:
    """Statistics of a (training) input set X (n x d) used by the bounds.

    For a weight-error row dw (d,) the output error of a linear unit is
        e(x) = dw^T x.

    B1 (distribution-free baseline, Anguita et al. 1999, Holder l1/linf):
        |e(x)| <= ||x||_1 ||dw||_inf <= max_x ||x||_1 * ||dw||_inf

    B2 (second-moment bound; Cauchy-Schwarz in the Sigma-geometry):
        e(x) = dw^T mu + dw^T (x - mu)
        |dw^T (x-mu)| <= sqrt(dw^T S dw) * sqrt((x-mu)^T S^-1 (x-mu))     for ANY S > 0
        =>  |e(x)| <= |dw^T mu| + R_S * sqrt(dw^T S dw),
        with R_S = max_x Mahalanobis radius (data assumption "bounded radius").
        We take S = Sigma + eps*I (always invertible) and minimise the bound
        over a grid of eps (each member of the grid is a valid bound, so the
        minimum is valid too).
    """

    def __init__(self, X, n_eps=13):
        n, d = X.shape
        self.n, self.d = n, d
        self.mu = X.mean(0)
        Xc = X - self.mu
        Sigma = Xc.T @ Xc / n
        lam, V = np.linalg.eigh(Sigma)
        self.lam = np.clip(lam, 0.0, None)
        self.V = V
        self.lam_max = float(self.lam.max())
        tr_d = max(float(self.lam.sum()) / d, 1e-300)
        self.eps = tr_d * np.logspace(-6, 1, n_eps)
        C = Xc @ V                                   # coordinates in eigenbasis
        self.R = np.array([np.sqrt(((C ** 2) / (self.lam + e)).sum(1).max())
                           for e in self.eps])       # Mahalanobis radius per eps
        self.l1max = float(np.abs(X).sum(1).max())   # max ||x||_1
        self.mu_l1 = float(np.abs(self.mu).sum())

    # ---- actual-dw bounds (row-wise) --------------------------------------
    def bound1(self, dw):
        dw = np.atleast_2d(dw)
        return self.l1max * np.abs(dw).max(1)

    def bound2(self, dw):
        dw = np.atleast_2d(dw)
        q = dw @ self.V
        nrm2 = (q ** 2).sum(1)
        quad = (q ** 2 * self.lam).sum(1)                       # dw^T Sigma dw
        cand = self.R[None, :] * np.sqrt(quad[:, None] + self.eps[None, :] * nrm2[:, None])
        return np.abs(dw @ self.mu) + cand.min(1)

    # ---- a-priori closed forms: only use the step delta (|dw_i|<=delta/2) -----
    def bound1_apriori(self, delta):
        return self.l1max * delta / 2.0

    def bound2_apriori(self, delta):
        # dw^T S dw <= (lam_max+eps) ||dw||_2^2 <= (lam_max+eps) d delta^2/4
        cand = self.R * np.sqrt((self.lam_max + self.eps) * self.d) * delta / 2.0
        return self.mu_l1 * delta / 2.0 + cand.min()


# ----------------------------------------------------------------------------
# 4. Linear model
# ----------------------------------------------------------------------------
def fit_ridge(X, y, lam_rel=1e-6):
    """Centered closed-form ridge (tiny relative ridge ~ OLS). Returns (w, b)."""
    mu, ym = X.mean(0), y.mean()
    Xc, yc = X - mu, y - ym
    G = Xc.T @ Xc
    lam = lam_rel * np.trace(G) / X.shape[1]
    w = np.linalg.solve(G + lam * np.eye(X.shape[1]), Xc.T @ yc)
    return w, ym - mu @ w


def gd_linear(Xtr, ytr, Xte, yte, epochs=500):
    """Full-batch gradient descent on MSE with the largest stable step 1/L.
    Used only for the training graphs. Returns (train_loss, test_loss)."""
    n = len(Xtr)
    A = np.hstack([Xtr, np.ones((n, 1))])
    At = np.hstack([Xte, np.ones((len(Xte), 1))])
    L = np.linalg.eigvalsh(A.T @ A / n).max()
    lr = 1.0 / L
    theta = np.zeros(A.shape[1])
    tr, te = [], []
    for _ in range(epochs):
        r = A @ theta - ytr
        tr.append(np.mean(r ** 2))
        te.append(np.mean((At @ theta - yte) ** 2))
        theta -= lr * (A.T @ r) / n          # gradient of MSE/2, stable step 1/L
    return np.array(tr), np.array(te)


# ----------------------------------------------------------------------------
# 5. NumPy MLP  (d -> H ReLU -> 1)   task = "reg" (MSE) or "clf" (BCE on logit)
# ----------------------------------------------------------------------------
def _bce_with_logits(z, y):
    return np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))


class MLP:
    def __init__(self, d, H, task, seed=0):
        rng = np.random.default_rng(seed)
        self.task = task
        self.W1 = rng.normal(0, np.sqrt(2.0 / d), (H, d))
        self.b1 = np.zeros(H)
        self.w2 = rng.normal(0, np.sqrt(1.0 / H), H)
        self.b2 = 0.0

    def forward(self, X, W1=None, w2=None):
        W1 = self.W1 if W1 is None else W1
        w2 = self.w2 if w2 is None else w2
        z1 = X @ W1.T + self.b1
        h = np.maximum(z1, 0)
        return h @ w2 + self.b2, h, z1

    def loss(self, out, y):
        return np.mean((out - y) ** 2) if self.task == "reg" else _bce_with_logits(out, y)

    def fit(self, Xtr, ytr, Xva, yva, epochs=600, lr=5e-3):
        params = ["W1", "b1", "w2", "b2"]
        m = {p: 0.0 for p in params}
        v = {p: 0.0 for p in params}
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        hist_tr, hist_va = [], []
        best = (np.inf, None)
        n = len(Xtr)
        for t in range(1, epochs + 1):
            out, h, z1 = self.forward(Xtr)
            hist_tr.append(self.loss(out, ytr))
            hist_va.append(self.loss(self.forward(Xva)[0], yva))
            if hist_va[-1] < best[0]:
                best = (hist_va[-1], (self.W1.copy(), self.b1.copy(), self.w2.copy(), float(self.b2)))
            if self.task == "reg":
                dout = 2.0 * (out - ytr) / n
            else:
                dout = (1.0 / (1.0 + np.exp(-out)) - ytr) / n
            g = {"w2": h.T @ dout, "b2": dout.sum()}
            dz1 = np.outer(dout, self.w2) * (z1 > 0)
            g["W1"], g["b1"] = dz1.T @ Xtr, dz1.sum(0)
            for p in params:
                m[p] = b1_ * m[p] + (1 - b1_) * g[p]
                v[p] = b2_ * v[p] + (1 - b2_) * g[p] ** 2
                upd = lr * (m[p] / (1 - b1_ ** t)) / (np.sqrt(v[p] / (1 - b2_ ** t)) + eps)
                setattr(self, p, getattr(self, p) - upd)
        self.W1, self.b1, self.w2, self.b2 = best[1]      # best-validation checkpoint
        return np.array(hist_tr), np.array(hist_va)


def mlp_error_and_bounds(net, Xtr, Xte, bits):
    """Quantize both weight tensors of the MLP and return output errors + bounds.

    e = out_q - out_fp
      = dw2^T h_q  +  w2^T (h_q - h),      |h_q - h|_j <= |(dW1 x)_j|   (ReLU is 1-Lipschitz)
    =>  |e| <= |dw2^T h_q|  +  sum_j |w2_j| |dW1_j^T x|
    Each of the two parts is a linear-unit error, so B1 / B2 apply to each part:
      layer 1 uses the statistics of the input x,
      layer 2 uses the statistics of the (quantized-net) hidden activation h_q.
    """
    W1q, d1 = quantize_uniform(net.W1, bits)
    w2q, d2 = quantize_uniform(net.w2, bits)
    dW1, dw2 = W1q - net.W1, w2q - net.w2

    out_fp, _, _ = net.forward(Xte)
    out_q, _, _ = net.forward(Xte, W1q, w2q)
    e = out_q - out_fp

    st1 = InputStats(Xtr)
    _, hq_tr, _ = net.forward(Xtr, W1q, w2q)
    st2 = InputStats(hq_tr)
    aw2 = np.abs(net.w2)

    B1 = st2.bound1(dw2)[0] + float((aw2 * st1.bound1(dW1)).sum())
    B2 = st2.bound2(dw2)[0] + float((aw2 * st1.bound2(dW1)).sum())
    return out_fp, out_q, e, B1, B2
