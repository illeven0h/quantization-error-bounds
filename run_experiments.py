"""
run_experiments.py  --  reproduces every result / figure of the initial experiments.

    python run_experiments.py

Outputs (written next to this script's parent folder):
    results/linear_results.csv, results/mlp_results.csv, results/*_summary.csv
    figures/*.png
"""
import os
import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer, load_digits

from qbounds import (quantize_uniform, Scaler, InputStats, fit_ridge, gd_linear,
                     MLP, mlp_error_and_bounds)

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

BITS = list(range(2, 9))
SCALINGS = ["raw", "minmax", "zscore"]
SCOL = {"raw": "tab:red", "minmax": "tab:orange", "zscore": "tab:blue"}
N_SEEDS_LIN, N_SEEDS_MLP = 10, 5
plt.rcParams.update({"font.size": 9, "figure.dpi": 130, "axes.grid": True, "grid.alpha": .3})


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------
def synthetic(regime, seed, n=4000, d=40):
    """Controlled input regimes. Features get heterogeneous scales and offsets
    (like real un-normalised data) so that normalization has something to fix."""
    rng = np.random.default_rng(seed)
    s = np.exp(rng.uniform(np.log(0.1), np.log(10), d))     # feature scales 0.1 .. 10
    m = s * rng.uniform(-5, 5, d)                            # feature offsets
    if regime == "isotropic":
        z = rng.normal(size=(n, d)); X = m + s * z
    elif regime == "lowrank":                                # rank-5 covariance + small noise
        A = rng.normal(size=(d, 5))
        z = rng.normal(size=(n, 5)) @ A.T + 0.1 * rng.normal(size=(n, d)); X = m + s * z
    elif regime == "heavytail":                              # Student-t(3) tails
        z = rng.standard_t(3, size=(n, d)); X = m + s * z
    elif regime == "sparse":                                 # ~10% non-zeros, non-negative
        z = (rng.random((n, d)) < 0.1) * rng.exponential(1.0, (n, d)); X = s * z
    else:
        raise ValueError(regime)
    beta = rng.normal(size=d) / np.sqrt(d)
    y = z @ beta + 0.3 * rng.normal(size=n)
    return X, y


def real_dataset(name):
    if name == "diabetes":
        D = load_diabetes(scaled=False); return D.data, D.target.astype(float)
    if name == "breast_cancer":                               # linear probability model
        D = load_breast_cancer(); return D.data, D.target.astype(float)
    if name == "digits":                                      # sparse pixels, regress digit value
        D = load_digits(); return D.data, D.target.astype(float)
    raise ValueError(name)


LIN_DATASETS = ["syn_isotropic", "syn_lowrank", "syn_heavytail", "syn_sparse",
                "diabetes", "breast_cancer", "digits"]


def get_linear_data(name, seed):
    if name.startswith("syn_"):
        X, y = synthetic(name[4:], seed)
    else:
        X, y = real_dataset(name)
    rng = np.random.default_rng(1000 + seed)
    idx = rng.permutation(len(X)); k = len(X) // 2
    return X[idx[:k]], y[idx[:k]], X[idx[k:]], y[idx[k:]]


# ----------------------------------------------------------------------------
# Experiment A: linear models
# ----------------------------------------------------------------------------
def run_linear():
    rows = []
    for ds in LIN_DATASETS:
        for seed in range(N_SEEDS_LIN):
            Xtr, ytr, Xte, yte = get_linear_data(ds, seed)
            for sc in SCALINGS:
                S = Scaler(sc).fit(Xtr)
                Atr, Ate = S.transform(Xtr), S.transform(Xte)
                w, b = fit_ridge(Atr, ytr)
                st = InputStats(Atr)
                yfp = Ate @ w + b
                mse_fp = np.mean((yfp - yte) ** 2)
                for bits in BITS:
                    wq, delta = quantize_uniform(w, bits)
                    dw = wq - w
                    e = Ate @ dw                                   # = y_q - y_fp
                    mx, rms = np.abs(e).max(), np.sqrt(np.mean(e ** 2))
                    B1, B2 = st.bound1(dw)[0], st.bound2(dw)[0]
                    B1a, B2a = st.bound1_apriori(delta), st.bound2_apriori(delta)
                    rows.append(dict(
                        dataset=ds, seed=seed, scaling=sc, bits=bits, delta=delta,
                        max_err=mx, rms_err=rms, B1=B1, B2=B2, B1_apriori=B1a, B2_apriori=B2a,
                        tight_B1=B1 / mx, tight_B2=B2 / mx, ratio_B2_B1=B2 / B1,
                        viol_B1=np.mean(np.abs(e) > B1 * (1 + 1e-9)),
                        viol_B2=np.mean(np.abs(e) > B2 * (1 + 1e-9)),
                        mse_fp=mse_fp, mse_q=np.mean((Ate @ wq + b - yte) ** 2),
                        R=st.R.min(), l1max=st.l1max))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "linear_results.csv"), index=False)
    return df


# ----------------------------------------------------------------------------
# Experiment B: one-hidden-layer MLP
# ----------------------------------------------------------------------------
MLP_DATASETS = ["syn_lowrank_nl", "diabetes", "breast_cancer"]


def get_mlp_data(name, seed):
    if name == "syn_lowrank_nl":                    # nonlinear target on low-rank inputs
        rng = np.random.default_rng(seed)
        d, n = 40, 3000
        s = np.exp(rng.uniform(np.log(0.1), np.log(10), d)); m = s * rng.uniform(-5, 5, d)
        A = rng.normal(size=(d, 5))
        g = rng.normal(size=(n, 5))
        z = g @ A.T + 0.1 * rng.normal(size=(n, d))
        X = m + s * z
        y = np.tanh(g[:, 0]) + 0.5 * np.maximum(g[:, 1], 0) + 0.3 * g[:, 2] * g[:, 3] + 0.1 * rng.normal(size=n)
        task = "reg"
    elif name == "diabetes":
        X, y = real_dataset("diabetes"); task = "reg"
    else:
        X, y = real_dataset("breast_cancer"); task = "clf"
    rng = np.random.default_rng(2000 + seed)
    idx = rng.permutation(len(X)); n = len(X)
    a, b = int(.6 * n), int(.8 * n)
    tr, va, te = idx[:a], idx[a:b], idx[b:]
    if task == "reg":
        mu, sd = y[tr].mean(), y[tr].std(); y = (y - mu) / sd      # errors in units of target std
    return X[tr], y[tr], X[va], y[va], X[te], y[te], task


def run_mlp():
    rows, curves = [], {}
    for ds in MLP_DATASETS:
        for seed in range(N_SEEDS_MLP):
            Xtr, ytr, Xva, yva, Xte, yte, task = get_mlp_data(ds, seed)
            for sc in SCALINGS:
                S = Scaler(sc).fit(Xtr)
                Atr, Ava, Ate = S.transform(Xtr), S.transform(Xva), S.transform(Xte)
                net = MLP(Atr.shape[1], 16, task, seed=seed)
                htr, hva = net.fit(Atr, ytr, Ava, yva, epochs=600, lr=5e-3)
                if seed == 0:
                    curves[(ds, sc)] = (htr, hva)
                for bits in BITS:
                    ofp, oq, e, B1, B2 = mlp_error_and_bounds(net, Atr, Ate, bits)
                    mx, rms = np.abs(e).max(), np.sqrt(np.mean(e ** 2))
                    if task == "reg":
                        m_fp, m_q = np.mean((ofp - yte) ** 2), np.mean((oq - yte) ** 2)
                    else:
                        m_fp, m_q = np.mean((ofp > 0) == (yte > .5)), np.mean((oq > 0) == (yte > .5))
                    rows.append(dict(dataset=ds, seed=seed, scaling=sc, bits=bits, task=task,
                                     max_err=mx, rms_err=rms, B1=B1, B2=B2,
                                     tight_B1=B1 / max(mx, 1e-300), tight_B2=B2 / max(mx, 1e-300),
                                     ratio_B2_B1=B2 / B1,
                                     viol_B1=np.mean(np.abs(e) > B1 * (1 + 1e-9)),
                                     viol_B2=np.mean(np.abs(e) > B2 * (1 + 1e-9)),
                                     metric_fp=m_fp, metric_q=m_q))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "mlp_results.csv"), index=False)
    return df, curves


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def gmean(x):
    return np.exp(np.mean(np.log(x)))


def fig_training_linear():
    names = ["syn_lowrank", "diabetes", "breast_cancer"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, ds in zip(axs, names):
        Xtr, ytr, Xte, yte = get_linear_data(ds, 0)
        for sc in SCALINGS:
            S = Scaler(sc).fit(Xtr)
            A, At = S.transform(Xtr), S.transform(Xte)
            tr, te = gd_linear(A, ytr, At, yte, epochs=500)
            w, b = fit_ridge(A, ytr)
            ax.plot(tr, color=SCOL[sc], label=f"{sc} (train)")
            ax.plot(te, color=SCOL[sc], ls="--", alpha=.7, label=f"{sc} (test)")
            ax.axhline(np.mean((A @ w + b - ytr) ** 2), color=SCOL[sc], ls=":", lw=.8)
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_title(ds); ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    axs[0].legend(fontsize=6.5)
    fig.suptitle("Linear model: gradient-descent training curves (step = 1/L). Dotted = closed-form optimum", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_training_linear.png"), bbox_inches="tight"); plt.close(fig)


def fig_training_mlp(curves):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, ds in zip(axs, MLP_DATASETS):
        for sc in SCALINGS:
            htr, hva = curves[(ds, sc)]
            ax.plot(htr, color=SCOL[sc], label=f"{sc} (train)")
            ax.plot(hva, color=SCOL[sc], ls="--", alpha=.7, label=f"{sc} (val)")
        ax.set_yscale("log"); ax.set_title(ds); ax.set_xlabel("epoch")
        ax.set_ylabel("loss (MSE)" if ds != "breast_cancer" else "loss (BCE)")
    axs[0].legend(fontsize=6.5)
    fig.suptitle("MLP (d-16-1, ReLU, Adam): training / validation loss", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_training_mlp.png"), bbox_inches="tight"); plt.close(fig)


def fig_error_vs_bits(df, name, title, datasets):
    fig, axs = plt.subplots(len(datasets), 3, figsize=(11, 2.6 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        for j, sc in enumerate(SCALINGS):
            ax = axs[i, j]
            g = df[(df.dataset == ds) & (df.scaling == sc)].groupby("bits")
            for col, lab, st in [("max_err", "measured max |e|", "k-o"), ("rms_err", "measured RMS", "k--s"),
                                 ("B1", "B1 distribution-free", "r-^"), ("B2", "B2 covariance-spectrum", "b-v")]:
                ax.plot(g[col].apply(gmean).index, g[col].apply(gmean).values, st, ms=3.5, label=lab)
            ax.set_yscale("log")
            if i == 0: ax.set_title(sc)
            if j == 0: ax.set_ylabel(ds + "\noutput error")
            if i == len(datasets) - 1: ax.set_xlabel("bits")
    axs[0, 0].legend(fontsize=6.5)
    fig.suptitle(title, y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close(fig)


def fig_heatmaps(df, name, datasets, title):
    piv = lambda col: df.groupby(["dataset", "scaling"])[col].apply(gmean).unstack()[SCALINGS].loc[datasets]
    mats = [("B2 / B1  (<1 : new bound tighter)", piv("ratio_B2_B1"), "RdYlGn_r", None),
            ("Tightness of B1 = B1 / max|e|", piv("tight_B1"), "viridis_r", None),
            ("Tightness of B2 = B2 / max|e|", piv("tight_B2"), "viridis_r", None)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 0.5 * len(datasets) + 2.2))
    for ax, (t, M, cm, _) in zip(axs, mats):
        im = ax.imshow(np.log10(M.values), cmap=cm, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(SCALINGS)
        ax.set_yticks(range(len(datasets))); ax.set_yticklabels(datasets)
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax.text(b, a, f"{M.values[a, b]:.2g}", ha="center", va="center", fontsize=8,
                        color="white" if abs(np.log10(M.values[a, b])) > .8 else "black")
        ax.set_title(t, fontsize=9); ax.grid(False)
    fig.suptitle(title + "  (geometric mean over seeds and 2-8 bits)", y=1.03)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close(fig)


def fig_normalization(df, name, datasets, bits=4):
    d = df[df.bits == bits].groupby(["dataset", "scaling"])[["max_err", "B1", "B2"]].agg(gmean)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), sharey=False)
    x = np.arange(len(datasets)); wdt = .26
    for ax, col, t in zip(axs, ["max_err", "B1", "B2"],
                          ["measured max |e|", "B1 distribution-free bound", "B2 covariance-spectrum bound"]):
        for k, sc in enumerate(SCALINGS):
            ax.bar(x + (k - 1) * wdt, [d.loc[(ds, sc), col] for ds in datasets], wdt, color=SCOL[sc], label=sc)
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=7)
        ax.set_title(t, fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle(f"Effect of normalization at {bits}-bit weights (output units, log scale)", y=1.03)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close(fig)


def fig_task_metric(dflin, dfmlp):
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, ds in zip(axs[:2], ["syn_lowrank", "diabetes"]):
        for sc in SCALINGS:
            g = dflin[(dflin.dataset == ds) & (dflin.scaling == sc)].groupby("bits")
            ax.plot(g.apply(lambda t: np.mean(t.mse_q / t.mse_fp)).index,
                    g.apply(lambda t: np.mean(t.mse_q / t.mse_fp)).values, "-o", ms=3, color=SCOL[sc], label=sc)
        ax.set_yscale("log"); ax.set_title(f"linear: {ds}"); ax.set_xlabel("bits"); ax.set_ylabel("test MSE (quantized) / MSE (fp)")
    for ax, ds in zip(axs[2:], ["syn_lowrank_nl", "breast_cancer"]):
        for sc in SCALINGS:
            g = dfmlp[(dfmlp.dataset == ds) & (dfmlp.scaling == sc)].groupby("bits")
            if ds == "breast_cancer":
                ax.plot(g.metric_q.mean().index, g.metric_q.mean().values, "-o", ms=3, color=SCOL[sc], label=sc)
                ax.set_ylabel("test accuracy (quantized)")
            else:
                ax.plot(g.apply(lambda t: np.mean(t.metric_q / t.metric_fp)).index,
                        g.apply(lambda t: np.mean(t.metric_q / t.metric_fp)).values, "-o", ms=3, color=SCOL[sc], label=sc)
                ax.set_yscale("log"); ax.set_ylabel("test MSE (quantized) / MSE (fp)")
        ax.set_title(f"MLP: {ds}"); ax.set_xlabel("bits")
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_task_metric_vs_bits.png"), bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Experiment A: linear models ...")
    dfl = run_linear()
    print("Experiment B: MLP ...")
    dfm, curves = run_mlp()

    print("Figures ...")
    fig_training_linear()
    fig_training_mlp(curves)
    fig_error_vs_bits(dfl, "fig3_linear_error_vs_bits.png",
                      "Linear models: measured output error vs. the two bounds (geometric mean over 10 seeds)",
                      LIN_DATASETS)
    fig_heatmaps(dfl, "fig4_linear_heatmaps.png", LIN_DATASETS, "Linear models: bound quality")
    fig_normalization(dfl, "fig5_linear_normalization.png", LIN_DATASETS, bits=4)
    fig_error_vs_bits(dfm, "fig6_mlp_error_vs_bits.png",
                      "MLP: measured logit/output error vs. composed bounds (geometric mean over 5 seeds)",
                      MLP_DATASETS)
    fig_heatmaps(dfm, "fig6b_mlp_heatmaps.png", MLP_DATASETS, "MLP: bound quality")
    fig_task_metric(dfl, dfm)

    # ---- summary tables ----
    agg = {"max_err": gmean, "rms_err": gmean, "B1": gmean, "B2": gmean,
           "tight_B1": gmean, "tight_B2": gmean, "ratio_B2_B1": gmean,
           "viol_B1": "mean", "viol_B2": "mean"}
    dfl.groupby(["dataset", "scaling", "bits"]).agg(agg).to_csv(os.path.join(RES, "linear_summary_by_bits.csv"))
    dfl.groupby(["dataset", "scaling"]).agg(agg).to_csv(os.path.join(RES, "linear_summary.csv"))
    dfm.groupby(["dataset", "scaling", "bits"]).agg(agg).to_csv(os.path.join(RES, "mlp_summary_by_bits.csv"))
    dfm.groupby(["dataset", "scaling"]).agg(agg).to_csv(os.path.join(RES, "mlp_summary.csv"))
    print("Done.")
