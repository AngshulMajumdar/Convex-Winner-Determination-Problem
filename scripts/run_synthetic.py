#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scwdp import (
    rejection_data, solve_rejection_lp, solve_acceptance_lp, solve_wdp_milp,
    repair, fractional_priority_recovery, descending_value_greedy, scwdp_outer,
    epsilon0_gamma, max_inf_distance_under_gap,
)

SEED = 20261012
RNG = np.random.default_rng(SEED)
DATA = ROOT / "data"
FIG = ROOT / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def gen_overlap(rng, n=6, m=14, kmax=4):
    A = np.zeros((n, m), dtype=float)
    for j in range(m):
        size = int(rng.integers(1, min(kmax, n) + 1))
        A[rng.choice(n, size=size, replace=False), j] = 1.0
    deg = A.sum(axis=1)
    for i in np.where(deg < 2)[0]:
        for _ in range(int(2 - deg[i])):
            A[i, int(rng.integers(m))] = 1.0
    sizes = A.sum(axis=0)
    v = 1.0 + 2.0 * sizes + rng.uniform(0.0, 5.0, size=m)
    return A, v


def gen_partition(rng, n_items=5, bids_per_item=3):
    m = n_items * bids_per_item
    A = np.zeros((n_items, m), dtype=float)
    v = np.zeros(m)
    for i in range(n_items):
        js = np.arange(i * bids_per_item, (i + 1) * bids_per_item)
        A[i, js] = 1.0
        # Deliberately separated values improve numerical conditioning while
        # keeping a unique integral LP optimum.
        vals = rng.uniform(1.0, 5.0, size=bids_per_item)
        winner = int(rng.integers(bids_per_item))
        vals[winner] += rng.uniform(6.0, 9.0)
        vals += np.arange(bids_per_item) * 1e-5
        v[js] = vals
    return A, v


def is_integral(x, tol=1e-8):
    return float(np.max(np.minimum(np.abs(x), np.abs(x - 1.0)))) <= tol


# ---------------------------------------------------------------------------
# S1: exact weighted-support thresholding + uniform repair proposition.
# ---------------------------------------------------------------------------
support_rows = []
repair_rows = []
for inst in range(25):
    A, v = gen_overlap(RNG)
    b = rejection_data(A)
    # Construct feasible fractional rejection points as convex combinations of
    # two feasible binary rejection points obtained from random greedy orders.
    for rep in range(20):
        order1 = RNG.permutation(len(v))
        order2 = RNG.permutation(len(v))
        # local random-order greedy acceptance
        def rand_greedy(order):
            used = np.zeros(A.shape[0], dtype=bool)
            x = np.zeros(len(v), dtype=int)
            for j in order:
                items = np.flatnonzero(A[:, j] > 0.5)
                if not used[items].any():
                    x[j] = 1
                    used[items] = True
            return x
        z1 = 1.0 - rand_greedy(order1)
        z2 = 1.0 - rand_greedy(order2)
        t = float(RNG.uniform(0.05, 0.95))
        z = t * z1 + (1.0 - t) * z2
        zhat = (z > 1e-14).astype(float)
        obj = float(v[z > 1e-14].sum())
        objhat = float(v[zhat > 0.5].sum())
        support_rows.append({
            "instance": inst, "sample": rep,
            "continuous_support_objective": obj,
            "thresholded_support_objective": objhat,
            "objective_difference": objhat - obj,
            "continuous_min_margin": float(np.min(A @ z - b)),
            "thresholded_min_margin": float(np.min(A @ zhat - b)),
        })

    for rep in range(80):
        q = RNG.uniform(0.0, 1.0, size=len(v))
        zr, theta, r = repair(A, b, v, q)
        repair_rows.append({
            "instance": inst, "sample": rep,
            "residual_inf": float(np.max(r)),
            "theta": theta,
            "postrepair_min_margin": float(np.min(A @ zr - b)),
            "repair_cost_increase": float(v @ zr - v @ q),
        })

support_df = pd.DataFrame(support_rows)
repair_df = pd.DataFrame(repair_rows)
support_df.to_csv(DATA / "support_thresholding.csv", index=False)
repair_df.to_csv(DATA / "repair_validation.csv", index=False)

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.scatter(repair_df["residual_inf"], repair_df["theta"], s=10, alpha=0.28)
mx = max(repair_df["residual_inf"].max(), repair_df["theta"].max())
ax.plot([0, mx], [0, mx], linestyle="--", linewidth=1.4, label=r"$\theta=\|r\|_\infty$")
ax.set_xlabel(r"Pre-repair violation $\|[b-Az]_+\|_\infty$")
ax.set_ylabel(r"Repair parameter $\theta(z)$")
ax.set_title("Uniform rejection repair: synthetic validation")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "figS1_repair.png", dpi=220)
fig.savefig(FIG / "figS1_repair.pdf")
plt.close(fig)


# ---------------------------------------------------------------------------
# S2: full numerical outer solver on conditioned synthetic partition WDPs.
# ---------------------------------------------------------------------------
bracket_rows = []
partition_instances = []
for inst in range(10):
    A, v = gen_partition(RNG, n_items=5, bids_per_item=3)
    Pstar, zlp = solve_rejection_lp(A, v)
    out = scwdp_outer(A, v, eps_abs=1e-4 * float(v.sum()), max_inner=20000)
    partition_instances.append((A, v, Pstar, out))
    for h in out["history"]:
        bracket_rows.append({
            "instance": inst, "outer": h["outer"],
            "width_over_W": h["width"] / out["W"],
            "theory_envelope": (2.0 / 3.0) ** h["outer"],
            "L": h["L"], "U": h["U"], "Pstar": Pstar,
            "event": h["event"], "inner_iters": h["inner_iters"],
            "invariant_holds": (h["L"] <= Pstar + 1e-9 and Pstar <= h["U"] + 1e-9),
        })
bracket_df = pd.DataFrame(bracket_rows)
bracket_df.to_csv(DATA / "bracket_contraction.csv", index=False)

fig, ax = plt.subplots(figsize=(6.4, 4.4))
for inst, grp in bracket_df.groupby("instance"):
    ax.semilogy(grp["outer"], grp["width_over_W"], linewidth=1.0, alpha=0.48)
Jmax = int(bracket_df["outer"].max())
js = np.arange(1, Jmax + 1)
ax.semilogy(js, (2.0/3.0) ** js, linestyle="--", linewidth=2.0,
            label=r"Theorem envelope $(2/3)^J$")
ax.set_xlabel("Completed outer iteration $J$")
ax.set_ylabel(r"Normalized bracket width $(U_J-L_J)/W$")
ax.set_title("Certified outer bracket contraction")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "figS2_bracket.png", dpi=220)
fig.savefig(FIG / "figS2_bracket.pdf")
plt.close(fig)


# ---------------------------------------------------------------------------
# S3: finite-tolerance exact-recovery geometry for unique integral LP optima.
# ---------------------------------------------------------------------------
unique_instances = []
trials = 0
while len(unique_instances) < 30 and trials < 500:
    A, v = gen_overlap(RNG, n=5, m=10, kmax=3)
    ULP, xlp = solve_acceptance_lp(A, v)
    if is_integral(xlp):
        xstar = np.rint(xlp).astype(int)
        zstar = 1.0 - xstar
        Pstar = float(v @ zstar)
        gamma, gw = epsilon0_gamma(A, v, zstar)
        if np.isfinite(gamma) and gamma > 1e-7:
            unique_instances.append((A, v, xstar, zstar, Pstar, gamma))
    trials += 1
if len(unique_instances) < 30:
    raise RuntimeError("Could not generate enough unique integral LP instances")

ratios = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00, 1.25, 2.0, 4.0])
recovery_rows = []
for inst, (A, v, xstar, zstar, Pstar, gamma) in enumerate(unique_instances):
    Vstar = float(v @ xstar)
    for rho in ratios:
        eps = float(rho * gamma)
        dist, witness = max_inf_distance_under_gap(A, v, zstar, Pstar, eps)
        if witness is None:
            raise RuntimeError("Auxiliary distance LP unexpectedly infeasible")
        xrec, vrec = fractional_priority_recovery(A, v, witness)
        recovery_rows.append({
            "instance": inst, "rho": rho, "epsilon": eps, "epsilon0": gamma,
            "max_inf_distance": dist,
            "recovery_exact": abs(vrec - Vstar) <= 1e-7,
            "recovered_value": vrec, "Vstar": Vstar,
        })
recovery_df = pd.DataFrame(recovery_rows)
recovery_df.to_csv(DATA / "finite_tolerance_recovery.csv", index=False)
agg = recovery_df.groupby("rho").agg(
    median_distance=("max_inf_distance", "median"),
    max_distance=("max_inf_distance", "max"),
    min_distance=("max_inf_distance", "min"),
    exact_rate=("recovery_exact", "mean")
).reset_index()
agg.to_csv(DATA / "finite_tolerance_recovery_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(agg["rho"], agg["max_distance"], marker="o", label="Maximum over 30 instances")
ax.plot(agg["rho"], agg["median_distance"], marker="s", label="Median")
ax.axhline(0.25, linestyle="--", linewidth=1.4, label=r"Recovery neighborhood $1/4$")
ax.axvline(1.0, linestyle=":", linewidth=1.4, label=r"$\varepsilon=\varepsilon_0$")
ax.set_xscale("symlog", linthresh=0.1)
ax.set_xlabel(r"Allowed LP objective excess $\varepsilon/\varepsilon_0$")
ax.set_ylabel(r"Worst feasible $\ell_\infty$ distance from $z^{\rm LP}$")
ax.set_title("Finite-tolerance exact-recovery geometry")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "figS3_exact_recovery.png", dpi=220)
fig.savefig(FIG / "figS3_exact_recovery.pdf")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(agg["rho"], 100.0 * agg["exact_rate"], marker="o")
ax.axvline(1.0, linestyle=":", linewidth=1.4, label=r"$\varepsilon=\varepsilon_0$")
ax.set_xscale("symlog", linthresh=0.1)
ax.set_ylim(-3, 103)
ax.set_xlabel(r"Allowed LP objective excess $\varepsilon/\varepsilon_0$")
ax.set_ylabel("Exact fractional-priority recovery rate (%)")
ax.set_title("Exact recovery under adversarial feasible witnesses")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "figS3b_recovery_rate.png", dpi=220)
fig.savefig(FIG / "figS3b_recovery_rate.pdf")
plt.close(fig)


# ---------------------------------------------------------------------------
# S4: retained-incumbent anytime dominance + online gap certificate.
# Fixed-seed representative overlapping instance with a strict recovery gain.
# ---------------------------------------------------------------------------
rep_rng = np.random.default_rng(777)
rep = None
for idx in range(20):
    A, v = gen_overlap(rep_rng, n=5, m=10, kmax=3)
    out = scwdp_outer(A, v, eps_abs=1e-3 * float(v.sum()), max_inner=10000)
    Vstar, xstar = solve_wdp_milp(A, v)
    if out["best"] > out["greedy"] + 1e-8:
        rep = (idx, A, v, out, Vstar)
        break
if rep is None:
    raise RuntimeError("Could not find representative recovery-improvement instance")
idx, A, v, out, Vstar = rep
W = float(v.sum())
trace_rows = []
# Initial safeguard checkpoint before any outer iteration. With L_0=0,
# Corollary 1 gives the computable bound 1-V_G/W.
init_true = (Vstar - float(out["greedy"])) / Vstar
init_cert = 1.0 - float(out["greedy"]) / W
trace_rows.append({
    "outer": 0, "incumbent": float(out["greedy"]),
    "greedy": float(out["greedy"]), "true_gap": init_true,
    "certificate_gap": init_cert, "L": 0.0, "U": W,
})
for h in out["history"]:
    cert = 1.0 - h["incumbent"] / (W - h["L"])
    cert = max(0.0, cert)
    truegap = (Vstar - h["incumbent"]) / Vstar
    trace_rows.append({
        "outer": h["outer"], "incumbent": h["incumbent"],
        "greedy": h["greedy"], "true_gap": truegap,
        "certificate_gap": cert, "L": h["L"], "U": h["U"],
    })
trace_df = pd.DataFrame(trace_rows)
trace_df.to_csv(DATA / "anytime_gap_certificate.csv", index=False)

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.semilogy(trace_df["outer"], np.maximum(trace_df["certificate_gap"], 1e-6),
            marker="o", label="Computable gap bound")
ax.semilogy(trace_df["outer"], np.maximum(trace_df["true_gap"], 1e-6),
            marker="s", label="True exact-WDP gap")
greedy_gap = (Vstar - float(out["greedy"])) / Vstar
ax.axhline(max(greedy_gap, 1e-6), linestyle="--", linewidth=1.4,
           label="Frozen greedy safeguard gap")
ax.set_xlabel("Completed outer iteration")
ax.set_ylabel("Relative value gap")
ax.set_title("Anytime safeguard and online gap certificate")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "figS4_gap_certificate.png", dpi=220)
fig.savefig(FIG / "figS4_gap_certificate.pdf")
plt.close(fig)

# Summary for supplement and regression checks.
summary = {
    "seed": SEED,
    "support_samples": len(support_df),
    "support_max_objective_difference": float(np.abs(support_df["objective_difference"]).max()),
    "support_min_thresholded_margin": float(support_df["thresholded_min_margin"].min()),
    "repair_samples": len(repair_df),
    "repair_max_theta_minus_residual": float((repair_df["theta"] - repair_df["residual_inf"]).max()),
    "repair_min_postrepair_margin": float(repair_df["postrepair_min_margin"].min()),
    "bracket_instances": int(bracket_df["instance"].nunique()),
    "bracket_all_invariants": bool(bracket_df["invariant_holds"].all()),
    "bracket_max_envelope_excess": float((bracket_df["width_over_W"] - bracket_df["theory_envelope"]).max()),
    "unique_integral_instances": len(unique_instances),
    "recovery_exact_rate_below_epsilon0": float(recovery_df[recovery_df["rho"] < 1.0]["recovery_exact"].mean()),
    "recovery_max_distance_below_epsilon0": float(recovery_df[recovery_df["rho"] < 1.0]["max_inf_distance"].max()),
    "representative_index": idx,
    "representative_greedy_gap": float(greedy_gap),
    "representative_final_true_gap": float(trace_df["true_gap"].iloc[-1]),
    "certificate_never_below_true_gap": bool((trace_df["certificate_gap"] + 1e-10 >= trace_df["true_gap"]).all()),
}
pd.DataFrame([summary]).to_csv(DATA / "summary.csv", index=False)
print(pd.Series(summary).to_string())
