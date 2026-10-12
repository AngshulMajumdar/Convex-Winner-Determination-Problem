"""Reference Python implementation of the SC-WDP components used only for
synthetic theorem-support experiments in the supplementary material.

This code is intentionally small and transparent. It is not the production
solver used for the main empirical benchmark.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds


def rejection_data(A: np.ndarray):
    m = A.shape[1]
    b = A @ np.ones(m) - np.ones(A.shape[0])
    return b


def solve_rejection_lp(A, v):
    b = rejection_data(A)
    res = linprog(v, A_ub=-A, b_ub=-b, bounds=[(0.0, 1.0)] * len(v), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.fun), np.asarray(res.x)


def solve_acceptance_lp(A, v):
    res = linprog(-v, A_ub=A, b_ub=np.ones(A.shape[0]), bounds=[(0.0, 1.0)] * len(v), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return float(-res.fun), np.asarray(res.x)


def solve_wdp_milp(A, v):
    n, m = A.shape
    con = LinearConstraint(A, -np.inf * np.ones(n), np.ones(n))
    res = milp(c=-v, integrality=np.ones(m), bounds=Bounds(np.zeros(m), np.ones(m)),
               constraints=con, options={"time_limit": 30.0})
    if not res.success:
        raise RuntimeError(res.message)
    x = np.rint(res.x).astype(int)
    return float(v @ x), x


def project_budget(q, v, tau, tol=1e-12, max_iter=80):
    """Projection onto {0<=z<=1, v^T z<=tau} using the scalar multiplier root."""
    c = np.clip(q, 0.0, 1.0)
    if float(v @ c) <= tau + tol * max(1.0, tau):
        return c
    lo = 0.0
    hi = max(1.0, float(np.max(np.maximum(q, 0.0) / v))) + 1.0
    for _ in range(max_iter):
        mu = 0.5 * (lo + hi)
        c = np.clip(q - mu * v, 0.0, 1.0)
        if float(v @ c) > tau:
            lo = mu
        else:
            hi = mu
        if hi - lo <= tol * max(1.0, hi):
            break
    return np.clip(q - hi * v, 0.0, 1.0)


def residual(A, b, z):
    r = np.maximum(b - A @ z, 0.0)
    return r, 0.5 * float(r @ r)


def lower_certificate(A, b, v, tau, z):
    """Maximize D_tau(y,lambda) over lambda for y = [b-Az]_+."""
    y = np.maximum(b - A @ z, 0.0)
    aty = A.T @ y
    q = aty / v
    candidates = np.concatenate(([0.0], q))
    const = float(b @ y - 0.5 * y @ y)
    # The concave piecewise-linear maximum occurs at 0 or a breakpoint q_j.
    vals = const - candidates * tau - np.maximum(
        aty[None, :] - candidates[:, None] * v[None, :], 0.0
    ).sum(axis=1)
    idx = int(np.argmax(vals))
    return float(vals[idx]), float(candidates[idx])


def repair(A, b, v, z):
    r = np.maximum(b - A @ z, 0.0)
    theta = float(np.max(r / (1.0 + r))) if len(r) else 0.0
    zr = z + theta * (1.0 - z)
    return zr, theta, r


def _greedy_from_order(A, v, order):
    used = np.zeros(A.shape[0], dtype=bool)
    x = np.zeros(len(v), dtype=int)
    for j in order:
        items = np.flatnonzero(A[:, j] > 0.5)
        if not used[items].any():
            x[j] = 1
            used[items] = True
    return x, float(v @ x)


def descending_value_greedy(A, v):
    order = np.lexsort((np.arange(len(v)), -v))
    return _greedy_from_order(A, v, order)


def fractional_priority_recovery(A, v, z):
    xfrac = 1.0 - z
    order = np.lexsort((np.arange(len(v)), -v, -xfrac))
    return _greedy_from_order(A, v, order)


def spg_step(A, b, v, tau, z, prev_z=None, prev_g=None,
             alpha0=1.0, alpha_min=1e-10, alpha_max=1e10,
             beta=0.5, c=1e-4):
    r = np.maximum(b - A @ z, 0.0)
    g = -A.T @ r
    alpha = alpha0
    if prev_z is not None and prev_g is not None:
        s = z - prev_z
        y = g - prev_g
        den = float(s @ y)
        if den > 0.0:
            alpha = float((s @ s) / den)
    alpha = min(alpha_max, max(alpha_min, alpha))
    p = project_budget(z - alpha * g, v, tau)
    d = p - z
    eta = float(np.max(np.abs(d)))
    f0 = 0.5 * float(r @ r)
    gd = float(g @ d)
    lam = 1.0
    nls = 0
    while True:
        ztrial = z + lam * d
        _, ftrial = residual(A, b, ztrial)
        nls += 1
        if ftrial <= f0 + c * lam * gd + 1e-15:
            break
        lam *= beta
        if nls > 100:
            raise RuntimeError("Armijo line search failed")
    return z + lam * d, r, g, eta, alpha, lam, nls


def scwdp_outer(A, v, eps_abs, max_outer=200, max_inner=50000,
                cert_positive_tol=1e-12, feasibility_tol=1e-10):
    """Numerical realization of Algorithms 1-2 for synthetic diagnostics.

    The exact theorem is in real arithmetic. Here positive lower-certificate and
    repaired-feasibility events are evaluated in double precision at stated
    tolerances; all bracket invariants are independently checked against the LP
    optimum by the experiment driver.
    """
    n, m = A.shape
    b = rejection_data(A)
    W = float(v.sum())
    xg, vg = descending_value_greedy(A, v)
    xbest, best = xg.copy(), vg
    L, U, zU = 0.0, W, np.ones(m)
    warm = None
    history = []
    recovery_events = [{"checkpoint": 0, "raw": vg, "retained": vg, "kind": "greedy"}]
    total_inner = 0

    for outer_idx in range(max_outer):
        if U - L <= eps_abs:
            break
        oldL, oldU = L, U
        w = U - L
        tau = L + w / 3.0
        bartau = L + 2.0 * w / 3.0
        z = np.zeros(m) if warm is None else project_budget(warm, v, tau)
        prev_z = None
        prev_g = None
        event = None

        for inner_idx in range(max_inner):
            znew, r, g, eta, alpha, lam, nls = spg_step(
                A, b, v, tau, z, prev_z, prev_g
            )
            xf, vf = fractional_priority_recovery(A, v, znew)
            if vf > best + 1e-12:
                best = vf
                xbest = xf.copy()
            recovery_events.append({"checkpoint": len(recovery_events), "raw": vf,
                                    "retained": best, "kind": "fractional"})

            low, lowlam = lower_certificate(A, b, v, tau, z)
            zr, theta, rr = repair(A, b, v, z)
            repcost = float(v @ zr)
            total_inner += 1

            if low > cert_positive_tol:
                L = tau
                warm = znew.copy()
                event = "lower"
                break

            if repcost <= bartau + 1e-12 and float(np.min(A @ zr - b)) >= -feasibility_tol:
                U = repcost
                zU = zr.copy()
                warm = znew.copy()
                xu, vu = fractional_priority_recovery(A, v, zU)
                if vu > best + 1e-12:
                    best = vu
                    xbest = xu.copy()
                recovery_events.append({"checkpoint": len(recovery_events), "raw": vu,
                                        "retained": best, "kind": "upper-witness"})
                event = "upper"
                break

            prev_z, prev_g = z.copy(), g.copy()
            z = znew

        if event is None:
            raise RuntimeError("No certified event fired within max_inner")

        history.append({
            "outer": outer_idx + 1, "oldL": oldL, "oldU": oldU,
            "L": L, "U": U, "width": U - L, "event": event,
            "tau": tau, "bartau": bartau, "incumbent": best,
            "greedy": vg, "inner_iters": inner_idx + 1,
        })

    return {
        "xbest": xbest, "best": best, "greedy": vg,
        "L": L, "U": U, "zU": zU, "W": W, "b": b,
        "history": history, "recovery_events": recovery_events,
        "total_inner": total_inner,
    }


def max_inf_distance_under_gap(A, v, zstar, Pstar, epsilon):
    """Maximize ||z-zstar||_inf over LP-feasible z with v^Tz<=Pstar+epsilon.

    zstar is binary, so the coordinatewise extrema are linear programs.
    Returns the maximum distance and one attaining witness.
    """
    b = rejection_data(A)
    m = len(v)
    A_ub_base = np.vstack([-A, v[None, :]])
    b_ub_base = np.concatenate([-b, [Pstar + epsilon]])
    best_dist = -1.0
    best_z = None
    for j in range(m):
        c = np.zeros(m)
        if zstar[j] < 0.5:
            c[j] = -1.0  # maximize z_j
            res = linprog(c, A_ub=A_ub_base, b_ub=b_ub_base,
                          bounds=[(0.0, 1.0)] * m, method="highs")
            if res.success:
                dist = float(res.x[j])
        else:
            c[j] = 1.0   # minimize z_j, maximize 1-z_j
            res = linprog(c, A_ub=A_ub_base, b_ub=b_ub_base,
                          bounds=[(0.0, 1.0)] * m, method="highs")
            if res.success:
                dist = float(1.0 - res.x[j])
        if res.success and dist > best_dist:
            best_dist = dist
            best_z = np.asarray(res.x)
    return best_dist, best_z


def epsilon0_gamma(A, v, zstar, threshold=0.25):
    """Compute the theorem's objective-separation gamma by 2m auxiliary LPs."""
    b = rejection_data(A)
    Pstar = float(v @ zstar)
    m = len(v)
    best = np.inf
    witness = None
    for j in range(m):
        bounds = [(0.0, 1.0)] * m
        if zstar[j] < 0.5:
            bounds[j] = (threshold, 1.0)
        else:
            bounds[j] = (0.0, 1.0 - threshold)
        res = linprog(v, A_ub=-A, b_ub=-b, bounds=bounds, method="highs")
        if res.success and res.fun < best:
            best = float(res.fun)
            witness = np.asarray(res.x)
    if not np.isfinite(best):
        return np.inf, None
    return float(best - Pstar), witness
