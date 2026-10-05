"""
Reference solution for the IFToMM benchmark problem "Cart-pole with a light cart".

Writes, next to this file:
  results.txt          reference trajectory, 1001 samples, the file uploaded to the library
  convergence.txt      tolerance ladder: error of each run against the reference
  convergence.tex      the same table as a LaTeX fragment for the description PDF
  schematic.png        system drawing (the PNG the submission form asks for)
  fig_trajectory.png   x and theta against time
  fig_spike.png        thetadot and det M through the first vertical crossing
  fig_convergence.png  error against tolerance

EQUATIONS OF MOTION
-------------------
Cart of mass M at x on a frictionless horizontal track; massless pole of length
l pinned to the cart, point mass m at its tip, theta measured from the downward
vertical. With q = (x, theta), the Lagrangian

    L = 1/2 (M+m) xd^2 + m l xd thd cos th + 1/2 m l^2 thd^2 + m g l cos th

gives M(q) qdd = F with det M = m l^2 (M + m sin^2 th). The 2x2 system is solved
by hand rather than numerically:

    xdd  =  m sin th (l thd^2 + g cos th)            / (M + m sin^2 th)
    thdd = -sin th ((M+m) g + m l thd^2 cos th)      / (l (M + m sin^2 th))

The only division is by M + m sin^2 th, which is a sum of non-negative terms, so
no cancellation occurs however small M becomes. Solving M(q) qdd = F with a
general linear solver instead loses about two and a half digits on this
problem; that comparison is reproduced in convergence.txt.

INDEPENDENT CHECK
-----------------
Horizontal momentum p and energy E are conserved. Starting from rest, p = 0 and
E = -m g l cos th0, which reduces the motion to the quadrature

    thd^2 = 2 g (M+m) (cos th - cos th0) / (l (M + m sin^2 th)),
    x     = m l (sin th0 - sin th) / (M+m).

Inverting t(th) with mpmath at 30 digits gives the trajectory without any time
integration. The integrated reference is compared against it.
"""

from __future__ import annotations

import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp


def mpmath_version():
    import mpmath
    return mpmath.__version__

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- problem data
m, l, g = 1.0, 1.0, 9.81
M = 1.0e-6
TH0 = math.pi / 2
Y0 = np.array([0.0, 0.0, TH0, 0.0])          # x, xd, th, thd
T_END = 10.0
DT_OUT = 0.01
T_OUT = np.round(np.arange(0.0, T_END + DT_OUT / 2, DT_OUT), 10)   # 1001 samples
THRESHOLD = 1.0e-6

REF_RTOL, REF_ATOL = 1e-13, 1e-15
REPS = 5          # CPU time is the fastest of this many identical runs


# ------------------------------------------------------------------- dynamics
def rhs(_t, y):
    x, xd, th, thd = y
    s, c = math.sin(th), math.cos(th)
    d = M + m * s * s
    return [xd,
            m * s * (l * thd * thd + g * c) / d,
            thd,
            -s * ((M + m) * g + m * l * thd * thd * c) / (l * d)]


def rhs_linsolve(_t, y):
    """Same dynamics, accelerations from a general solve of M(q) qdd = F."""
    x, xd, th, thd = y
    s, c = math.sin(th), math.cos(th)
    A = np.array([[M + m, m * l * c], [m * l * c, m * l * l]])
    F = np.array([m * l * thd * thd * s, -m * g * l * s])
    xdd, thdd = np.linalg.solve(A, F)
    return [xd, xdd, thd, thdd]


def energy(Y):
    x, xd, th, thd = Y
    T = 0.5 * (M + m) * xd**2 + m * l * xd * thd * np.cos(th) + 0.5 * m * l**2 * thd**2
    return T - m * g * l * np.cos(th)


def run(method, rtol, atol, f=rhs, reps=1):
    best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        sol = solve_ivp(f, (0.0, T_END), Y0, method=method, rtol=rtol, atol=atol,
                        t_eval=T_OUT)
        best = min(best, time.perf_counter() - t0)
    if not sol.success:
        raise RuntimeError(f"{method} rtol={rtol:g}: {sol.message}")
    return sol, best


def error(Y, Yref):
    """Benchmark error: RMS over samples of the position errors in x/l and theta."""
    ex = (Y[0] - Yref[0]) / l
    et = Y[2] - Yref[2]
    return float(np.sqrt(np.mean(ex**2 + et**2)))


# ------------------------------------------------------- quadrature reference
def quadrature_trajectory():
    import mpmath as mp
    mp.mp.dps = 30
    Mm, mm, lm, gm = mp.mpf(M), mp.mpf(m), mp.mpf(l), mp.mpf(g)
    th0 = mp.pi / 2

    def speed_w(w):         # |thd| at th = th0 - w^2, written without cancellation:
        th = th0 - w * w    # cos th - cos th0 = 2 sin(th0 - w^2/2) sin(w^2/2)
        dc = 2 * mp.sin(th0 - w * w / 2) * mp.sin(w * w / 2)
        return mp.sqrt(2 * gm * (Mm + mm) * dc / (lm * (Mm + mm * mp.sin(th) ** 2)))

    def speed(th):          # |thd| as a function of th
        return speed_w(mp.sqrt(max(th0 - th, 0)))

    def t_of(th):           # time to fall from th0 to th, 0 <= th <= th0
        # substitute th = th0 - w^2 to remove the turning-point singularity;
        # 2w / speed_w(w) tends to a finite limit as w -> 0
        wmax = mp.sqrt(th0 - th)
        brk = [mp.sqrt(th0 - b) for b in (mp.mpf("0.1"), mp.mpf("1e-2"),
                                          mp.mpf("1e-3"), mp.mpf("1e-4"))]
        pts = [mp.mpf(0)] + [w for w in brk if w < wmax] + [wmax]
        return mp.quad(lambda w: 2 * w / speed_w(w), pts)

    Q = t_of(mp.mpf(0))     # quarter period

    def theta_desc(tau, guess):
        """theta on the first quarter, given elapsed time tau in [0, Q]."""
        if tau <= 0:
            return th0
        th = mp.mpf(guess)
        for _ in range(30):  # Newton on t(th) - tau, with dt/dth = -1/|thd|
            step = (t_of(th) - tau) * speed(th)
            new = th + step
            if new >= th0:            # keep the iterate inside (0, th0]
                new = (th + th0) / 2
            elif new <= 0:
                new = th / 2
            step, th = new - th, new
            if abs(step) < mp.mpf("1e-24"):
                break
        else:
            raise RuntimeError(f"Newton did not converge at tau={tau}")
        return th

    # initial guesses from a quick numerical solve
    guess_sol = solve_ivp(rhs, (0.0, T_END), Y0, method="DOP853", rtol=1e-10,
                          atol=1e-12, t_eval=T_OUT)
    out = np.empty((4, len(T_OUT)))
    for i, t in enumerate(T_OUT):
        tm = mp.mpf(str(t))
        k = int(mp.floor(tm / Q))
        r = tm - k * Q                          # time into current quarter
        quarter = k % 4
        if quarter == 0:
            tau, sgn_th, sgn_v = r, 1, -1
        elif quarter == 1:
            tau, sgn_th, sgn_v = Q - r, -1, -1
        elif quarter == 2:
            tau, sgn_th, sgn_v = r, -1, 1
        else:
            tau, sgn_th, sgn_v = Q - r, 1, 1
        th = theta_desc(tau, max(abs(guess_sol.y[2, i]), 1e-12)) * sgn_th
        thd = sgn_v * speed(abs(th)) if abs(th) < th0 else mp.mpf(0)
        x = mm * lm * (mp.sin(th0) - mp.sin(th)) / (Mm + mm)
        xd = -mm * lm * mp.cos(th) * thd / (Mm + mm)
        out[:, i] = [float(x), float(xd), float(th), float(thd)]
    return out, float(Q)


# -------------------------------------------------------------------- figures
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)


def draw_schematic(path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, Arc, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=200)
    ax.set_aspect("equal")
    ax.axis("off")
    # track
    ax.plot([-1.6, 2.2], [0, 0], color=INK, lw=1.2)
    for xh in np.arange(-1.55, 2.2, 0.15):
        ax.plot([xh, xh - 0.1], [0, -0.1], color=MUTED, lw=0.6)
    # cart
    cx, cw, ch = 0.0, 0.5, 0.26
    ax.add_patch(Rectangle((cx - cw / 2, 0.02), cw, ch, fc="#f3f2ec", ec=INK, lw=1.2))
    ax.text(cx - 0.14, 0.15, "M", ha="center", va="center", fontsize=11, color=INK)
    pivot = np.array([cx + 0.08, 0.15])
    th = 0.75
    tip = pivot + 1.25 * np.array([math.sin(th), -math.cos(th)])
    ax.plot([pivot[0], tip[0]], [pivot[1], tip[1]], color=INK, lw=1.6)
    ax.add_patch(Circle(pivot, 0.035, fc="white", ec=INK, lw=1.0, zorder=5))
    ax.add_patch(Circle(tip, 0.09, fc=BLUE, ec=INK, lw=1.0, zorder=5))
    ax.text(tip[0] + 0.14, tip[1] - 0.02, "m", fontsize=11, color=INK, va="center")
    ax.text(*(pivot + 0.62 * np.array([math.sin(th), -math.cos(th)]) + [0.1, 0.06]),
            "l", fontsize=11, color=INK, style="italic")
    # vertical reference and angle
    ax.plot([pivot[0], pivot[0]], [pivot[1], pivot[1] - 1.1], color=MUTED, lw=0.8, ls="--")
    ax.add_patch(Arc(pivot, 0.9, 0.9, theta1=-90, theta2=-90 + math.degrees(th),
                     color=INK, lw=0.9))
    ax.text(pivot[0] + 0.14, pivot[1] - 0.6, r"$\theta$", fontsize=12, color=INK)
    # x coordinate
    ax.plot([-1.4, -1.4], [-0.05, 0.55], color=MUTED, lw=0.8)
    ax.add_patch(FancyArrowPatch((-1.4, 0.45), (pivot[0], 0.45), arrowstyle="-|>",
                                 mutation_scale=10, color=INK, lw=0.9))
    ax.text(-0.66, 0.52, r"$x$", fontsize=12, color=INK, ha="center")
    # gravity
    ax.add_patch(FancyArrowPatch((1.95, 0.6), (1.95, 0.1), arrowstyle="-|>",
                                 mutation_scale=10, color=INK, lw=0.9))
    ax.text(2.03, 0.35, r"$g$", fontsize=12, color=INK)
    ax.set_xlim(-1.7, 2.3)
    ax.set_ylim(-1.25, 0.7)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_trajectory(Y, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 3.8), dpi=200, sharex=True)
    axes[0].plot(T_OUT, Y[0], color=BLUE, lw=1.4)
    axes[0].set_ylabel(r"$x$ (m)", color=INK)
    axes[1].plot(T_OUT, Y[2], color=BLUE, lw=1.4)
    axes[1].set_ylabel(r"$\theta$ (rad)", color=INK)
    axes[1].set_xlabel("t (s)", color=INK)
    for ax in axes:
        _style(ax)
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_spike(dense, t_cross, path):
    import matplotlib.pyplot as plt
    tt = np.linspace(t_cross - 1e-5, t_cross + 1e-5, 4001)
    Yd = dense(tt)
    det_ratio = (M + m * np.sin(Yd[2]) ** 2) / (M + m)   # det M / max det M
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 3.8), dpi=200, sharex=True)
    us = (tt - t_cross) * 1e6
    axes[0].plot(us, Yd[3], color=BLUE, lw=1.4)
    axes[0].set_ylabel(r"$\dot\theta$ (rad/s)", color=INK)
    axes[1].semilogy(us, det_ratio, color=ORANGE, lw=1.4)
    axes[1].set_ylabel(r"$\det M\;/\;\max\,\det M$", color=INK)
    axes[1].set_xlabel(f"time from first vertical crossing, t = {t_cross:.6f} s (µs)",
                       color=INK)
    for ax in axes:
        _style(ax)
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_convergence(rows, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=200)
    series = [("DOP853", "closed form", BLUE, "o"),
              ("Radau", "closed form", ORANGE, "s"),
              ("DOP853", "linear solve", AQUA, "^")]
    for meth, form, col, mk in series:
        pts = [(r["rtol"], r["e_quad"]) for r in rows
               if r["method"] == meth and r["form"] == form]
        if pts:
            xs, ys = zip(*pts)
            ax.loglog(xs, ys, color=col, lw=1.4, marker=mk, ms=6,
                      label=f"{meth}, {form}")
    ax.axhline(THRESHOLD, color=MUTED, lw=1.0, ls="--")
    ax.text(1e-11, THRESHOLD * 1.6, "acceptance threshold", color=MUTED, fontsize=8,
            ha="center")
    ax.invert_xaxis()
    ax.set_xlabel("relative tolerance (rtol)", color=INK)
    ax.set_ylabel("error vs. quadrature solution", color=INK)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------- main
def main():
    print("integrating reference ...")
    ref_sol, ref_cpu = run("DOP853", REF_RTOL, REF_ATOL, reps=REPS)
    Yref = ref_sol.y
    dense = solve_ivp(rhs, (0.0, T_END), Y0, method="DOP853", rtol=REF_RTOL,
                      atol=REF_ATOL, dense_output=True).sol

    print("computing quadrature solution (mpmath, 30 digits) ...")
    t0 = time.perf_counter()
    Yq, Q = quadrature_trajectory()
    print(f"  done in {time.perf_counter() - t0:.0f} s, quarter period {Q:.12f} s")
    e_ref_quad = error(Yref, Yq)
    print(f"  reference vs quadrature: e = {e_ref_quad:.2e}")

    rows = []
    ladder = [("DOP853", "closed form", rhs, [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                                              1e-10, 1e-11, 1e-12, 1e-13]),
              ("Radau", "closed form", rhs, [1e-6, 1e-7, 1e-8, 1e-9, 1e-10,
                                             1e-11, 1e-12, 1e-13]),
              ("DOP853", "linear solve", rhs_linsolve, [1e-6, 1e-8, 1e-10,
                                                        1e-12, 1e-13])]
    for meth, form, f, rtols in ladder:
        for rt in rtols:
            if (meth, f, rt, rt * 1e-2) == ("DOP853", rhs, REF_RTOL, REF_ATOL):
                sol, cpu = ref_sol, ref_cpu     # same run as the reference; one timing
            else:
                sol, cpu = run(meth, rt, rt * 1e-2, f=f, reps=REPS)
            E = energy(sol.y)
            rows.append(dict(method=meth, form=form, rtol=rt, nfev=sol.nfev, cpu=cpu,
                             e_ref=error(sol.y, Yref), e_quad=error(sol.y, Yq),
                             dE=float(np.max(np.abs(E - E[0])))))
            r = rows[-1]
            print(f"  {meth:6s} {form:12s} rtol={rt:7.0e}  e_quad={r['e_quad']:.2e}"
                  f"  dE={r['dE']:.1e}  nfev={r['nfev']:6d}  cpu={cpu:.3f}s")

    # each SciPy solver at its default tolerances (rtol 1e-3, atol 1e-6)
    defaults = []
    for meth in ("RK45", "RK23", "DOP853", "LSODA", "BDF", "Radau"):
        sol = solve_ivp(rhs, (0.0, T_END), Y0, method=meth, t_eval=T_OUT)
        defaults.append(dict(method=meth, success=bool(sol.success), nfev=sol.nfev,
                             e_quad=error(sol.y, Yq)))
        print(f"  {meth:6s} default tolerances: success={sol.success}"
              f"  e_quad={defaults[-1]['e_quad']:.3g}")

    # ---- results.txt
    E = energy(Yref)
    data = np.column_stack([T_OUT, Yref[0], Yref[2], Yref[1], Yref[3], E - E[0]])
    header = (f"{'Time':>23s}{'X':>23s}{'THETA':>23s}{'XDOT':>23s}"
              f"{'THETADOT':>23s}{'ENERGY':>23s}")
    np.savetxt(HERE / "results.txt", data, fmt="%23.15E", header=header,
               comments="")

    # ---- crossings of theta = 0 from the dense reference
    fine = np.linspace(0, T_END, 2_000_001)
    thf = dense(fine)[2]
    idx = np.nonzero(np.sign(thf[1:]) != np.sign(thf[:-1]))[0]
    crossings = []
    for i in idx:
        a, b = fine[i], fine[i + 1]
        for _ in range(60):                     # bisection to machine precision
            mid = 0.5 * (a + b)
            if np.sign(dense(mid)[2]) == np.sign(dense(a)[2]):
                a = mid
            else:
                b = mid
        crossings.append(0.5 * (a + b))
    exact_cross = [Q * (2 * k + 1) for k in range(len(crossings))]
    worst_cross = max(abs(c - e) for c, e in zip(crossings, exact_cross))
    # peak |thd| is reached at theta = 0; from the energy integral
    peak = math.sqrt(2 * g * (M + m) * (1 - math.cos(TH0)) / (l * M))
    # duration of the spike: time spent where |thd| exceeds half its peak
    from scipy.integrate import quad

    def inv_speed(th):
        return math.sqrt(l * (M + m * math.sin(th) ** 2)
                         / (2 * g * (M + m) * (math.cos(th) - math.cos(TH0))))

    a, b = 0.0, 0.1
    for _ in range(200):
        mid = 0.5 * (a + b)
        a, b = (mid, b) if 1.0 / inv_speed(mid) > peak / 2 else (a, mid)
    fwhm = 2 * quad(inv_speed, 0.0, a, epsabs=1e-16, epsrel=1e-12)[0]
    # the tails are long: |thd| 10 microseconds after a crossing
    a, b = 0.0, 0.5
    for _ in range(200):
        mid = 0.5 * (a + b)
        if quad(inv_speed, 0.0, mid, epsabs=1e-16, epsrel=1e-12)[0] < 1e-5:
            a = mid
        else:
            b = mid
    tail = 1.0 / inv_speed(a)

    # ---- convergence.txt
    with open(HERE / "convergence.txt", "w", encoding="utf-8") as fh:
        fh.write("Cart-pole with a light cart -- reference solution convergence\n")
        fh.write(f"M = {M:g} kg, m = {m:g} kg, l = {l:g} m, g = {g:g} m/s^2, "
                 f"theta0 = pi/2, T = {T_END:g} s, {len(T_OUT)} samples\n")
        fh.write(f"Reference: DOP853 closed form, rtol={REF_RTOL:g}, atol={REF_ATOL:g}, "
                 f"CPU {ref_cpu:.3f} s\n")
        fh.write(f"Reference vs 30-digit quadrature solution: e = {e_ref_quad:.3e}\n")
        fh.write(f"Quarter period (quadrature): {Q:.15f} s\n")
        fh.write(f"Vertical crossings: {len(crossings)}; worst timing error vs "
                 f"quadrature: {worst_cross:.2e} s\n")
        fh.write(f"Peak |thetadot|: {peak:.1f} rad/s at each crossing, full width "
                 f"at half maximum {fwhm:.3e} s, {tail:.0f} rad/s at 10 us\n\n")
        fh.write(f"{'method':8s}{'form':14s}{'rtol':>9s}{'atol':>9s}{'e_vs_quad':>12s}"
                 f"{'e_vs_ref':>12s}{'max|dE| (J)':>13s}{'nfev':>9s}{'CPU (s)':>9s}\n")
        for r in rows:
            fh.write(f"{r['method']:8s}{r['form']:14s}{r['rtol']:9.0e}"
                     f"{r['rtol'] * 1e-2:9.0e}{r['e_quad']:12.3e}{r['e_ref']:12.3e}"
                     f"{r['dE']:13.2e}{r['nfev']:9d}{r['cpu']:9.3f}\n")
        fh.write("\nSciPy solvers at their default tolerances (rtol 1e-3, atol 1e-6):\n")
        for d in defaults:
            fh.write(f"  {d['method']:8s}reported success: {str(d['success']):5s}"
                     f"  e_vs_quad = {d['e_quad']:.3g}  nfev = {d['nfev']}\n")
        fh.write(f"\nCPU (s) is the fastest of {REPS} identical runs, wall-clock, one thread.\n"
                 f"Python {platform.python_version()}, NumPy {np.__version__}, "
                 f"SciPy {scipy.__version__}, mpmath {mpmath_version()}\n"
                 f"{platform.platform()}, {platform.processor()}\n"
                 "Errors do not depend on the machine; at loose tolerances they can "
                 "differ between SciPy versions.\n")

    # ---- convergence.tex
    def sci(v):
        if v == 0:
            return "$0$"
        mant, ex = f"{v:.1e}".split("e")
        return f"${mant}\\times10^{{{int(ex)}}}$"

    with open(HERE / "convergence.tex", "w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n")
        fh.write("Integrator & Accelerations & rtol & $e$ vs.\\ quadrature & "
                 "$\\max|\\Delta E|$ (J) & Evaluations \\\\\n\\midrule\n")
        for r in rows:
            if r["form"] == "linear solve" and r["rtol"] not in (1e-8, 1e-13):
                continue
            if r["method"] == "Radau" and r["rtol"] not in (1e-8, 1e-10, 1e-13):
                continue
            if r["method"] == "DOP853" and r["form"] == "closed form" and \
                    r["rtol"] not in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-13):
                continue
            fh.write(f"{r['method']} & {r['form']} & $10^{{{round(math.log10(r['rtol']))}}}$ & "
                     f"{sci(r['e_quad'])} & {sci(r['dE'])} & {r['nfev']:,} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # ---- summary values for the PDF
    with open(HERE / "values.tex", "w", encoding="utf-8") as fh:
        fh.write(f"\\newcommand{{\\Equad}}{{{sci(e_ref_quad)}}}\n")
        fh.write(f"\\newcommand{{\\Qperiod}}{{{Q:.10f}}}\n")
        fh.write(f"\\newcommand{{\\Ncross}}{{{len(crossings)}}}\n")
        fh.write(f"\\newcommand{{\\Tcross}}{{{crossings[0]:.6f}}}\n")
        fh.write(f"\\newcommand{{\\CrossErr}}{{{sci(worst_cross)}}}\n")
        fh.write(f"\\newcommand{{\\Peak}}{{{peak:,.0f}}}\n")
        fh.write(f"\\newcommand{{\\Fwhm}}{{{fwhm * 1e6:.2f}}}\n")
        fh.write(f"\\newcommand{{\\RefCPU}}{{{ref_cpu:.2f}}}\n")
        fh.write(f"\\newcommand{{\\Tail}}{{{tail:.0f}}}\n")
        radau = [r for r in rows if r["method"] == "Radau" and r["rtol"] == 1e-13][0]
        fh.write(f"\\newcommand{{\\ERadau}}{{{sci(radau['e_quad'])}}}\n")
        rk = [d for d in defaults if d["method"] == "RK45"][0]
        fh.write(f"\\newcommand{{\\EDefaultRK}}{{{rk['e_quad']:.1f}}}\n")
        fh.write(f"\\newcommand{{\\EDefaultMin}}{{{min(d['e_quad'] for d in defaults):.2f}}}\n")
        fh.write(f"\\newcommand{{\\EDefaultMax}}{{{max(d['e_quad'] for d in defaults):.1f}}}\n")
        fh.write(f"\\newcommand{{\\NDefaultOK}}{{{sum(d['success'] for d in defaults)}}}\n")
        words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"]
        fh.write(f"\\newcommand{{\\NSolvers}}{{{words[len(defaults)]}}}\n")
        fh.write(f"\\newcommand{{\\Reps}}{{{words[REPS]}}}\n")
        fh.write(f"\\newcommand{{\\ScipyVersion}}{{{scipy.__version__}}}\n")
        fh.write(f"\\newcommand{{\\PythonVersion}}{{{platform.python_version()}}}\n")

    print("drawing figures ...")
    draw_schematic(HERE / "schematic.png")
    plot_trajectory(Yref, HERE / "fig_trajectory.png")
    plot_spike(dense, crossings[0], HERE / "fig_spike.png")
    plot_convergence(rows, HERE / "fig_convergence.png")
    print(f"crossings {len(crossings)}, worst timing error {worst_cross:.2e} s, "
          f"peak thd {peak:.0f} rad/s")
    print("done.")


if __name__ == "__main__":
    main()
