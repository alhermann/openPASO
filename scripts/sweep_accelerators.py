"""Re-measure the accelerator comparison the coupling knowledge serves.

WHY THIS FILE EXISTS AT ALL. The sentence in `coupling_knowledge` that makes
"aitken" both the default and the recommendation — "Aitken matched or beat a
constant theta almost everywhere, and in a quarter of those settings it
converged to the right interface value where the SAME constant theta diverged"
— came out of a 40-cell sweep run in commit 33689b2d. That sweep's harness was
never committed: the commit touched only `coupling_knowledge.py` and its test.
So when `feature/coupling-robustness` later replaced the PER-PARTICIPANT Aitken
theta with ONE global theta for the whole interface state, nothing could tell
anyone that the served numbers no longer described the shipped driver, and they
did not. Two tier-2 fixtures went red and stayed red for want of this file.

A measured claim needs the measurement to be re-runnable, or it decays into a
quotation. That is the whole point of this script.

WHAT IT MEASURES. For each (rho, theta) on the grid, the same split conduction
problem is coupled twice through the REGISTERED `couple` tool — once with
accelerator="aitken", once with "constant" — under an IDENTICAL max_iter and
tol. Both arms are the real shipped skfem participant scripts on non-matching
interface meshes (17 interface points against 13), staged exactly the way
couplinglib.probe_theta stages them, so this measures the driver as it ships
rather than an analytic stand-in for it. (The 2026-08-05 sweep used "an
analytic Dirichlet-Neumann participant pair reproducing the same linear map";
that is a second reason its numbers did not transfer.)

Per cell it records whether the run CONVERGED (residual < tol), how many
iterations it took, and the DEVIATION of the interface values from the
closed-form answer relative to it. The three are not the same question and the
claim needs all three:

  * `converged` alone cannot separate a run that found the answer from one that
    settled on the wrong fixed point;
  * the residual is normalised by the export magnitude, so a diverging run
    SATURATES it near a constant of order one instead of sending it to infinity;
  * the deviation grows as the amplification factor to the power of the
    iteration count with nothing to hide behind — but a run can land on the
    right value and still sit above tol, which is a real and common outcome for
    Aitken here and is reported separately rather than folded into either side.

RUNNING IT. Each cell is its own process, so one bad cell cannot take the grid
with it and an interrupted grid resumes:

    OPENPASO_REPO=<checkout> \
    PYTHONPATH=<checkout>/scripts/tier2_fixtures/coupling/_lib \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python scripts/sweep_accelerators.py --out /tmp/sweep --workers 8

PIN THE BLAS THREAD COUNT. The driver is bit-deterministic at a fixed thread
count — the same cell run three times reproduced its residual and its deviation
to every digit — but a DIVERGING Aitken run is not reproducible ACROSS thread
counts. Aitken's theta is a nonlinear function of the residual history and it is
clamped, so a last-bit difference in a reduction can change which adaptation
hits the clamp and from there the two runs are different trajectories. The
VERDICT (converged, landed, ran away) was stable across thread counts in every
cell checked; the digits of a runaway were not. Report verdicts, not digits, for
cells that ran away.

`--report` re-reads a finished grid and prints it, so the numbers can be checked
without paying for the runs again.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "scripts" / "tier2_fixtures" / "coupling" / "_lib"

# The grid. rho spans the range the served sentence names (1/4 to 9) and
# includes both ratios the two accelerator fixtures sit at (4 and 6); theta is
# the knowledge's own coarse grid, 0.1 to 1.0 by 0.1.
RHOS = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 9.0]
THETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ACCELS = ["aitken", "constant"]

MAX_ITER = 300
TOL = 1e-4

# Deviation from the exact interface temperature, relative to it. The same two
# thresholds the two accelerator fixtures use, so the grid and the fixtures
# grade a run the same way.
SETTLED = 1e-3          # at or below this the run LANDED on the answer
RUNAWAY = 1.0           # above this it is not even the right order of magnitude

# Physics against the closed form, again the fixtures' own numbers.
T_ATOL = 0.05
Q_ATOL = 0.2
BALANCE_RTOL = 1e-2


# ── one cell ───────────────────────────────────────────────────────────────

def _quiet_stage(L, root, name, backend, edits):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return L.stage(root, name, backend, edits)


def run_cell(rho: float, theta: float, accelerator: str,
             max_iter: int = MAX_ITER, tol: float = TOL,
             dirichlet: str = "left", backend: str = "skfem",
             mesh_l=(16, 16), mesh_r=(14, 12)) -> dict:
    """One (rho, theta, accelerator), staged and coupled the way the fixtures
    stage and couple it."""
    sys.path.insert(0, str(LIB))
    import couplinglib as L                                  # noqa: PLC0415

    p = L.problem_with_rho(rho)
    roles = {"left": "dirichlet" if dirichlet == "left" else "neumann",
             "right": "dirichlet" if dirichlet == "right" else "neumann"}
    root = L.workroot(f"sweep_{accelerator}")
    specs = []
    for pos, mesh in (("left", mesh_l), ("right", mesh_r)):
        partner = "right" if pos == "left" else "left"
        e = L.heat_edits(p, pos, roles[pos], partner, mesh)
        # away from every ratio's answer, so each cell starts comparably wrong
        e["T_INIT"] = f"{L.SWEEP_T_INIT}"
        specs.append(_quiet_stage(L, root, pos, backend, e))
    t0 = time.time()
    res = L.pair(specs, max_iter=max_iter, tol=tol,
                 accelerator=accelerator, theta=theta)
    wall = time.time() - t0

    worst = 0.0
    for ex in (res.get("exports") or {}).values():
        for v in ex.get("values", []) or []:
            d = abs(float(v) - p.t_iface)
            worst = max(worst, d) if math.isfinite(d) else float("inf")
    deviation = worst / abs(p.t_iface) if math.isfinite(worst) else float("inf")

    out = {"rho": rho, "theta": theta, "accelerator": accelerator,
           "max_iter": max_iter, "tol": tol, "wall_s": round(wall, 1),
           "converged": bool(res.get("converged")),
           "iterations": int(res.get("iterations", 0)),
           "residual": float(res.get("residual", float("nan"))),
           "deviation": deviation if math.isfinite(deviation) else None,
           "amplification": p.amplification(dirichlet, theta),
           "theta_opt": p.theta_opt(dirichlet),
           "stability_limit": 2.0 / (1.0 + rho),
           "relaxation": res.get("relaxation"),
           "validation": res.get("validation") or [],
           "error": str(res.get("error"))[:300] if res.get("error") else None}

    # The physics: a partitioned scheme converges to a fixed point, which is the
    # SOLUTION only if the two sides exchanged the right quantity with the right
    # sign. Checked from BOTH sides, with the two outward normals' opposite
    # signs, plus the conservation balance.
    #
    # `validation` IS RECORDED BUT IS NOT PART OF `solved`, and that is a
    # deliberate call rather than a convenience. At tol=1e-4 the driver's
    # block-residual check fires on well-converged cells at MILD ratios —
    # "block(s) right.normal_fluxes are still changing by more than 1.0e-03
    # relative, while the global norm ... reports convergence" — because the
    # global norm is dominated by a temperature in kelvin. That is a true and
    # useful warning ABOUT THE RESIDUAL, not a statement that the answer is
    # wrong: those same cells sit 0.007 K and 0.005 W/m^2 from the closed form
    # with a flux balance of 1.6e-04. Folding it into the verdict scored Aitken
    # as LOSING at rho=1/4 in cells where it was both closer to the answer and
    # six times faster than the constant arm. `validation_clean` is reported on
    # its own so the warning is not lost.
    ex = res.get("exports") or {}
    phys: dict = {"has_exports": bool(ex)}
    ok = True
    if ex:
        nl, nr = len(ex["left"]["coordinates"]), len(ex["right"]["coordinates"])
        phys["n_points"] = [nl, nr]
        phys["nonmatching"] = bool(nl != nr)
        ok = ok and nl != nr
        for side in ("left", "right"):
            lo, hi = L.span(ex[side]["values"])
            err = abs(0.5 * (lo + hi) - p.t_iface)
            phys[f"{side}_T_err"] = err if math.isfinite(err) else None
            ok = ok and err <= T_ATOL
        for side, sign in (("left", +1.0), ("right", -1.0)):
            lo, hi = L.span(ex[side]["normal_fluxes"])
            err = abs(0.5 * (lo + hi) - sign * p.q)
            phys[f"{side}_q_err"] = err if math.isfinite(err) else None
            ok = ok and err <= Q_ATOL
        net_l, net_r = L.net_flux(ex["left"]), L.net_flux(ex["right"])
        rel = abs(net_l + net_r) / max(abs(net_l), abs(net_r), 1e-30)
        phys["flux_balance_rel"] = rel if math.isfinite(rel) else None
        ok = ok and math.isfinite(rel) and rel < BALANCE_RTOL
    else:
        ok = False
    out["physics"] = phys
    out["physics_ok"] = bool(ok)
    out["validation_clean"] = not res.get("validation")
    out["landed"] = bool(deviation < SETTLED)
    out["solved"] = bool(out["converged"] and ok and deviation < SETTLED)
    out["ran_away"] = bool((not math.isfinite(deviation)) or deviation > RUNAWAY)
    shutil.rmtree(root, ignore_errors=True)
    return out


# ── the grid ───────────────────────────────────────────────────────────────

def tag(rho: float, theta: float, accel: str) -> str:
    return f"r{rho:g}_t{theta:g}_{accel}".replace(".", "p")


def run_grid(out: Path, workers: int, max_iter: int, tol: float) -> None:
    cells = out / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    jobs = [(r, t, a, cells / f"{tag(r, t, a)}.json")
            for r in RHOS for t in THETAS for a in ACCELS
            if not (cells / f"{tag(r, t, a)}.json").exists()]
    # A cell above the constant stability limit burns the whole budget, so
    # start the expensive ones first or the tail of the grid runs serially.
    jobs.sort(key=lambda j: -(j[1] * (1.0 + j[0])))
    print(f"{len(jobs)} cells to run, {workers} workers, "
          f"max_iter={max_iter} tol={tol:g}", flush=True)
    env = dict(os.environ)
    env.setdefault("OPENPASO_REPO", str(REPO))
    env.setdefault("PYTHONPATH", str(LIB))
    running: list = []
    done = 0
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < workers:
            r, t, a, dest = jobs.pop(0)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--cell",
                   repr(r), repr(t), a, str(max_iter), repr(tol), str(dest)]
            running.append((subprocess.Popen(cmd, env=env, cwd=str(out),
                                             stdout=subprocess.DEVNULL,
                                             stderr=subprocess.PIPE),
                            (r, t, a), time.time()))
        time.sleep(1.0)
        still = []
        for proc, key, started in running:
            if proc.poll() is None:
                still.append((proc, key, started))
                continue
            done += 1
            note = ""
            if proc.returncode:
                note = (f" rc={proc.returncode} "
                        f"{proc.stderr.read().decode('utf-8', 'replace')[-200:]!r}")
            print(f"[{done}] rho={key[0]:g} theta={key[1]:g} {key[2]} in "
                  f"{time.time() - started:.0f}s{note}", flush=True)
        running = still
    print(f"grid finished in {(time.time() - t0) / 60:.1f} min", flush=True)


# ── reading a finished grid ────────────────────────────────────────────────

INF = float("inf")


def _dev(c: dict) -> float:
    v = c.get("deviation")
    return INF if v is None else float(v)


def load(out: Path) -> dict:
    g = {}
    for f in (out / "cells").glob("*.json"):
        d = json.loads(f.read_text())
        if "crashed" in d:
            print(f"CRASHED {f.name}: {d['crashed']}")
            continue
        g[(d["rho"], d["theta"], d["accelerator"])] = d
    return g


def report(out: Path) -> None:
    g = load(out)
    cells = [(r, t) for r in RHOS for t in THETAS
             if (r, t, "aitken") in g and (r, t, "constant") in g]
    n = len(cells)
    print(f"complete cells: {n}/{len(RHOS) * len(THETAS)}   "
          f"(max_iter={MAX_ITER}, tol={TOL:g}, skfem pair, 17/13 interface)\n")
    if not n:
        return

    hdr = "  rho | " + " ".join(f"{t:>6.1f}" for t in THETAS)
    for accel in ACCELS:
        print(f"=== {accel.upper()}: S=solved (met tol AND on the closed form), "
              f"L=landed on the closed form but above tol, .=neither "
              f"/ iterations ===")
        print(hdr)
        for r in RHOS:
            row = []
            for t in THETAS:
                c = g.get((r, t, accel))
                if c is None:
                    row.append("     ?")
                    continue
                mark = "S" if c["solved"] else "L" if c["landed"] else "."
                row.append(f"{mark}{c['iterations']:>5d}")
            print(f"{r:>5g} | " + " ".join(row))
        print()

    print("=== deviation from the closed-form interface value ===")
    print("  rho theta         aitken         constant  verdict")
    for r in RHOS:
        for t in THETAS:
            a, c = g.get((r, t, "aitken")), g.get((r, t, "constant"))
            if not a or not c:
                continue
            if a["solved"] and not c["solved"]:
                v = "AITKEN RESCUES"
            elif c["solved"] and not a["solved"]:
                v = "aitken LOSES"
            elif a["solved"]:
                v = ("both, aitken " + ("faster" if a["iterations"] < c["iterations"]
                                        else "same" if a["iterations"] == c["iterations"]
                                        else "slower"))
            else:
                v = "neither solved"
            print(f"{r:>5g} {t:>5.1f} {_dev(a):>14.4e} {_dev(c):>16.4e}  {v}")
    print()

    # matched or beat
    mob, lost, slower = 0, [], []
    for r, t in cells:
        a, c = g[(r, t, "aitken")], g[(r, t, "constant")]
        ar, cr = a["solved"], c["solved"]
        if ar and not cr:
            ok = True
        elif cr and not ar:
            ok = False
        elif ar and cr:
            ok = True
            if a["iterations"] > c["iterations"]:
                slower.append((r, t, a["iterations"], c["iterations"]))
        else:
            ok = _dev(a) <= _dev(c)
        mob += ok
        if not ok:
            lost.append((r, t, a["iterations"], c["iterations"], _dev(a), _dev(c)))
    print(f"MATCHED OR BEAT (outcome): {mob}/{n} = {100 * mob / n:.0f}%")
    for r, t, ai, ci, ad, cd in lost:
        print(f"   LOST  rho={r:g} theta={t:.1f}  aitken {ai} it dev {ad:.3e} | "
              f"constant {ci} it dev {cd:.3e}")
    print(f"both solved but aitken needed MORE iterations: {len(slower)}")
    for r, t, ai, ci in slower:
        print(f"   rho={r:g} theta={t:.1f}  aitken {ai} vs constant {ci}")
    va = sum(1 for r, t in cells if not g[(r, t, "aitken")]["validation_clean"])
    vc = sum(1 for r, t in cells if not g[(r, t, "constant")]["validation_clean"])
    print(f"cells whose `validation` block was non-empty (reported, NOT part of "
          f"the verdict — see run_cell): aitken {va}/{n}, constant {vc}/{n}")
    print()

    resc = [(r, t) for r, t in cells if g[(r, t, "aitken")]["solved"]
            and g[(r, t, "constant")]["ran_away"]]
    land = [(r, t) for r, t in cells if g[(r, t, "aitken")]["landed"]
            and g[(r, t, "constant")]["ran_away"]]
    caway = [(r, t) for r, t in cells if g[(r, t, "constant")]["ran_away"]]
    both = [(r, t) for r, t in caway if g[(r, t, "aitken")]["ran_away"]]
    print(f"constant ran away in {len(caway)}/{n} cells")
    print(f"  aitken SOLVED it there:               {len(resc)}/{n} = "
          f"{100 * len(resc) / n:.0f}%   {resc}")
    print(f"  aitken LANDED on the value there:     {len(land)}/{n} = "
          f"{100 * len(land) / n:.0f}%   {land}")
    print(f"  aitken ALSO ran away there:           {len(both)}/{len(caway)}")
    print()

    print("=== the recovery boundary ===")
    for r in RHOS:
        ar = [t for t in THETAS if (r, t, "aitken") in g
              and g[(r, t, "aitken")]["solved"]]
        al = [t for t in THETAS if (r, t, "aitken") in g
              and g[(r, t, "aitken")]["landed"]]
        cr = [t for t in THETAS if (r, t, "constant") in g
              and g[(r, t, "constant")]["solved"]]
        print(f"  rho={r:>4g}  limit 2/(1+rho)={2 / (1 + r):.3f}  "
              f"aitken solved {[f'{t:g}' for t in ar] or '-'}  "
              f"landed {[f'{t:g}' for t in al] or '-'}  "
              f"constant solved {[f'{t:g}' for t in cr] or '-'}")
    print("\n  theta=0.5 column (the knowledge's 'cannot estimate rho' default):")
    for r in RHOS:
        a, c = g.get((r, 0.5, "aitken")), g.get((r, 0.5, "constant"))
        if not a or not c:
            continue
        print(f"    rho={r:>4g}  aitken solved={a['solved']} "
              f"landed={a['landed']} it={a['iterations']} dev={_dev(a):.3e} | "
              f"constant solved={c['solved']} dev={_dev(c):.3e}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--cell":
        rho, theta, accel = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
        max_iter, tol, dest = int(sys.argv[5]), float(sys.argv[6]), Path(sys.argv[7])
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rec = run_cell(rho, theta, accel, max_iter, tol)
        except Exception as e:                                # noqa: BLE001
            rec = {"rho": rho, "theta": theta, "accelerator": accel,
                   "crashed": f"{type(e).__name__}: {e}"}
        dest.write_text(json.dumps(rec))
        return
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True,
                    help="directory for the cell JSONs (use ext4, not exFAT)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--report", action="store_true",
                    help="only re-read and print a finished grid")
    a = ap.parse_args()
    if not a.report:
        run_grid(a.out, a.workers, a.max_iter, a.tol)
    report(a.out)


if __name__ == "__main__":
    main()
