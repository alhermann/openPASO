#!/usr/bin/env python
"""Walk the execution path of a balanced coupled instance, once, and discard it.

WHY THIS EXISTS. A coupled task whose intended execution path has never run
measures the path, not the agent: a tool bug in it reads as agent failure and is
charged to the arm under test. ``run_blind.py`` refuses any problem not recorded
``true`` in ``path_readiness.json``, and this is what earns that record.

CUSTODY. The walk does NOT open the published instance. It rebuilds the same
builder with a FRESH CSPRNG draw, so the geometry, the materials, the roles, the
mesh sequence, the transmission conditions and both participants are identical
while the hidden polynomial coefficients are different ones. Everything the path
exercises is the same; no sealed key is touched; and the field the walk measures
against exists only in this process and is never written to the run tree.

WHAT IT MEASURES. That both participants run in their own interpreters, that the
partitioned iteration converges to the prescribed tolerance, and that the
recovered field converges to the manufactured solution under refinement. The
error is taken at each participant's OWN NODES, so the number reported is the
solver's and not an interpolator's.

  campaign3_blind/walk_balanced.py C9 --levels 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = Path(os.environ.get("OPENPASO_REPO",
                           "/home/alexander/Schreibtisch/ofa-balanced"))
sys.path.insert(0, str(REPO / "src"))

import build_balanced as B                                       # noqa: E402
from core.coupling_driver import Participant, run_coupling       # noqa: E402

TMP = Path(os.environ.get("OPENPASO_WALK_TMP",
                          "/tmp/claude-1001/-home-alexander-4C/balancedtmp/walk"))

PY = {
    "fenics": "/home/alexander/miniconda3/envs/fenics/bin/python",
    "ngsolve": "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python",
    "skfem": "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python",
    "kratos": "/mnt/kratos-tier2/kv/bin/python",
    "dune": "/home/alexander/miniconda3/envs/dune-fem-env/bin/python",
    "4C": "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python",
    "febio": "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python",
    "dealii": "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python",
}
SCRIPT = {c: f"w_{c.lower().replace('4c', 'fourc')}.py" for c in PY}
SCRIPT["4C"] = "w_fourc.py"

# Per-cell relaxation. rho = c_D / c_N with c = k / L decides the admissible
# theta; it is derivable from the published coefficients and widths, and it is
# recorded here rather than searched for so a walk that fails fails for a reason.
RELAX = {
    "C1": ("aitken", 0.7), "C2": ("aitken", 0.5), "C3": ("aitken", 0.5),
    "C4": ("aitken", 0.5), "C5": ("constant", 0.20), "C6": ("aitken", 0.5),
    "C7": ("aitken", 0.5), "C8": ("aitken", 0.5), "C9": ("constant", 0.6),
    "C10": ("aitken", 0.5), "C11": ("aitken", 0.5), "C12": ("aitken", 0.5),
}

BUILDER = {fn.__name__.split("_")[-1]: fn for fn in B.BUILDERS}


def _cells(extent, N):
    """Cell counts per axis for a subdomain meshed at h = 1/N."""
    out = []
    for lo, hi in extent:
        k = (hi - lo) * N
        if abs(k - round(k)) > 1e-9:
            raise ValueError(f"extent {(lo, hi)} is not a multiple of h = 1/{N}")
        out.append(int(round(k)))
    return out


def make_cfg(spec, info, sources, side, level, N):
    codes = spec["codes"]
    idx = 0 if side == "A" else 1
    extent = [list(map(float, a))
              for a in (spec["extent_a"] if side == "A" else spec["extent_b"])]
    axis = {"x": 0, "y": 1}.get(spec.get("interface_axis"))
    if axis is None and info["family"] != "notched":
        raise NotImplementedError(f"no interface axis for {spec['id']}")
    cfg = dict(side=spec["roles"][side], sidename=side,
               partner="B" if side == "A" else "A",
               code=codes[idx], level=level, dim=spec["dim"], axis=axis,
               xi=float(B.XI), extent=extent, n=_cells(extent, N),
               physics=("vector" if spec.get("components") == ["ux", "uy"]
                        else "scalar"))
    fam = info["family"]
    if fam == "vector":
        (lamA, muA), (lamB, muB) = info["mat"]
        cfg.update(lam=float(lamA), mu=float([muA, muB][idx]),
                   source=[sp.sstr(c) for c in sources[side]])
    elif fam == "scalar":
        K = info["K"][idx]
        cfg.update(K=[[float(K[i, j]) for j in range(spec["dim"])]
                      for i in range(spec["dim"])],
                   reaction=float(info.get("reaction", (0, 0))[idx]),
                   source=sp.sstr(sources[side]))
        if info.get("transient"):
            # the task prescribes dt = h/4 = 1/(4N); with t_end = 1/4 that is
            # exactly N steps per level (8, 16, 32)
            cfg.update(transient=dict(t_end=float(info["t_end"]),
                                      dt=1.0 / (4 * N)))
    elif fam == "thermoelastic":
        (lamA, muA), (lamB, muB) = info["mat"]
        K = info["K"][idx]
        cfg.update(physics="thermoelastic",
                   k=float(K[0, 0]), lam=float(lamA),
                   mu=float([muA, muB][idx]), beta=float(info["beta"]),
                   source_T=sp.sstr(sources[side][0]),
                   source_u=[sp.sstr(c) for c in sources[side][1:]])
    elif fam == "notched":
        # The NON-RECTANGULAR subdomain and the BENT interface. A is the unit
        # square minus subdomain B minus the notch, and carries THREE material
        # cells with three different sources; B is the rectangle (1/2,1)x(0,1/2)
        # and carries one. The interface is the polyline that separates them, so
        # it has two outward normals and no single axis -- `axis` is None and
        # every participant that reads it must cope.
        half = 0.5
        legs = [[0, half, [0.0, half]], [1, half, [half, 1.0]]]
        kv, cells = info["kv"], info["cells"]
        cfg.update(axis=None, xi=half, bent=legs, dim=2, physics="scalar")
        if side == "A":
            cfg.update(
                extent=[[0.0, 1.0], [0.0, 1.0]], n=[N, N],
                remove=[[[0.5, 1.0], [0.0, 0.5]], [[0.75, 1.0], [0.75, 1.0]]],
                cells=[{"box": [[0.0, 0.5], [0.0, 0.5]],
                        "k": float(kv[(0, 0)]),
                        "source": sp.sstr(sources[(0, 0)])},
                       {"box": [[0.0, 0.5], [0.5, 1.0]],
                        "k": float(kv[(0, 1)]),
                        "source": sp.sstr(sources[(0, 1)])},
                       {"box": [[0.5, 1.0], [0.5, 1.0]],
                        "k": float(kv[(1, 1)]),
                        "source": sp.sstr(sources[(1, 1)])}],
                K=[[float(kv[(0, 0)]), 0.0], [0.0, float(kv[(0, 0)])]],
                reaction=0.0, source=sp.sstr(sources[(0, 0)]))
        else:
            k = float(kv[(1, 0)])
            cfg.update(extent=[[0.5, 1.0], [0.0, 0.5]],
                       n=[N // 2, N // 2],
                       K=[[k, 0.0], [0.0, k]], reaction=0.0,
                       source=sp.sstr(sources[(1, 0)]))
    else:
        raise NotImplementedError(f"walk config for family {fam!r}")
    return cfg


def stage(spec, info, sources, level, N, root: Path):
    parts = []
    for i, side in enumerate(("A", "B")):
        code = spec["codes"][i]
        wd = root / f"{side}_{code}"
        wd.mkdir(parents=True, exist_ok=True)
        shutil.copy(HERE / "walkers" / "wcommon.py", wd)
        # deal.II has no Python API, so its participant is a wrapper around a
        # compiled program; the source has to travel with it.
        if code == "dealii":
            shutil.copy(HERE / "walkers" / "iface_dealii.cc", wd)
        script = HERE / "walkers" / SCRIPT[code]
        # the thermoelastic pair needs 4C's TSI machinery, which lives in its
        # own participant; every other physics goes through the plain one
        if code == "4C" and info["family"] == "thermoelastic":
            script = HERE / "walkers" / "w_fourc_tsi.py"
        if not script.is_file():
            raise FileNotFoundError(f"no walk participant for {code}: {script}")
        shutil.copy(script, wd)
        (wd / "cfg.json").write_text(json.dumps(
            make_cfg(spec, info, sources, side, level, N), indent=2))
        parts.append(Participant(name=side, command=[PY[code], script.name],
                                 work_dir=wd,
                                 imports_from=["B" if side == "A" else "A"],
                                 timeout=3600))
    return parts


def node_error(wd: Path, exprs, coords, avoid=()):
    """RMS error at the participant's own nodes.

    ``avoid`` lists coordinate values a node must not sit exactly on. It exists
    for the four-material instance, whose exact field on the non-rectangular
    subdomain is a Piecewise over three material cells selected by STRICT
    inequalities: a node lying exactly on x = 1/2 or y = 1/2 falls through to
    the wrong branch, which extrapolates that cell's flux potential across the
    material line and is simply a different function there. The nodes affected
    are O(N) out of O(N^2), so the artefact enters the RMS with weight O(h) and
    an O(1) error -- an observed order of 1/2. MEASURED before this was added:
    0.791 then 0.628 on the notched subdomain while the rectangular partner gave
    2.129 and 2.127 on the same run.

    The graded probe grid does not have the problem and does not need this: 44 is
    chosen precisely so that no probe lands on x = 1/2 or 3/4 (45 would put one
    at 22.5/45 = 0.5). This is a defect of the WALK's measurement, not of the key.
    """
    import csv
    fns = [sp.lambdify(coords, sp.sympify(e), "math") for e in exprs]
    acc, n, skipped = 0.0, 0, 0
    with open(wd / "nodes.csv", newline="") as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            v = [float(a) for a in row]
            p = v[:len(coords)]
            got = v[len(coords):]
            if any(abs(c - a) < 1e-9 for c in p for a in avoid):
                skipped += 1
                continue
            for f, g in zip(fns, got):
                acc += (float(f(*p)) - g) ** 2
                n += 1
    return math.sqrt(acc / n) if n else None


def walk(pid: str, levels: int, seed=None, max_iter=200):
    fn = BUILDER[pid]
    d = B.V.Draw(seed)
    spec, info, fields, sources, coords = fn(d)
    root = TMP / pid
    if root.exists():
        shutil.rmtree(root)
    acc, mode_theta = [], RELAX.get(pid, ("aitken", 0.5))
    print(f"=== walk {pid}: {'+'.join(spec['codes'])}  "
          f"A={spec['roles']['A']} B={spec['roles']['B']}  "
          f"{spec['physics']}")
    print(f"    relaxation: {mode_theta[0]} theta={mode_theta[1]}  "
          f"(fresh draw, seed withheld from disk)")
    for k in range(levels):
        N = spec["mesh_N"][k]
        t0 = time.time()
        parts = stage(spec, info, sources, k + 1, N, root / f"level{k + 1}")
        r = run_coupling(parts, max_iter=max_iter, tol=1e-6,
                         accelerator=mode_theta[0], theta0=mode_theta[1],
                         probe=False)
        dt = time.time() - t0
        errs = {}
        for i, side in enumerate(("A", "B")):
            wd = root / f"level{k + 1}" / f"{side}_{spec['codes'][i]}"
            if not (wd / "nodes.csv").is_file():
                errs[side] = None
                continue
            f = fields[side]
            if info.get("transient"):
                # the participants report the FINAL-TIME field; the space-time
                # expression must be evaluated there or lambdify leaves t as an
                # unbound global and the measurement crashes
                # substitute BY NAME: the builder's t is Symbol("t",
                # real=True) and a bare Symbol("t") is a different symbol that
                # substitutes nothing, silently
                te = sp.nsimplify(info["t_end"])

                def _at_tend(e):
                    e = sp.sympify(e)
                    return e.subs({s_: te for s_ in e.free_symbols
                                   if s_.name == "t"})
                f = (_at_tend(f) if not isinstance(f, (list, tuple))
                     else [_at_tend(c) for c in f])
            errs[side] = node_error(
                wd, f if isinstance(f, (list, tuple)) else [f], coords,
                avoid=info.get("avoid_lines", ()))
        acc.append(dict(level=k + 1, N=N, converged=r.converged,
                        history=[float(v) for v in (r.history or [])],
                        iterations=r.iterations, residual=r.residual,
                        wall_s=round(dt, 1), errors=errs, error=r.error))
        print(f"    level {k + 1} (h = 1/{N}): converged={r.converged} "
              f"iters={r.iterations} res={r.residual:.3e} {dt:.1f}s "
              f"errA={errs.get('A')} errB={errs.get('B')}")
        if r.error:
            print(f"      driver error: {r.error}")
        if not r.converged:
            break
    if len(acc) >= 2 and all(a["errors"].get("A") for a in acc[:2]):
        for side in ("A", "B"):
            e = [a["errors"][side] for a in acc if a["errors"].get(side)]
            if len(e) >= 2:
                o = [math.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
                print(f"    observed order, subdomain {side}: "
                      + ", ".join(f"{v:.3f}" for v in o))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problems", nargs="+")
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--accel", default=None,
                    choices=["aitken", "constant"])
    a = ap.parse_args()
    out = {}
    for pid in a.problems:
        try:
            if a.theta is not None or a.accel:
                m, th = RELAX.get(pid, ("aitken", 0.5))
                RELAX[pid] = (a.accel or m,
                              a.theta if a.theta is not None else th)
            out[pid] = walk(pid, a.levels, a.seed, a.max_iter)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            out[pid] = [{"error": f"{type(exc).__name__}: {exc}"}]
    # MERGE, never overwrite: each walk is expensive and a later run of one
    # problem must not erase the record of an earlier run of another.
    f = TMP / "walk_summary.json"
    prev = json.loads(f.read_text()) if f.is_file() else {}
    prev.update(out)
    f.write_text(json.dumps(prev, indent=2, default=str))
    print(f"\nwrote {f} ({len(prev)} instance(s) on record)")


if __name__ == "__main__":
    TMP.mkdir(parents=True, exist_ok=True)
    main()
