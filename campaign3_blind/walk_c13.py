#!/usr/bin/env python
"""Walk C13's execution path: FEniCSx solid + SPARTA DSMC gas, once, discarded.

The pre-registered band was computed by build_offpool.c13_theory() and COMMITTED
before this script first ran (commit 555d4a1f); nothing here can touch it. What
this walk establishes is only that the path works: both codes run in their own
interpreters, the partitioned iteration contracts, the conservation identity
holds, and — reported, not tuned — where the coupled answer lands relative to
the frozen band.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(os.environ.get(
    "OPENPASO_REPO", "/home/alexander/Schreibtisch/ofa-balanced")) / "src"))

import build_offpool as B                                        # noqa: E402
from core.coupling_driver import Participant, run_coupling       # noqa: E402

TMP = Path(os.environ.get("OPENPASO_WALK_TMP",
                          "/tmp/claude-1001/-home-alexander-4C/balancedtmp/walkc13"))

CFG_COMMON = dict(ls=B.LS, lg=B.LG, h=B.HH, t_hot=B.T_HOT, t_cold=B.T_COLD,
                  ks=B.KS, n_mean=B.N_MEAN)


def main(max_iter=25, tol=2e-2):
    root = TMP
    if root.exists():
        shutil.rmtree(root)
    wa, wb = root / "A_fenics", root / "B_sparta"
    for wd, script, extra in (
            (wa, "w_fenics_c13.py",
             dict(side="neumann", sidename="A", partner="B", code="fenics",
                  level=1, n=[40, 24])),
            (wb, "w_sparta_c13.py",
             dict(side="dirichlet", sidename="B", partner="A", code="sparta",
                  level=1, cells=[30, 18], nequil=2000, nave=4000,
                  dt=1e-7, seed=90210, t_init=300.0))):
        wd.mkdir(parents=True)
        shutil.copy(HERE / "walkers" / "wcommon.py", wd)
        shutil.copy(HERE / "walkers" / script, wd)
        (wd / "cfg.json").write_text(json.dumps({**CFG_COMMON, **extra},
                                                indent=1))
    parts = [
        Participant(name="A", command=[
            "/home/alexander/miniconda3/envs/fenics/bin/python",
            "w_fenics_c13.py"], work_dir=wa, imports_from=["B"], timeout=1200),
        Participant(name="B", command=[
            "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python",
            "w_sparta_c13.py"], work_dir=wb, imports_from=["A"], timeout=1800),
    ]
    t0 = time.time()
    r = run_coupling(parts, max_iter=max_iter, tol=tol, accelerator="constant",
                     theta0=1.0, probe=False)
    dt = time.time() - t0

    # read the walk's own QoI and identity numbers from the participants' logs
    logA = (wa / "run_level1_A.log").read_text()
    logB = (wb / "run_level1_B.log").read_text()

    def _get(txt, name):
        for ln in txt.splitlines():
            if ln.startswith(name):
                return float(ln.split("=")[1])
        return None

    theta = _get(logA, "THETA_INTERFACE")
    flux_solid = _get(logA, "FLUX_SOLID")
    etot_iface = _get(logB, "etot_xlo")
    imbal = _get(logB, "steady_imbalance_rel")
    flux_gas = -etot_iface if etot_iface is not None else None
    ident = (abs(flux_solid - flux_gas)
             / max(abs(flux_solid), abs(flux_gas), 1e-300)
             if flux_solid is not None and flux_gas is not None else None)
    band = B.build_C13()[2]["qoi"]["band"]
    out = dict(converged=bool(r.converged), iterations=r.iterations,
               residual=float(r.residual), wall_s=round(dt, 1),
               theta_interface=theta, frozen_band=band,
               inside_band=(band[0] <= theta <= band[1]
                            if theta is not None else None),
               flux_solid=flux_solid, flux_gas=flux_gas,
               identity_rel=ident, dsmc_steady_imbalance=imbal,
               history=[float(v) for v in (r.history or [])])
    print(json.dumps(out, indent=1))
    (TMP / "c13_walk.json").write_text(json.dumps(out, indent=1))
    return 0 if r.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
