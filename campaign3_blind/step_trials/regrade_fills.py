"""Re-run saved step-trial fills and grade them the way the campaign's own tasks do:
the interface INTERIOR only for the recovered flux (the ends are Dirichlet-Neumann corners the
tasks exclude), the whole trace for the field. No API calls: this only re-executes saved scripts."""
import json, math, os, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROLE = {"D": "dirichlet", "N": "neumann"}
INTERP = {"ngs": "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python",
          "skf": "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python",
          "kratos": "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python",
          "fen": "/home/alexander/miniconda3/envs/fenics/bin/python"}

def refs(role):
    if role == "dirichlet":
        return ("participant_A.py", "right", lambda y: 0.6 * math.sin(math.pi * y),
                lambda y: -math.sin(math.pi * y),
                {"values": lambda y: 0.6 * math.sin(math.pi * y), "normal_fluxes": lambda y: 0.0})
    return ("participant_B.py", "left", lambda y: 0.8 * math.sin(math.pi * y),
            lambda y: -5.0 * math.sin(math.pi * y),
            {"values": lambda y: 0.0, "normal_fluxes": lambda y: 5.0 * math.sin(math.pi * y)})

def grade(path, code, role):
    fname, partner, u_ref, q_ref, imp = refs(role)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); (d / fname).write_text(Path(path).read_text())
        ys = [i / 10 for i in range(11)]
        (d / "imports.json").write_text(json.dumps({partner: {
            "field_name": "u", "n_points": len(ys), "coordinates": [[0.6, y] for y in ys],
            "values": [imp["values"](y) for y in ys], "normal_fluxes": [imp["normal_fluxes"](y) for y in ys]}}))
        try:
            r = subprocess.run([INTERP[code], fname], cwd=td, capture_output=True, text=True, timeout=600,
                               env=dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="2"))
        except subprocess.TimeoutExpired:
            return "TIMEOUT", ""
        ex = d / "exports.json"
        if not ex.is_file():
            tail = (r.stderr or r.stdout or "").strip().splitlines()
            return "NO EXPORTS", (tail[-1][:120] if tail else f"rc={r.returncode}")
        e = json.loads(ex.read_text())
        co = e["coordinates"]; q = [float(x) for x in e["normal_fluxes"]]; v = [float(x) for x in e["values"]]
        inner = [i for i, c in enumerate(co) if 1e-9 < c[1] < 1 - 1e-9]
        eq = max(abs(q[i] - q_ref(co[i][1])) for i in inner) if inner else float("inf")
        ev = max(abs(vv - u_ref(c[1])) for c, vv in zip(co, v))
        ok = eq < 0.25 and ev < 0.06
        return ("PASS" if ok else "FAIL"), f"n={len(q)} flux_interior={eq:.2e} trace={ev:.2e}"

if __name__ == "__main__":
    rows = []
    for pat, code, role in (("participant_skfD*.py", "skf", "dirichlet"),
                            ("participant_ngsD*.py", "ngs", "dirichlet"),
                            ("participant_ngsN*.py", "ngs", "neumann"),
                            ("participant_kratosN[567]_*.py", "kratos", "neumann"),
                            ("participant_fenN*.py", "fen", "neumann")):
        for f in sorted(HERE.glob(pat)):
            verdict, info = grade(f, code, role)
            rows.append((f.name, verdict, info))
            print(f"{f.name:42} {verdict:10} {info}", flush=True)
    Path(HERE / "regrade_fills.json").write_text(json.dumps(rows, indent=1))
    print("PASS:", sum(1 for _, v, _ in rows if v == "PASS"), "of", len(rows))
