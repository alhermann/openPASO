"""deal.II participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Pure glue: the PDE solve is done by a compiled deal.II executable THAT YOU
WRITE AND BUILD YOURSELF. No C++ source ships with this contract and none is on
this install, so do not search for one: write a program that reads the side,
the geometry, the material, the mesh size, the source samples and the imported
interface samples from one plain-text file your solve region writes, and
writes the interface trace and the CONSISTENT flux (the residual of the
assembled system with no boundary condition applied, divided by the nodal
interface weight) to a plain-text file your solve region reads back. That pair
of files is private to you. This wrapper is the contract around it: the
imports.json handshake, the sign convention, the exports schema and the
self-check. deal.II has no Python API, so unlike every other backend there is
a BUILD STEP before this can run at all:

    cmake -S <dir with YOUR .cc and a 6-line CMakeLists> -B <build> \
          -DDEAL_II_DIR=<deal.II BUILD or INSTALL tree>
    make -C <build>

and DEALII_EXE below must point at YOUR binary.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
# ... (explanation continues in the annotated block)
SIDE      = "neumann"   # "dirichlet" | "neumann"
PARTNER   = "left"      # name of the partner participant in couple(...)
X0, X1    = 0.6, 1.4
Y0, Y1    = 0.0, 1.0
IFACE_X   = 0.6
K         = 5.0


def F_SRC(x, y):
    """Volumetric source, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above
    and is almost never what your problem wants. THIS KNOB USED TO BE A SCALAR
    CONSTANT, AND A CONSTANT CANNOT REPRESENT A SOURCE THAT VARIES WITH
    POSITION: the source of a manufactured solution is a POLYNOMIAL in x and y,
    and no single number is that polynomial. Left at zero the temperature is
    harmonic, the outer Dirichlet values are the only data left in the problem,
    and the answer degenerates to the 1-D profile between them — the interface
    flux is one constant along the whole interface, and it is identically zero
    when the two subdomains carry the same outer value. The coupling will
    converge beautifully to that, and it is not the problem you were given.

    If your problem states a source, or gives you a manufactured solution whose
    source term you derived, put it here. `x` and `y` are NumPy arrays, so
    build the answer with NumPy and return ONE array of the same shape (write
    `0.0 * x + c` for a genuine constant, never a bare `c`):

        # -div(K grad T) for the manufactured T = x**3 * y**2
        return -K * (6.0 * x * y**2 + 2.0 * x**3)

    THIS PARTICIPANT DRIVES A COMPILED SOLVER: your solve region below has to
    carry the samples of F_SRC to your binary (for example on a uniform grid in
    the input file) and your binary has to assemble the volume integral of f times v from them. A
    solver that never reads the source returns the boundary-data-only solution
    with no error anywhere, so have the binary announce what it read and make
    your wrapper refuse a non-zero F_SRC that produced no announcement.
    """
    return 5.0 * np.pi**2 * (1.4 - x) * np.cos(np.pi * y)
T_OUTER   = 0.0
NX, NY    = 8, 10
T_INIT    = 0.0
Q_INIT    = 0.0           # iteration-1 fallback interface flux
DEALII_EXE = "./build/solver"   # YOUR compiled solver; you write and build it (docstring)
# ─────────────────────────────────────────────────────────────────────────

DEGREE = 1                # FE_Q degree used by the deal.II solver


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None


def sample(imp, key, fallback):
    """Partner samples as sorted (y, value) pairs; constant fallback if absent.

    The deal.II solver interpolates them piecewise-linearly (clamped outside
    the range) onto its own interface nodes — same semantics as numpy.interp.
    """
    if imp and imp.get("coordinates"):
        ys = [float(c[1]) for c in imp["coordinates"]]
        vs = [float(v) for v in (imp.get(key) or [])]
        if len(vs) == len(ys) and len(ys) > 0:
            return sorted(zip(ys, vs))
    return [(float(Y0), float(fallback)), (float(Y1), float(fallback))]


imp = read_imports()
if SIDE == "dirichlet":
    side_flag, pairs = 0, sample(imp, "values", T_INIT)
else:
    side_flag, pairs = 1, sample(imp, "normal_fluxes", Q_INIT)

# ─────────────────────────────────────────────────────────────────────────
# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE.
#
# Build the mesh, the function space, the weak form and the linear solve for
# ... (explanation continues in the annotated block)
# ordinary finite-element work and OASiS has no business dictating it.
#
# What OASiS does document — because you cannot guess it and it is what the
# interface check compares against — is everything AROUND the solve: the
# imports/exports handshake above, the interface sign convention, and the flux
# ... (explanation continues in the annotated block)

# Write solver input file
with open("solver_input.txt", "w") as f:
    f.write(f"{X0} {X1} {Y0} {Y1} {IFACE_X} {K} {NX} {NY} {DEGREE} {side_flag}\n")
    f.write(f"{len(pairs)}\n")
    for y, val in pairs:
        f.write(f"{IFACE_X} {y} {val}\n")

# Run solver
result = subprocess.run([DEALII_EXE], capture_output=True, text=True)
print(result.stdout, file=sys.stdout)
print(result.stderr, file=sys.stderr)
if result.returncode != 0:
    raise SystemExit(f"SOLVER FAILED WITH CODE {result.returncode}")

# Read solver output file
with open("solver_output.txt", "r") as f:
    lines = f.readlines()
n_points = int(lines[0].strip())
coords = []
temps = []
fluxes = []
for line in lines[1:]:
    parts = line.strip().split()
    x, y, val, flx = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    coords.append([x, y])
    temps.append(val)
    fluxes.append(flx)

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
# ... (explanation continues in the annotated block)
_chk_vals = np.asarray(temps, float).ravel()
_chk_flux = np.asarray(fluxes, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes; "
                     "the solve did not produce a usable field, so nothing was "
                     "exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 \
        and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 "
                     "against a nonzero imported flux: the imported load never "
                     "entered the assembled system (the facet term / boundary "
                     "condition that integrates it is missing). Fix the "
                     "application; do not couple on")
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied flux can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": len(coords),
    "coordinates": coords,
    "values": temps,
    "normal_fluxes": fluxes,
}, indent=2))

# ── WHAT YOUR SOLVE MUST LEAVE BEHIND ─────────────────────────
# The code above and below the elided block uses these names. Your
# block has to define every one of them, or the rest will not run:
#
#     coords
#     fluxes
#     temps
#     v
#
# That is the whole contract. Read the surviving lines to see the
# shape each one has to have -- they are already indexed, assembled
# or written out there. OASiS does not serve the solve itself, but
# it will not make you guess which variables the hole was filling.
