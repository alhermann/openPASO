"""Kratos Multiphysics participant for the OASiS `couple` driver (NEUMANN side).

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1, so an
iteration-1 fallback is mandatory), writes exports.json LAST and exits 0.
Needs KratosMultiphysics + ConvectionDiffusionApplication importable in the
interpreter named in `command`. DO NOT GUESS THAT INTERPRETER: take it from
`discover(query='list')`, which reports the one this install actually imports
Kratos with. A system python3 is a common trap — it usually exists, so nothing
looks wrong, and it raises ModuleNotFoundError because the Kratos build targets
a different Python version than the system one.

THE OTHER HALF OF participant_kratos.py. That file is the DIRICHLET side: it
imports the partner's `values` (interface temperature), fixes them as nodal
TEMPERATURE, and exports the consistent REACTION_FLUX. This file is the NEUMANN
side: it imports the partner's `normal_fluxes` and applies them as the natural
boundary condition, then exports the interface TEMPERATURE its solve produced,
which is what a Dirichlet partner consumes.

Physics: steady conduction  -div(K grad T) = f  on one rectangular subdomain of
a domain split by a straight interface at x = IFACE_X. Top and bottom edges are
natural (zero flux). The non-interface x-boundary carries a Dirichlet value
T_OUTER.

  x = OUTER_X : Dirichlet T = T_OUTER
  x = IFACE_X : INTERFACE, natural BC = the partner's imported flux
  top/bottom  : insulated (natural, nothing to do)

THE SIGN, WHICH IS THE ONLY THING A NEUMANN PARTICIPANT CAN GET SILENTLY WRONG.
Every participant exports its outward normal flux density with respect to ITS
OWN outward normal,
      q = -(K grad T) . n_own,
so on a shared interface the two sides carry OPPOSITE signs. Integrating the
weak form on THIS subdomain,
      int_Omega K grad T . grad v  =  int_Omega f v  +  int_dOmega (K grad T . n) v ds,
and on the interface n = n_own = -n_partner, so
      K grad T . n_own = -q_own = +q_partner.
The partner's number is therefore applied UNCHANGED — no minus sign anywhere.
Kratos's FluxCondition2D2N enforces exactly  K grad T . n = FACE_HEAT_FLUX
(verified by a patch test: T fixed to 0 at x=0, FACE_HEAT_FLUX=1 on x=1, K=1,
no source -> T(1)=+1.000000, i.e. dT/dx=+1 not -1), so

      FACE_HEAT_FLUX  :=  the imported `normal_fluxes`, verbatim.

If you ever flip that sign to "make the temperatures look right", you have
built a coupling that drives heat the wrong way across the interface and still
converges. The self-check printed at the end of this script is the guard: it
evaluates the discrete divergence theorem on this subdomain and must come out
at round-off.
"""
import json
import sys
from pathlib import Path

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER   = "left"        # the partner's `name` in your couple(...) call
X0, X1    = 0.6, 1.4      # this subdomain's x-extent
Y0, Y1    = 0.0, 1.0      # this subdomain's y-extent
IFACE_X   = 0.6           # the shared interface; must equal X0 or X1
K         = 5.0           # conductivity of THIS subdomain


def F_SRC(x, y):
    """Volumetric source f in  -div(K grad T) = f, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above.
    A CONSTANT CANNOT REPRESENT A POLYNOMIAL SOURCE: if your problem states
    one, or you derived it from a manufactured solution, a single number here
    silently solves a different problem. With the whole outer boundary
    prescribed and no source, the answer degenerates to the profile between
    the outer values.

    Unlike its FEniCSx and DUNE siblings, this one is called ONCE PER NODE
    with SCALAR coordinates (see the SetSolutionStepValue loop below), so
    write it with plain math or NumPy scalars — do not assume arrays:

        return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    """
    return 5.0 * np.pi**2 * (1.4 - x) * np.sin(np.pi * y)

T_OUTER   = 0.0           # Dirichlet value on the NON-interface x-boundary
FULL_OUTER_DIRICHLET = True  # True: T_OUTER also on y=Y0,Y1; corners stay outer
NX, NY    = 8, 10         # this subdomain's OWN mesh; need not match the partner
Q_INIT    = 0.0           # iteration-1 fallback interface flux density


def source(x, y):
    """Volumetric source at the nodes — the interpolation point for F_SRC.

    Sampling at the nodes is the P1 interpolant of the source, an O(h^2) load
    error, the same order as the discretization error, so it does not touch
    the second-order rate.
    """
    return F_SRC(x, y)
# ─────────────────────────────────────────────────────────────────────────

ON_MAX_X = abs(IFACE_X - X1) < abs(IFACE_X - X0)   # interface is this side's x-max?
OUTER_X = X0 if ON_MAX_X else X1
S = 1.0 if ON_MAX_X else -1.0          # outward normal at the interface = S * e_x
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)


def read_imports():
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text() or "{}").get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
    """Map the partner's samples onto THIS participant's interface points.
    The driver does no interpolation — non-matching meshes are handled here."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    # `or []` and not `.get(key, [])`: a partner that writes the key with an
    # explicit null gets [] here instead of a TypeError out of np.asarray, and
    # falls through to the fallback like any other unusable import.
    vs = np.asarray(imp.get(key) or [], float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


def build_model():
    """Structured triangulation of [X0,X1] x [Y0,Y1]; returns (mp, nid)."""
    model = KM.Model()
    mp = model.CreateModelPart('thermal')
    
    # Model setup order (measured fact 11)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    
    settings = KM.ConvectionDiffusionSettings()
    settings.SetUnknownVariable(KM.TEMPERATURE)
    settings.SetDiffusionVariable(KM.CONDUCTIVITY)
    settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
    settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)
    
    # Add nodal solution step variables BEFORE creating nodes (fact 11)
    mp.AddNodalSolutionStepVariable(KM.TEMPERATURE)
    mp.AddNodalSolutionStepVariable(KM.CONDUCTIVITY)
    mp.AddNodalSolutionStepVariable(KM.HEAT_FLUX)
    mp.AddNodalSolutionStepVariable(KM.FACE_HEAT_FLUX)
    mp.AddNodalSolutionStepVariable(KM.REACTION_FLUX)
    
    mp.SetBufferSize(1)
    props = mp.CreateNewProperties(1)
    
    # Structured grid node indexing
    hx = (X1 - X0) / NX
    hy = (Y1 - Y0) / NY
    
    nid = {}  # (i, j) -> node_id
    node_id = 1
    
    for j in range(NY + 1):
        y = Y0 + j * hy
        for i in range(NX + 1):
            x = X0 + i * hx
            nid[(i, j)] = node_id
            mp.CreateNewNode(node_id, x, y, 0.0)
            node_id += 1
    
    # Create triangular elements (fact 1: LaplacianElement2D3N exists)
    elem_id = 1
    for j in range(NY):
        for i in range(NX):
            # Split each quad into two triangles
            n_bl = nid[(i, j)]
            n_br = nid[(i + 1, j)]
            n_tr = nid[(i + 1, j + 1)]
            n_tl = nid[(i, j + 1)]
            
            # Triangle 1: bottom-left, bottom-right, top-right
            mp.CreateNewElement('LaplacianElement2D3N', elem_id,
                                [n_bl, n_br, n_tr], props)
            elem_id += 1
            
            # Triangle 2: bottom-left, top-right, top-left
            mp.CreateNewElement('LaplacianElement2D3N', elem_id,
                                [n_bl, n_tr, n_tl], props)
            elem_id += 1
    
    return mp, nid


def main():
    """Solve once and write exports.json. Returns (model_part, node_index_map)
    so a verification script can `import` this file, call main(), and integrate
    the volume field against a manufactured solution — the coupling itself only
    ever needs the file handshake."""
    if min(abs(IFACE_X - X0), abs(IFACE_X - X1)) > TOL:
        sys.exit(f"IFACE_X={IFACE_X} is not an x-boundary of this subdomain "
                 f"[{X0},{X1}] — nothing is shared with the partner")
    # NX < 1 would put the interface column and the outer Dirichlet column on
    # the SAME nodes; the Dirichlet condition wins, the imported flux is
    # discarded, and the run still exits 0 with a plausible-looking export.
    if NX < 1 or NY < 1:
        sys.exit(f"NX,NY = {NX},{NY}: need at least one element in each "
                 "direction, and NX >= 1 so the interface and the outer "
                 "Dirichlet boundary do not land on the same nodes")

    # Build the model
    mp, nid = build_model()
    
    # Calculate mesh spacings for use in main()
    hx = (X1 - X0) / NX
    hy = (Y1 - Y0) / NY
    
    # Determine interface and outer boundary indices
    if ON_MAX_X:
        i_if = NX      # interface at x = X1
        i_out = 0      # outer Dirichlet at x = X0
    else:
        i_if = 0       # interface at x = X0
        i_out = NX     # outer Dirichlet at x = X1
    
    # Interface y-coordinates for sampling
    y_if = np.array([Y0 + j * hy for j in range(NY + 1)])
    
    # Read partner's flux data
    q_in = sample(read_imports(), "normal_fluxes", Q_INIT, y_if)

# ─────────────────────────────────────────────────────────────────────────
# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE.
#
# Build the mesh, the function space, the weak form and the linear solve for
# the problem you were given, in this backend, however you judge best. That is
# ordinary finite-element work and OASiS has no business dictating it.
#
# What OASiS does document — because you cannot guess it and it is what the
# interface check compares against — is everything AROUND the solve: the
# imports/exports handshake above, the interface sign convention, and the flux
# recovery below. Those are this tool's own interface, not your method.
#
# At this point you are expected to have produced:
#   * the discrete solution on this subdomain, with the partner's interface
#     data applied according to SIDE, and
#   * the assembled operator and the VOLUME load separately, because the flux
#     recovery below subtracts the volume load alone.
# ─────────────────────────────────────────────────────────────────────────

    # Apply volumetric source (HEAT_FLUX) at all nodes
    for (i, j), node_id in nid.items():
        x = X0 + i * hx
        y = Y0 + j * hy
        mp.Nodes[node_id].SetSolutionStepValue(KM.HEAT_FLUX, source(x, y))
    
    # Apply conductivity at all nodes
    for (i, j), node_id in nid.items():
        mp.Nodes[node_id].SetSolutionStepValue(KM.CONDUCTIVITY, K)
    
    # ── the interface: the partner's flux, applied UNCHANGED (see the header) ──
    for j in range(NY + 1):
        mp.Nodes[nid[(i_if, j)]].SetSolutionStepValue(
            KM.FACE_HEAT_FLUX, float(q_in[j]))
    props = mp.GetProperties()[1]
    for j in range(NY):
        mp.CreateNewCondition("FluxCondition2D2N", j + 1,
                              [nid[(i_if, j)], nid[(i_if, j + 1)]], props)

# ─────────────────────────────────────────────────────────────────────────
# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE.
#
# Build the mesh, the function space, the weak form and the linear solve for
# the problem you were given, in this backend, however you judge best. That is
# ordinary finite-element work and OASiS has no business dictating it.
#
# What OASiS does document — because you cannot guess it and it is what the
# interface check compares against — is everything AROUND the solve: the
# imports/exports handshake above, the interface sign convention, and the flux
# recovery below. Those are this tool's own interface, not your method.
#
# At this point you are expected to have produced:
#   * the discrete solution on this subdomain, with the partner's interface
#     data applied according to SIDE, and
#   * the assembled operator and the VOLUME load separately, because the flux
#     recovery below subtracts the volume load alone.
# ─────────────────────────────────────────────────────────────────────────

    # Apply Dirichlet boundary conditions on outer boundaries (fact 10)
    # Outer x-boundary
    for j in range(NY + 1):
        n = mp.Nodes[nid[(i_out, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
        n.Fix(KM.TEMPERATURE)
    
    # If FULL_OUTER_DIRICHLET, also fix top and bottom edges (excluding corners already done)
    if FULL_OUTER_DIRICHLET:
        # Bottom edge y = Y0 (j = 0), excluding interface and outer corners
        for i in range(1, NX):
            n = mp.Nodes[nid[(i, 0)]]
            n.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
            n.Fix(KM.TEMPERATURE)
        
        # Top edge y = Y1 (j = NY), excluding interface and outer corners
        for i in range(1, NX):
            n = mp.Nodes[nid[(i, NY)]]
            n.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
            n.Fix(KM.TEMPERATURE)
    
    # Add DOFs with reaction support (fact 3, 9)
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    
    # Set up solver stack (fact 9)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, True, False, False, False)
    
    # Solve
    strategy.Initialize()
    strategy.Solve()

    # Extract interface temperatures
    T = np.array([mp.Nodes[nid[(i_if, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                  for j in range(NY + 1)])

    # ── this side's own outward normal flux, from what the CONDITIONS assembled
    #
    # THE ARGUMENT THIS REPLACES WAS HALF RIGHT. It said the reaction formula
    # must not be used on a Neumann side because those interface dofs are free,
    # so r = A u - b is ~0 there and -r/w would export zero. True — but only
    # because Kratos stores REACTION_FLUX on FIXED dofs alone, and only when
    # the residual is taken against a load that ALREADY CONTAINS the interface
    # term. It then fell back to an L2 projection of the elementwise P1
    # gradient, defended as "acceptable here and only here, because a Dirichlet
    # partner reads this participant's values, never its normal_fluxes".
    #
    # The grader reads them. The two-sided interface jump is a GATE on a
    # coupled cell, and this side's export is half of it.
    #
    # MEASURED on the 3-D sibling, which had the identical construction, by
    # imposing q(y,z) = 2 + 3 sin(4y) cos(3z) and asking for it back:
    #     gradient averaging  1.27e+00, 8.37e-01, 3.95e-01   order 0.60, 1.08
    #     assembled conditions 5.16e-01, 1.77e-01, 4.77e-02  order 1.55, 1.89
    # The projection is O(h) where the field is O(h^2), so it, and not the
    # physics, sets the graded interface order.
    #
    # AN ECHO OF q_in WOULD NOT DO EITHER, though it is algebraically the exact
    # answer here: FluxCondition2D2N enforces K grad T . n = FACE_HEAT_FLUX and
    # FACE_HEAT_FLUX is the partner's array verbatim. An echo never passes
    # through the discretisation, so applied on the wrong facets or with the
    # wrong sign it reads the same and the balance gate still reports roundoff.
    # Summing each condition's own right-hand side is a MEASUREMENT of what
    # entered the linear system: it is int_Gamma g phi_i ds when the interface
    # is built correctly and something else the moment it is not. On those rows
    # the discrete equation gives r_i = +(assembled interface load), so the
    # outward density is -r_i/w_i — the same expression the Dirichlet side uses.
    info = mp.ProcessInfo
    rhs_i = np.zeros(len(mp.Nodes) + 1)
    w_i = np.zeros(len(mp.Nodes) + 1)
    for cond in mp.Conditions:
        vec = KM.Vector()
        cond.CalculateRightHandSide(vec, info)
        nds = cond.GetNodes()
        # w_i = int_Gamma phi_i ds: in 2-D the interface facets are segments,
        # so a node's weight is half of each segment it belongs to.
        p0, p1 = nds[0], nds[1]
        seg = ((p1.X - p0.X) ** 2 + (p1.Y - p0.Y) ** 2) ** 0.5
        for k, nd in enumerate(nds):
            rhs_i[nd.Id] += float(vec[k])
            w_i[nd.Id] += 0.5 * seg
    ids_if = [nid[(i_if, j)] for j in range(NY + 1)]
    r_if = np.array([rhs_i[i] for i in ids_if])
    wq = np.array([w_i[i] for i in ids_if])
    Q = np.where(np.abs(wq) > 1e-14, -r_if / np.maximum(np.abs(wq), 1e-300)
                 * np.sign(np.where(wq == 0, 1.0, wq)), 0.0)
    if FULL_OUTER_DIRICHLET:
        # Corner reactions also contain the perpendicular outer-boundary flux
        # and cannot be separated into one interface contribution. C2 excludes
        # them from grading; retain the points for exchange but not that mixed
        # reaction.
        Q[[0, -1]] = 0.0

    # ── CONSERVATION SELF-CHECK: the discrete divergence theorem ──────────────
    # Summing the unconstrained residual r = A u - b over ALL nodes gives
    # sum_i r_i = -sum_i b_i, because sum_i A_ij = int K grad(sum_i phi_i).grad
    # phi_j = 0 (the phi_i are a partition of unity). r vanishes on free rows,
    # so over the fixed rows alone
    #       sum_{outer} r_i  +  int_Omega f dOmega  +  int_Gamma q_applied ds  =  0
    # exactly, at round-off, for ANY mesh. It is not a discretisation check: it
    # fails only if the flux was applied with the wrong sign or magnitude, or
    # not applied at all — which is precisely the failure mode this side has.
    load_iface = float(hy * (0.5 * q_in[0] + q_in[1:-1].sum() + 0.5 * q_in[-1]))
    load_vol = 0.0
    for el in mp.Elements:
        nds = el.GetNodes()
        x = [n.X for n in nds]
        y = [n.Y for n in nds]
        det = (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
        load_vol += (0.5 * abs(det)) * sum(
            n.GetSolutionStepValue(KM.HEAT_FLUX) for n in nds) / 3.0
    react = sum(mp.Nodes[nid[(i_out, j)]].GetSolutionStepValue(KM.REACTION_FLUX)
                for j in range(NY + 1))
    imb_abs = abs(react + load_vol + load_iface)
    scale = max(abs(react), abs(load_vol), abs(load_iface))
    # On iteration 1 with Q_INIT = 0 and no source there is no heat flow at all,
    # every term is round-off, and a RATIO of round-off to round-off is O(1)
    # while meaning nothing. Say "trivial" instead of printing a 1.0 that reads
    # as a 100% conservation error.
    if scale <= 1e-10 * K * max(1.0, abs(T_OUTER)) * (Y1 - Y0):
        bal = f"balance trivial (no heat flow yet, |imbalance|={imb_abs:.3e})"
    else:
        bal = (f"balance |sum(reactions)+vol+iface| = {imb_abs:.3e} abs / "
               f"{imb_abs / scale:.3e} rel")

    print(f"[kratos neumann] interface n={len(T)} "
          f"q_applied=[{q_in.min():.6g},{q_in.max():.6g}] "
          f"T=[{T.min():.6g},{T.max():.6g}] {bal}")
    print(f"NDOF = {len(mp.Nodes)}")

    field_coords = [[float(n.X), float(n.Y)] for n in mp.Nodes]
    field_values = [float(n.GetSolutionStepValue(KM.TEMPERATURE))
                    for n in mp.Nodes]
    Path("field.json").write_text(json.dumps({
        "coordinates": field_coords,
        "values": field_values,
    }, indent=2))

    # exports.json LAST: the driver takes its existence as proof of success.
    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": int(len(T)),
        "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
        "values": [float(t) for t in T],
        "normal_fluxes": [float(q) for q in Q],
    }, indent=2))
    return mp, nid


if __name__ == "__main__":
    main()

# ── WHAT YOUR SOLVE MUST LEAVE BEHIND ─────────────────────────
# The code above and below the elided block uses these names. Your
# block has to define every one of them, or the rest will not run:
#
#     i_if
#     i_out
#     mp
#     n
#     props
#     y_if
#
# That is the whole contract. Read the surviving lines to see the
# shape each one has to have -- they are already indexed, assembled
# or written out there. OASiS does not serve the solve itself, but
# it will not make you guess which variables the hole was filling.
