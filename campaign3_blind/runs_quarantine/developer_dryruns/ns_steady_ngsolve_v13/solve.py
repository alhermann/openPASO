"""
Steady Incompressible Navier-Stokes Solver using NGSolve
Taylor-Hood P2/P1 elements with Newton's method

Problem: -nu*lap(u) + (u.grad)u + grad(p) = f, div(u) = 0
Domain: unit square (0,1)x(0,1)
BCs: u = 0 on entire boundary
Pressure pinning: integral(p) = 0 over domain (via single DOF pinning)
"""

from ngsolve import *
from ngsolve import solvers
import numpy as np
import os

# Problem parameters
nu = 1.0  # kinematic viscosity

# Source term components f_ux and f_uy
def f_ux_expr(x, y):
    return (5*x**9*y**6/4 - 15*x**9*y**5/4 + 35*x**9*y**4/8 - 5*x**9*y**3/2 + 
            5*x**9*y**2/8 - 8*x**8*y**7/5 - 44*x**8*y**6/5 + 351*x**8*y**5/10 - 
            883*x**8*y**4/20 + 131*x**8*y**3/5 - 27*x**8*y**2/4 + 4*x**7*y**8/5 + 
            236*x**7*y**7/25 + 108*x**7*y**6/5 - 3068*x**7*y**5/25 + 4147*x**7*y**4/25 - 
            102*x**7*y**3 + 27*x**7*y**2 - 14*x**6*y**8/5 - 516*x**6*y**7/25 - 
            49*x**6*y**6/2 + 10441*x**6*y**5/50 - 29833*x**6*y**4/100 + 941*x**6*y**3/5 - 
            203*x**6*y**2/4 + 18*x**5*y**8/5 + 532*x**5*y**7/25 + 263*x**5*y**6/20 - 
            18489*x**5*y**5/100 + 55117*x**5*y**4/200 - 1771*x**5*y**3/10 + 387*x**5*y**2/8 - 
            6*x**5*y + 3*x**5 - 2*x**4*y**8 - 52*x**4*y**7/5 - 27*x**4*y**6/10 + 
            406*x**4*y**5/5 - 626*x**4*y**4/5 + 408*x**4*y**3/5 - 21*x**4*y**2/2 + 
            132*x**4*y/5 - 84*x**4/5 + 2*x**3*y**8/5 + 48*x**3*y**7/25 - 
            344*x**3*y**5/25 + 546*x**3*y**4/25 - 172*x**3*y**3/5 + 10*x**3*y**2 - 
            224*x**3*y/5 + 123*x**3/5 + 12*x**2*y**4 + 264*x**2*y**3/5 - 
            444*x**2*y**2/5 + 252*x**2*y/5 - 54*x**2/5 - 12*x*y**4 - 174*x*y**3/5 + 
            369*x*y**2/5 - 85*x*y/3 + 2*y**4 + 24*y**3/5 - 54*y**2/5 + 
            pi*y*sin(pi*x)/3 + 7*y)

def f_uy_expr(x, y):
    return (5*x**8*y**7/4 - 35*x**8*y**6/8 + 45*x**8*y**5/8 - 25*x**8*y**4/8 + 
            5*x**8*y**3/8 - 8*x**7*y**8/5 - 32*x**7*y**7/5 + 174*x**7*y**6/5 - 
            50*x**7*y**5 + 146*x**7*y**4/5 - 6*x**7*y**3 + 4*x**6*y**9/5 + 
            451*x**6*y**8/50 + 171*x**6*y**7/25 - 5257*x**6*y**6/50 + 8609*x**6*y**5/50 - 
            531*x**6*y**4/5 + 45*x**6*y**3/2 - 12*x**5*y**9/5 - 459*x**5*y**8/25 + 
            127*x**5*y**7/25 + 7921*x**5*y**6/50 - 14307*x**5*y**5/50 + 1829*x**5*y**4/10 - 
            79*x**5*y**3/2 + 14*x**4*y**9/5 + 911*x**4*y**8/50 - 1421*x**4*y**7/100 - 
            26213*x**4*y**6/200 + 50851*x**4*y**5/200 - 1331*x**4*y**4/8 + 291*x**4*y**3/8 + 
            15*x**4*y**2 - 15*x**4*y + 5*x**4/2 - 8*x**3*y**9/5 - 236*x**3*y**8/25 + 
            258*x**3*y**7/25 + 1492*x**3*y**6/25 - 3064*x**3*y**5/25 + 408*x**3*y**4/5 - 
            34*x**3*y**3 - 264*x**3*y**2/5 + 336*x**3*y/5 - 12*x**3 + 2*x**2*y**9/5 + 
            54*x**2*y**8/25 - 72*x**2*y**7/25 - 308*x**2*y**6/25 + 666*x**2*y**5/25 - 
            3*x**2*y**4 - 2*x**2*y**3 + 336*x**2*y**2/5 - 369*x**2*y/5 + 77*x**2/6 - 
            24*x*y**5/5 - 132*x*y**4/5 + 296*x*y**3/5 - 252*x*y**2/5 + 108*x*y/5 - x + 
            12*y**5/5 + 87*y**4/10 - 123*y**3/5 + 27*y**2/2 - cos(pi*x)/3)


def solve_navier_stokes(maxh, level_num, output_dir="."):
    """
    Solve steady Navier-Stokes on uniform mesh with given max element size.
    
    Uses Taylor-Hood elements: P2 for velocity, P1 for pressure.
    Newton's method for nonlinear system via NGSolve's built-in solver.
    Pressure pinned at a single interior node.
    """
    print(f"\n{'='*60}")
    print(f"Level {level_num}: Solving with maxh = {maxh}")
    print(f"{'='*60}")
    
    # Create uniform mesh on unit square
    from netgen.geom2d import unit_square
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    print(f"Mesh created: {mesh.ne} elements, {mesh.nv} vertices")
    
    # Taylor-Hood elements: VectorH1(order=2) for velocity, H1(order=1) for pressure
    V = VectorH1(mesh, order=2, dirichlet=".*")  # Dirichlet on all boundaries
    Q = H1(mesh, order=1)  # Pressure space (no BC)
    X = V * Q  # Mixed space
    
    # Trial and test functions
    (u, p), (v, q) = X.TnT()
    
    # Define source term as CoefficientFunction using built-in x, y coordinates
    f_vec = CoefficientFunction((f_ux_expr(x, y), f_uy_expr(x, y)))
    
    # Nonlinear form A(w)[test] = 0 where w = (u, p)
    a = BilinearForm(X, symmetric=False)
    
    # Weak form of steady Navier-Stokes:
    # Strong form: -nu*lap(u) + (u.grad)u + grad(p) = f
    # After integration by parts:
    # nu*grad(u):grad(v) + ((u.grad)u).v - p*div(v) - f.v + div(u)*q = 0
    
    # Viscous term: nu * InnerProduct(Grad(u), Grad(v))
    a += nu * InnerProduct(Grad(u), Grad(v)) * dx
    
    # Convective term: ((u.grad)u).v = InnerProduct(u, Grad(u)*v)
    a += InnerProduct(u, Grad(u)*v) * dx
    
    # Pressure gradient term: -p*div(v)
    a += -p * div(v) * dx
    
    # Source term: -f.v (moved to LHS)
    a += -InnerProduct(f_vec, v) * dx
    
    # Continuity equation: div(u)*q
    a += div(u) * q * dx
    
    # Initial guess: zero velocity and pressure
    gfu = GridFunction(X)
    
    # Get free DOFs - for product space, we need to handle pressure pinning
    freedofs = X.FreeDofs()
    
    print(f"Total DOFs: {X.ndof}")
    print(f"V.ndof: {V.ndof}, Q.ndof: {Q.ndof}")
    print(f"Free DOFs before pinning: {len(freedofs)}")
    
    # Pressure pinning: pin the first pressure DOF (at index V.ndof)
    # This removes the constant pressure null space
    pin_dof = V.ndof
    print(f"Pinning pressure at DOF {pin_dof}")
    
    # Remove pinned DOF from free DOFs using Clear
    freedofs.Clear(pin_dof)
    print(f"Free DOFs after pinning: {len(freedofs)}")
    
    # Set pinned value to 0
    gfu.vec[pin_dof] = 0.0
    
    # Set up Newton iteration using NGSolve's built-in solver
    max_newton_iter = 50
    newton_tol = 1e-10
    
    print("Starting Newton iteration...")
    
    # Use NGSolve's Newton solver
    result = solvers.Newton(a, gfu, freedofs=freedofs, maxit=max_newton_iter, 
                           maxerr=newton_tol, inverse='umfpack', printing=True)
    
    converged_flag, num_iters = result
    converged = (converged_flag == 0)
    
    if converged:
        print(f"Newton converged in {num_iters} iterations!")
    else:
        print(f"Warning: Newton did not converge in {max_newton_iter} iterations")
    
    # Ensure pinned DOF stays at zero
    gfu.vec[pin_dof] = 0.0
    
    # Extract velocity and pressure
    velocity = gfu.components[0]
    pressure = gfu.components[1]
    
    # Shift pressure to have zero mean
    p_mean = Integrate(pressure, mesh) / Integrate(1, mesh)
    print(f"Mean pressure: {p_mean:.6e}")
    
    # Create shifted pressure grid function
    pressure_shifted = GridFunction(Q)
    for i in range(len(pressure.vec)):
        pressure_shifted.vec[i] = pressure.vec[i] - p_mean
    
    # Compute final residual for logging
    R = LinearForm(X)
    curr_u = gfu.components[0]
    curr_p = gfu.components[1]
    R += nu * InnerProduct(Grad(curr_u), Grad(v)) * dx
    R += InnerProduct(curr_u, Grad(curr_u)*v) * dx
    R += -curr_p * div(v) * dx
    R += -InnerProduct(f_vec, v) * dx
    R += div(curr_u) * q * dx
    R.Assemble()
    res_norm_all = R.vec.Norm()
    
    # Write run log
    log_file = os.path.join(output_dir, f"run_level{level_num}.log")
    with open(log_file, 'w') as f:
        f.write(f"NDOF = {X.ndof}\n")
        f.write(f"Elements = {mesh.ne}\n")
        f.write(f"Vertices = {mesh.nv}\n")
        f.write(f"Newton iterations = {num_iters}\n")
        f.write(f"Final residual = {res_norm_all:.6e}\n")
        f.write(f"Maxh = {maxh}\n")
    print(f"Run log written to {log_file}")
    
    # Generate probe points
    probe_points = []
    for ix in range(44):
        for iy in range(44):
            px = (ix + 0.5) / 44.0
            py = (iy + 0.5) / 44.0
            probe_points.append((px, py))
    
    print(f"Generated {len(probe_points)} probe points")
    
    # Evaluate solution at probe points and write CSV
    csv_file = os.path.join(output_dir, f"solution_level{level_num}.csv")
    with open(csv_file, 'w') as f:
        f.write("x,y,ux,uy\n")
        for (px, py) in probe_points:
            # Evaluate GridFunction directly at point
            vel_at_pt = velocity(px, py)
            ux_val = vel_at_pt[0]
            uy_val = vel_at_pt[1]
            f.write(f"{px:.15e},{py:.15e},{ux_val:.15e},{uy_val:.15e}\n")
    
    print(f"Solution CSV written to {csv_file}")
    
    # Also write VTK for visualization
    vtk = VTKOutput(mesh, coefs=[velocity, pressure_shifted], 
                    names=["velocity", "pressure"], 
                    filename=os.path.join(output_dir, f"result_level{level_num}"),
                    subdivision=1)
    vtk.Do()
    print(f"VTK output written")
    
    return {
        'ndof': X.ndof,
        'elements': mesh.ne,
        'newton_iters': num_iters,
        'final_residual': res_norm_all,
        'velocity': velocity,
        'pressure': pressure_shifted,
        'mesh': mesh
    }


def main():
    """Main driver: solve at three mesh levels and check convergence."""
    
    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    # Mesh levels: h = 1/8, 1/16, 1/32
    mesh_levels = [1/8, 1/16, 1/32]
    
    results = []
    for i, maxh in enumerate(mesh_levels):
        level_num = i + 1
        result = solve_navier_stokes(maxh, level_num, output_dir)
        results.append(result)
    
    # Check mesh independence
    print("\n" + "="*60)
    print("MESH INDEPENDENCE ANALYSIS")
    print("="*60)
    
    csv_files = [f"solution_level{k}.csv" for k in range(1, 4)]
    
    def read_csv(filename):
        """Read CSV and return arrays of x, y, ux, uy."""
        data = []
        with open(filename, 'r') as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split(',')
                data.append([float(p) for p in parts])
        data = np.array(data)
        return data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    
    # Read finest two levels
    x2, y2, ux2, uy2 = read_csv(csv_files[1])  # Level 2 (h=1/16)
    x3, y3, ux3, uy3 = read_csv(csv_files[2])  # Level 3 (h=1/32)
    
    # Compute relative changes
    diff_ux = ux3 - ux2
    diff_uy = uy3 - uy2
    
    # L2 relative change
    l2_ux2 = np.sqrt(np.mean(ux2**2))
    l2_uy2 = np.sqrt(np.mean(uy2**2))
    rel_change_ux_l2 = np.sqrt(np.mean(diff_ux**2)) / (l2_ux2 + 1e-15)
    rel_change_uy_l2 = np.sqrt(np.mean(diff_uy**2)) / (l2_uy2 + 1e-15)
    
    # Max relative change at any point
    max_rel_ux = np.max(np.abs(diff_ux) / (np.abs(ux2) + 1e-15))
    max_rel_uy = np.max(np.abs(diff_uy) / (np.abs(uy2) + 1e-15))
    
    max_rel_change = max(rel_change_ux_l2, rel_change_uy_l2, max_rel_ux, max_rel_uy)
    
    print(f"L2 relative change ux: {rel_change_ux_l2:.6e}")
    print(f"L2 relative change uy: {rel_change_uy_l2:.6e}")
    print(f"Max relative change ux: {max_rel_ux:.6e}")
    print(f"Max relative change uy: {max_rel_uy:.6e}")
    print(f"Overall max relative change: {max_rel_change:.6e}")
    
    # Determine convergence (threshold: 1% relative change)
    tol = 0.01
    converged = max_rel_change < tol
    
    print(f"\nMesh independence criterion (tol={tol}): {'CONVERGED' if converged else 'NOT_CONVERGED'}")
    
    # Write RESULT.txt
    with open(os.path.join(output_dir, "RESULT.txt"), 'w') as f:
        f.write(f"LEVELS = 3\n")
        f.write(f"FILES = {', '.join(csv_files)}\n")
        f.write(f"MESH_INDEPENDENCE = {'CONVERGED' if converged else 'NOT_CONVERGED'}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")
    
    print(f"\nRESULT.txt written")
    print("Done!")


if __name__ == "__main__":
    main()
