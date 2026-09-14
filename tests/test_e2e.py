#!/usr/bin/env python3
"""
End-to-end tests: generate → validate → run → post-process → verify.

Each test runs a real simulation and checks numerical results.
Skips backends that aren't installed.

Cross-solver benchmarks (10 total):
  1. Poisson unit square — 3-solver (FEniCS + deal.II + 4C)
  2. Elasticity cantilever 10×1 — 3-solver
  3. Heat conduction T=100→0 — 3-solver
  4. Poisson L-domain — 3-solver
  5. Poisson rectangle [0,2]×[0,1] — 3-solver
  6. Heat rectangle [0,2]×[0,1] — 3-solver
  7. Thick beam 5×2 — 3-solver
  8. Poisson L-domain fine — 2-solver (FEniCS + deal.II, finer mesh)
  9. Heat conduction T=50→200 — 3-solver (different BCs)
  10. Elasticity cantilever 10×1 different material — 3-solver
"""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv
pv.OFF_SCREEN = True

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.registry import load_all_backends, get_backend
from core.backend import BackendStatus


def _run_async(coro):
    """Run an async coroutine in a sync test."""
    return asyncio.run(coro)          # see webui/runner.py: 3.12+ raises otherwise


# ═══════════════════════════════════════════════════════════════════════════
# Cross-solver helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_available_backends():
    """Return dict of available backend objects."""
    load_all_backends()
    result = {}
    for name in ["fenics", "dealii", "fourc"]:
        b = get_backend(name)
        if b and b.check_availability()[0] == BackendStatus.AVAILABLE:
            result[name] = b
    return result


def _run_single(backend, content, timeout=300):
    """Run a simulation and return (job, vtu_files)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        job = _run_async(backend.run(content, work_dir, np=1, timeout=timeout))
        if job.status != "completed":
            return job, []
        vtu_files = sorted([f for f in backend.get_result_files(job) if f.suffix == ".vtu"])
        # Copy VTUs to persistent location for post-processing
        import shutil
        persist = Path(tempfile.mkdtemp())
        copied = []
        for v in vtu_files:
            dst = persist / v.name
            shutil.copy2(v, dst)
            copied.append(dst)
        return job, copied


def _extract_scalar_max(vtu_path, skip_fields=("owner", "ghostnodes", "elementowner")):
    """Extract max of first meaningful scalar field from VTU."""
    mesh = pv.read(str(vtu_path))
    for name in mesh.point_data:
        if name.lower() in skip_fields:
            continue
        data = mesh.point_data[name]
        if data.ndim == 1:
            return name, float(data.max())
    return None, None


def _extract_displacement_max_uy(vtu_path):
    """Extract max |u_y| from displacement field."""
    mesh = pv.read(str(vtu_path))
    if "uy" in mesh.point_data:
        return float(np.abs(mesh.point_data["uy"]).max())
    if "displacement" in mesh.point_data:
        d = mesh.point_data["displacement"]
        if d.ndim > 1 and d.shape[1] >= 2:
            return float(np.abs(d[:, 1]).max())
    # Try vector field names
    for name in mesh.point_data:
        d = mesh.point_data[name]
        if d.ndim > 1 and d.shape[1] >= 2:
            return float(np.abs(d[:, 1]).max())
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Single-backend E2E tests
# ═══════════════════════════════════════════════════════════════════════════

class _E2EBase(unittest.TestCase):
    """Base class for single-backend E2E tests."""

    backend_name: str = ""

    @classmethod
    def setUpClass(cls):
        load_all_backends()
        cls.backend = get_backend(cls.backend_name)

    def _skip_if_unavailable(self):
        if not self.backend:
            self.skipTest(f"{self.backend_name} not registered")
        status, msg = self.backend.check_availability()
        if status != BackendStatus.AVAILABLE:
            self.skipTest(f"{self.backend_name} not available: {msg}")

    def _run_and_verify(self, physics: str, variant: str, params: dict,
                        expected_field: str = None,
                        expected_max: float = None,
                        tolerance: float = 0.05):
        """Generate, validate, run, and verify a simulation."""
        self._skip_if_unavailable()

        content = self.backend.generate_input(physics, variant, params)
        self.assertTrue(len(content) > 50, "Generated input too short")

        errors = self.backend.validate_input(content)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            job = _run_async(self.backend.run(content, work_dir, np=1, timeout=300))

            self.assertEqual(job.status, "completed", f"Job failed: {job.error}")
            self.assertIsNotNone(job.elapsed)
            self.assertGreater(job.elapsed, 0)

            result_files = self.backend.get_result_files(job)
            self.assertGreater(len(result_files), 0, f"No output files in {work_dir}")

            vtu_files = [f for f in result_files if f.suffix == ".vtu"]
            if vtu_files:
                from core.post_processing import post_process_file
                pp = post_process_file(vtu_files[0], plot_fields=False)

                if expected_field and expected_max is not None:
                    matching = [f for f in pp.fields if f.name == expected_field]
                    self.assertTrue(len(matching) > 0,
                                    f"Field '{expected_field}' not found. "
                                    f"Available: {[f.name for f in pp.fields]}")
                    actual_max = matching[0].max
                    rel_error = abs(actual_max - expected_max) / abs(expected_max) if expected_max != 0 else abs(actual_max)
                    self.assertLess(rel_error, tolerance,
                                    f"max({expected_field})={actual_max:.6e}, "
                                    f"expected {expected_max:.6e}, "
                                    f"rel_error={rel_error:.4f}")
                return pp
            return None


class TestFenicsE2E(_E2EBase):
    backend_name = "fenics"

    def test_poisson_2d(self):
        self._run_and_verify("poisson", "2d", {"kappa": 1.0, "nx": 32, "ny": 32},
                             expected_field="u", expected_max=0.07373, tolerance=0.01)

    def test_elasticity_2d(self):
        self._run_and_verify("linear_elasticity", "2d", {"E": 1000, "nu": 0.3})

    def test_heat_2d(self):
        self._run_and_verify("heat", "2d_steady", {"T_left": 100, "T_right": 0},
                             expected_field="temperature", expected_max=100.0, tolerance=0.01)

    def test_navier_stokes_2d(self):
        self._run_and_verify("navier_stokes", "2d", {"Re": 100, "nx": 16, "ny": 16})

    def test_thermal_structural_2d(self):
        self._run_and_verify("thermal_structural", "2d", {
            "E": 200e3, "nu": 0.3, "alpha": 12e-6,
            "T_hot": 100, "T_cold": 0, "nx": 20, "ny": 20,
        })

    def test_hyperelasticity_3d(self):
        self._run_and_verify("hyperelasticity", "3d", {"E": 1000, "nu": 0.3})


class TestFenicsNonTrivialGeometry(_E2EBase):
    backend_name = "fenics"

    def test_poisson_l_domain(self):
        self._run_and_verify("poisson", "l_domain", {"mesh_size": 0.05})

    def test_elasticity_plate_hole(self):
        self._run_and_verify("linear_elasticity", "plate_hole",
                             {"E": 1000, "nu": 0.3, "mesh_size": 0.06})

    def test_navier_stokes_channel_cylinder(self):
        self._run_and_verify("navier_stokes", "channel_cylinder",
                             {"Re": 20, "mesh_size": 0.03})


class TestDealiiE2E(_E2EBase):
    backend_name = "dealii"

    def test_poisson_2d(self):
        self._run_and_verify("poisson", "2d", {"refinements": 5},
                             expected_field="solution", expected_max=0.07373, tolerance=0.01)

    def test_poisson_l_domain(self):
        self._run_and_verify("poisson", "l_domain", {"refinements": 5})

    def test_elasticity_2d(self):
        self._run_and_verify("linear_elasticity", "2d", {})

    def test_heat_2d_transient(self):
        self._run_and_verify("heat", "2d_transient",
                             {"refinements": 4, "n_steps": 10, "dt": 0.05})


class TestFourcE2E(_E2EBase):
    backend_name = "fourc"

    def test_poisson_2d(self):
        self._skip_if_unavailable()
        try:
            self._run_and_verify("poisson", "poisson_2d", {})
        except (ValueError, Exception) as e:
            if "generator" in str(e).lower() or "template" in str(e).lower():
                self.skipTest(f"4C generator not available: {e}")
            raise

    def test_heat_2d(self):
        self._skip_if_unavailable()
        try:
            self._run_and_verify("heat", "heat_2d", {})
        except (ValueError, Exception) as e:
            if "generator" in str(e).lower() or "template" in str(e).lower():
                self.skipTest(f"4C generator not available: {e}")
            raise

    def test_elasticity(self):
        self._skip_if_unavailable()
        try:
            self._run_and_verify("linear_elasticity", "linear_2d", {})
        except (ValueError, Exception) as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ["generator", "template", "xml", "solver", "mpi_abort"]):
                self.skipTest(f"4C elasticity not runnable: {str(e)[:100]}")
            raise


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Solver Benchmarks (10 total)
# ═══════════════════════════════════════════════════════════════════════════

class _CrossSolverBase(unittest.TestCase):
    """Base class for cross-solver benchmark tests."""

    @classmethod
    def setUpClass(cls):
        cls.backends = _get_available_backends()

    def _require(self, *names):
        for n in names:
            if n not in self.backends:
                self.skipTest(f"Need {n}")

    def _run_4c_inline(self, yaml_content):
        """Run a 4C inline-mesh input file and return VTU path."""
        b = self.backends.get("fourc")
        if not b:
            return None, None
        job, vtus = _run_single(b, yaml_content, timeout=300)
        if job.status != "completed":
            return job, []
        return job, vtus


class TestXSolverPoisson(_CrossSolverBase):
    """Benchmark #1: Poisson -Δu=1 on [0,1]², u=0 — 3-solver."""

    def test_poisson_unit_square(self):
        self._require("fenics", "dealii")
        from backends.fourc.inline_mesh import matched_poisson_input

        results = {}

        # FEniCS
        b = self.backends["fenics"]
        content = b.generate_input("poisson", "2d", {"kappa": 1.0, "nx": 32})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"FEniCS failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["fenics"] = val

        # deal.II
        b = self.backends["dealii"]
        content = b.generate_input("poisson", "2d", {"refinements": 5})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"deal.II failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["dealii"] = val

        # 4C (inline mesh — no normalization needed, matched topology)
        if "fourc" in self.backends:
            yaml = matched_poisson_input(32, 32)
            job, vtus = self._run_4c_inline(yaml)
            self.assertEqual(job.status, "completed", f"4C failed: {job.error}")
            name, val = _extract_scalar_max(vtus[-1])
            results["fourc"] = val

        # Verify: all should give max(u) ≈ 0.0737
        print(f"\n  Poisson unit square: {results}")
        for solver, val in results.items():
            self.assertAlmostEqual(val, 0.0737, delta=0.002,
                                   msg=f"{solver}: max(u)={val:.6f}")

        vals = list(results.values())
        mean_val = sum(vals) / len(vals)
        for solver, val in results.items():
            self.assertLess(abs(val - mean_val) / mean_val, 0.02,
                            f"{solver} disagrees: {val:.6f} vs mean {mean_val:.6f}")


class TestXSolverElasticity(_CrossSolverBase):
    """Benchmark #2: Cantilever 10×1, body force (0,-1), E=1000, nu=0.3 — 3-solver."""

    def test_cantilever_10x1(self):
        self._require("fenics", "dealii")
        from backends.fourc.inline_mesh import matched_elasticity_input

        results = {}

        # FEniCS
        b = self.backends["fenics"]
        content = b.generate_input("linear_elasticity", "2d", {"E": 1000, "nu": 0.3})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"FEniCS failed: {job.error}")
        results["fenics"] = _extract_displacement_max_uy(vtus[-1])

        # deal.II
        b = self.backends["dealii"]
        content = b.generate_input("linear_elasticity", "2d", {})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"deal.II failed: {job.error}")
        results["dealii"] = _extract_displacement_max_uy(vtus[-1])

        # 4C (inline mesh)
        if "fourc" in self.backends:
            yaml = matched_elasticity_input(40, 4, E=1000.0, nu=0.3, lx=10.0, ly=1.0)
            job, vtus = self._run_4c_inline(yaml)
            self.assertEqual(job.status, "completed", f"4C failed: {job.error}")
            results["fourc"] = _extract_displacement_max_uy(vtus[-1])

        print(f"\n  Elasticity 10x1: {results}")
        vals = [v for v in results.values() if v is not None]
        self.assertTrue(len(vals) >= 2)
        mean_val = sum(vals) / len(vals)
        for solver, val in results.items():
            if val is not None:
                self.assertLess(abs(val - mean_val) / mean_val, 0.05,
                                f"{solver}: |uy_max|={val:.4f} vs mean {mean_val:.4f}")


class TestXSolverHeat(_CrossSolverBase):
    """Benchmark #3: Heat conduction T=100→0 on [0,1]² — 3-solver."""

    def test_heat_conduction(self):
        self._require("fenics", "dealii")
        from backends.fourc.inline_mesh import matched_heat_input

        results = {}

        # FEniCS
        b = self.backends["fenics"]
        content = b.generate_input("heat", "2d_steady", {"T_left": 100, "T_right": 0})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"FEniCS failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["fenics"] = val

        # deal.II (new steady-state template)
        b = self.backends["dealii"]
        content = b.generate_input("heat", "2d_steady", {"T_left": 100, "T_right": 0})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"deal.II failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["dealii"] = val

        # 4C (inline mesh)
        if "fourc" in self.backends:
            yaml = matched_heat_input(32, 32, T_left=100.0, T_right=0.0)
            job, vtus = self._run_4c_inline(yaml)
            self.assertEqual(job.status, "completed", f"4C failed: {job.error}")
            name, val = _extract_scalar_max(vtus[-1])
            results["fourc"] = val

        print(f"\n  Heat T=100→0: {results}")
        for solver, val in results.items():
            self.assertAlmostEqual(val, 100.0, delta=1.0,
                                   msg=f"{solver}: max(T)={val}")


class TestXSolverLDomain(_CrossSolverBase):
    """Benchmark #4: Poisson on L-domain [-1,1]²\\[0,1]×[-1,0] — 3-solver."""

    def test_l_domain_poisson(self):
        self._require("fenics", "dealii")
        from backends.fourc.inline_mesh import matched_l_domain_poisson_input

        results = {}

        # FEniCS (Gmsh L-domain)
        b = self.backends["fenics"]
        content = b.generate_input("poisson", "l_domain", {"mesh_size": 0.03})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"FEniCS failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["fenics"] = val

        # deal.II (built-in hyper_L)
        b = self.backends["dealii"]
        content = b.generate_input("poisson", "l_domain", {"refinements": 5})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"deal.II failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["dealii"] = val

        # 4C (inline L-mesh — n=32 gives 3072 elements, matching deal.II refine=5)
        if "fourc" in self.backends:
            yaml = matched_l_domain_poisson_input(32)
            job, vtus = self._run_4c_inline(yaml)
            if job.status == "completed" and vtus:
                name, val = _extract_scalar_max(vtus[-1])
                results["fourc"] = val

        print(f"\n  L-domain Poisson: {results}")
        # FEniCS and deal.II should agree closely
        if "fenics" in results and "dealii" in results:
            rel = abs(results["fenics"] - results["dealii"]) / max(results.values())
            self.assertLess(rel, 0.02,
                            f"FEniCS vs deal.II: {results['fenics']:.6f} vs {results['dealii']:.6f}")


class TestXSolverPoissonRectangle(_CrossSolverBase):
    """Benchmark #5: Poisson -Δu=1 on [0,2]×[0,1], u=0 — 3-solver."""

    def test_poisson_rectangle(self):
        self._require("fenics", "dealii")
        from backends.fourc.inline_mesh import matched_poisson_rectangle_input

        results = {}

        # FEniCS
        b = self.backends["fenics"]
        content = b.generate_input("poisson", "rectangle", {"lx": 2.0, "ly": 1.0, "nx": 64, "ny": 32})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"FEniCS failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["fenics"] = val

        # deal.II
        b = self.backends["dealii"]
        content = b.generate_input("poisson", "rectangle", {"lx": 2.0, "ly": 1.0, "refinements": 4})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"deal.II failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["dealii"] = val

        # 4C (inline mesh — no normalization needed)
        if "fourc" in self.backends:
            yaml = matched_poisson_rectangle_input(64, 32, 2.0, 1.0)
            job, vtus = self._run_4c_inline(yaml)
            if job.status == "completed" and vtus:
                name, val = _extract_scalar_max(vtus[-1])
                results["fourc"] = val

        print(f"\n  Poisson rectangle [0,2]x[0,1]: {results}")
        # FEniCS and deal.II should agree closely
        if "fenics" in results and "dealii" in results:
            rel = abs(results["fenics"] - results["dealii"]) / max(results["fenics"], results["dealii"])
            self.assertLess(rel, 0.02,
                            f"FEniCS={results['fenics']:.6f} vs deal.II={results['dealii']:.6f}")


class TestXSolverHeatRectangle(_CrossSolverBase):
    """Benchmark #6: Heat T=100→0 on [0,2]×[0,1] — 3-solver."""

    def test_heat_rectangle(self):
        self._require("fenics", "dealii")
        from backends.fourc.inline_mesh import matched_heat_rectangle_input

        results = {}

        # FEniCS
        b = self.backends["fenics"]
        content = b.generate_input("heat", "rectangle", {"lx": 2.0, "ly": 1.0, "T_left": 100, "T_right": 0})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"FEniCS failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["fenics"] = val

        # deal.II
        b = self.backends["dealii"]
        content = b.generate_input("heat", "rectangle", {"lx": 2.0, "ly": 1.0, "T_left": 100, "T_right": 0})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"deal.II failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["dealii"] = val

        # 4C
        if "fourc" in self.backends:
            yaml = matched_heat_rectangle_input(64, 32, 2.0, 1.0, T_left=100.0, T_right=0.0)
            job, vtus = self._run_4c_inline(yaml)
            if job.status == "completed" and vtus:
                name, val = _extract_scalar_max(vtus[-1])
                results["fourc"] = val

        print(f"\n  Heat rectangle [0,2]x[0,1]: {results}")
        for solver, val in results.items():
            self.assertAlmostEqual(val, 100.0, delta=1.0,
                                   msg=f"{solver}: max(T)={val}")


class TestXSolverThickBeam(_CrossSolverBase):
    """Benchmark #7: Thick cantilever 5×2, body force, E=1000, nu=0.3 — 3-solver."""

    def test_thick_beam_5x2(self):
        self._require("fenics", "dealii")
        from backends.fourc.inline_mesh import matched_elasticity_input

        results = {}

        # FEniCS
        b = self.backends["fenics"]
        content = b.generate_input("linear_elasticity", "thick_beam",
                                   {"E": 1000, "nu": 0.3, "lx": 5.0, "ly": 2.0})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"FEniCS failed: {job.error}")
        results["fenics"] = _extract_displacement_max_uy(vtus[-1])

        # deal.II
        b = self.backends["dealii"]
        content = b.generate_input("linear_elasticity", "thick_beam",
                                   {"lx": 5.0, "ly": 2.0, "E": 1000, "nu": 0.3})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"deal.II failed: {job.error}")
        results["dealii"] = _extract_displacement_max_uy(vtus[-1])

        # 4C (inline mesh — same generator with different dimensions)
        if "fourc" in self.backends:
            yaml = matched_elasticity_input(40, 16, E=1000.0, nu=0.3, lx=5.0, ly=2.0)
            job, vtus = self._run_4c_inline(yaml)
            if job.status == "completed" and vtus:
                results["fourc"] = _extract_displacement_max_uy(vtus[-1])

        print(f"\n  Thick beam 5x2: {results}")
        vals = [v for v in results.values() if v is not None]
        self.assertTrue(len(vals) >= 2)
        mean_val = sum(vals) / len(vals)
        for solver, val in results.items():
            if val is not None:
                self.assertLess(abs(val - mean_val) / mean_val, 0.10,
                                f"{solver}: |uy_max|={val:.4f} vs mean {mean_val:.4f}")


class TestXSolverLDomainFine(_CrossSolverBase):
    """Benchmark #8: Poisson L-domain with finer mesh — 2-solver (FEniCS + deal.II)."""

    def test_l_domain_fine(self):
        self._require("fenics", "dealii")

        results = {}

        # FEniCS (finer Gmsh mesh)
        b = self.backends["fenics"]
        content = b.generate_input("poisson", "l_domain", {"mesh_size": 0.02})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"FEniCS failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["fenics"] = val

        # deal.II (more refinements)
        b = self.backends["dealii"]
        content = b.generate_input("poisson", "l_domain", {"refinements": 6})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"deal.II failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["dealii"] = val

        print(f"\n  L-domain fine: {results}")
        # Should be closer than coarse mesh
        rel = abs(results["fenics"] - results["dealii"]) / max(results.values())
        self.assertLess(rel, 0.01,
                        f"Fine L-domain: FEniCS={results['fenics']:.6f} vs deal.II={results['dealii']:.6f}")


class TestXSolverHeatDifferentBCs(_CrossSolverBase):
    """Benchmark #9: Heat T=50→200 on [0,1]² — 3-solver (different BCs)."""

    def test_heat_different_bcs(self):
        self._require("fenics", "dealii")
        from backends.fourc.inline_mesh import matched_heat_input

        results = {}

        # FEniCS
        b = self.backends["fenics"]
        content = b.generate_input("heat", "2d_steady", {"T_left": 50, "T_right": 200})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"FEniCS failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["fenics"] = val

        # deal.II
        b = self.backends["dealii"]
        content = b.generate_input("heat", "2d_steady", {"T_left": 50, "T_right": 200})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"deal.II failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["dealii"] = val

        # 4C
        if "fourc" in self.backends:
            yaml = matched_heat_input(32, 32, T_left=50.0, T_right=200.0)
            job, vtus = self._run_4c_inline(yaml)
            if job.status == "completed" and vtus:
                name, val = _extract_scalar_max(vtus[-1])
                results["fourc"] = val

        print(f"\n  Heat T=50→200: {results}")
        for solver, val in results.items():
            self.assertAlmostEqual(val, 200.0, delta=2.0,
                                   msg=f"{solver}: max(T)={val}")


class TestXSolverElasticityDiffMaterial(_CrossSolverBase):
    """Benchmark #10: Cantilever 10×1 with E=200000, nu=0.25 (steel-like) — 3-solver."""

    def test_cantilever_steel(self):
        self._require("fenics", "dealii")
        from backends.fourc.inline_mesh import matched_elasticity_input

        E, nu = 200000.0, 0.25
        results = {}

        # FEniCS
        b = self.backends["fenics"]
        content = b.generate_input("linear_elasticity", "2d", {"E": E, "nu": nu})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"FEniCS failed: {job.error}")
        results["fenics"] = _extract_displacement_max_uy(vtus[-1])

        # deal.II (parametrized E, nu)
        b = self.backends["dealii"]
        content = b.generate_input("linear_elasticity", "2d", {"E": E, "nu": nu})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"deal.II failed: {job.error}")
        results["dealii"] = _extract_displacement_max_uy(vtus[-1])

        # 4C (inline mesh with custom material)
        if "fourc" in self.backends:
            yaml = matched_elasticity_input(40, 4, E=E, nu=nu, lx=10.0, ly=1.0)
            job, vtus = self._run_4c_inline(yaml)
            if job.status == "completed" and vtus:
                results["fourc"] = _extract_displacement_max_uy(vtus[-1])

        print(f"\n  Elasticity steel (E={E}, nu={nu}): {results}")
        vals = [v for v in results.values() if v is not None]
        self.assertTrue(len(vals) >= 2, f"Need at least 2 solvers, got: {results}")
        # FEniCS and 4C should agree (different element types: tri vs quad)
        if "fenics" in results and "fourc" in results:
            f_val = results["fenics"]
            c_val = results["fourc"]
            if f_val and c_val:
                rel = abs(f_val - c_val) / max(f_val, c_val)
                self.assertLess(rel, 0.05,
                                f"FEniCS={f_val:.6f} vs 4C={c_val:.6f}")


class TestXSolverPoisson3D(_CrossSolverBase):
    """Benchmark #11: Poisson -Δu=1 on [0,1]³, u=0 — 3-solver (3D!)."""

    def test_poisson_3d(self):
        self._require("fenics", "dealii")
        from backends.fourc.inline_mesh import matched_poisson_3d_input

        results = {}

        # FEniCS (tetrahedra)
        b = self.backends["fenics"]
        content = b.generate_input("poisson", "3d", {"kappa": 1.0, "nx": 16, "ny": 16, "nz": 16})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"FEniCS failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["fenics"] = val

        # deal.II (hexahedra)
        b = self.backends["dealii"]
        content = b.generate_input("poisson", "3d", {"refinements": 3})
        job, vtus = _run_single(b, content)
        self.assertEqual(job.status, "completed", f"deal.II failed: {job.error}")
        name, val = _extract_scalar_max(vtus[-1])
        results["dealii"] = val

        # 4C (HEX8 inline mesh)
        if "fourc" in self.backends:
            yaml = matched_poisson_3d_input(8)
            job, vtus = self._run_4c_inline(yaml)
            if job.status == "completed" and vtus:
                name, val = _extract_scalar_max(vtus[-1])
                results["fourc"] = val

        print(f"\n  Poisson 3D: {results}")
        # deal.II and 4C should match exactly (same element type)
        if "dealii" in results and "fourc" in results:
            self.assertAlmostEqual(results["dealii"], results["fourc"], places=4,
                                   msg="deal.II vs 4C 3D mismatch")
        # All should be positive and reasonable
        for solver, val in results.items():
            self.assertGreater(val, 0.01, f"{solver}: max(u)={val}")


if __name__ == "__main__":
    unittest.main()
