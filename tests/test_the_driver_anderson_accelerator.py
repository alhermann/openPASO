"""Anderson mixing on the whole interface state: on a linear contraction it converges in a handful of
steps where constant relaxation needs dozens, and it falls back to relaxation until two residuals exist."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _iterate(step, G, x0, tol=1e-8, maxit=300):
    x = x0.copy()
    for k in range(1, maxit + 1):
        g = G(x)
        if np.linalg.norm(g - x) < tol * max(1.0, np.linalg.norm(g)):
            return k
        x = step(x, g)
    return maxit


def test_anderson_beats_constant_relaxation_on_a_stiff_linear_map():
    from core.coupling_driver import _Anderson, _relax
    rng = np.random.default_rng(3)
    n = 12
    A = np.diag(np.linspace(0.05, 0.92, n))                 # a fixed-point map with a slow mode near 1
    Q, _ = np.linalg.qr(rng.standard_normal((n, n))); A = Q @ A @ Q.T
    b = rng.standard_normal(n)
    G = lambda x: A @ x + b
    x0 = np.zeros(n)
    it_relax = _iterate(lambda x, g: _relax(x, g, 0.5), G, x0)
    it_anderson = _iterate(_Anderson(m=5, beta=0.5).step, G, x0)
    # measured: relaxation 300 (not converged), Anderson(5) 33, Anderson(12) 14 at tol 1e-8
    assert it_anderson < it_relax / 5, (it_anderson, it_relax)
    assert it_anderson <= 40


def test_anderson_first_step_is_plain_relaxation_and_stays_finite():
    from core.coupling_driver import _Anderson
    a = _Anderson(m=5, beta=0.5)
    x = np.array([1.0, 2.0]); g = np.array([3.0, 6.0])
    assert np.allclose(a.step(x, g), 0.5 * x + 0.5 * g)
    out = a.step(np.array([2.0, 4.0]), np.array([2.0, 4.0]))       # a zero residual: no division blow-up
    assert np.all(np.isfinite(out))
