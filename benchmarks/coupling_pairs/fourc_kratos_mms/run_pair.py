"""Development benchmark for the C2 4C/Kratos manufactured diffusion pair.

This is not an evaluation solution path. C2 is a spent development problem;
the benchmark exercises the general participant templates and the openPASO file
handshake before another model run is paid for.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

PAIR_DIR = Path(__file__).resolve().parent
REPO = PAIR_DIR.parents[2]
PARTICIPANTS = REPO / "data" / "coupling_participants"
SPEC_PATH = REPO / "campaign3_blind" / "problems" / "C2" / "spec_public.json"
SPEC = json.loads(SPEC_PATH.read_text())

sys.path.insert(0, str(REPO / "src"))
from core.coupling_driver import Participant, run_coupling  # noqa: E402

PYTHON = os.environ.get(
    "OPENPASO_SOLVER_PYTHON",
    "/home/user/Schreibtisch/open-fem-agent/.venv/bin/python",
)
FOURC = os.environ.get("FOURC_BINARY", "/home/user/4C/build/4C")
FOURC_LD = os.environ.get("FOURC_LD_LIBRARY_PATH", "/opt/4C-dependencies/lib")


def exact(side: str, x, y):
    """Developer-only oracle derived from the public PDE constraints."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if side == "A":
        return (x * y * (1600 * x - 1007) * (y - 1)
                * (15 * x * y - 9 * x + 20 * y - 90) / 24000)
    return (y * (2 * x - 3) * (8 * x + 995) * (y - 1)
            * (120 * x * y - 72 * x + 46925 * y - 152955)
            / 15360000000)


def exact_flux_a(y):
    """Exact outward flux of side A at x=5/8."""
    y = np.asarray(y, float)
    return -39 * y * (y - 1) * (199 * y - 649) / 6400


def _edit(text: str, edits: dict[str, str]) -> str:
    for key, value in edits.items():
        pattern = re.compile(
            rf"^({re.escape(key)}\s*=\s*)(.*?)(\s*(?:#.*)?)$", re.M)
        if not pattern.search(text):
            raise AssertionError(f"participant edit key not found: {key}")
        text = pattern.sub(
            lambda match: match.group(1) + value + match.group(3),
            text, count=1)
    return text


def _replace_source(text: str, marker: str, expression: str) -> str:
    replacement = "    return (" + expression + ")"
    if text.count(marker) != 1:
        raise AssertionError(f"source marker occurs {text.count(marker)} times")
    return text.replace(marker, replacement)


def render_participants(base: Path, mesh_n: int) -> tuple[Participant, Participant]:
    """Render one C2 mesh level from the shipped general participants."""
    nx_a = round(0.625 * mesh_n)
    nx_b = round(0.875 * mesh_n)
    if not math.isclose(nx_a / mesh_n, 0.625):
        raise ValueError(f"mesh N={mesh_n} does not align with x=5/8")

    work_a = base / "A"
    work_b = base / "B"
    work_a.mkdir(parents=True)
    work_b.mkdir(parents=True)

    fourc = (PARTICIPANTS / "participant_fourc.py").read_text()
    fourc = _edit(fourc, {
        "SIDE": '"dirichlet"',
        "PARTNER": '"KratosB"',
        "X0, X1": "0.0, 0.625",
        "Y0, Y1": "0.0, 1.0",
        "IFACE_X": "0.625",
        "K": "1.0",
        "T_OUTER": "0.0",
        "FULL_OUTER_DIRICHLET": "True",
        "NX, NY": f"{nx_a}, {mesh_n}",
        "T_INIT": "0.0",
        "Q_INIT": "0.0",
        "FOURC_BIN": repr(FOURC),
        "FOURC_LD": repr(FOURC_LD),
        "FIT_DEG": "6",
        "SRC_DEG_MAX": "8",
    })
    fourc = _replace_source(
        fourc, "    return np.zeros_like(x)", SPEC["source_public"]["A"])
    (work_a / "participant.py").write_text(fourc)

    kratos = (PARTICIPANTS / "participant_kratos_neumann.py").read_text()
    kratos = _edit(kratos, {
        "PARTNER": '"FourCA"',
        "X0, X1": "0.625, 1.5",
        "Y0, Y1": "0.0, 1.0",
        "IFACE_X": "0.625",
        "K": "200.0",
        "T_OUTER": "0.0",
        "FULL_OUTER_DIRICHLET": "True",
        "NX, NY": f"{nx_b}, {mesh_n}",
        "Q_INIT": "0.0",
    })
    kratos = _replace_source(
        kratos, "    return 0.0 * x", SPEC["source_public"]["B"])
    (work_b / "participant.py").write_text(kratos)

    return (
        Participant("FourCA", [PYTHON, "participant.py"], work_a,
                    imports_from=["KratosB"], timeout=900),
        Participant("KratosB", [PYTHON, "participant.py"], work_b,
                    imports_from=["FourCA"], timeout=900),
    )


def _field(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    coordinates = np.asarray(data["coordinates"], float)
    values = np.asarray(data["values"], float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"bad field coordinates in {path}")
    if len(coordinates) != len(values) or not np.isfinite(values).all():
        raise ValueError(f"bad field values in {path}")
    return coordinates, values


def _grid(coordinates: np.ndarray, values: np.ndarray):
    xs = np.unique(np.round(coordinates[:, 0], 10))
    ys = np.unique(np.round(coordinates[:, 1], 10))
    grid = np.full((len(ys), len(xs)), np.nan)
    ix = {value: i for i, value in enumerate(xs)}
    iy = {value: i for i, value in enumerate(ys)}
    for point, value in zip(np.round(coordinates, 10), values):
        grid[iy[point[1]], ix[point[0]]] = value
    if not np.isfinite(grid).all():
        raise ValueError("field is not a complete structured grid")
    return xs, ys, grid


def sample_field(path: Path, points: np.ndarray, *, triangles: bool) -> np.ndarray:
    """Evaluate the actual Q1/P1 discrete field at fixed interior points."""
    coordinates, values = _field(path)
    xs, ys, field = _grid(coordinates, values)
    result = []
    for x, y in np.asarray(points, float):
        i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
        j = int(np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2))
        xi = (x - xs[i]) / (xs[i + 1] - xs[i])
        eta = (y - ys[j]) / (ys[j + 1] - ys[j])
        a, b = field[j, i], field[j, i + 1]
        d, c = field[j + 1, i], field[j + 1, i + 1]
        if not triangles:
            value = ((1 - xi) * (1 - eta) * a + xi * (1 - eta) * b
                     + xi * eta * c + (1 - xi) * eta * d)
        elif xi + eta <= 1:
            value = (1 - xi - eta) * a + xi * b + eta * d
        else:
            value = (1 - eta) * b + (xi + eta - 1) * c + (1 - xi) * d
        result.append(value)
    return np.asarray(result)


def fixed_probes(side: str) -> np.ndarray:
    bounds = SPEC["extent_a" if side == "A" else "extent_b"]
    m = int(SPEC["probe_M"])
    axes = [np.array([lo + (i + 0.5) * (hi - lo) / m for i in range(m)])
            for lo, hi in bounds]
    return np.array([(x, y) for x in axes[0] for y in axes[1]])


def interface_probes() -> np.ndarray:
    lo, hi = map(float, SPEC["iface_graded_band"])
    m = int(SPEC["probe_M"])
    return np.array([(0.625, lo + (i + 0.5) * (hi - lo) / m)
                     for i in range(m)])


def _interface(export: dict, y: np.ndarray):
    coordinates = np.asarray(export["coordinates"], float)
    values = np.asarray(export["values"], float).reshape(-1)
    fluxes = np.asarray(export["normal_fluxes"], float).reshape(-1)
    order = np.argsort(coordinates[:, 1])
    source_y = coordinates[order, 1]
    return (np.interp(y, source_y, values[order]),
            np.interp(y, source_y, fluxes[order]))


def _write_csv(path: Path, header: str, rows) -> None:
    path.write_text(header + "\n" + "".join(
        ",".join(f"{float(value):.15e}" for value in row) + "\n"
        for row in rows))


def run_level(base: Path, mesh_n: int, *, tol: float = 1e-10):
    participants = render_participants(base, mesh_n)
    result = run_coupling(
        list(participants), max_iter=100, tol=tol,
        accelerator="aitken", theta0=0.95, probe=False)
    if not result.converged:
        raise RuntimeError(result.error or f"coupling residual {result.residual}")
    if not (participants[0].work_dir / "field.json").is_file():
        raise RuntimeError("4C participant produced no complete field")
    if not (participants[1].work_dir / "field.json").is_file():
        raise RuntimeError("Kratos participant produced no complete field")
    return result, participants


def error_report(result, participants) -> dict:
    out = {}
    for side, participant, triangles in (
            ("A", participants[0], False),
            ("B", participants[1], True)):
        points = fixed_probes(side)
        values = sample_field(participant.work_dir / "field.json", points,
                              triangles=triangles)
        reference = exact(side, points[:, 0], points[:, 1])
        error = float(np.sqrt(np.mean((values - reference) ** 2)))
        scale = float(np.sqrt(np.mean(reference ** 2)))
        out[side] = {"rms_error": error, "relative_error": error / scale,
                     "max_abs": float(np.max(np.abs(values)))}
    out["coupling"] = {
        "iterations": result.iterations,
        "residual": result.residual,
        "finite_history": [float(value) for value in result.history
                           if math.isfinite(float(value))],
    }
    probes = interface_probes()
    _, flux_a = _interface(result.exports["FourCA"], probes[:, 1])
    _, flux_b = _interface(result.exports["KratosB"], probes[:, 1])
    reference_flux = exact_flux_a(probes[:, 1])
    scale = max(float(np.max(np.abs(reference_flux))), 1e-30)
    out["interface"] = {
        "a_max_error": float(np.max(np.abs(flux_a - reference_flux))),
        "balance_relative": float(np.max(np.abs(flux_a + flux_b)) / scale),
    }
    return out


def write_level_artifacts(output: Path, level: int, result, participants) -> dict:
    probes_by_side = {side: fixed_probes(side) for side in ("A", "B")}
    values_by_side = {
        "A": sample_field(participants[0].work_dir / "field.json",
                          probes_by_side["A"], triangles=False),
        "B": sample_field(participants[1].work_dir / "field.json",
                          probes_by_side["B"], triangles=True),
    }
    for side in ("A", "B"):
        points = probes_by_side[side]
        values = values_by_side[side]
        _write_csv(output / f"solution_level{level}_{side}.csv", "x,y,u",
                   ((x, y, value) for (x, y), value in zip(points, values)))

    interface_points = interface_probes()
    for side, name in (("A", "FourCA"), ("B", "KratosB")):
        values, fluxes = _interface(result.exports[name], interface_points[:, 1])
        _write_csv(output / f"interface_level{level}_{side}.csv", "x,y,u,qn",
                   ((x, y, value, flux) for (x, y), value, flux
                    in zip(interface_points, values, fluxes)))

    finite_history = [(iteration, float(value))
                      for iteration, value in enumerate(result.history, 1)
                      if math.isfinite(float(value))]
    _write_csv(output / f"residual_level{level}.csv",
               "iteration,interface_residual", finite_history)

    for side, participant in zip(("A", "B"), participants):
        source = participant.work_dir / "participant_output.log"
        if not source.is_file():
            raise RuntimeError(f"missing native participant log for side {side}")
        shutil.copy2(source, output / f"run_level{level}_{side}.log")

    return {"points": probes_by_side, "values": values_by_side,
            "report": error_report(result, participants),
            "iterations": result.iterations, "residual": result.residual}


def run_all(output: Path, mesh_levels=(8, 16, 32)) -> dict:
    """Run all C2 levels and write the complete public submission contract."""
    output.mkdir(parents=True, exist_ok=True)
    levels = []
    for level, mesh_n in enumerate(mesh_levels, 1):
        result, participants = run_level(output / f"level_{level}", mesh_n)
        levels.append(write_level_artifacts(
            output, level, result, participants))

    changes = []
    for side in ("A", "B"):
        coarse = levels[-2]["values"][side]
        fine = levels[-1]["values"][side]
        changes.append(float(np.max(np.abs(fine - coarse))
                             / max(np.max(np.abs(fine)), 1e-30)))
    max_change = max(changes)

    csv_files = []
    for level in range(1, len(levels) + 1):
        csv_files.extend([
            f"solution_level{level}_A.csv", f"solution_level{level}_B.csv",
            f"interface_level{level}_A.csv", f"interface_level{level}_B.csv",
            f"residual_level{level}.csv",
        ])
    finest = levels[-1]
    (output / "RESULT.txt").write_text(
        f"LEVELS = {len(levels)}\n"
        f"FILES = {','.join(csv_files)}\n"
        f"INTERFACE_RESIDUAL = {finest['residual']:.15e}\n"
        f"COUPLING_ITERATIONS = {finest['iterations']}\n"
        f"MESH_INDEPENDENCE = {'CONVERGED' if max_change < 0.05 else 'NOT_CONVERGED'}\n"
        f"MAX_REL_CHANGE = {max_change:.15e}\n")

    errors = {side: [level["report"][side]["rms_error"] for level in levels]
              for side in ("A", "B")}
    orders = {side: [math.log(values[i] / values[i + 1], 2)
                     for i in range(len(values) - 1)]
              for side, values in errors.items()}
    return {"levels": levels, "errors": errors, "orders": orders,
            "max_relative_change": max_change}


if __name__ == "__main__":
    import tempfile

    mesh_n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with tempfile.TemporaryDirectory(prefix=f"fourc-kratos-c2-n{mesh_n}-") as tmp:
        coupled, pair = run_level(Path(tmp), mesh_n)
        print(json.dumps(error_report(coupled, pair), indent=2))