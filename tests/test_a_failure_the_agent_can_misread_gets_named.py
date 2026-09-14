"""Two failures that a competent agent reads as "this is impossible".

Both were measured on real runs of the coupled cell C2, and in both the run
had the truth within reach and drew the opposite conclusion.

ONE. C2_27b_MCP_seed1203 was told the condition it needed, by name, twice --
once in the served pitfall and once in the write-time check, which hands over
the exact factory line. It wrote instead

    cond = KM.ConvectionDiffusionApplication.ThermalFace2D2N(condition_id, ...)

got `has no attribute 'ThermalFace2D2N'`, concluded the condition "doesn't
exist in the current Kratos version", started looking for a third code to
replace the two its task prescribes, and submitted nothing. Measured on this
install, by construction through the factory:

    LaplacianElement2D3N   python-attribute=False   factory-by-name=True
    ThermalFace2D2N        python-attribute=False   factory-by-name=True
    FluxCondition2D2N      python-attribute=False   factory-by-name=True

LaplacianElement2D3N is the element that same script solves with. So the
AttributeError carries no information about existence at all -- registered
components live in a C++ registry and none of them is a Python attribute. A
name that genuinely is absent fails differently and inside the factory:
`The Condition "ThermalFace2D" is not registered!`.

TWO. C2_27b_MCP_seed1201's whole record of its 4C run is: `4C stdout:` empty,
the MPI_ABORT boilerplate, `4C return code: 1`. It reported "the 4C binary
requires specific MPI environment configuration" as blocker number one. The
reason was not withheld, it was destroyed -- 4C's stdout is block-buffered and
MPI_Abort tears the process down before the flush. Same rejected deck, three
invocations, measured:

    plain capture            429 bytes    no reason
    `2>&1` merged            429 bytes    no reason
    stdbuf -oL -eL          2164 bytes    Section 'NOT_A_REAL_SECTION' is not
                                          a valid section name.
    mpirun -np 1            2164 bytes    the same

openPASO's own runner has wrapped 4C in `stdbuf -oL` for a long time
(src/backends/fourc/backend.py). An agent that invokes the binary itself never
saw it. That is the recurring shape: a mechanism exists, is instrumented, and
does not reach the case it was built for.

SO THESE TESTS ASK WHAT THE AGENT ENDS UP WITH, not what the helpers return:
they drive the real write_file and run_bash tools and read the reply.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("langchain_core")

from langgraph_eval.agent import (                        # noqa: E402
    _eaten_error_check, _read_write_tools_for, _registry_attribute_check,
    _bash_tool_for, _registry_error_check,
)

ATTR_CALL = ('import KratosMultiphysics as KM\n'
             'cond = KM.ConvectionDiffusionApplication.ThermalFace2D2N('
             'condition_id, [n1, n2], prop)\n')
FACTORY_CALL = ('mp.CreateNewCondition("ThermalFace2D2N", cid, [n1, n2], '
                'prop)\n')
REAL_TRACEBACK = (
    "  cond = KM.ConvectionDiffusionApplication.ThermalFace2D2N(cid,\n"
    "AttributeError: module 'KratosMultiphysics.ConvectionDiffusionApplication'"
    " has no attribute 'ThermalFace2D2N'\n")
REAL_ABORT = (
    "4C stdout: \n"
    "4C stderr: No protocol specified\n"
    "------------------------------------------------------------\n"
    "MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD\n"
    "with errorcode 1.\n"
    "------------------------------------------------------------\n"
    "4C return code: 1\n")


def _tools(tmp_path, *, openpaso_arm: bool):
    rw = _read_write_tools_for(tmp_path, audit_on_submit=openpaso_arm)
    write = next(t for t in rw if t.name == "write_file")
    shell = _bash_tool_for(tmp_path, audit_on_submit=openpaso_arm)
    return write, shell


# ─────────────────────────── the registry truth ───────────────────────────

def test_the_attribute_form_is_named_when_the_agent_writes_it(tmp_path):
    write, _ = _tools(tmp_path, openpaso_arm=True)
    reply = write.invoke({"path": "participant_B.py", "content": ATTR_CALL})
    assert "IS NOT HOW A REGISTERED KRATOS COMPONENT IS REACHED" in reply
    assert "CreateNewCondition(\"ThermalFace2D2N\"" in reply
    # the measurement, so the claim is checkable by the reader
    assert "factory-by-name=True" in reply
    assert "LaplacianElement2D3N" in reply
    # and the consequence that actually cost a round
    assert "DO NOT CHANGE CODES OVER THIS" in reply


def test_the_error_channel_catches_what_the_disk_channel_cannot(tmp_path):
    """The channel that matters: seed1203 left no such call in any file.

    All three of its scripts hold zero attribute-constructor calls -- the
    broken version was overwritten before the run ended. What the agent
    ended up with was the traceback.
    """
    assert _registry_attribute_check(Path("p.py"), FACTORY_CALL) == ""
    got = _registry_error_check(REAL_TRACEBACK)
    assert "not evidence" in got.lower()
    assert "not available in version 10.3.0" in got


def test_a_genuinely_absent_name_is_not_contradicted():
    """`ThermalFace2D` really does not exist. Do not tell the agent it does."""
    absent = 'Error: The Condition "ThermalFace2D" is not registered!'
    assert _registry_error_check(absent) == ""


def test_the_registry_check_cannot_fire_on_a_real_python_attribute():
    """dir(KratosMultiphysics) holds exactly three names matching the registry
    signature and all three are ExactMortarIntegrationUtility*; the
    ConvectionDiffusion application holds none. Excluding the utility family
    covers all of them, so a legitimate call stays silent."""
    for src in ('u = KM.ExactMortarIntegrationUtility2D2N(2)\n',
                'u = KM.ExactMortarIntegrationUtility3D4N(2)\n',
                'g = KM.Line2D2(n1, n2)\n',
                'p = KM.Parameters("{}")\n'):
        assert _registry_attribute_check(Path("p.py"), src) == "", src


# ───────────────────────── the eaten reason ──────────────────────────────

def test_an_abort_with_no_reason_is_not_left_as_an_mpi_problem(tmp_path):
    got = _eaten_error_check(REAL_ABORT)
    assert "NOT AN MPI OR ENVIRONMENT PROBLEM" in got
    assert "stdbuf -oL -eL" in got
    assert "429" in got and "2164" in got
    # the X11 lines the run listed among its blockers
    assert "MIT-MAGIC-COOKIE" in got


def test_the_real_seed1201_log_fires_verbatim():
    """Not a paraphrase: the file the run actually left on disk."""
    log = (ROOT / "campaign3_blind/runs/C2_27b_MCP_seed1201/work"
                  "/run_log.txt")
    if not log.exists():                    # run data pruned; claim untestable
        pytest.skip("seed1201 run data absent")
    assert _eaten_error_check(log.read_text(errors="replace"))


def test_an_abort_that_carries_its_reason_is_left_alone():
    for txt in (
        "MPI_ABORT was invoked\nPROC 0 ERROR in 4C_io_input_file.cpp, line "
        "546:\nSection 'X' is not a valid section name.\n",
        "MPI_ABORT was invoked\nTraceback (most recent call last):\n"
        "ValueError: bad\n",
        "MPI_ABORT was invoked\nError: The Condition \"Y\" is not "
        "registered!\n",
    ):
        assert _eaten_error_check(txt) == "", txt


def test_a_successful_run_with_x11_noise_says_nothing():
    assert _eaten_error_check("processor 0 finished normally\n"
                              "No protocol specified\n") == ""


# ───────────────────────── the bare arm is untouched ─────────────────────

def test_neither_check_reaches_the_bare_arm(tmp_path):
    write, shell = _tools(tmp_path, openpaso_arm=False)
    reply = write.invoke({"path": "participant_B.py", "content": ATTR_CALL})
    assert "REGISTERED KRATOS COMPONENT" not in reply
    out = shell.invoke({"command": "printf '%s' " + repr(REAL_ABORT)})
    assert "NOT AN MPI OR ENVIRONMENT PROBLEM" not in out


def test_the_openpaso_arm_gets_it_through_the_shell(tmp_path):
    """The abort arrives as command OUTPUT, so the shell tool must carry it."""
    _, shell = _tools(tmp_path, openpaso_arm=True)
    script = tmp_path / "fake_4c.sh"
    script.write_text("#!/bin/sh\ncat <<'EOF'\n" + REAL_ABORT + "EOF\n")
    script.chmod(0o755)
    out = shell.invoke({"command": "sh ./fake_4c.sh"})
    assert "NOT AN MPI OR ENVIRONMENT PROBLEM" in out
    assert "stdbuf -oL -eL" in out


# ══════════════════ a give-up filed over finished work, by SHELL ═════════════
#
# C2_27b_MCP_seed1202 finished the work. Its directory holds six
# solution_level*_[AB].csv, six interface_level*_[AB].csv, three residual
# histories converging 3.44e-01 -> 3.35e-08 in four iterations, and six run
# logs -- the complete deliverable set. It then wrote COULD_NOT_COMPLETE, and
# the check built for exactly that fired ZERO times, because it hung on
# write_file while this very module's sibling records that 57% of submitters
# write RESULT.txt by shell only.

GIVEUP = ("COULD_NOT_COMPLETE\nThe coupled simulation was not successfully "
          "completed due to a fundamental issue with the Kratos Neumann "
          "boundary condition implementation.\n")


def _plant_finished_work(w: Path) -> None:
    w.mkdir(parents=True, exist_ok=True)
    for k in (1, 2, 3):
        for side in ("A", "B"):
            (w / f"solution_level{k}_{side}.csv").write_text(
                "x, y, u\n" + "".join(
                    f"0.{i}, 0.{i}, {k}.{i}e-03\n" for i in range(1, 9)))
            (w / f"interface_level{k}_{side}.csv").write_text(
                "y, u, q\n0.5, 1.0e-03, 2.0e-03\n0.6, 1.1e-03, 2.1e-03\n")
        (w / f"residual_level{k}.csv").write_text(
            "iteration,interface_residual\n2,3.44e-01\n3,1.56e-03\n"
            "4,7.18e-06\n5,3.35e-08\n")


def test_a_give_up_written_by_heredoc_is_contradicted(tmp_path):
    """The channel seed1202 actually used."""
    w = tmp_path / "work"
    _plant_finished_work(w)
    _, shell = _tools(w, openpaso_arm=True)
    out = shell.invoke({"command": "cat > RESULT.txt <<'XEOF'\n" + GIVEUP
                                   + "XEOF"})
    assert "FILING A GIVE-UP ON TOP OF WORK THAT IS ON DISK" in out
    assert "per-level field file" in out and "residual_level1.csv" in out
    assert "counts for nothing" in out


def test_the_bare_arm_is_not_told_by_the_shell_either(tmp_path):
    w = tmp_path / "work"
    _plant_finished_work(w)
    _, shell = _tools(w, openpaso_arm=False)
    out = shell.invoke({"command": "cat > RESULT.txt <<'XEOF'\n" + GIVEUP
                                   + "XEOF"})
    assert "FILING A GIVE-UP" not in out


def test_a_real_submission_by_heredoc_is_not_called_a_give_up(tmp_path):
    w = tmp_path / "work"
    _plant_finished_work(w)
    _, shell = _tools(w, openpaso_arm=True)
    out = shell.invoke({"command": "cat > RESULT.txt <<'XEOF'\nLEVELS = 3\n"
                                   "ORDER = 1.98\nXEOF"})
    assert "FILING A GIVE-UP" not in out


# ══════════════════ the same field written at every level ═══════════════════

def test_bit_identical_levels_are_named(tmp_path):
    from langgraph_eval.agent import _identical_levels_check
    w = tmp_path / "work"
    w.mkdir()
    same = "x, y, u\n0.1, 0.1, 1.5e-03\n0.2, 0.2, 2.5e-03\n0.3, 0.3, 3.5e-03\n"
    for k in (1, 2, 3):
        (w / f"solution_level{k}_A.csv").write_text(same)
    got = _identical_levels_check(w, w / "solution_level2_A.csv")
    assert "BIT-IDENTICAL" in got
    assert "log2(|L1-L2|/|L2-L3|)" in got
    assert "0/0" in got


def test_levels_that_refine_are_left_alone(tmp_path):
    from langgraph_eval.agent import _identical_levels_check
    w = tmp_path / "work"
    w.mkdir()
    for k in (1, 2, 3):
        (w / f"solution_level{k}_B.csv").write_text(
            "x, y, u\n" + "".join(f"0.{i}, 0.{i}, {k}.{i}e-03\n"
                                  for i in range(1, 4)))
    for k in (1, 2, 3):
        assert _identical_levels_check(
            w, w / f"solution_level{k}_B.csv") == "", k


def test_the_real_seed1202_side_a_fires_and_side_b_does_not():
    """Measured: side A max|du| = 0.000e+00 for every pair; side B refines
    2.307290e-03 -> 2.364044e-03 -> 2.370374e-03."""
    from langgraph_eval.agent import _identical_levels_check
    w = (ROOT / "campaign3_blind/runs/C2_27b_MCP_seed1202/work")
    if not (w / "solution_level1_A.csv").exists():
        pytest.skip("seed1202 run data absent")
    for k in (1, 2, 3):
        assert _identical_levels_check(w, w / f"solution_level{k}_A.csv")
        assert _identical_levels_check(w, w / f"solution_level{k}_B.csv") == ""


# ════════════════ the proof that was captured and then dropped ══════════════
#
# C2_27b_MCP_seed1301 is the furthest any openPASO run has reached on the coupled
# cell: both participants really ran, the partitioned iteration converged
# 1.3901141511 -> 4.3834e-07 in eight iterations at level 1, and the graded
# order came out 1.9367. Its participant_A.py invoked the binary correctly --
# `['stdbuf', '-oL', '-eL', '/home/.../4C', deck, prefix]` with
# capture_output=True -- and then wrote its own three-line summary into the log
# instead of result.stdout. 56 bytes of prose. On the same cell with the same
# two codes, a real capture is 2947 and 1476 bytes.

PROSE_LOG = ("NDOF = 54\n4C Multiphysics solver\nElements: TRANSP QUAD4\n")
CAPTURED_4C = (
    "NDOF = 54\n"
    "******************************************************\n"
    "*                         4C                         *\n"
    "*                version 2026.2.0-dev                *\n"
    "processor 0 finished normally\n")
CAPTURED_KRATOS = (
    "NDOF = 72\n KRATOS ___ ___  _  ___   __   ___ ___ ___ ___ \n"
    "BlockBuildDofArrayUtility: Setting up the DOFs\n"
    "ResidualBasedLinearStrategy: Setup Dofs Time: 0.00215229 [s]\n")


def test_a_log_of_the_agents_own_prose_is_named(tmp_path):
    write, _ = _tools(tmp_path, openpaso_arm=True)
    reply = write.invoke({"path": "run_level1_A.log", "content": PROSE_LOG})
    assert "CARRIES YOUR OWN WORDS, NOT THE SOLVER'S OUTPUT" in reply
    assert "capture_output=True" in reply
    assert "2947 and 1476 bytes" in reply
    # it must say why it matters, in the task's own terms
    assert "cannot be credited to that code" in reply


def test_a_real_capture_is_left_alone(tmp_path):
    write, _ = _tools(tmp_path, openpaso_arm=True)
    for name, body in (("run_level1_A.log", CAPTURED_4C),
                       ("run_level1_B.log", CAPTURED_KRATOS)):
        reply = write.invoke({"path": name, "content": body})
        assert "CARRIES YOUR OWN WORDS" not in reply, name


def test_it_only_looks_at_the_prescribed_log_name(tmp_path):
    from langgraph_eval.agent import _discarded_proof_check
    for name in ("run_log.txt", "solution_level1_A.csv", "residual_level1.csv",
                 "notes.log", "RESULT.txt"):
        assert _discarded_proof_check(Path(name), PROSE_LOG) == "", name


def test_the_real_seed1301_logs_all_fire_and_the_reference_does_not():
    """Both directions, on files that exist: six prose logs against six real
    captures of the same two codes on the same cell."""
    from langgraph_eval.agent import _discarded_proof_check as C
    w = ROOT / "campaign3_blind/runs/C2_27b_MCP_seed1301/work"
    if not w.exists():
        pytest.skip("seed1301 run data absent")
    logs = sorted(w.glob("run_level*.log"))
    assert len(logs) == 6
    assert all(C(f, f.read_text(errors="replace")) for f in logs)


def test_the_bare_arm_never_hears_about_it(tmp_path):
    write, _ = _tools(tmp_path, openpaso_arm=False)
    reply = write.invoke({"path": "run_level1_A.log", "content": PROSE_LOG})
    assert "CARRIES YOUR OWN WORDS" not in reply


def test_every_backend_has_at_least_one_marker():
    """A code with no marker can never be credited, so the list must cover
    all nine rather than the two this cell happens to use."""
    from langgraph_eval.agent import _looks_like_captured_output as L
    samples = {
        "4C": "processor 0 finished normally",
        "kratos": "ResidualBasedLinearStrategy: Setup Dofs Time: 0.002 [s]",
        "fenicsx": "Solving linear variational problem",
        "dealii": "deallog: Starting value 1.0",
        "ngsolve": "assemble VOL element 3/24",
        "dune": "Newton iteration 2: residual 1e-8",
        "skfem": "Basis(ElementTriP1) with 81 DOFs",
        "febio": "N O R M A L   T E R M I N A T I O N",
        "sparta": "Loop time of 1.234 on 1 procs",
    }
    missing = [k for k, v in samples.items() if not L(v)]
    assert missing == [], f"no marker covers: {missing}"



def test_an_env_assignment_after_a_wrapper_is_named(tmp_path):
    """`stdbuf -oL VAR=x prog` runs VAR=x as the program.

    C2_27b_MCP_seed1401 was served `stdbuf -oL -eL <binary> deck out`, wanted a
    library path too, and wrote the assignment after stdbuf. Measured
    diagnostic lines recovered from one rejected deck: wrapper-then-assignment
    0 (nothing ran at all), assignment-first 2, `env` form 2. The primitive
    meant to make 4C's errors visible is what stopped its 4C from running.
    """
    from langgraph_eval.agent import _env_after_wrapper_check as C
    real = ("stdbuf -oL -eL LD_LIBRARY_PATH=/opt/4C-dependencies/lib "
            "/home/alexander/4C/build/4C deck.4C.yaml out")
    got = C(real)
    assert "RUNS `LD_LIBRARY_PATH=...` AS THE PROGRAM" in got
    assert "already exports" in got.lower()
    # both working forms must be shown
    assert "stdbuf -oL -eL env LD_LIBRARY_PATH=..." in got
    for cmd in ("LD_LIBRARY_PATH=/opt/lib stdbuf -oL -eL /a/4C d o",
                "stdbuf -oL -eL env LD_LIBRARY_PATH=/opt/lib /a/4C d o",
                "stdbuf -oL -eL /a/4C deck.4C.yaml out",
                "timeout 900 python solve.py --tol=1e-8",
                "timeout 60 python x.py a=b",
                "timeout 30s ./run",
                "nice -n 19 python run.py"):
        assert C(cmd) == "", cmd


def test_a_wrappers_own_argument_is_not_mistaken_for_the_program():
    """timeout takes a duration and nice a level before the command."""
    from langgraph_eval.agent import _env_after_wrapper_check as C
    assert C("timeout 900 OMP_NUM_THREADS=4 ./solver in.txt")
    assert C("nice -n 19 OMP_NUM_THREADS=4 ./solver")


def test_neither_new_check_reaches_the_bare_arm(tmp_path):
    write, shell = _tools(tmp_path, openpaso_arm=False)
    out = shell.invoke({"command": "stdbuf -oL -eL FOO=1 /bin/echo hi"})
    assert "AS THE PROGRAM" not in out


# ═════ past halfway with nothing gradeable on disk, fired once ══════════════
#
# Measured over the nine openPASO runs of the coupled cell in rounds 12-14, by
# file mtime as a fraction of each run's own wall clock: three never wrote a
# solution file at all (seed1201, seed1203, seed1401), and the six that did
# started at 27%, 32%, 49%, 72%, 75% and 78%. The two earliest both produced
# the complete six-file set; the one that started at 72% produced two. SEVEN OF
# NINE were past 45% with nothing gradeable on disk.
#
# _time_left_note fires its "write the deliverable now" at 75% spent, which is
# after the point where these runs had already decided -- two of them filed a
# give-up blaming the clock at 47% and 49%.


# ═══════ files written DURING an MCP tool call reach the same checks ═════════
#
# C2_27b_MCP_seed1701 drove its coupling through couple() -- exactly what the
# must-read asks -- and its participant wrote run_level1_B.log, 179 bytes of
# its own prose, DURING the MCP call. The discarded-proof check was wired to
# run_bash and write_file, so the artefact appeared between hook points and
# nothing fired. The MCP tools are now wrapped with the same before/after
# artefact hook; the check bodies stay in openPASO.

def test_a_prose_log_written_during_an_mcp_call_is_named(tmp_path):
    import asyncio
    from langgraph_eval.agent import _wrap_mcp_tool_with_artefact_hook
    from langchain_core.tools import StructuredTool

    async def fake_couple(command: str = "") -> str:
        (tmp_path / "run_level1_B.log").write_text(
            "Kratos solve complete: max|u| = 5.48e-03\nNDOF = 72\n")
        return "coupling converged in 8 iterations"

    t = StructuredTool.from_function(coroutine=fake_couple, name="couple",
                                     description="d")
    t = _wrap_mcp_tool_with_artefact_hook(t, tmp_path)
    out = asyncio.run(t.coroutine(command="x"))
    assert "coupling converged in 8 iterations" in out
    assert "CARRIES YOUR OWN WORDS, NOT THE SOLVER'S OUTPUT" in out


def test_a_real_capture_written_during_an_mcp_call_stays_silent(tmp_path):
    import asyncio
    from langgraph_eval.agent import _wrap_mcp_tool_with_artefact_hook
    from langchain_core.tools import StructuredTool

    async def fake_couple(command: str = "") -> str:
        (tmp_path / "run_level1_B.log").write_text(
            "NDOF = 72\n KRATOS ___ banner\n"
            "ResidualBasedLinearStrategy: Setup Dofs Time: 0.002 [s]\n")
        return "ok"

    t = StructuredTool.from_function(coroutine=fake_couple, name="couple",
                                     description="d")
    t = _wrap_mcp_tool_with_artefact_hook(t, tmp_path)
    out = asyncio.run(t.coroutine(command="x"))
    assert out == "ok"


def test_a_structured_reply_passes_through_untouched(tmp_path):
    import asyncio
    from langgraph_eval.agent import _wrap_mcp_tool_with_artefact_hook
    from langchain_core.tools import StructuredTool

    async def fake_tool() -> dict:
        (tmp_path / "run_level1_A.log").write_text("my own words\nNDOF = 9\n")
        return {"k": 1}

    t = StructuredTool.from_function(coroutine=fake_tool, name="x",
                                     description="d")
    t = _wrap_mcp_tool_with_artefact_hook(t, tmp_path)
    assert asyncio.run(t.coroutine()) == {"k": 1}
