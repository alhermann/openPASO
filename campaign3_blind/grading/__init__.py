"""Campaign-3 blind grader v2 — the coherent rebuild.

One module per concern, every constant in `constants.py`, every check
exercised by a firing test in `tests/test_grader_v2_fires.py`:

    constants   every number, with its derivation or an honest TO-DERIVE
    loading     locations (OPENPASO_BLIND_KEYS / this checkout), hard errors,
                checkout-pinned helper imports
    probes      the grader-owned evaluation set; task/grader grid agreement
    submission  CSV and RESULT.txt reading; claim semantics
    evidence2   canonical NDOF contract line per level; growth under halving
    checks      per-field true error, order fit, band and magnitude bounds
    iface       per-leg interface grading; NOT CHECKED is not PASSED
    outcomes    result types per evidence grade; aggregation refuses mixing

The orchestrator is `campaign3_blind/grade_blind_v2.py`.
"""
