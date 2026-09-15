# Does the equation check separate a right coupled answer from a wrong one?

Measured 2026-09-15 on openPASO, by replaying the evaluation campaign's own graded
cells through the real `verify_pde_consistency`. No sealed key was opened, no seed was
allocated, no model was called, and the campaign tree was only read.

Reproduce:

```bash
python benchmarks/pde_check_calibration/replay.py --problem C2 --problem C8 --problem C10
python benchmarks/pde_check_calibration/replay.py --seed 8631 --seed 8632 --seed 8472
```

## Why this was measured

The campaign is about to make this check run automatically inside `couple()`, because
two wordings of the invitation to call it were measured and both failed — 2 calls against
14 asks, then 0 against 7. The argument is that only the field against its own equation
separates three self-consistent coupled runs, one right and two wrong.

That argument had not been measured. The cells are still on disk with their graded
outcomes, so it can be.

## What the numbers say

**219 graded coupled cells replayed. 94 have solution files to check.**

| | cells | check says INCONSISTENT |
|---|---|---|
| graded **CORRECT** | 14 | **1** — a false alarm |
| graded **wrong** (confidently wrong + unphysical) | 17 | **6** — caught |
| | | 11 wrong cells: silent |

Three findings, in order of how much they matter.

### 1. It refuses the family that motivated it

The three cells quoted in the campaign log as the reason for by-default adoption —
8631 CORRECT, 8632 COMPLETED_UNPHYSICAL, 8472 CONFIDENTLY_WRONG — are all **C9, linear
elasticity**. Replayed, all three come back `REFUSED`:

```
C9  8631   CORRECT                -> REFUSED
C9  8632   COMPLETED_UNPHYSICAL   -> REFUSED
C9  8472   CONFIDENTLY_WRONG      -> REFUSED
```

The check implements one operator, `-div(K grad u) = f` with constant symmetric K, and
refuses anything else on purpose — the guard exists because unguarded it once called an
elasticity result INCONSISTENT and blessed a biharmonic one. C9 states
`-div(sigma(u)) = f`, so the guard fires. **C3 is refused too** (`-div(k grad u) + c*u = f`
on one subdomain), and C3 is the most productive coupled family in the record, 17 of the
33 CORRECT cells.

Accepted: C2, C8, C10. Refused: C3, C9.

### 2. On a coupled problem, one side is almost always NOT_APPLICABLE

Across the 94 cells with files, side B came back `NOT_APPLICABLE` **92 times**. The check
explains this itself:

> "the delivered field is not near zero on the boundary of the box. This check's identity
> needs u = 0 on the whole boundary, so it does not apply here — **which is the normal
> case for one side of a coupled problem, where the interface carries the partner's
> data.** Nothing is asserted."

That is not a defect in the check. It is the check being honest about a limit that
happens to coincide with the shape of every coupled problem in the campaign.

### 3. It fires on a cell that is right

`C2 seed 6751`, graded **CORRECT**, side A:

```
verdict     : INCONSISTENT
observed    : rate 0.16, weak residual FLAT 1.324e-02 -> 1.060e-02 over 3 levels
explanation : "the field is converging to something that is not the solution of the
               stated problem. Look at the source term first ..."
```

Side B of the same cell is `NOT_APPLICABLE` for the boundary reason above — so the
boundary test caught the coupling on one side and did not on the other, and where it did
not, it produced a confident wrong verdict.

This matters against the campaign's own admission rule. Every gate admitted overnight had
to speak on **0 of the 32/33 CORRECT cells**; `run_log_identity_findings` was withdrawn at
9 of 32, `ndof_ladder_findings` at 8 of 32. By that standard this check, run by default on
coupled levels, speaks on **1 of 14**.

The risk is the one the check's own source names: *"The first tells an agent to throw away
work that may be right."*

## What this does and does not establish

**Does:** run by default on every converged coupled level, this check would be silent on
C3 and C9 entirely, silent on one side of every other coupled problem, catch about a third
of wrong cells, and occasionally tell an agent its correct answer is wrong.

**Does not:** say the idea is wrong. Making the check automatic instead of asking for it
is well supported — two wordings failed. What the numbers question is whether *this*
check, with *this* operator and *this* boundary requirement, is the one to make automatic.

**Caveat, stated plainly.** The arguments were reconstructed from `spec_public.json`:
per-side sources from `source_public`, per-side rectangles from `extent_a`/`extent_b`, and
`k` parsed out of the prose line "thermal conductivity k = 1 in subdomain A; k = 1000 in
subdomain B". An agent reading its task would supply these itself and might phrase them
differently. The refusals in finding 1 do not depend on that — they are decided by the
`equation` string alone — but the rates in finding 3 could move.

**A smaller thing worth noting:** the public spec gives `k` only in prose, while the check
wants a number or a JSON matrix. Whatever else is decided, an agent handed
"thermal conductivity k = 1 in subdomain A; ... k = 1000 in subdomain B" has to convert
that itself before the call succeeds. That is friction on the path to a check nobody calls.
