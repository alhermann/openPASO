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

## Headline

**219 graded coupled cells replayed. 94 have solution files to check.** The first
measurement found two defects in the check; both were fixed here, and the table is the
same 94 cells before and after.

| version of the check | false alarms on CORRECT | wrong cells caught | no verdict |
|---|---|---|---|
| as shipped (test function = product of sines) | **1 of 14** | 6 of 17 | 78 |
| sines, with sin² behind a size test | 1 of 14 | 16 of 17 | 35 |
| **sin² always (now shipped)** | **0 of 14** | **14 of 17** | 35 |

The middle row catches two more wrong cells and is the one **not** adopted: the campaign's
own admission rule is that a gate must speak on **zero** of the CORRECT cells —
`run_log_identity_findings` was withdrawn at 9 of 32, `ndof_ladder_findings` at 8 of 32.
A gate at 1 of 14 does not qualify however many wrong cells it catches, because a false
accusation tells an agent to throw away work that is right.

## Finding 1 — it refuses the family that motivated it

Unchanged, and not fixable by anything here. The three cells quoted in the campaign log as
the reason for by-default adoption — 8631 CORRECT, 8632 COMPLETED_UNPHYSICAL, 8472
CONFIDENTLY_WRONG — are all **C9, linear elasticity**. All three come back `REFUSED`:

```
C9  8631   CORRECT                -> REFUSED
C9  8632   COMPLETED_UNPHYSICAL   -> REFUSED
C9  8472   CONFIDENTLY_WRONG      -> REFUSED
```

The check implements one operator, `-div(K grad u) = f` with constant symmetric K, and
refuses anything else on purpose — the guard exists because unguarded it once called an
elasticity result INCONSISTENT and blessed a biharmonic one. C9 states
`-div(sigma(u)) = f`, so the guard fires. **C3 is refused too**
(`-div(k grad u) + c*u = f` on one subdomain), and C3 is the most productive coupled
family in the record, 17 of the 33 CORRECT cells.

Accepted: C2, C8, C10. Refused: C3, C9.

## Finding 2 — one side of a coupled problem was almost always NOT_APPLICABLE, and is not any more

In the first measurement, side B came back `NOT_APPLICABLE` **92 times out of 94**. The
check explained itself honestly: its identity needed `u = 0` on the whole boundary, which
one side of a partitioned coupling never has, because the interface carries the partner's
data. So the check could not speak about almost any coupled problem — which is exactly
where it was about to run by default.

**That limit was removable.** Integrating the identity by parts twice leaves two boundary
terms:

```
closed_int  -K grad(u).n v   -> zero because v = 0 on the face
closed_int   u K grad(v).n   -> survives unless u = 0 on the face
```

A product of half-period sines kills only the first. A product of **sin²** has value *and*
normal slope zero on every face, so both go for any `u` at all, and no assumption about
the field's boundary trace is needed.

Calibrated on a manufactured case deliberately not zero on the boundary
(`u = sin(pi x) sin(pi y) + x`, so `-lap u = 2 pi² sin sin`), levels 16 → 128:

| test function | correct field | wrong field (½ amplitude) |
|---|---|---|
| sines | 8.13e-01 flat | 3.13e-01 flat |
| **sin²** | **4.87e-03 → 7.53e-05 (falls 64.7×)** | 5.02e-01 flat |

The sines cannot tell them apart at all and rank the correct field *worse*. sin² separates
them and falls at the documented factor of four per refinement. It holds at `k = 200` on an
off-unit box, catches a 3 % amplitude error (3.47e-02 flat) and catches the near-zero field
shape C9 keeps producing. Where `u` really is zero on the boundary the sines are exact
(1.8e-16) and sin² merely converges — that is the entire cost, and the verdict reads the
*fall*, not an absolute floor.

`NOT_APPLICABLE` drops from 78 to 35. The 35 that remain are cells with fewer than two
usable levels or a non-uniform export grid, not a limitation of the identity.

## Finding 3 — the one false alarm was the routing, not the field

`C2 seed 6751`, graded **CORRECT**, side A, was called `INCONSISTENT` with a residual that
is flat and non-monotone: `1.324e-02 → 5.787e-03 → 1.060e-02`, rate 0.16.

The first fix put sin² behind a size test — use it only when the outermost probe layer is
large relative to the field — and 6751 **survived it**. Diagnosing that is what produced
the real answer.

Its boundary layer is 0.067 of the field's own scale, under the 0.25 threshold, so it went
to the sines. Forced through sin², the **same three files** give:

| level | sines | sin² |
|---|---|---|
| 1 | 1.3238e-02 | 1.2506e-02 |
| 2 | 5.7869e-03 | 3.7640e-03 |
| 3 | 1.0603e-02 | **1.5867e-03** |

Monotone, rate ≈ 2.4, verdict `CONSISTENT`. The field was right the whole time.

**The size test cannot be made to work, at any threshold.** On a grid of cell midpoints the
outermost probe sits `h/2` inside the face, so a field that *is* zero on the boundary still
reads `|grad u| h/2` there — percent-level, the same order as a face genuinely carrying a
partner's data. The two cases are not separable by magnitude, and the probe grid here is
fixed across levels (1936 points at every level), so they are not separable by how the
layer shrinks either. So the size test was deleted along with the sines, and sin² is
unconditional.

Two quadrature checks were run before concluding this, to rule out the cheaper
explanations: the grid is exactly the midpoint grid of `extent_a` (44 × 44, first node at
`h/2`), and the grid value of `int f v` agrees with the exact symbolic integral to
3.5e-07 relative. The residual was measuring the field, not the integration.

## What this establishes

Run by default on every converged coupled level, the check as it now stands is silent on
C3 and C9 entirely (operator guard), answers **both** sides of the families it does accept,
catches **14 of 17** wrong cells, and accuses **none** of the 14 correct ones. That clears
the campaign's own admission bar, which the shipped version did not.

The three wrong cells it misses are all `COMPLETED_UNPHYSICAL` (C2 6633, 6952, 7212) and
all read `CONSISTENT` on both sides. That is a real and explainable limit rather than
noise: a test function that vanishes on the boundary annihilates boundary data by
construction, so a field that satisfies its equation in the interior but carries the wrong
condition on a face is invisible to this identity. Catching those needs a different
instrument, not a threshold.

**Caveat, stated plainly.** The arguments were reconstructed from `spec_public.json`:
per-side sources from `source_public`, per-side rectangles from `extent_a`/`extent_b`, and
`k` parsed out of the prose line "thermal conductivity k = 1 in subdomain A; k = 1000 in
subdomain B". An agent reading its task would supply these itself and might phrase them
differently. Finding 1 does not depend on that — it is decided by the `equation` string
alone — but the rates could move.

**A smaller thing worth noting:** the public spec gives `k` only in prose, while the check
wants a number or a JSON matrix. Whatever else is decided, an agent handed
"thermal conductivity k = 1 in subdomain A; ... k = 1000 in subdomain B" has to convert
that itself before the call succeeds. That is friction on the path to a check nobody calls.

---

## What the check provably cannot see: a coupling that exchanged nothing

Measured 2026-09-16, after the elasticity operator was added.

The campaign found a C9 cell, seed 8741, where **both sides exchanged literally nothing**
— the interface traction columns read `9.5e-18` and `0.0`. Each side returned the answer
it would have returned with no partner at all. Every per-participant guard stayed silent,
because each fires only when ITS OWN recovery is zero against a NONZERO partner; when the
exchange is dead in both directions both guards are satisfied. The iteration then
"converges" immediately, because two sides exchanging nothing cannot disagree.

**Prediction, stated before measuring:** the equation check misses it. A dead exchange
leaves each side solving its own equation in the interior with a homogeneous natural
condition on the interface, and an identity whose test function vanishes on the boundary
cannot see a wrong condition on a face.

**It held, and the margin is not close:**

| cell | graded | side A | side B | verdict |
|---|---|---|---|---|
| 8631 | CORRECT | 1.88e-01 → 1.13e-02 (16.6×) | 9.58e-02 → 6.21e-03 (15.4×) | CONSISTENT |
| 8791 | CORRECT | 1.86e-01 → 1.13e-02 (16.4×) | 1.16e-01 → 6.84e-03 (16.9×) | CONSISTENT |
| 8632 | COMPLETED_UNPHYSICAL | 1.08e+00 → 5.73e-01 (1.9×) | 8.55e+00 → 8.82e+00 (1.0×) | **INCONSISTENT** |
| 8741 | COMPLETED_UNPHYSICAL | 2.10e-01 → 1.15e-02 (18.3×) | 1.28e-01 → 6.09e-03 (21.1×) | CONSISTENT |

8741 does not merely slip past the threshold. Its residuals **fall faster than either
correct cell's** and its sequence is indistinguishable from theirs. No tuning of the
decay rule separates them, because there is nothing in the interior to separate: each
field really does satisfy its own equation.

**So the two instruments are strictly complementary, and neither substitutes for the
other.** The campaign's `_interface_transmitted_nothing` reads the interface files and
catches exactly this; the equation check reads the interior and catches 8632, whose side B
exports a trace 127 % different from the one it imported. Running only one of them leaves
a whole failure mode unwatched.

This is the general form of the three C2 cells the scalar check misses, all
COMPLETED_UNPHYSICAL and all reading CONSISTENT on both sides: **a field that is right in
the interior and wrong on a face is invisible to this identity, whatever the operator.**
That is a property of the method, not a threshold, and it is the reason this check should
be described as one instrument among several rather than as the one that separates right
from wrong.
