# Where the blind re-run actually stands, against the targets

Written 2026-08-20, after five development rounds at 27B (576 runs, of which
366 ledgers survive in the tree plus 18 credit casualties quarantined). This
file exists because the targets were agreed in conversation and written down
NOWHERE, and an unwritten target is one that drifts to meet whatever was
measured.

Updated 2026-08-29 after seven rounds: the measured table below now carries
all seven, every number recounted from dev_grades_27b_seed*.json on that
date, and the sections dated 2026-08-27 and 2026-08-29 at the end retract
what the earlier text got wrong. The 2026-08-20 prose between the table and
those sections is left standing as the record of what was believed then;
where it is wrong, a dated bracket says so in place.

## The targets, as agreed

    at 27B      OASiS > 70%, bare ~36%, uplift > +33 points, McNemar significant
    at 122B     OASiS > 61%
    at 397B     OASiS > 66%
    coupling    bare completes zero real coupled runs and fabricates;
                OASiS completes them
    substitution  27B + OASiS >= 397B bare (67%)
    cost/energy   better than 1.2x lower per solved task
    reliability   fabrication near zero in the OASiS arm

## What seven rounds at 27B measured (recounted 2026-08-29 from the grade files)

    16 FEM single-code cells (grade 1, order-based)
      round 3   bare 22.9%   OASiS 33.3%   +10.4      (seeds 1+2+3 pooled)
      round 4   bare 25.0%   OASiS 37.5%   +12.5
      round 5   bare 18.8%   OASiS 31.2%   +12.5
      round 6   bare 25.0%   OASiS 31.2%   +6.2
      round 7   bare 18.8%   OASiS 37.5%   +18.8
      pooled, rounds 4-6: bare 22/96 (22.9%), OASiS 32/96 (33.3%), +10.4
      pooled, all graded seeds 1-11: bare 37/160 (23.1%), OASiS 56/160
      (35.0%), +11.9
      (The per-round McNemar p-values this table used to carry are retired:
       the 2026-08-27 correction below shows that test is anti-conservative
       here — seeds are nested in cells. Significance claims must use the
       cluster-correct tests described there.)

    12 pooled coupled cells (grade 1), CORRECT per arm
      round 3   bare 1/36*   OASiS 0/36     round 6   bare 0/24   OASiS 0/24
      round 4   bare 0/24    OASiS 0/24     round 7   bare 0/24   OASiS 0/24
      round 5   bare 0/18    OASiS 0/16
      all graded seeds 1-11: bare 1/126, OASiS 0/124
      * C7 seed 2 — the campaign's single coupled CORRECT, a BARE run,
        hidden by a grader defect until the 2026-08-27 regrade (7a3284a4).
        See the 2026-08-29 section below.

    SPARTA (grade 3, band-only, WITHIN_BAND per arm)
      round 3  bare 4/6  OASiS 0/6      round 6  bare 1/4  OASiS 2/4
      round 4  bare 1/4  OASiS 3/4      round 7  bare 1/4  OASiS 3/4
      round 5  bare 1/4  OASiS 2/4

## Read honestly

**The headline uplift is about a third of the target and is not significant.**
+10.4, +12.5, +12.5 against a target of +33. The consistency across three
independent rounds is the strongest thing about it — the direction never
reverses — but no single round reaches significance, and OASiS at 31-37% is
nowhere near 70%.

**These are NOT the paper's numbers and must not be compared as if they were.**
The paper's rates come from its own evaluation draw. These 32 cells are the
DEVELOPMENT set: deliberately diagnostic, deliberately burnt, and looked at
repeatedly precisely so that knowledge could be fixed against them. A
development rate is a measure of the instrument and the knowledge, not the
number that goes in a table. The evaluation phase draws fresh problems after a
freeze, and only those numbers are comparable to anything published.

That is a real caveat, not an excuse. The uplift is the quantity least sensitive
to which problems were drawn, and it is +12.5, not +33.

**The coupling claim is supported in behaviour and unsupported in score.**
Across three rounds the bare arm has driven a coupling to convergence ZERO
times in 55 attempts, while the OASiS arm has done it 21 times (7, 11, and 3 in
the measured slices) and has climbed on every intermediate step: reaching the
coupling tool 29% -> 46% -> 57%, building the prescribed per-level structure
50% -> 61% -> 79%. But the graded score is 0 in BOTH arms, so on the paper's own
scoring the coupling claim currently has no numerical support. What we can say
today is a process claim, not an outcome claim.
[This paragraph is doubly wrong. The behaviour half was retracted 2026-08-27
— the instrument could only see one arm (below). The "0 in BOTH arms" score
half fell 2026-08-29: the bare arm had one CORRECT, hidden by a grader
defect (below).]

**The reliability claim IS supported, and it is the cleanest result we have.**
On coupled cells in round 4 the bare arm fabricated 10 of 28 runs; the OASiS arm
fabricated 2. Across all five rounds no OASiS run has been graded
FABRICATED_NO_RUN at a rate approaching bare's. Fabrication near zero in the
OASiS arm is the one target currently met.
[Qualified 2026-08-27: round 6 broke this claim with exact parity, 9/24 vs
9/24. Updated 2026-08-29: round 7 recovers it, 12/24 vs 5/24 — see below.]

**Untested:** the 122B and 397B tiers, substitution, and cost/energy. The ladder
rule requires a converged tier before advancing, and 27B has not converged.

## What has to happen for the targets to be reachable

[SUPERSEDED. CONVERGENCE.md's later correction measured this across all 54
round-5 OASiS runs instead of one 5-run slice: solver-output-with-no-summary
is 4% of runs, and the notice's host tool is used by ~15% — completion is
not the gap, correctness is. Round 6 therefore could not test the notice and
was re-purposed for power. The current blocker analysis is in the
2026-08-29 section below.]

The blocker is not knowledge coverage. Five rounds have produced 16 primitives
and the last round produced no new knowledge gap at all. The blocker is that
runs which solve correctly do not finish: three of the five failing single-code
OASiS runs in round 5 seed 6 made 85-107 tool calls, produced solver output, and
wrote no summary. Every solved-but-unwritten run is a target point lost for a
reason that has nothing to do with whether OASiS knows enough.

If round 6's just-in-time notice converts even half of those, the single-code
rate moves without a single new fact being added. That is the cheapest available
distance to the target, and it is why round 6 measures the unwritten-summary
rate rather than the solve rate.

---

## CORRECTIONS, 2026-08-27 — three claims above are wrong or unsupported

An independent re-derivation of every number from the raw grades, with code
that shares nothing with the analysis script, found errors I had repeated for
days. Each is verified by me before being written here.

**RETRACTED — the coupling behaviour claim.** I reported "the bare arm has
driven a coupling to convergence ZERO times in 55 attempts, while the OASiS arm
has done it 21 times" as the strongest coupling evidence we had. It is an
artefact of the instrument. I measured it by grepping run transcripts for
`converged: true` — a string emitted by the OASiS `couple` tool's JSON return.
THE BARE ARM HAS NO SUCH TOOL AND CANNOT PRINT IT. The test could only ever
find OASiS runs; it was biased by construction, and the direction of the bias
is the direction of the claim.

The opposite measurement is biased the opposite way. The grader's own
code-agnostic coupling evidence (a falling partitioned residual history in
FILES) gives BARE 20/90 PROVEN against OASiS 5/88 — but 68 of 88 OASiS records
carry NO coupling block at all, because `couple` returns its history in-band
and only 26 of the 61 runs that called it ever wrote a residual file. Each
instrument sees the arm whose idiom it was written for.

So the honest statement today is: WE DO NOT KNOW how the arms compare on
achieving coupling convergence. What survives is grader-adjudicated and
arm-symmetric: the 12 pooled coupled cells are 0/102 (bare) and 0/100 (OASiS)
across rounds 3-6 — neither arm has produced a gradeable coupled answer. A
code-agnostic convergence measure has to be built before any coupling
behaviour claim goes near the paper.
[CORRECTED 2026-08-29: the replacement figure was itself stale within the
hour — written 30 minutes before the 7a3284a4 grader fix landed. Recounted
from the re-graded files, the same pool is bare 1/102, OASiS 0/100 (and the
"rounds 3-6" label was wrong on its own terms: 102/100 pools seeds 1-9, and
seed 1 was graded in round 2). The one CORRECT is C7_BARE seed 2. Totals
over all graded seeds 1-11: bare 1/126, OASiS 0/124. "Neither arm has
produced a gradeable coupled answer" is therefore false for the bare arm.
See the 2026-08-29 section below.]

**BROKEN IN ROUND 6 — the reliability claim.** "No OASiS run has been graded
FABRICATED_NO_RUN at a rate approaching bare's" holds for rounds 3-5 (17/36 vs
5/36, 10/24 vs 2/24, 9/18 vs 2/16) and FAILS in round 6: BARE 9/24 and OASiS
9/24, exact parity. The claim is now "held in rounds 3-5, lost in round 6",
and why it regressed is an open question for round 7's grading, not a
footnote.

**OVERSTATED — the +10.4 uplift is one cell.** Per cell over seeds 4-9:
DL1 goes 0/6 -> 6/6 (+6) and NG1 goes 6/6 -> 1/6 (-5); 7 of 16 cells are tied
at 0/6-vs-0/6 or equal. Net +10 paired wins, of which one cell supplies +6.
The instance-level cluster bootstrap interval is [-9.4, +30.2] points and
contains zero comfortably.

**And the significance was computed wrongly.** McNemar over 96 pairs treats six
seeds per cell as independent trials; they are nested in 16 cells. Cluster-
correct (exhaustive enumeration of all 2^16 within-cell arm-label swaps) gives
p = 0.383, and a cell-level sign test gives p = 0.508, against the 0.13 I have
been quoting. p = 0.13 is anti-conservative by roughly a factor of three. Every
future rate must carry the clustered interval, not the paired-binomial one.

**Also fixed:** the fabrication line elsewhere in the campaign log used a
28-run all-14-cell denominator against a grade-1-only numerator (C13/C14
cannot carry FABRICATED_NO_RUN — different verdict vocabulary), and a
campaign-wide fabrication count keyed on the top-level status field misses 24
real fabrications visible only in `evidence.fatal`. True campaign total 106,
not 82. The duplicate `dev_grades_27b_seed1.json` — byte-identical to the
`_round2` file and one `*seed*` glob away from double-counting seed 1 — has
been deleted.

**What this does NOT change:** the FEM uplift arithmetic (+10.4, +12.5, +12.5,
+6.25 per round; pooled 22/96 vs 32/96) reproduced exactly, and the coupled
0-in-both-arms result reproduced exactly. The errors are in what the numbers
were claimed to MEAN, not in the numbers themselves — which is the more
dangerous kind.
[2026-08-29: the coupled half of that sentence did not survive the day it
was written. The re-derivation "reproduced exactly" the output of a grader
whose interface discovery was broken; 30 minutes later 7a3284a4 fixed it and
the bare arm has one CORRECT. Independent recomputation is not independent
of the instrument it recomputes from.]

---

## STATUS, 2026-08-29 — round 7, the regrade, and everything since

Written after re-deriving every number below from dev_grades_27b_seed*.json
and the runs_quarantine/ ledgers on 2026-08-29. Git references are to
commits verified in this tree.

**Round 7 (seeds 10+11, graded 2026-08-27), recounted.** Both seed files
carry all 64 cells; 21 infrastructure casualties (19 credit-402s in seed 11,
one output truncation NG1_BARE seed 10, one stdout-corruption crash DU1_MCP
seed 11 — each now documented by a WHY.md beside its ledger) were
quarantined and re-run before grading.

    16 FEM single-code    bare 6/32 (18.8%)   OASiS 12/32 (37.5%)   +18.8
    12 pooled coupled     bare 0/24           OASiS 0/24
    SPARTA (band-only)    bare 1/4            OASiS 3/4

+18.8 is the campaign's largest FEM uplift and, at n=32 pairs with the
clustered interval on a 10-point uplift already spanning [-9.4, +30.2], it
is noise until replicated. Behind it, recounted: submitted-and-wrong
(COMPLETED_UNPHYSICAL + CONFIDENTLY_WRONG) on FEM/OASiS went 7 -> 2 runs
(21.9% -> 6.2%) from round 6, but only 2 of the 5 became CORRECT (10 -> 12);
the others became FAILED (4 -> 6) or MALFORMED_SUBMISSION (3 -> 5).
CONFIDENTLY_WRONG hit zero (2 -> 0), which is the real reliability movement.
CONVERGENCE.md records audit adoption at 1 of 51 OASiS runs, so the audit
tool did not cause any of it.

**Coupled fabrication, recounted per round (FABRICATED_NO_RUN, grade-1
cells, bare vs OASiS):** 17/36 vs 5/36 (seeds 1-3), 10/24 vs 2/24 (r4),
9/18 vs 2/16 (r5), 9/24 vs 9/24 (r6 — exact parity), 12/24 vs 5/24 (r7).
Campaign totals on these cells: bare 57/126, OASiS 23/124. The honest claim
is a strong tendency with one measured exception, not an invariant — and
"fabrication near zero in the OASiS arm" is NOT currently met on coupled
cells.

**The campaign-long "coupled is 0 in both arms" was false.** A non-recursive
glob in grading/iface.py hid interface files that agents wrote into
subdirectories; the 2026-08-27 regrade (7a3284a4) moved C7_BARE seed 2 to
CORRECT — observed order 1.976, r^2 = 1.000, interface jumps 0.000 at all
three levels, submitted by an agent whose own summary claimed NOT_CONVERGED.
Recounted totals: bare 1/126, OASiS 0/124 over all graded seeds. The full
table and the two smaller false statements found while recounting are in
CONVERGENCE.md's correction of the same date. What this does to the paper's
coupling story: the OASiS arm's zero is intact, and the one gradeable
coupled success of the campaign sits in the BARE arm.

**These totals are still not final.** fbd583ef (2026-08-29) corrected the
interface gate: it enforced a fixed jump tolerance — a mesh ruler — rather
than the jump-stays-O(1)-under-refinement test its message claimed, and the
flux recovery that passes the fixed tolerance was described in the OASiS
payload but not in the task, an arm asymmetry. Three coupled cells failed
solely on that gate; the commit's own regrade measurement has C8 seed 4
CORRECT in BOTH arms under the corrected test. The grade files in the tree
are unchanged in git since the 2026-08-27 regrade and still carry
COMPLETED_UNPHYSICAL (bare) / CONFIDENTLY_WRONG (OASiS) for that cell, so
every coupled table in this file must be recounted after the next regrade.

**Every uplift above carries a harness qualifier (9835afa8, 2026-08-27).**
LangGraph ended a run the moment a reply carried no tool call — right when
the agent is done, wrong when it was mid-sentence. The commit's measurement
over the 813 runs to date: 112 (13.8%) ended silently with no result on
disk, and the exposure is NOT arm-symmetric — 79/405 OASiS (19.5%) against
33/408 bare (8.1%), because the OASiS arm accumulates context faster. Every
uplift in this file was measured with OASiS silently losing about one run in
five to the harness. Fixed (the run continues); no round before 8 had the
fix.

**Development switched from rounds to cells (b9326dfa, 2026-08-27).** One
cell, three seeds, ~4 credits, ~20 minutes — cheap enough to repeat after
every fix. The per-cell probes on coupled cells (MCP arm, seeds 12-13, then
C9 at seeds 21-23) produced the findings below; probes invalidated by our
own defects are quarantined under runs_quarantine/ with WHY.md files and
enter no count.

**What landed 2026-08-27..29** (each with its own firing test, none yet
measured at round scale — round 8 is that measurement):

  * Round-8 fixes batch (1728e668): structural participant delivery,
    converged-run verdict split, 4C stdbuf, DUNE probe, universal block on
    the live tool, audit hardening.
  * The give-up gate (8c80f60f): a COULD_NOT_COMPLETE written over work that
    is on disk is answered with what the disk already holds. Its notice
    originally promised that a partial submission "is graded on the part you
    supply" — false to the grader, which returns MALFORMED_SUBMISSION
    (WRONG_LEVEL_COUNT / MISSING_SUBDOMAIN_FILE) for missing levels or
    sides; corrected 2026-08-29 to state the completeness requirement.
  * Option B (26fa7c3a, b7db46f3, 64a922af, 18417077): the coupling
    knowledge had been serving COMPLETE participant scripts, solve included
    — making the measured uplift partly "was handed a working solver". The
    solve is now elided at serving time for all nine backends, the automatic
    file delivery was reverted, and the manual door
    (materialize_participant, called 2-4 times per probe run) is removed;
    a test walks every MCP tool and fails on any that installs a solver.
    The three contaminated C9 probe runs are quarantined
    (20260829_materialize_contaminated) and excluded from every count.
    2026-09-10: two further doors found and closed -- the participant
    PARTS route (`signal='participant[:role]:partN'`) served the raw file,
    solve included, with its SHA-256 (used by one development run, C2
    seed 1503, which is therefore void as an honest data point), and the
    three Kratos participants kept `strategy.Solve()` outside the SOLVE
    markers. Every coupled CORRECT recorded before that date on C3
    (2/3, 3/6) was measured with complete participants served and is not
    a paper number; the honest C3 tally at that date is 0/3.
  * The clock (4dd880c5): every command result now carries
    `[clock: N min left of M]`, both arms by construction. Probe seeds 22
    and 23 had filed give-ups blaming a 45-minute budget after using ~18
    minutes of 45 — the agent had no way to know the time and guessed 2.5x
    wrong in the direction that costs a run.
  * The self-check was blind to every coupled submission (f283b6cd): its
    more-than-one-file-per-level guard fired on every coupled level (five
    files match), so the near-zero/floor/order checks were dead on exactly
    the half of the campaign that scores zero. Re-calibrated; the commit's
    measurement: 0 false alarms on 93 graded-CORRECT single-code runs,
    catch 64 of 83 submitted-and-wrong (66 of 83 after e8c8e9ed).
  * OASiS was telling the model it was being tested (152f0118): served
    payloads and tool descriptions carried "this campaign", run counts, and
    one line revealing a cell's existence — to the treatment arm only. The
    provenance is removed, the lessons kept, and a test screens every
    served payload and tool description.
  * Three doors the knowledge never came through (e8c8e9ed): the universal
    block rode knowledge(topic='physics') but not prepare_simulation (43 of
    98 runs never received it); the physics matcher and the block were
    paired the wrong way round (38-character replies where 24727 characters
    existed); audit_results was advertised only inside the block that did
    not arrive (0 calls in 98 runs). All three doors now serve.
  * The interface gate (fbd583ef): above.

**Round 8 is in flight as of this writing** — rebuild re-baseline, seeds
14+15, 32 cells, both arms, 128 runs, launched 2026-08-29 21:13. Nothing
from it is graded or in this file.

**The standing discipline, unchanged:** these are DEVELOPMENT-set numbers.
The cells are burnt — diagnosed against, fixed against, looked at
repeatedly — and none of these rates is the paper's number. The paper's
numbers come from fresh evaluation draws after a freeze, and the freeze
criterion (CONVERGENCE.md) has not been met. Against the targets at the top
of this file: uplift ~+10 to +12 pooled against +33; OASiS 31-37% against
70%; coupled OASiS 0/124 against "OASiS completes them"; substitution,
cost/energy, 122B and 397B untested.
