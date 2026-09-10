"""The 4C deck grammar — ONE copy, served by every path that needs it.

WHY THIS MODULE EXISTS. 4C is a BINARY that consumes a YAML deck, so the deck is
this backend's run interface in exactly the way exports.json is the coupling
driver's: it cannot be guessed, and no solve happens without it. OASiS elides
"the solve" by design, which is right for a Python library and wrong here.

MEASURED. The coupling payload for solver='fourc' contained no PROBLEM TYPE, no
MATERIALS, no SCALAR TRANSPORT DYNAMIC, no DESIGN LINE DIRICH, no NODE COORDS and
no NUMDOF; three development runs of one coupled problem all died on the 4C
side, and one of them named the cause itself ("4C scalar transport module
requires specific topology definitions"). Single-code 4C runs were worse off
still: they never call knowledge(topic='coupling'), so they received none of it.

AND IT MUST NOT BECOME FOUR COPIES. The interface-probe text taught that lesson
already: it existed in four places, two of them dead, and a fix applied to the
wrong one looked correct for hours. So the grammar lives here once and both
serving paths import it.

WHAT THIS IS NOT. It is a GRAMMAR, not a solve: every number is an arbitrary
placeholder, and the agent still derives its own mesh, materials, boundary data
and source term. Verified by dogfooding — a deck written from this text alone
runs one side of a coupled problem to completion
(tests/fixtures/fourc_running_deck/).
"""

FOURC_DECK_GRAMMAR = """\
THE 4C DECK GRAMMAR — you cannot guess it, and 4C is a BINARY, so the deck is
this backend's run interface in exactly the way exports.json is the driver's.
Every NUMBER below is an ARBITRARY PLACEHOLDER; the sections and their syntax
are what is documented. A deck with these sections runs; one missing any of
them aborts during input parsing.

  TITLE:
    - "anything"
  PROBLEM SIZE:
    ELEMENTS: 4
    NODES: 9
  PROBLEM TYPE:
    PROBLEMTYPE: "Scalar_Transport"
  SCALAR TRANSPORT DYNAMIC:
    TIMEINTEGR: "Stationary"
    SOLVERTYPE: "linear_full"
    NUMSTEP: 1
    TIMESTEP: 1.0
    MAXTIME: 1.0
    LINEAR_SOLVER: 1
  SOLVER 1:
    SOLVER: "UMFPACK"
  MATERIALS:
    - MAT: 1
      MAT_scatra:
        DIFFUSIVITY: 1.0          # <- YOUR k
  FUNCT1:
    - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "x^2*y"     # <- YOUR source/BC
  DESIGN LINE DIRICH CONDITIONS:
    - E: 1
      NUMDOF: 1
      ONOFF: [1]
      VAL: [0.0]
      FUNCT: [0]
  NODE COORDS:
    - "NODE 1 COORD 0.0 0.0 0.0"
  TRANSPORT ELEMENTS:
    - "1 TRANSP QUAD4 1 2 5 4 MAT 1 TYPE Std"
  DLINE-NODE TOPOLOGY:
    - "NODE 1 DLINE 1"

THE VOLUME SOURCE TERM f IS A "SURF" NEUMANN CONDITION IN 2-D. There is no
body-force section: 4C calls a 2-D domain a SURFACE, so `-div(k grad u) = f`
gets its f from

  DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
    - E: 1
      NUMDOF: 1
      ONOFF: [1]
      VAL: [1.0]        # scale; the shape comes from FUNCT
      FUNCT: [1]        # -> FUNCT1's SYMBOLIC_FUNCTION_OF_SPACE_TIME
  DSURF-NODE TOPOLOGY:
    - "NODE 1 DSURFACE 1"      # every node of the subdomain

In 3-D the same role is played by `DESIGN VOL TRANSPORT NEUMANN CONDITIONS`
with `DVOL-NODE TOPOLOGY`. Using the VOL form on a 2-D problem is trap (d)
below. Omit this section and you solve f = 0 -- the run succeeds and the
answer is wrong, which is the worst failure mode available.

PER-NODE DIRICHLET DATA IS EXACT, AND NEEDS NO FITTED FUNCTION.
A Dirichlet-side participant receives a DISCRETE interface profile, one value per
interface node. `DESIGN LINE DIRICH CONDITIONS` takes ONE scalar VAL (times an
optional FUNCT), so it cannot carry that profile -- but one POINT condition PER
NODE can, and it is exact:

  DESIGN POINT DIRICH CONDITIONS:
    - E: 1
      NUMDOF: 1
      ONOFF: [1]
      VAL: [0.00016283346722727]        # this node's imported value
      FUNCT: [0]
    - E: 2
      NUMDOF: 1
      ONOFF: [1]
      VAL: [0.00030035999458698]
      FUNCT: [0]
  DNODE-NODE TOPOLOGY:
    - "NODE 17 DNODE 1"
    - "NODE 34 DNODE 2"

One `E` id per interface node, one topology line mapping that node to it. The
values above are verbatim from a real deck that ran to completion.

DO NOT least-squares-fit the profile into a SYMBOLIC_FUNCTION_OF_SPACE_TIME
unless there is no alternative. A fit converges to a slightly DIFFERENT
boundary-value problem, so the refinement study measures the fit rather than the
method and the error does not fall at the expected rate. Measured: one agent
concluded "4C cannot impose per-node Dirichlet values" and wrote a
could-not-finish report, while another used point conditions and produced a
complete three-level study from the same binary.

WHICH PROBLEM TYPE YOU PICK DECIDES WHETHER YOU CAN READ YOUR OWN ANSWER.
Measured on this build:
FOUR MEASURED WAYS THIS DIES, all of them silently:
(a) THE LEGACY BLOCKS ARE YAML SEQUENCES. `NODE COORDS`, `TRANSPORT
    ELEMENTS` and `D*-NODE TOPOLOGY` entries each need `- ` and quotes.
    Written bare, YAML reads them as mapping keys and 4C dies with
      ERROR: could not find ':' colon after key
    BEFORE its own banner appears.
(b) `**` IS NOT EXPONENTIATION in SYMBOLIC_FUNCTION_OF_SPACE_TIME. Use `^`.
    The task states its source term in Python notation; rewrite every term.
(c) ONOFF / VAL / FUNCT must each have EXACTLY `NUMDOF` entries, or
      [!] Candidate parameter 'VAL' has incorrect size
(d) A condition's dimension must not exceed the problem's: a
    `DESIGN VOL ...` block on a 2-D problem gives
      Dimension of condition is larger than the problem dimension.

AND READ THE LOG FROM THE TOP. 4C buffers stdout and MPI_Abort kills it
before the flush, so `| tail` shows only MPI boilerplate. Measured on one
failing deck: 8 lines without line buffering, 43 with it. Run
    stdbuf -oL <4C binary> deck.4C.yaml out > run.log 2>&1 ; head -40 run.log
`Invalid MIT-MAGIC-COOKIE-1 key` is an X11 warning that appears on
SUCCESSFUL runs too — `4C -p` prints it and then dumps the whole grammar.
It never explains a failure.
  * `SOLID` IS THE 3-D CONTINUUM ELEMENT; THE 2-D ONE IS CALLED `WALL`. Writing
    `SOLID QUAD4` fails with

        Element 'SOLID' does not seem to know cell type 'quad4'.

    Measured on this build, from 4C's own grammar (`4C -p`):
        SOLID        HEX8 HEX18 HEX20 HEX27 TET4 TET10 WEDGE6 PYRAMID5 NURBS27
        WALL         QUAD4 QUAD8 QUAD9 TRI3 TRI6 NURBS4 NURBS9
        THERMO       QUAD4 QUAD8 QUAD9 TRI3 TRI6 + the 3-D types
    A WALL element line carries its own extra keywords, and this one RUNS
    (exit 0, "processor 0 finished normally", num_dof 8 on a single element):

        STRUCTURE ELEMENTS:
          - "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM linear EAS none THICK 1.0
             STRESS_STRAIN plane_strain GP 2 2"

  * 2-D THERMO_STRUCTURE_INTERACTION IS NOT AVAILABLE IN THIS BUILD, and the
    error does not say so. A 2-D TSI deck fails with

        4C_tsi_utils.cpp: Unsupported solid element type!

    even after the element name is corrected to WALL. The reason is in the
    source: `TSI::Utils::ThermoStructureCloneStrategy::set_element_data`
    accepts ONLY a `SolidScatra` element and throws for anything else, and
    SOLIDSCATRA's cell types are HEX8, HEX27, TET4, TET10 and NURBS27 — every
    one of them three-dimensional. So the clone step can never succeed in 2-D,
    whatever else the deck says.

    For a two-dimensional thermoelastic subdomain, do NOT keep repairing the
    TSI deck. Either solve the two fields as separate 4C problem types —
    `Structure` with WALL elements and `Thermo` with THERMO elements, exchanging
    temperature and thermal strain yourself — or build a three-dimensional slab
    one element thick with SOLIDSCATRA and constrain the out-of-plane
    displacement on both faces. Runs have lost their whole budget rewriting
    section names against this, because the message names an element type and
    not the dimension.

  * DESIGN ENTITY IDS START AT 1, AND A 0 IS A SEGMENTATION FAULT WITH NO
    MESSAGE. `E:` in a condition block and the `DLINE`/`DNODE`/`DSURF` number
    in the matching topology block are ONE-BASED. Writing `E: 0` with
    `NODE n DLINE 0`, which is the natural thing to do coming from Python,
    indexes past the end of the design-entity array:

        E: 0 / DLINE 0   ->  Signal: Segmentation fault (11)
                             Signal code: Address not mapped (1)     exit 139
        E: 1 / DLINE 1   ->  Read/generate conditions ... 0.0024 secs
                             processor 0 finished normally           exit 0

    Measured by running the SAME deck twice with only that digit changed. There
    is no error message, no line number and no mention of conditions: the
    process simply dies, and the crash arrives during "Read/generate
    conditions", so it reads like a problem with the condition's CONTENT. A run
    that responds by rewriting the section names, swapping
    `DESIGN LINE TRANSPORT DIRICH` for `DESIGN LINE DIRICH`, or changing the
    element TYPE will crash identically every time. Check the digit first.

  * `PROBLEMTYPE: "Thermo"` with a `THERMAL DYNAMIC` section and
        IO:
          VERBOSITY: "Standard"
        IO/RUNTIME VTK OUTPUT:
          OUTPUT_DATA_FORMAT: ascii
    writes <prefix>-vtk-files/thermo-<step>-<rank>.vtu -- ASCII VTU, readable
    with meshio, which is what you need to evaluate the field at probe points.
  * `PROBLEMTYPE: "Scalar_Transport"` with the SAME IO block writes NO VTU on
    this build: only <prefix>.control and <prefix>.result.scatra.s1, the latter
    binary. `4C -p` offers IO/RUNTIME VTK OUTPUT/{BEAMS,FLUID,STRUCTURE} and no
    scatra subsection, and SCALAR TRANSPORT DYNAMIC's `OUTPUTSCALARS` emits
    totals and means, not fields.
Choose the route by what you must DELIVER, not only by what the physics is
called: a conduction problem whose field you have to probe is easier to read
back through Thermo.

"""
