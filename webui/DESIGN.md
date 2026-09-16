# openPASO console design system

The reference for how this interface looks and behaves. Written so a later
change stays consistent instead of re-deciding from scratch.

**Design read.** An operator console for a scientific agent, for engineers and
researchers who do not want to learn nine input formats, with a restrained
instrument language, leaning toward a dark technical UI with a single coral
accent.

**Dials.** `DESIGN_VARIANCE 5` · `MOTION_INTENSITY 6` · `VISUAL_DENSITY 7`.
Density sits above a landing-page baseline on purpose: this is a cockpit
watching a live run, so numbers are monospaced and 1px rules do the grouping
that card boxes would do elsewhere.

---

## 1. Overview

Three fixed columns on one dark ground, no page scroll. Left is the navigator
(session, model, mode, servers, tokens, history). Centre is the run: a header
line, the log, and the prompt. Right is the sandbox: the files the run produced
and a panel that renders whichever one you click.

The ground is flat, not a gradient. The logo was drawn against `#0D1117`, so the
mark sits on the page with no visible edge.

Six characteristics:

- one saturated colour, and it means one thing
- 1px rules instead of card boxes
- every number monospaced
- one corner radius everywhere
- motion only where it carries information
- dark only, and deliberately so

---

## 2. Colour

### The reservation rule

**Coral `#FF6B4A` is openPASO's own voice.** The brand mark, the primary action,
and openPASO speaking in the log. Nothing else.

The console still needs to tell a tool call from a sub-agent from an error, so
those keep their own hues, desaturated until coral is the loudest thing on the
page. Saturation is the hierarchy: coral sits at 100%, every state hue between
42% and 62%.

| token | value | role |
|---|---|---|
| `ink-950` | `#0D1117` | canvas. The brand dark |
| `ink-800` | `#161B22` | raised surface |
| `ink-700` | `#21262D` | hairline, and the main grouping device |
| `ink-600` … `ink-400` | `#2D333B` … `#4A545F` | rising surfaces |
| `accent-500` | `#FF6B4A` | coral. Reserved |
| `accent-400` / `600` | `#FF8A6E` / `#E85530` | hover / pressed |
| `accent-300` | `#FFA890` | coral text on dark |
| `graphit` | `#64748B` | the brand neutral. Identical to Tailwind `slate-500` |
| `state-call` | `#BF944A` | a tool call |
| `state-sub` | `#A08BD0` | a sub-agent |
| `state-err` | `#D85A6F` | an error |
| text | `#E6EDF3` / `#9BA7B4` / `#6E7B8B` | primary / muted / subtle |

---

## 3. Typography

**Inter** for the interface, **JetBrains Mono** for anything a machine produced.
Both SIL OFL, so they can be embedded and shipped.

Monospace is not decoration: it marks machine output. Session ids, token counts,
tool arguments, file sizes, residuals, paths and log bodies are all mono. Prose
is Inter.

Section labels are 10px, uppercase, wide tracking, subtle ink. Body is 14px.
The log body is 13px mono and wraps rather than truncating.

---

## 4. Layout

Left rail 288px, right panel 448px, centre takes the rest and never goes below
0 width. Interior padding 20px on the rails, 24px in the centre column.
Log entries are separated by 12px and reach at most `max-w-3xl`, so a long line
stays readable on a wide screen.

---

## 5. Elevation

Four steps, by surface and rule, never by drop shadow. Canvas `ink-950`, panels
`ink-800`, log entries a translucent `ink-800`, and the raised control state
`ink-700`. Glow is used once, on nothing by default; there are no atmospheric
gradients and no glassmorphism.

---

## 6. Shape

**One radius: 4px.** Buttons, inputs, cards, pills, the lot. The only round
thing on the page is the connection dot, which is a dot and not a corner.

---

## 7. Components

- **Buttons.** Primary is flat `accent-500` with `ink-950` text. Secondary is
  `ink-800` with a hairline. Both 4px, both get a visible focus ring.
- **Mode control.** Three segments; the selected one is `ink-700` with coral
  text, not a coral fill. A pill that is always coral is not a signal.
- **Log entry.** A hairline, a translucent fill, an icon, a label, and a
  monospace body. The type of the event decides the hue.
- **Prompt.** A bordered row that takes a coral border on focus-within, a
  labelled textarea, and the primary button.
- **Result panel.** One of text, JSON, table, plot, image, field animation or
  mesh, decided by the file.

---

## 8. Motion

The exemplar this structure came from has no motion section. This one needs it,
because the log streams and the interface is filmed.

**Motivated only.** Before adding an animation, say in one sentence what it
communicates. Valid: something arrived, something changed state, focus moved.
Not valid: it looked good.

- `transform` and `opacity` only
- in 160ms, out 120ms, `cubic-bezier(0.16, 1, 0.3, 1)`
- a log entry animates in once and then holds still
- exactly one perpetual loop is allowed: the connection dot
- **`prefers-reduced-motion` collapses all of it to instant.** Not optional

---

## 9. Do and do not

**Do**

- keep coral for openPASO acting, and let saturation carry the hierarchy
- group with a 1px rule and space before reaching for a card
- set every number in mono, so columns line up and digits do not shift
- give every control a visible `:focus-visible` ring
- write what happened as a sentence

**Do not**

- introduce a second saturated accent, for marketing or anything else
- print raw JSON, a stack trace, or an absolute path into the log
- use a gradient as a surface, or `backdrop-blur` anywhere
- mix corner radii
- animate something because the library makes it easy
- ship a light mode half-done; this interface is dark by decision

---

## 10. Working on it

1. Take one component at a time and decide its surface step first.
2. Reach for a hairline before a card.
3. Treat coral as scarce. If you are adding a third coral thing to a view, one
   of them is wrong.
4. Check the result against the Web Interface Guidelines before calling it done.
5. Look at it at 1920 wide and at phone width.

## 11. Known gaps

- Light mode is not designed and is not planned.
- Third-party assets still load from CDNs, so the page does not render offline.
- The result panel's mesh view renders a single file and has no time axis.
- Responsive behaviour below about 900px has not been designed; the three
  columns do not yet collapse.
