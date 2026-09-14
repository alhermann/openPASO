"""For every coupled problem: what the door gives each side, and whether it arrives whole.

Campaign-side by design -- OASiS knows nothing about these problem ids. For each problem it reads the
public spec, works out the physics word an agent would pass for that family, calls the coupling door the
way a worker does (second call: the pointer-mode reply), and reports whether the first fenced block is
the handshake contract and which variant leads it.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

FAMILY_TO_PHYSICS = {
    "diffusion": "", "conjugate_heat_transfer": "", "fem_dsmc": "",
    "transient_diffusion": "transient",
    "elasticity": "elasticity",
    "thermal_structural": "thermoelastic",
    "fluid_structure_interaction": "",
}


def physics_for(spec: dict) -> str:
    fam = spec.get("physics_family", "")
    word = FAMILY_TO_PHYSICS.get(fam, "")
    if spec.get("dim") == 3 and not word:
        word = "3d"
    return word


def main() -> int:
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools

    class _MCP:
        def __init__(self):
            self.tools = {}

        def tool(self, *a, **k):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco

    mcp = _MCP()
    register_consolidated_tools(mcp)
    kn = mcp.tools["knowledge"]
    bad = 0
    for pid in [f"C{i}" for i in range(1, 15)]:
        spec_path = HERE / "problems" / pid / "spec_public.json"
        if not spec_path.is_file():
            continue
        spec = json.loads(spec_path.read_text())
        phys = physics_for(spec)
        for side, code in zip(("A", "B"), spec.get("codes", [])):
            role = (spec.get("roles") or {}).get(side, "")
            kn(topic="coupling", solver=code, physics=phys) if phys else kn(topic="coupling", solver=code)
            out = kn(topic="coupling", solver=code, physics=phys) if phys else kn(topic="coupling", solver=code)
            m = re.search(r"```python\n(.*?)```", out, re.S)
            block = m.group(1) if m else ""
            whole = bool(block) and "imports.json" in block and "exports.json" in block
            lead = ""
            d = re.search(r'"""(.+)', block) if block else None
            if d:
                lead = d.group(1)[:52]
            elided = "OASiS DOES NOT SERVE THIS" in block or "THE SOLVE ITSELF IS YOURS" in block
            if not whole or not elided:
                bad += 1
            flag = "ok " if (whole and elided) else "BAD"
            print(f"{flag} {pid:4} side {side} {code:8} role {role:9} physics {phys or '-':13} "
                  f"reply {len(out):6} block {len(block):6} | {lead}")
    print(f"\n{'READY' if bad == 0 else str(bad) + ' SIDE(S) NOT SERVED WHOLE'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
