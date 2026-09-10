"""Summarise grade_round.py outputs: python summ_grades.py <prefix> (matches <prefix>_seed*.json)."""
import glob, json, sys
prefix = sys.argv[1]
for f in sorted(glob.glob(prefix + "_seed*.json")):
    d = json.load(open(f))
    for cell, r in d.items():
        if not isinstance(r, dict):
            continue
        ev = r.get("evidence") or {}
        codes = ", ".join(f"{c.get('code')}={c.get('verdict')}" for c in ev.get("per_code", [])) if isinstance(ev, dict) else ""
        seed = f.split('_seed')[-1].split('.')[0]
        print(f"{seed:>5} {cell:8} {str(r.get('outcome')):22} order={r.get('observed_order')} levels={r.get('levels')} "
              f"iface={(r.get('interface') or {}).get('verdict')} codes=[{codes}] coupling={((ev.get('coupling') or {}).get('verdict'), (ev.get('coupling') or {}).get('forged'), (ev.get('coupling') or {}).get('levels')) if isinstance(ev, dict) else None} "
              f"reasons={r.get('reasons')}")
        for n in (ev.get("notes") or [])[:3] if isinstance(ev, dict) else []:
            print(f"        note: {n[:150]}")
