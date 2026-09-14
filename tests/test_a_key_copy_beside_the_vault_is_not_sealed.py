"""A sealed vault says nothing about a copy of the keys beside it.

WHAT HAPPENED, MEASURED. The live vault was sealed:

    d--------- .../qwen_uplift_test/campaign3_blind/keys

and its sibling was not:

    drwxr-xr-x .../qwen_uplift_test/campaign3_blind/keys_backup_20260816

The sibling held 32 plaintext `key.json` files — every live problem ID, C1–C14
and all eighteen single-code cells — 28 of them carrying `exact_solution`. It was
created 2026-08-14, before every run in this campaign. `is_sealed()` reported
SEALED throughout, and it was telling the truth: it was only ever asked about one
directory.

TWO INDEPENDENT REASONS IT PERSISTED FOR SEVENTEEN DAYS:

  1. `shield_keys.sh` sealed `"$D/keys"` where `D` was the SCRIPT's own
     directory — i.e. `<repo>/campaign3_blind/keys`, which does not exist,
     because the keys deliberately live outside the repository. Every chmod was
     swallowed by `2>/dev/null`. The script had never touched the real vault.
  2. Nothing looked at siblings.

The blindness of a benchmark cannot be established by checking one path when a
decrypted copy sits next to it. So the question is now asked of the whole
parent: any sibling directory whose name begins with `keys` is a key store, and
one that is readable makes a run's blindness unprovable.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blind_eval import keyvault as kv  # noqa: E402

SHIELD = REPO_ROOT / "campaign3_blind" / "shield_keys.sh"


def _store(d: Path, ids=("C2", "FE1")):
    """A key store shaped like the real one."""
    d.mkdir(parents=True, exist_ok=True)
    for pid in ids:
        (d / pid).mkdir(exist_ok=True)
        (d / pid / "key.json").write_text(
            '{"exact_solution": {"A": "x*y", "B": "x+y"}, "tol": 0.4}')
    return d


class TestTheSiblingCheck(unittest.TestCase):
    def test_an_open_backup_beside_a_sealed_vault_is_reported(self):
        """The exact shape of the breach."""
        with TemporaryDirectory() as t:
            root = Path(t)
            keys = _store(root / "keys")
            _store(root / "keys_backup_20260816")
            kv.seal(keys)
            try:
                found = kv.unsealed_sibling_key_stores(keys)
            finally:
                kv.unseal(keys)
        self.assertEqual(
            [p.name for p, _ in found], ["keys_backup_20260816"],
            "a readable copy of the keys beside the sealed vault was not "
            "reported; this is what stayed open for 17 days")

    def test_a_sealed_sibling_is_not_reported(self):
        with TemporaryDirectory() as t:
            root = Path(t)
            keys = _store(root / "keys")
            backup = _store(root / "keys_backup")
            kv.seal(keys)
            kv.seal(backup)
            try:
                found = kv.unsealed_sibling_key_stores(keys)
            finally:
                kv.unseal(keys)
                kv.unseal(backup)
        self.assertEqual(found, [])

    def test_a_non_key_sibling_is_ignored(self):
        """`problems/` and `runs/` live beside the vault and must not trip it."""
        with TemporaryDirectory() as t:
            root = Path(t)
            keys = _store(root / "keys")
            (root / "problems").mkdir()
            (root / "runs").mkdir()
            kv.seal(keys)
            try:
                found = kv.unsealed_sibling_key_stores(keys)
            finally:
                kv.unseal(keys)
        self.assertEqual(found, [])

    def test_the_primary_itself_is_not_double_reported(self):
        with TemporaryDirectory() as t:
            keys = _store(Path(t) / "keys")
            found = kv.unsealed_sibling_key_stores(keys)   # primary is OPEN
        self.assertEqual(found, [], "the primary reported itself as a sibling")


class TestTheShieldingScript(unittest.TestCase):
    def _run(self, arg, env_keys):
        return subprocess.run(
            ["bash", str(SHIELD), arg], capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "OPENPASO_BLIND_KEYS": str(env_keys)})

    def test_it_seals_every_sibling_store_not_just_the_primary(self):
        with TemporaryDirectory() as t:
            root = Path(t)
            keys = _store(root / "keys")
            backup = _store(root / "keys_backup_20260816")
            try:
                r = self._run("seal", keys)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(kv.seal_state(keys), "SEALED")
                self.assertEqual(
                    kv.seal_state(backup), "SEALED",
                    "the backup store was left readable; the script only knew "
                    f"about the primary. stdout={r.stdout}")
            finally:
                for d in (keys, backup):
                    kv.unseal(d)

    def test_status_exits_nonzero_when_any_store_is_open(self):
        """So a driver script cannot proceed past `status` on a breach."""
        with TemporaryDirectory() as t:
            root = Path(t)
            keys = _store(root / "keys")
            backup = _store(root / "keys_backup")
            kv.seal(keys)
            try:
                r = self._run("status", keys)
                self.assertNotEqual(
                    r.returncode, 0,
                    f"status reported success with an OPEN key store: "
                    f"{r.stdout}")
                self.assertIn("READABLE KEY STORE", r.stdout)
            finally:
                kv.unseal(keys)
                kv.unseal(backup)

    def test_it_refuses_to_guess_where_the_keys_are(self):
        """It used to seal its OWN directory, which does not hold the keys, and
        hid every failure with 2>/dev/null."""
        r = subprocess.run(["bash", str(SHIELD), "status"],
                           capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("REFUSING", r.stderr)

    def test_absence_is_not_a_seal(self):
        with TemporaryDirectory() as t:
            r = self._run("status", Path(t) / "does_not_exist")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not a directory", r.stderr)


class TestTheLiveVaultRightNow(unittest.TestCase):
    def test_no_readable_key_store_exists_beside_the_live_vault(self):
        """The one that asks about the real machine rather than a fixture."""
        import os
        live = os.environ.get(
            "OPENPASO_BLIND_KEYS",
            "/home/user/Schreibtisch/qwen_uplift_test/campaign3_blind/keys")
        p = Path(live)
        if not p.parent.is_dir():
            self.skipTest(f"{p.parent} not present on this machine")
        found = kv.unsealed_sibling_key_stores(p)
        self.assertEqual(
            [(str(d), st) for d, st in found], [],
            "a readable copy of the answer keys exists beside the vault; no "
            "run started while it is open can be shown to have been blind")


if __name__ == "__main__":
    unittest.main()
