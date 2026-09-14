#!/usr/bin/env bash
# Seal / unseal the answer keys. Keys are unreadable while agents run.
#
# TWO DEFECTS FIXED HERE, BOTH MEASURED.
#
# 1. IT WAS SEALING A DIRECTORY THAT DOES NOT EXIST. `D` was the SCRIPT's own
#    directory, so it acted on <repo>/campaign3_blind/keys -- and there is no
#    such directory: the keys deliberately live outside the repository, at
#    $OPENPASO_BLIND_KEYS. Every `chmod` was suppressed by `2>/dev/null` and every
#    `stat` printed an error, so `seal` did nothing at all to the real vault.
#    OPENPASO_BLIND_KEYS is the single authority, exactly as in the grader, the
#    runner and loading.py.
#
# 2. IT ONLY EVER KNEW ABOUT "keys". A copy of the keys is a copy of the
#    answers. Measured on this machine: keys/ was d--------- while its sibling
#    keys_backup_20260816/ was drwxr-xr-x and held 32 plaintext key.json files
#    -- every live problem ID, C1-C14 and all eighteen single-code cells, 28 of
#    them carrying `exact_solution` -- created 2026-08-14, i.e. before every run
#    in this campaign. is_sealed() reported SEALED throughout, truthfully,
#    because it was only ever asked about one directory.
#
#    So every sibling whose name starts with `keys` is treated as a key store.
set -u
KEYS="${OPENPASO_BLIND_KEYS:-}"
if [ -z "$KEYS" ]; then
  echo "REFUSING: OPENPASO_BLIND_KEYS is not set. The keys live outside the" >&2
  echo "repository and this script must not guess where." >&2
  exit 2
fi
if [ ! -d "$KEYS" ]; then
  echo "REFUSING: \$OPENPASO_BLIND_KEYS=$KEYS is not a directory. Absence is" >&2
  echo "not a seal -- a missing keys tree means nothing to grade against." >&2
  exit 2
fi
PARENT="$(cd "$(dirname "$KEYS")" && pwd)"

stores() {                      # every key store: the primary and its siblings
  find "$PARENT" -maxdepth 1 -type d -name 'keys*' | sort
}

case "${1:-}" in
  seal)
    for d in $(stores); do
      chmod -R a-rwx "$d"; chmod 000 "$d"
      echo "SEALED    $(stat -c %A "$d")  $d"
    done ;;
  unseal)
    for d in $(stores); do
      chmod 700 "$d"; chmod -R u+rwX "$d"
      echo "UNSEALED  $(stat -c %A "$d")  $d"
    done ;;
  status)
    n_open=0
    for d in $(stores); do
      m="$(stat -c %A "$d")"
      case "$m" in
        d---------) echo "SEALED    $m  $d" ;;
        *)          echo "OPEN      $m  $d  <-- READABLE KEY STORE"
                    n_open=$((n_open + 1)) ;;
      esac
    done
    [ "$n_open" -gt 0 ] && exit 1
    exit 0 ;;
  *)
    echo "usage: OPENPASO_BLIND_KEYS=<dir> shield_keys.sh {seal|unseal|status}"
    exit 2 ;;
esac
