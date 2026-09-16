#!/usr/bin/env bash
# The standalone-tool gates (ported from DIALibGen): each tool runs with
# NOTHING from the build environment, reports ITS OWN version, keeps OpenMS's
# update check off, and parses its parameters in isolation.
#
#   standalone_test.sh <DIALibraryRefiner> <DIALibTune-or-empty> <expected-version>
set -u
REFINER="$1"; TUNE="${2:-}"; WANT="$3"
fail() { echo "FAIL: $*" >&2; exit 1; }
TMP=$(mktemp -d "${TMPDIR:-/tmp}/dlr-standalone.XXXXXX") || exit 1
trap 'rm -rf "$TMP"' EXIT
KEEP=(HOME="${HOME:-$TMP}")
[ -n "${OPENMS_DATA_PATH:-}" ] && KEEP+=(OPENMS_DATA_PATH="$OPENMS_DATA_PATH")
[ -n "${LD_LIBRARY_PATH:-}" ] && KEEP+=(LD_LIBRARY_PATH="$LD_LIBRARY_PATH")
[ -n "${DYLD_FALLBACK_LIBRARY_PATH:-}" ] && KEEP+=(DYLD_FALLBACK_LIBRARY_PATH="$DYLD_FALLBACK_LIBRARY_PATH")
run_bare() { env -i "${KEEP[@]}" "$@"; }

check_tool() {
  local bin="$1" name="$2"
  run_bare "$bin" --help > "$TMP/$name.help" 2>&1 || { cat "$TMP/$name.help" >&2; fail "$name does not run in a bare environment"; }
  got=$(sed -n 's/^Version: \([^ ]*\).*/\1/p' "$TMP/$name.help" | head -1)
  [ "$got" = "$WANT" ] || fail "$name reports version '$got', expected '$WANT' (OpenMS's leaking through?)"
  run_bare "$bin" --helphelp > "$TMP/$name.helphelp" 2>&1
  grep -q "$WANT" "$TMP/$name.helphelp" || fail "$name --helphelp does not carry version $WANT"
  grep -qi "OpenMS " "$TMP/$name.helphelp" || fail "$name --helphelp does not name the OpenMS version"
  for f in "$TMP/$name.help" "$TMP/$name.helphelp"; do
    ! grep -q "QIODevice\|QNetworkReply" "$f" || fail "$name: OpenMS's update check is live"
  done
  run_bare "$bin" -write_ini "$TMP/$name.ini" > "$TMP/$name.ini.log" 2>&1 || { cat "$TMP/$name.ini.log" >&2; fail "$name -write_ini failed"; }
  [ -s "$TMP/$name.ini" ] || fail "$name -write_ini wrote nothing"
  mkdir -p "$TMP/$name.ctd"
  run_bare "$bin" -write_ctd "$TMP/$name.ctd" > "$TMP/$name.ctd.log" 2>&1 || { cat "$TMP/$name.ctd.log" >&2; fail "$name -write_ctd failed (ToolHandler registration missing?)"; }
  [ -s "$TMP/$name.ctd/$name.ctd" ] || fail "$name -write_ctd produced no $name.ctd"
  echo "ok   $name $got: runs bare, own version, update check off, -write_ini and -write_ctd work"
}
check_tool "$REFINER" DIALibraryRefiner
# The refiner's effective config must be accepted back (defaults inside their own ranges).
run_bare "$REFINER" -in x.parquet -ids y.parquet -out z.parquet -write_config "$TMP/eff.json" > "$TMP/eff.log" 2>&1
[ -s "$TMP/eff.json" ] || fail "DIALibraryRefiner -write_config wrote nothing"
grep -q '"schema_version"' "$TMP/eff.json" || fail "effective config has no schema_version"
[ -n "$TUNE" ] && check_tool "$TUNE" DIALibTune
echo "PASSED"
