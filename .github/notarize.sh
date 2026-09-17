#!/usr/bin/env bash
# Submit one artefact to Apple for notarization: say WHY when Apple rejects it,
# and RETRY when the network did.
#
#   .github/notarize.sh <a .zip, .dmg or .pkg>
#
# needs MACOS_APPLE_ID, MACOS_TEAM_ID and MACOS_NOTARY_PASSWORD in the env.
#
# Two failures wear the same exit code and must not be treated alike:
#
#   REJECTED -- Apple looked and said Invalid. `notarytool submit --wait` then
#     prints a submission id and nothing else: the reasons (the unsigned
#     binary, the missing hardened runtime, the bad entitlement) live behind
#     `notarytool log <id>`, a second round trip nobody is at the keyboard to
#     make. Fetching it here is the difference between "notarization failed"
#     and a named cause, on a step that takes tens of minutes to reach again.
#     Retrying would only be rejected again.
#
#   UNREACHABLE -- the request timed out or the service answered 5xx
#     (`NSURLErrorDomain Code=-1001`, seen on macos-x64 on 2026-09-17 while
#     macos-arm64 notarized the same tree in the same run). Nothing is wrong
#     with the artefact, and failing the job means a tag build has to be
#     re-run by hand. This retries.
#
# When a submission id exists, a retry WAITS on that id instead of submitting
# again: the upload already happened, and Apple's quota is 75 notarizations a
# day, of which one tag spends four.
#
# A shared script rather than the same block pasted into two steps: the CLI and
# the disk image are submitted separately, and a diagnostic that exists in only
# one of them is the one that will be missing when it matters.
set -uo pipefail

: "${MACOS_APPLE_ID:?not set}" "${MACOS_TEAM_ID:?not set}" "${MACOS_NOTARY_PASSWORD:?not set}"
ART=${1:?no artefact given}
[ -s "$ART" ] || { echo "::error::$ART does not exist or is empty"; exit 1; }

CREDS=(--apple-id "$MACOS_APPLE_ID" --team-id "$MACOS_TEAM_ID" --password "$MACOS_NOTARY_PASSWORD")
ATTEMPTS=${NOTARIZE_ATTEMPTS:-3}
WAIT_TIMEOUT=${NOTARIZE_TIMEOUT:-15m}   # per attempt; the step's own budget is 60 minutes
id=""

for attempt in $(seq 1 "$ATTEMPTS"); do
  if [ -z "$id" ]; then
    [ "$attempt" -eq 1 ] || echo "--- resubmitting $(basename "$ART") (attempt $attempt/$ATTEMPTS)"
    out=$(xcrun notarytool submit "$ART" "${CREDS[@]}" --wait --timeout "$WAIT_TIMEOUT" 2>&1); rc=$?
    # The id is printed on submission, so it is in `out` even when the wait failed.
    id=$(printf '%s\n' "$out" | awk '/^ *id: /{print $2; exit}')
  else
    echo "--- waiting again on submission $id (attempt $attempt/$ATTEMPTS)"
    out=$(xcrun notarytool wait "$id" "${CREDS[@]}" --timeout "$WAIT_TIMEOUT" 2>&1); rc=$?
  fi
  printf '%s\n' "$out"
  [ "$rc" -eq 0 ] && exit 0

  # Apple looked and said no: the log is the answer, and another try is not.
  if printf '%s\n' "$out" | grep -qE '^ *status: (Invalid|Rejected)'; then
    if [ -n "$id" ]; then
      echo "--- notarytool log $id ---"
      # || true: the log itself failing must not replace the real error with its own.
      xcrun notarytool log "$id" "${CREDS[@]}" || true
    fi
    echo "::error::Apple rejected $(basename "$ART") -- the issues are listed above"
    exit 1
  fi

  # Anything else is treated as transport: a timeout, a 5xx, a dropped
  # connection, or a wait that ran out while the submission was still In
  # Progress. Back off and try again -- on the id when we have one.
  if [ "$attempt" -lt "$ATTEMPTS" ]; then
    sleep $(( attempt * 60 ))
  fi
done

echo "::error::could not get a verdict for $(basename "$ART") from Apple in $ATTEMPTS attempts -- the last output is above. This is Apple's service or the network, not the artefact; re-run this job."
exit 1
