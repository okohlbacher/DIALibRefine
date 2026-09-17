#!/usr/bin/env bash
# Put the seven MACOS_* code-signing secrets on one or more GitHub repositories.
#
#   set-macos-secrets.sh [SOURCE] [--dry-run] [--no-check-cert] OWNER/REPO [OWNER/REPO ...]
#
# GitHub Actions secrets are WRITE-ONLY: there is no reading them back from a
# repository that already has them (DIALibGen, BALL, FASTag) to copy them over.
# They are re-set from the local source of truth -- the Developer ID .p12
# export and the passwords kept beside it on the signing Mac -- and this script
# is the one place that knows the seven names the workflows expect:
#
#   MACOS_CERTIFICATE_BASE64     Developer ID Application cert + key, .p12, base64
#   MACOS_CERTIFICATE_PASSWORD   password of that .p12
#   MACOS_KEYCHAIN_PASSWORD      any password for the throwaway CI keychain
#   MACOS_SIGNING_IDENTITY       "Developer ID Application: Name (TEAMID)"
#   MACOS_APPLE_ID               Apple ID used for notarytool
#   MACOS_TEAM_ID                the team id
#   MACOS_NOTARY_PASSWORD        app-specific password for notarization
#
# SOURCE, one of:
#   --from-dir DIR    one file per secret, named exactly like it, mode 600
#                     (default: "$HOME/Documents/Admin/Software Signing/secrets").
#                     The certificate may instead be present as the one *.p12
#                     in DIR, base64-encoded on the fly. Trailing newlines are
#                     stripped (editors add them; a password that ends in one
#                     fails inside CI with a message that names nothing); a
#                     UTF-8 BOM is stripped; CR line endings are refused.
#   --from-env        the seven variables are exported in the calling shell
#                     (e.g. `set -a; . ~/secrets.env; set +a` on a chmod-600
#                     file). MACOS_CERTIFICATE_P12 may hold the .p12 path
#                     instead of MACOS_CERTIFICATE_BASE64 holding the content.
#   --interactive     prompt for each value with echo off; the certificate is
#                     asked for as the path to the .p12.
#
# What never happens here: a value on a command line (argv is visible in `ps`
# to every user; --body is never used), a value in a temp file (here-strings
# would create one on bash 3.2), a value echoed, xtrace (switched off first
# thing, even if inherited), a value in the shell history.
#
# Before writing anything it checks that every repo can be administered by
# the token, that the base64 decodes to a DER blob, that the identity reads
# "Developer ID Application: ... (TEAMID)", and -- with openssl, the .p12
# password passed through the environment, never argv -- that the .p12 opens
# and its certificate carries the identity string. --dry-run stops there.
# Afterwards `gh secret list -R OWNER/REPO` shows the seven names + timestamps.
#
# Needs: gh (logged in with repo scope: `gh auth status`), base64, openssl
# (macOS's /usr/bin/openssl is preferred: Keychain Access exports use RC2-40,
# which OpenSSL 3 refuses without -legacy), bash >= 3.2. Run it on the Mac
# that holds the .p12.
set +x; unset BASH_XTRACEFD
case "$-" in *x*) echo "error: refusing to run with xtrace on" >&2; exit 1 ;; esac
set -euo pipefail
umask 077

NAMES=(MACOS_CERTIFICATE_BASE64 MACOS_CERTIFICATE_PASSWORD MACOS_KEYCHAIN_PASSWORD \
       MACOS_SIGNING_IDENTITY MACOS_APPLE_ID MACOS_TEAM_ID MACOS_NOTARY_PASSWORD)
DEFAULT_DIR="$HOME/Documents/Admin/Software Signing/secrets"

usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"; exit "${1:-2}"; }
die()   { echo "error: $*" >&2; exit 1; }
warn()  { echo "warning: $*" >&2; }

source_mode=""; dir="$DEFAULT_DIR"; dry_run=0; check_cert=1; repos=()
while [ $# -gt 0 ]; do
  case "$1" in
    --from-dir)      source_mode=dir; dir="${2:?--from-dir needs a directory}"; shift 2 ;;
    --from-env)      source_mode=env; shift ;;
    --interactive)   source_mode=interactive; shift ;;
    --dry-run)       dry_run=1; shift ;;
    --no-check-cert) check_cert=0; shift ;;
    -h|--help)       usage 0 ;;
    -*)              die "unknown option $1" ;;
    *)               repos+=("$1"); shift ;;
  esac
done
[ "${#repos[@]}" -gt 0 ] || usage
for r in "${repos[@]}"; do
  case "$r" in */*) ;; *) die "'$r' is not OWNER/REPO" ;; esac
done
if [ -z "$source_mode" ]; then
  source_mode=dir
  [ -d "$dir" ] || die "no secrets directory at '$dir' (use --from-dir, --from-env or --interactive)"
fi
command -v gh >/dev/null 2>&1 || die "gh is not installed"
gh auth status >/dev/null 2>&1 || die "gh is not logged in (gh auth login)"

# ---- pre-flight the repositories: a half-written set is the worst outcome ---
for r in "${repos[@]}"; do
  [ "$(gh api "repos/$r" --jq .permissions.admin 2>/dev/null || true)" = true ] \
    || die "cannot administer $r with this token (does it exist? repo scope?)"
done

# ---- helpers that keep the values inside this shell -------------------------
# Files must be private: a 604 password file or a 644 .p12 (Keychain Access's
# default export mode, and it holds the private key) is refused.
require_private() {
  local f="$1" perms
  perms=$(stat -f '%Lp' "$f" 2>/dev/null || stat -c '%a' "$f")
  case "$perms" in *00) ;; *) die "$f is readable by others (mode $perms); chmod 600 it" ;; esac
}
# Editor artefacts: a UTF-8 BOM is dropped, CR line endings are an error,
# trailing newlines are dropped, and an empty result is an error.
clean_text() { # NAME  (edits the variable in place)
  local n="$1" v="${!1}"
  v="${v#$'\xef\xbb\xbf'}"
  case "$v" in *$'\r'*) die "$n contains a CR (Windows/TextEdit line endings) -- convert the file" ;; esac
  while [ "${v%$'\n'}" != "$v" ]; do v="${v%$'\n'}"; done
  [ -n "$v" ] || die "$n is empty"
  printf -v "$n" '%s' "$v"
}
p12_path=""
encode_p12() { # PATH -> MACOS_CERTIFICATE_BASE64
  [ -f "$1" ] || die "no file at $1"
  require_private "$1"
  MACOS_CERTIFICATE_BASE64=$(base64 < "$1")
}

case "$source_mode" in
  dir)
    for n in "${NAMES[@]}"; do
      f="$dir/$n"
      if [ "$n" = MACOS_CERTIFICATE_BASE64 ] && [ ! -f "$f" ]; then
        set -- "$dir"/*.p12
        [ -f "$1" ] || die "neither $f nor a .p12 in $dir"
        [ $# -eq 1 ] || die "more than one .p12 in $dir -- which one?"
        p12_path="$1"; encode_p12 "$p12_path"
        continue
      fi
      [ -f "$f" ] || die "missing $f"
      require_private "$f"
      printf -v "$n" '%s' "$(cat "$f")"
      clean_text "$n"
    done
    ;;
  env)
    if [ -z "${MACOS_CERTIFICATE_BASE64:-}" ] && [ -n "${MACOS_CERTIFICATE_P12:-}" ]; then
      p12_path="$MACOS_CERTIFICATE_P12"; encode_p12 "$p12_path"
    fi
    for n in "${NAMES[@]}"; do
      [ -n "${!n:-}" ] || die "$n is not set in the environment"
      clean_text "$n"
      export -n "$n"     # not inherited by gh, base64, openssl, stat, ... from here on
    done
    export -n MACOS_CERTIFICATE_P12 2>/dev/null || true
    ;;
  interactive)
    [ -t 0 ] || die "--interactive needs a terminal"
    # -e, no -r: a path dragged from Finder arrives with backslash-escaped spaces
    read -e -p "path to the Developer ID Application .p12: " p12_path
    encode_p12 "$p12_path"
    for n in "${NAMES[@]}"; do
      [ "$n" = MACOS_CERTIFICATE_BASE64 ] && continue
      case "$n" in
        MACOS_SIGNING_IDENTITY|MACOS_APPLE_ID|MACOS_TEAM_ID) read -r -p "$n: " v ;;
        *) IFS= read -r -s -p "$n (no echo): " v; echo ;;   # IFS=: a password may begin or end with a space
      esac
      printf -v "$n" '%s' "$v"
      clean_text "$n"
    done
    ;;
esac

# ---- validate before touching GitHub ---------------------------------------
# One line of base64 is what every decoder accepts (a file made with GNU
# base64 is wrapped at 76 columns; those newlines would go into the secret).
MACOS_CERTIFICATE_BASE64=$(printf '%s' "$MACOS_CERTIFICATE_BASE64" | tr -d '\n\r ')
first=$(printf '%s' "$MACOS_CERTIFICATE_BASE64" | base64 --decode 2>/dev/null | head -c 1 | od -An -tx1 | tr -d ' \n') || true
[ "$first" = 30 ] || die "MACOS_CERTIFICATE_BASE64 does not decode to a DER PKCS#12 blob (a .cer/.pem renamed .p12? a pasted text with a stray character?)"
case "$MACOS_SIGNING_IDENTITY" in
  "Developer ID Application: "*) ;;
  *) die "MACOS_SIGNING_IDENTITY must read 'Developer ID Application: Name (TEAMID)'" ;;
esac
case "$MACOS_SIGNING_IDENTITY" in *"($MACOS_TEAM_ID)"*) ;; *) die "MACOS_SIGNING_IDENTITY does not end in (MACOS_TEAM_ID = $MACOS_TEAM_ID)" ;; esac
case "$MACOS_APPLE_ID" in *@*) ;; *) die "MACOS_APPLE_ID does not look like an Apple ID" ;; esac

# ---- the identity string must be the certificate's own -------------------
# CI checks this too, but there it costs a ten-minute macOS job to find out.
# The .p12 password reaches openssl through the environment of that one
# process (a prefix assignment, no export), never argv. Keychain Access
# exports use RC2-40 for the certificate bag, which OpenSSL 3 refuses without
# -legacy and LibreSSL (/usr/bin/openssl on macOS) opens as is -- so that one
# is preferred, and OpenSSL 3 gets a second try with -legacy. Its stderr never
# contains the password and is the diagnosis when it fails.
if [ "$check_cert" -eq 1 ]; then
  OPENSSL=""
  for c in /usr/bin/openssl "$(command -v openssl 2>/dev/null || true)"; do [ -n "$c" ] && [ -x "$c" ] && { OPENSSL="$c"; break; }; done
  if [ -z "$OPENSSL" ]; then
    warn "openssl not found: the certificate is NOT verified against the identity string"
  else
    cert_subject() { # extra openssl pkcs12 flags in "$@"; DER on stdin
      _P12_PW="$MACOS_CERTIFICATE_PASSWORD" "$OPENSSL" pkcs12 -nokeys -clcerts -passin env:_P12_PW "$@" \
        | "$OPENSSL" x509 -noout -subject -nameopt oneline,-esc_msb
    }
    subject=$(printf '%s' "$MACOS_CERTIFICATE_BASE64" | base64 --decode | cert_subject 2>/dev/null || true)
    if [ -z "$subject" ]; then
      subject=$(printf '%s' "$MACOS_CERTIFICATE_BASE64" | base64 --decode | cert_subject -legacy 2>/dev/null || true)
      [ -n "$subject" ] && warn "the .p12 needed 'openssl pkcs12 -legacy' (RC2-40 export); CI's security import accepts it"
    fi
    if [ -z "$subject" ]; then
      err=$(printf '%s' "$MACOS_CERTIFICATE_BASE64" | base64 --decode | { _P12_PW="$MACOS_CERTIFICATE_PASSWORD" "$OPENSSL" pkcs12 -nokeys -clcerts -passin env:_P12_PW 2>&1 >/dev/null || true; } | tail -3)
      die "the .p12 cannot be opened ($OPENSSL): wrong MACOS_CERTIFICATE_PASSWORD, or an algorithm this openssl lacks. openssl says:
$err
(with --no-check-cert the check is skipped; CI verifies again on import)"
    fi
    case "$subject" in
      *"$MACOS_SIGNING_IDENTITY"*) ;;
      *) die "the certificate's subject does not carry MACOS_SIGNING_IDENTITY verbatim; certificate says: ${subject#*subject=}" ;;
    esac
    echo "certificate opens and matches the signing identity ($OPENSSL)"
  fi
fi

# ---- set them ---------------------------------------------------------------
done_repos=()
for r in "${repos[@]}"; do
  echo "== $r"
  for n in "${NAMES[@]}"; do
    if [ "$dry_run" -eq 1 ]; then echo "   would set $n (value withheld)"; continue; fi
    # printf is a builtin: the value is not an argv of any process. gh reads
    # the value from stdin when --body is absent.
    printf '%s' "${!n}" | gh secret set "$n" -R "$r" >/dev/null \
      || die "gh secret set $n failed for $r. Complete so far: ${done_repos[*]:-none}. Rerun with the remaining repositories."
    echo "   set $n"
  done
  [ "$dry_run" -eq 1 ] && continue
  # names and timestamps only -- values cannot be read back, by design
  listing=$(gh secret list -R "$r")
  for n in "${NAMES[@]}"; do
    printf '%s\n' "$listing" | grep -q "^$n[[:space:]]" || die "$n is not listed on $r after setting it"
  done
  printf '%s\n' "$listing" | grep -E "^MACOS_" | sed 's/^/   /'
  done_repos+=("$r")
done
if [ "$dry_run" -eq 1 ]; then echo "dry run: nothing written."; else echo "done. Dry-run the packaging before tagging: Actions -> build -> Run workflow -> package_dry_run."; fi
