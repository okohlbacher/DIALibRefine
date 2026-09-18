# Cutting a release

Releases are built by `.github/workflows/ci.yml` on a `v*` tag: four legs
(linux-x64, linux-arm64, macos-arm64, macos-x64) each produce a portable
tarball; the macOS legs also produce a signed, notarized, stapled `.dmg`.
A final job asserts that every expected asset is attached to the release.

## Before tagging

1. Bump `VERSION` in `CMakeLists.txt` and add the section to `CHANGELOG.md`.
   Both tools print that version (`--help`), and the standalone gate fails if
   they print OpenMS's instead.
2. **Dry-run the packaging without a tag**: Actions → *build* → *Run
   workflow* with *package_dry_run* on. Every packaging and signing step runs
   and the bundles come back as workflow artifacts; nothing touches a release.
   This is the only way the release path gets exercised before a tag depends
   on it.
3. Tag and push: `git tag -a v0.2.0 -m "0.2.0" && git push origin v0.2.0`.

## macOS signing

Signing is optional at the workflow level — without the secrets the macOS
bundles are ad-hoc signed, exactly as on a pull request — and mandatory for
a release a user can open: Gatekeeper refuses an unsigned or signed-but-
unnotarized download, and clearing quarantine does not help.

Seven repository secrets (Settings → Secrets and variables → Actions), the
same names DIALibGen and the other tools on this team use:

| secret | |
|---|---|
| `MACOS_CERTIFICATE_BASE64` | Developer ID **Application** certificate + key, `.p12`, base64 |
| `MACOS_CERTIFICATE_PASSWORD` | the `.p12` password |
| `MACOS_KEYCHAIN_PASSWORD` | any password for the throwaway keychain CI creates (generated if unset) |
| `MACOS_SIGNING_IDENTITY` | `Developer ID Application: Name (TEAMID)` — checked against the certificate |
| `MACOS_APPLE_ID` | the Apple ID for `notarytool` |
| `MACOS_TEAM_ID` | the team id |
| `MACOS_NOTARY_PASSWORD` | an app-specific password for notarization |

Set them with the premade script, on the Mac that holds the `.p12`:

```bash
.github/set-macos-secrets.sh okohlbacher/DIALibRefine
```

It reads the values from one file per secret (default
`~/Documents/Admin/Software Signing/secrets/`, or `--from-dir`, `--from-env`,
`--interactive`), checks that the certificate opens with its password and
carries the identity string, and pushes each value to `gh secret set` on
stdin — never on a command line, never in a temp file, never on the
terminal. `--dry-run` does everything but the write. Secrets are write-only
on GitHub: they cannot be copied from another repository, which is why the
script exists. The web UI works too. **No key is ever stored in this
repository or on a build runner**: the workflow imports the certificate into
an ephemeral keychain and deletes it in an `always()` step.

What the workflow does with them, in order: import the certificate; collect
the dylib closure of both executables into `lib/` and rewrite load commands
to `@executable_path/../lib`; `codesign --force --timestamp --options
runtime` every dylib and both binaries; prove the tree runs under `env -i`
and fine-tunes; zip it and submit to `notarytool` (through
`.github/notarize.sh`, which fetches the log on rejection so the *reason*
is in the job output); build a `.dmg` with `hdiutil`, sign it, notarize it,
staple it, and verify with `spctl` and `stapler validate`.

The gate is a step *output*, never `secrets.*` in an `if:` — that form makes
the whole workflow invalid and every push a zero-job failure.

Apple allows 75 notarizations a day; a tag submits four (two zips, two
images).

## What is not released

- A CUDA build (standard runners have no GPU and the bundle
  would be 500–800 MB). Build it from source with `DLR_LIBTORCH_DIR`
  ([install.md](install.md#cuda)).
- Windows. DIALibGen's `windows.yml` (OpenMS from source, SignPath) is the
  template if it becomes wanted; libtorch win-64 would be a new dependency
  there.

## Verifying a release

```bash
tar xzf DIALibRefine-linux-x64.tar.gz -C /tmp/dlr && env -i /tmp/dlr/bin/DIALibRefine --help
# macOS
spctl --assess --type open --context context:primary-signature -vv DIALibRefine-macos-arm64.dmg
xcrun stapler validate DIALibRefine-macos-arm64.dmg
codesign --verify --strict --verbose=2 /Volumes/DIALibRefine/DIALibRefine/bin/DIALibRefine
```
