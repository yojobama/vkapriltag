# Release workflow: pre-release tags

Branch `release/2.0.0-pre1-workflow`, from main 3f5ccc1. Only `.github/workflows/release.yml` changed.

## Changes

- Tag check accepts `vMAJOR.MINOR.PATCH` and `vMAJOR.MINOR.PATCH-preN`; anything else fails with an `::error::`.
- The step exports `PKG_VERSION` (tag without `v`), `DEB_VERSION` (`2.0.0~pre1` for a pre-release, else the same as `PKG_VERSION`) and `DEB_FILE_NAME` (`vkapriltag_<version>_<arch>.deb` for a pre-release, else `DEB-DEFAULT`).
- Package step runs `cpack -D CPACK_DEBIAN_PACKAGE_VERSION=... -D CPACK_DEBIAN_FILE_NAME=...`. The CMake code is unchanged:
  `apriltags_vulkan/CMakeLists.txt` sets `CPACK_DEBIAN_FILE_NAME` to `DEB-DEFAULT` unconditionally, so it is overridden at cpack time.
  Configure still gets `CPACK_PACKAGE_VERSION=2.0.0-pre1`; the Debian Version field comes from `CPACK_DEBIAN_PACKAGE_VERSION`.
- Release job: checks out the repository, extracts the CHANGELOG section into `release-notes.md`, then
  `prerelease` and `make_latest` follow the tag (`false` for a pre-release, `legacy`, the action default, for a final), name `vkapriltag <version>`,
  `body_path` when the section exists, else `generate_release_notes: true`.

## Checks (local, Git Bash; snippets extracted from the YAML)

| Tag | Result | PKG_VERSION | DEB_VERSION | DEB_FILE_NAME (arm64) | prerelease |
|---|---|---|---|---|---|
| v1.10.1 | accepted | 1.10.1 | 1.10.1 | DEB-DEFAULT (vkapriltag_1.10.1_arm64.deb) | false |
| v2.0.0-pre1 | accepted | 2.0.0-pre1 | 2.0.0~pre1 | vkapriltag_2.0.0-pre1_arm64.deb | true |
| v2.0.0-pre12 | accepted | 2.0.0-pre12 | 2.0.0~pre12 | vkapriltag_2.0.0-pre12_arm64.deb | true |
| 2.0.0-pre1 | rejected (error) | | | | |
| v2.0.0-rc1 | rejected (error) | | | | |
| v2.0 | rejected (error) | | | | |

- CHANGELOG extraction against the real file: `2.0.0-pre1` gives 165 lines, first line is the `- **Quad fit in float32**` bullet, last bullet is
  "The keep-alive works on Windows", nothing from `## 1.10.1`. `1.10.1` gives 12 lines ("Release packaging only; ..." to the `checks that the package contains` line).
  `9.9.9` gives an empty file and `found=false` (fallback to generated notes).
- CPack, on a toy project with the same CPack variables (the real library needs Vulkan and glslang, not available here): `cpack -D` with the
  pre-release values produced `vkapriltag_2.0.0-pre1_arm64.deb` with control `Version: 2.0.0~pre1`; with `DEB-DEFAULT` and `2.0.0` it produced `vkapriltag_2.0.0_arm64.deb`,
  the same name as before. A default `DEB-DEFAULT` name with a `~` version would have been `vkapriltag_2.0.0~pre1_<arch>.deb`, hence the explicit name.
- YAML parses with PyYAML. `actionlint` is not installed.

## Not verified

- No GitHub run is possible before the tag. `make_latest: legacy` for final tags and the empty `body_path` when no section is found rely on the action's documented input handling.
