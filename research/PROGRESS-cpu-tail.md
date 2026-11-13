# PROGRESS cpu-tail

Branch `research/2026-11-cpu-tail` from 1879a92. Base source worktree: `$A/src-cpu-tail-base` (1879a92).

## Step 1: go/no-go (head + blob_diff only against the full frame)

Temporary hack (uncommitted, not kept): `HACK_HEAD_ONLY` environment variable makes `RecordFusedTail` stop after the
boundary-count copy (host-read barrier and completion flag set). Builds: `$A/build-cpu-tail` (hack), `$A/build-cpu-tail-base`.
Status: building.
