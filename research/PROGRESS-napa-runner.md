# napa-runner

Native C runner for NAPA 0.5.1 (`tools/napa_corpus/napa_eval/`), same dumps and timings.json as `napa_run.py`.

Build: `cmake -S apriltags_vulkan/tools/napa_corpus/napa_eval -B <build> -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release -DNAPA_DIR=<NAPA checkout>`
Run: `napa_eval ROOT OUT [--min-tag-px N] [--reps N] [--gap-ms MS] [--only S] [--list-name NAME]`
Summary: `python tools/napa_corpus/napa_summary.py OUT/timings.json ROOT OUT`

Desktop, 5 reps, min_tag_px 32: 7152 of 7348 (97.3%), 0 false, error 0.021 / 0.100 px; frame median 0.094, p95 0.391 ms.
min_tag_px 16: 97.7%, 0.021 / 0.102 px; median 0.273, p95 1.329 ms (4203 frames).
Pi smoke test (one-tag, 1 rep) passes; whole corpus not timed there.
