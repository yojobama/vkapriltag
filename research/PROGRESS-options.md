# Progress: options registry (branch research/2026-11-options, base 545bc4d)

Refactor only: with the environment unchanged, and with any single switch set, behaviour must equal 545bc4d.

## Inventory of every variable the library reads at 545bc4d

Parse = how the value is read today (`nz` = set and first character is not `0`, an empty string counts as on;
`flag` = set, non-empty and first character not `0`; `atoi`/`strtol`/`strtoul`/`atof`; `strict int` = throws if
malformed). Precedence = what wins when the configuration field and the variable are both given.
Categories: F feature (changes detections), T tuning (same detections), D diagnostic, V device forcing.

### vk/Context.cpp (Context::Init, ContextOptions)
| Variable | Type, parse | Default | Config field | Precedence | Cat |
| --- | --- | --- | --- | --- | --- |
| APRILTAG_VK_ALLOW_CPU | bool, flag | off | allow_cpu_device | env can only turn on (`=0` does not turn a true config off) | V |
| APRILTAG_VK_VALIDATION | bool, flag | off | enable_validation | env can only turn on | D |
| APRILTAG_VK_DEVICE | int, strict | -1 (score) | device_index | env wins | V |
| APRILTAG_VK_WG | int, strict, >0 | 0 (device) | workgroup_size_override | env wins | T |
| APRILTAG_VK_WG2D | WxH | 0x0 (device) | workgroup_size_2d_x/y | env wins when well formed | T |
| APRILTAG_VK_FORCE_NO_8BIT / _SUBGROUP / _INT64_ATOMIC / _CONDITIONAL_RENDERING | bool, flag | off | force_no_* | env can only turn on | V |
| APRILTAG_VK_MAX_INVOCATIONS | int, strict, >0 | 0 (device) | max_invocations_override | env wins | V |
| APRILTAG_VK_PIPELINE_CACHE | bool, set and not flag turns off | on | use_pipeline_cache | env can only turn off | T |
| APRILTAG_VK_SPIN_WAIT | bool, flag, process static in `WaitFence` | off | none | env only | T |

### vk/PipelineCache.cpp, vk/ComputePipeline.cpp, common/Profiler.cpp
| Variable | Type, parse | Default | Config field | Precedence | Cat |
| --- | --- | --- | --- | --- | --- |
| APRILTAG_VK_CACHE_DIR | string, non-empty | platform dir | none | env only | T |
| LOCALAPPDATA / XDG_CACHE_HOME / HOME | string (platform, not APRILTAG) | unset | none | fallbacks for the cache dir | - |
| APRILTAG_VK_DISPATCH_TIMES | presence, process static | off | none | env only | D |
| APRILTAG_VK_TRACE | string path, non-empty (`EnableFromEnv`) | off | none | env only | D |

### common/WorkerPool.{h,cpp}
| Variable | Type, parse | Default | Config field | Precedence | Cat |
| --- | --- | --- | --- | --- | --- |
| APRILTAG_CPU_THREADS | int, strtol >0, clamp 65536 | 0 (hardware) | cpu_threads | CONFIG wins when non-zero (`ResolveThreadCount`) | T |
| APRILTAG_CPU_BIG_CORES | bool, nz | on (config) | cpu_big_cores | env wins | T |

### gpu/GpuDetector.cpp (constructor, CreateBuffers, CreatePipelines)
| Variable | Type, parse | Default | Config field | Precedence | Cat |
| --- | --- | --- | --- | --- | --- |
| APRILTAG_VK_MIN_DIFF | int, atoi, floor 0 | 5 | min_white_black_diff | env wins | F |
| APRILTAG_VK_MAX_POINTS | int, strtol >0 | 0 (dense) | max_boundary_points | env wins | T |
| APRILTAG_VK_UF_CHUNK | int, strtol >0 | 2 | uf_iterations_per_chunk | env wins | T |
| APRILTAG_VK_MIN_TAG_PX | int, strtol >0 | 0 | min_tag_pixels | env wins | F |
| APRILTAG_VK_SMALL_RESCUE | bool, nz | off | small_tag_rescue | env wins (also read in QuadDecode and TagDecoder) | F |
| APRILTAG_VK_SORT_WG | int, strtoul >=1 | device max, 256 integrated / 1024 | none | env only | T |
| APRILTAG_VK_SORT_SPLIT | int, strtoul >=1 | 1024 (bounded by shared memory) | none | env only | T |
| APRILTAG_VK_SORT_RANK_ALL | int, strtoul, &~3 | 256 | none | env only | T |
| APRILTAG_VK_FLAG_POLL | bool, nz | on if device-coherent host memory | none | env only | T |
| APRILTAG_VK_KEEPALIVE_MS | int, atoi | -1 (6 integrated, 0 other) | keepalive_ms | env wins | T |
| APRILTAG_VK_PREWARM_LEAD_US | int, atoi | -1 (3000) | prewarm_lead_us | env wins | T |
| APRILTAG_VK_PREWARM_BURST | int, atoi, >=1 | 2 | prewarm_burst | env wins | T |
| APRILTAG_VK_PREWARM_GAP_US | int, atoi, >=0 | 1000 | prewarm_gap_us | env wins | T |
| APRILTAG_VK_PREWARM_GROUPS | int, atoi, >=1 | KeepAlive default | none | env only | T |
| APRILTAG_VK_TIMESTAMPS | int, strtol !=0 (or tracing) | off | none | env only | D |
| APRILTAG_VK_FUSE_SUBMITS | bool, nz and host readbacks | on when readbacks direct | none | env only (can only turn off) | T |
| APRILTAG_VK_UF_VERIFY | bool, nz | off | none | env only | T |
| APRILTAG_VK_PRERECORD | bool, nz and fused, no transfer lane | on when fused | none | env only (can only turn off) | T |
| APRILTAG_VK_BOX_DECIMATE | bool, nz | on | box_decimate | env wins | F |
| APRILTAG_VK_DOT | bool, only `0` read (needs dot-product shader and device support) | on | none | env only (can only turn off) | T |
| APRILTAG_VK_FUSED_THRESHOLD | bool, only `0` read | on with 8-bit storage | none | env only (can only turn off) | T |
| APRILTAG_VK_ADAPTIVE_FLOOR | float, atof | 2.0 | adaptive_floor_scale | env wins | F |
| APRILTAG_VK_UF_INIT_SCAN | bool, nz and device support | on with subgroup arithmetic | none | env only (can only turn off) | T |
| APRILTAG_VK_COUNT_RUNS | bool, nz and uf_init_scan | integrated GPUs | none | env only (device default overridden) | T |
| APRILTAG_VK_SPARSE | bool, nz and fused, scan, count_runs | on (config) | sparse_labelling | env wins | T |
| APRILTAG_VK_MERGE_X4 | bool, nz and u8, width % 4 | integrated GPUs | none | env only | T |
| APRILTAG_VK_MERGE_BAND | int, strtoul, pow2 >=2 else 0 | 8 discrete, 0 integrated or sparse | none | env only | T |
| APRILTAG_VK_LEAN_TAIL | int, strtoul bit mask | 3 integrated, 0 other | none | env only | T |

### gpu/QuadDecode.cpp (constructor)
| Variable | Type, parse | Default | Config field | Precedence | Cat |
| --- | --- | --- | --- | --- | --- |
| APRILTAG_VK_QUADFIT | enum dp / peaks, other values ignored | dp | quad_fit_method | env wins (comment said config wins: wrong) | F |
| APRILTAG_VK_SMALL_RESCUE | as above | off | small_tag_rescue | env wins | F |
| APRILTAG_VK_HULL_QUAD | bool, nz | off | hull_quad | env wins | F |

### TagDecoder.cpp (constructor; no config object)
| Variable | Type, parse | Default | Config field | Precedence | Cat |
| --- | --- | --- | --- | --- | --- |
| APRILTAG_VK_GREY_MARGIN | bool, nz | on | none | env only | F |
| APRILTAG_VK_SMALL_RESCUE | bool, nz | off | none (bug: config field does not reach it) | env only | F |
| APRILTAG_VK_POLISH | bool, nz | off | `SetPolish()` after construction | setter wins (called later) | F |
| APRILTAG_VK_SIDE_ONE_OUT, _THIRD_BIT, _FAMILY_LIMITS, _BORDER_PRECHECK, _DARK_REJECT, _EARLY_REJECT | bool, nz | on | none | env only | F |
| APRILTAG_VK_PROBE_STATS | bool, nz | off | none | env only | D |
| APRILTAG_VK_PROBE_MARGIN | float, atof | 0 (ctor argument probe_margin) | ctor argument | env wins | F |
| APRILTAG_VK_PROBE_CSV | string path (also turns on stats) | off | none | env only | D |
| APRILTAG_VK_REFINE (RefineEdges.cpp, `ResolveRefineEdgesMethod`) | enum exact / profile, other values ignored | profile | ctor argument | env wins (header comment said argument wins: wrong) | F |

### TrackingDetector.cpp (`ApplyEnv`, TrackingConfig)
| Variable | Type, parse | Default | Config field | Precedence | Cat |
| --- | --- | --- | --- | --- | --- |
| APRILTAG_VK_TRACKING | bool, nz | off | enabled | env wins | F |
| APRILTAG_VK_TRACKING_KEYFRAME | int, atoi, >=1 | 8 | keyframe_interval | env wins | F |
| APRILTAG_VK_TRACKING_MIN_MARGIN | float, atof | 10 | min_margin | env wins | F |
| APRILTAG_VK_TRACKING_CONVERGE | float, atof | 0.05 | converge_px | env wins | F |
| APRILTAG_VK_TRACKING_ITERATIONS | int, atoi, >=1 | 3 | max_iterations | env wins | F |
| APRILTAG_VK_TRACKING_ASYNC | bool, nz | off | async | env wins | F |
| APRILTAG_VK_TRACKING_ASYNC_INTERVAL | int, atoi, >=1 | 1 | async_interval | env wins | F |
| APRILTAG_VK_TRACKING_ASYNC_THREADS | int, atoi, >=0 | 1 | async_threads | env wins | T |
| APRILTAG_VK_TRACKING_ASYNC_COPY | bool, nz | on | async_copy | env wins | F |
| APRILTAG_VK_TRACKING_ASYNC_DROP | bool, nz | (config) | async_drop_lost | env wins | F |

### PolishFit.cpp (process-wide statics)
| Variable | Type, parse | Default | Config field | Precedence | Cat |
| --- | --- | --- | --- | --- | --- |
| APRILTAG_VK_POLISH_DEV | uint mask, strtoul, static at first use | 0 | none | env only | D |
| APRILTAG_VK_POLISH_N | int, atoi, static at first use | 8000 | none | env only | T |
| APRILTAG_VK_POLISH_BLUREXTRA / _WINMULT / _MINWIN / _SMAXM / _TOL / _ITERS / _REACH | float, atof, static `kKnobs` (read before main) | 2.0 / 0.8 / 0.5 / 1.3 / 0.004 / 6 / 1.6 | none | env only | T |

### RefineEdges.cpp (process-wide static)
| Variable | Type, parse | Default | Config field | Precedence | Cat |
| --- | --- | --- | --- | --- | --- |
| APRILTAG_VK_PROFILE_FEATURES | uint mask, strtoul, static at first use | 31 | none | env only | F |

### Known inconsistencies (resolution)
1. Env beats config for almost everything; `cpu_threads` is the reverse. Kept (changing it alters behaviour for a caller
   that sets both); made explicit in the registry (precedence column) and flagged in the report.
2. Small-tag rescue was read in GpuDetector, QuadDecode and TagDecoder. One resolver now serves all three, and
   TagDecoder takes the configured value through a new trailing constructor argument.
3. Wrong comments: `QuadDecode.cpp` ("explicit config wins") and `RefineEdges.h` ("explicit argument wins"): both fixed.
4. Boolean "can only turn on / off" variables (device forcing, `PIPELINE_CACHE`, `FUSE_SUBMITS`, ...) are not "env
   overrides config": kept as they are and labelled in the table.
5. Statics: `ProfileFeatures()`, `DevFlags()`, `MaxPixels()`, `kKnobs`, `WaitFence` spin, `DispatchProbe::Enabled()`
   keep read-once-per-process semantics, through the registry.

## What was built
- `library/include/vkapriltag/common/Options.h`, `library/src/common/Options.cpp`: `OptionTable()` (82 rows: 79 APRILTAG
  variables plus the three platform variables the cache directory reads), each with type, default text, category,
  precedence, configuration field and a one-line doc; `Options::FromEnvironment()` copies every registered variable once;
  accessors keep the parsing each reader had (`Bool` = first character not `0`, `Flag` = strict, `Int` = atoi,
  `Long`/`ULong`, `Float`, `StrictInt` throws as before, `WxH`, `Choice`, `String`, `Has`). An unregistered name throws.
- Resolution points: `Context::Init`, `GpuDetector` (member `env_`, one snapshot per detector), `QuadDecode` and
  `TagDecoder` constructors, `TrackingDetector::ApplyEnv`, `ResolveThreadCount`, `ResolveWorkerPoolOptions`,
  `ResolveRefineEdgesMethod`, `CacheDirectory`, `EnableFromEnv`. Process-wide statics (`ProfileFeatures`, `DevFlags`,
  `MaxPixels`, `kKnobs`, `WaitFence` spin, `DispatchProbe::Enabled`) take one snapshot at first use, as before.
- `ResolveThreadCount` moved from an inline in `WorkerPool.h` to `WorkerPool.cpp` (public header no longer uses getenv).
- `TagDecoder` gained a trailing constructor argument `small_tag_rescue` (the configured value); `TrackingDetector` passes
  `config_.small_tag_rescue` to its three decoders. Library callers that never set the field behave as before.
- `vk_eval --list-options` (markdown) and `--list-option-names`; `PERFORMANCE.md` section 2 is the generated output between
  `<!-- options:begin -->` markers.
- ctest: `unit_options` (table checks, parsing, the `ResolveSmallTagRescue` and `cpu_threads` precedence) and
  `options_lint` (`tests/options_lint.py`: no getenv of APRILTAG_ outside Options.cpp, no getenv at all in the library, every
  APRILTAG_VK_/CPU_ name in the sources is registered, PERFORMANCE.md equals `--list-options`).

## Commands
Builds (wrappers in `$S/options/`, S = scratchpad): `cfg-cur.bat`, `build-cur.bat`, `ctest-cur.bat` for this branch
(`ext/avk-compare/build-options`) and `cfg-base.bat`, `build-base.bat` for the base, which is `git archive 545bc4d`
extracted to `ext/avk-compare/src-options-base` and built in `ext/avk-compare/build-options-base`.
Dump comparison: `bash $S/options/gate.sh LABEL CORPUS DEC [VAR=val ...]` runs base, current and base again and `cmp`s the
dump files; `bash $S/options/battery.sh` runs the single-switch list over the committed subset.

## Steps
