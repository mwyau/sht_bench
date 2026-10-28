# NumPy FFT backend comparison

`numpy_fft_bench.py` measures the public `numpy.fft` API against a selected
NumPy build. It preserves the original 26 workloads, input generation, and
timing path used for the PocketFFT-versus-DUCC comparison. Default allocation
and reusable `out=` calls are separate rows.

## Provenance

The original executable was recovered from `/tmp/numpy_fft_bench.py`, SHA-256
`a40b067cee75c37a6d512fb0f789f5fe1d41dd23f0befb3bca3437d31760d4a1`. Its
mtime and matching Python bytecode predate the three mainline matrices. The
saved baseline report at `/tmp/numpy_fft_baseline_report.md` invokes that path.
The retained session log
`/home/albert/.codex2/sessions/2026/10/27/rollout-2026-10-27T16-25-11-01a17f09-cdad-72d2-b0b9-421c8b256275.jsonl`
records the three mainline commands, all running `/tmp/numpy_fft_bench.py`
with `--mode both --repeat 7` and writing the named CSV/JSON outputs. Their
JSON metadata identifies the build and CPU profile, and all three CSVs have
52 identical `(case, mode)` keys. The checked-in script is a direct copy with
only a new module docstring and
optional `--case` / `--list-cases` CLI options; the workload definitions,
global RNG seed (`824203`), array generation, warmups, `timeit` measurement,
summaries, and CSV fields remain unchanged. Case filtering still constructs
all inputs in their original order before skipping unselected timings, so it
does not shift the RNG stream for selected cases.

The measurements were created directly by this script. The separate
`/tmp/numpy_fft_mainline_compare.py` consumes the three CSV/JSON pairs and
produces `/tmp/numpy_fft_mainline_comparison.md` and
`/tmp/numpy_fft_mainline_comparison.csv`; it is aggregation, not measurement.
Earlier `/tmp/compare_numpy_fft.py`, `/tmp/compare_numpy_fft_cache.py`,
`/tmp/numpy_fft_safe_cache_compare.py`, and
`/tmp/numpy_fft_final_comparison*.py` are also comparison/report scripts. The
named mainline outputs are CSV and Markdown; no plotting script or plot
artifact was found for them. No extra measurement helper is required beyond
Python and NumPy.

Each original matrix has 52 rows, with identical `(case, mode)` keys across
PocketFFT, DUCC X86_V3, and DUCC X86_V2. The CSV header is:

```text
case,function,input_dtype,output_dtype,input_shape,output_shape,layout,mode,iterations_per_sample,repeat,median_us,p10_us,p90_us,min_us,max_us,points_per_call,points_per_second
```

## Measurement method

Input creation and one setup transform happen before timing. Each mode gets
three warmups. `timeit.Timer.autorange()` uses its 0.2 second default target;
the iteration count is capped at 65,536. The harness records seven repeated
per-call samples by default, then emits median, p10/p90, minimum, maximum, and
points per second. `--min-time` is retained as output metadata; changing it
does not alter `timeit`'s autorange target. The `out` mode reuses one output
array for all timed calls.

The script records Python, NumPy, OS, machine, CPU model, NumPy CPU dispatch
and feature flags, thread environment variables, NumPy configuration, and—if
`NUMPY_BUILD_DIR` is set—the FFT compile command from `compile_commands.json`.
CSV and JSON paths and repeat count are configurable. `taskset` and environment
variables control affinity and thread limits, as in the original runs.

## Build environment used for the original comparison

The three NumPy builds use commit
`b833c1b9bf6712cc8a03bb301b77c48984a2373d`, Python 3.12.13, GCC 15.2, and
OpenBLAS 0.3.32. The machine was an AMD Ryzen 9 5950X. PocketFFT was an
unmodified Release build with the default X86_V2 baseline and configured
dispatch targets. DUCC used the Release FFT backend port at NumPy commit
`90d90a976aa92a9f40d81c3aa8f7a90999eb5abb` (parent `b833c1b9`), with the
worker pool disabled and its plan-cache mutex enabled. Its X86_V3 variant was
forced for V3, and X86_V3/X86_V4 were disabled at runtime for V2. AVX-512 was
unavailable on the host.

From the respective NumPy worktree roots, the validated build commands were:

```bash
PATH=/tmp/numpy-pocketfft-venv/bin:$PATH \
  /tmp/numpy-pocketfft-venv/bin/spin build -j 12 \
  -C build-mainline-baseline -- -Dbuildtype=release

PATH=/tmp/numpy-pocketfft-venv/bin:$PATH \
  /tmp/numpy-pocketfft-venv/bin/spin build -j 12 \
  -C build-mainline -- -Dbuildtype=release
```

The baseline install was at
`/tmp/numpy-pocketfft-origin-main/build-mainline-baseline-install`; the DUCC
install was at `/tmp/numpy-ducc-origin-main/build-mainline-install`.

## Reproduce all three profiles

Run from the `sht_bench` root after building the two isolated NumPy trees above.
Use fresh output paths: when resuming an existing CSV, the harness skips rows
whose `(case, mode)` pair is already present.

```bash
PY=/tmp/numpy-pocketfft-venv/bin/python
SCRIPT="$PWD/benchmarks/numpy_fft/numpy_fft_bench.py"
BASE=/tmp/numpy-pocketfft-origin-main
DUCC=/tmp/numpy-ducc-origin-main

# Unmodified PocketFFT baseline, native NumPy dispatch.
taskset -c 2 env \
  -u NPY_ENABLE_CPU_FEATURES -u NPY_DISABLE_CPU_FEATURES \
  PYTHONPATH="$BASE/build-mainline-baseline-install/usr/lib/python3.12/site-packages" \
  NUMPY_BUILD_DIR="$BASE/build-mainline-baseline" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DUCC0_NUM_THREADS=1 \
  "$PY" "$SCRIPT" --mode both --repeat 7 \
  --csv /tmp/numpy_fft_mainline_baseline-repro.csv \
  --json /tmp/numpy_fft_mainline_baseline-repro.json

# DUCC X86_V3.
taskset -c 2 env \
  -u NPY_DISABLE_CPU_FEATURES NPY_ENABLE_CPU_FEATURES=X86_V3 \
  PYTHONPATH="$DUCC/build-mainline-install/usr/lib/python3.12/site-packages" \
  NUMPY_BUILD_DIR="$DUCC/build-mainline" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DUCC0_NUM_THREADS=1 \
  "$PY" "$SCRIPT" --mode both --repeat 7 \
  --csv /tmp/numpy_fft_mainline_ducc_v3-repro.csv \
  --json /tmp/numpy_fft_mainline_ducc_v3-repro.json

# DUCC X86_V2 fallback: suppress higher dispatch targets at runtime.
taskset -c 2 env \
  -u NPY_ENABLE_CPU_FEATURES NPY_DISABLE_CPU_FEATURES=X86_V3,X86_V4 \
  PYTHONPATH="$DUCC/build-mainline-install/usr/lib/python3.12/site-packages" \
  NUMPY_BUILD_DIR="$DUCC/build-mainline" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DUCC0_NUM_THREADS=1 \
  "$PY" "$SCRIPT" --mode both --repeat 7 \
  --csv /tmp/numpy_fft_mainline_ducc_v2-repro.csv \
  --json /tmp/numpy_fft_mainline_ducc_v2-repro.json
```

Use `--repeat N` and `--csv PATH --json PATH` to change run settings. To
enumerate the full 26 × 2 matrix without timing it:

```bash
"$PY" "$SCRIPT" --mode both --list-cases | wc -l  # 52
```

`--case NAME` can be repeated to time a subset while retaining the original
case's generated input. Pin to another logical CPU with `taskset -c N`; set
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and `DUCC0_NUM_THREADS` in the
environment to change thread limits. The script records these values.
