# FFT benchmark

`fft-bench` measures forward, one-dimensional DUCC FFTs through `ducc0.fft`.
The initial comparison is deliberately limited to DUCC and covers:

- `r2c`: `float32`, `float64`, and `longdouble` input;
- `c2c`: `complex64`, `complex128`, and `clongdouble` input through the same
  portable precision labels.

The `longdouble` label is portable; each result records the actual NumPy dtype
and its runtime representation. Typical Linux x86-64 reports
`x87 extended precision (64-bit significand, 16-byte storage)`, while an
AArch64 system that provides the wider IEEE format reports `IEEE binary128`.
The label is never interpreted from the dtype name alone.

## Run a single process

```bash
uv run fft-bench run \
  --kind r2c,c2c \
  --dtype float32,float64,longdouble \
  --sizes 256,1024,4096,16384 \
  --threads 1 \
  --output results/fft/run.json
```

The default sizes are `256, 1024, 4096, 16384, 65536, 262144, 1048576`.
Inputs and output buffers are prepared before timing. The shared benchmark
timing method performs warm-up, calibration, repeated `perf_counter_ns`
samples with GC disabled, and records minimum, median, mean, standard
deviation, and raw samples.

## Run a process-isolated thread matrix and report

```bash
uv run --extra plot fft-bench matrix \
  --kind r2c,c2c \
  --dtype float32,float64,longdouble \
  --threads 1,2,4 \
  --output results/fft \
  --plot
```

Each thread-count cell is a fresh Python process and passes the requested
`nthreads` explicitly to DUCC. The report is written to `results/fft/report/`
and contains only `README.md` and PNG figures; the JSON files in the parent
directory are raw working data and are not part of the report artifact.

If the installed DUCC binding cannot accept genuine NumPy long-double input,
the case is recorded as unsupported with a reason. It is not relabeled as
`float64` and no timing result is fabricated.
