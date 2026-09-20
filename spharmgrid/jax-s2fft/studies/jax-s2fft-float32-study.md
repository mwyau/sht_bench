# S2FFT float32 precision study

This record describes the S2FFT 1.4.0 precision investigation for the JAX
backend. It is a development result, not a runtime test or a scientific
validation dataset. The complete measurements are stored in the JSON files in
[`plans/results`](results/).

## Environment and method

The primary measurements used Python 3.11.15, `jax==0.5.0`,
`jaxlib==0.5.0`, `s2fft==1.4.0`, and NumPy from the isolated uv environment.
The exact-minimum spharmgrid test environment additionally used `ducc0==0.41.0`
and `xarray==2026.7.0`. The measurements ran on the local Linux CPU host; the
JSON metadata records the platform and package versions for each run.

The field contains resolved scalar spherical harmonic content through degree
three. The spin `+1` and `-1` fields are the independently constructed
spin-weighted fields

```text
q_+ = -v + i u
q_- = -v - i u,
```

where `u` and `v` are the analytic geographic components of a mixed degree-one
and degree-two gradient field. The float64 S2FFT NumPy transform of the same
field is the numerical reference. Each JAX run uses float32 real input or
complex64 spin input; with x64 enabled, S2FFT may use float64 intermediates.

For each case, the reported error is the maximum absolute difference divided
by the maximum absolute value of the corresponding float64 reference. The
forward error compares coefficients, the inverse error applies the inverse
transform to float64-reference coefficients, and the round-trip error applies
the inverse transform to the JAX forward result. MWSS inverse errors are also
reported separately for the north-pole row, south-pole row, and non-polar rows.

The reproducible script is [`jax-s2fft-float32-study.py`](jax-s2fft-float32-study.py).
For the no-JIT diagnostic runs, it substitutes S2FFT's NumPy GL quadrature
weights and fixes S2FFT 1.4.0's negative scalar-spin power expression. These
patches only avoid diagnostic harness failures; they do not replace the
Price–McEwen recurrence, FFTs, resampling, or reference transform.

Commands used for the ordinary path were:

```bash
JAX_ENABLE_X64=0 JAX_DISABLE_JIT=1 JAX_PLATFORM_NAME=cpu \
    uv run --python 3.11 --no-project \
    --with 'jax==0.5.0' --with 'jaxlib==0.5.0' --with 's2fft==1.4.0' \
    python plans/jax-s2fft-float32-study.py \
    --output plans/results/jax-s2fft-float32-x64-off.json \
    --bandlimits 8 16 32 64

JAX_ENABLE_X64=1 JAX_DISABLE_JIT=1 JAX_PLATFORM_NAME=cpu \
    uv run --python 3.11 --no-project \
    --with 'jax==0.5.0' --with 'jaxlib==0.5.0' --with 's2fft==1.4.0' \
    python plans/jax-s2fft-float32-study.py \
    --output plans/results/jax-s2fft-float32-x64-on.json \
    --bandlimits 8 16 32 64
```

The L=128 extensions are in
[`jax-s2fft-float32-x64-off-l128.json`](results/jax-s2fft-float32-x64-off-l128.json)
and
[`jax-s2fft-float32-x64-on-l128.json`](results/jax-s2fft-float32-x64-on-l128.json).

The script records `elapsed_seconds` for the three transforms in each case.
Those timings include the ordinary or O(L²) transform calls after diagnostic
precomputations have been constructed; they are not end-to-end kernel-build
benchmarks. On the local CPU host, the ordinary x64-off run took 48.69 seconds
across 24 cases at L=8–64 and 41.93 seconds across the six L=128 cases. The
corresponding x64-on run took 48.40 and 43.75 seconds. The public O(L³)
diagnostic run took 2.89 seconds across 24 cases because its kernel
construction was outside the timed region; that number is not comparable with
the ordinary path.

## Ordinary JAX results with x64 disabled

The complete L=8–64 records are in
[`jax-s2fft-float32-x64-off.json`](results/jax-s2fft-float32-x64-off.json).
The columns below are forward coefficient, inverse spatial, and round-trip
relative maximum errors.

|   L | sampling | spin | forward | inverse | round trip |
| --: | :------- | ---: | ------: | ------: | ---------: |
|   8 | GL       |    0 | 3.05e-7 | 1.27e-7 |    7.95e-7 |
|   8 | GL       |   +1 | 4.87e-7 | 2.69e-7 |    1.10e-6 |
|   8 | GL       |   −1 | 5.06e-7 | 2.69e-7 |    1.10e-6 |
|  16 | GL       |    0 | 8.16e-7 | 3.17e-7 |    3.89e-6 |
|  16 | GL       |   +1 | 1.05e-6 | 2.05e-7 |    3.06e-6 |
|  16 | GL       |   −1 | 1.05e-6 | 2.05e-7 |    3.06e-6 |
|  32 | GL       |    0 | 2.76e-6 | 2.50e-7 |    2.18e-5 |
|  32 | GL       |   +1 | 4.92e-6 | 2.63e-7 |    2.83e-5 |
|  32 | GL       |   −1 | 4.92e-6 | 2.75e-7 |    2.84e-5 |
|  64 | GL       |    0 | 6.01e-6 | 5.72e-7 |    5.67e-5 |
|  64 | GL       |   +1 | 1.13e-5 | 6.32e-7 |    6.95e-5 |
|  64 | GL       |   −1 | 1.13e-5 | 6.66e-7 |    6.94e-5 |
|   8 | MWSS     |    0 | 9.36e-2 | 8.69e-2 |    1.47e-1 |
|   8 | MWSS     |   +1 | 6.42e-7 | 2.43e-7 |    9.74e-7 |
|   8 | MWSS     |   −1 | 6.31e-7 | 2.43e-7 |    1.05e-6 |
|  16 | MWSS     |    0 | 4.68e-2 | 8.69e-2 |    9.55e-2 |
|  16 | MWSS     |   +1 | 5.76e-7 | 4.08e-7 |    1.62e-6 |
|  16 | MWSS     |   −1 | 5.76e-7 | 4.24e-7 |    1.62e-6 |
|  32 | MWSS     |    0 | 2.34e-2 | 8.68e-2 |    8.68e-2 |
|  32 | MWSS     |   +1 | 2.26e-6 | 4.63e-7 |    2.42e-5 |
|  32 | MWSS     |   −1 | 2.25e-6 | 4.63e-7 |    2.40e-5 |
|  64 | MWSS     |    0 | 1.17e-2 | 8.68e-2 |    8.68e-2 |
|  64 | MWSS     |   +1 | 8.75e-6 | 5.12e-7 |    3.57e-4 |
|  64 | MWSS     |   −1 | 8.76e-6 | 5.15e-7 |    3.57e-4 |

At L=128, the scalar MWSS forward, inverse, and round-trip errors were
`5.85e-3`, `8.68e-2`, and `8.68e-2`. The corresponding spin `+1` values were
`1.89e-5`, `5.69e-7`, and `3.19e-4`, and spin `−1` values were `1.89e-5`,
`5.65e-7`, and `3.19e-4`. GL round-trip errors reached `2.46e-4` for spin 0
and `3.50e-4` for both spin signs.

The MWSS inverse row errors show that the scalar failure is interior, not a
pole singularity:

|   L | spin | north pole | south pole | non-polar rows |
| --: | ---: | ---------: | ---------: | -------------: |
|   8 |    0 |    0.00e+0 |    4.98e-8 |        8.69e-2 |
|  16 |    0 |    0.00e+0 |    4.98e-8 |        8.69e-2 |
|  32 |    0 |    0.00e+0 |    4.97e-8 |        8.68e-2 |
|  64 |    0 |    0.00e+0 |    4.97e-8 |        8.68e-2 |
| 128 |    0 |    0.00e+0 |    4.97e-8 |        8.68e-2 |
|   8 |   +1 |    2.32e-8 |    3.39e-8 |        2.43e-7 |
|  32 |   +1 |    4.41e-8 |    5.42e-8 |        4.63e-7 |
|  64 |   +1 |    8.00e-8 |    9.01e-8 |        5.12e-7 |
| 128 |   +1 |    9.49e-8 |    9.01e-8 |        5.69e-7 |
|   8 |   −1 |    2.32e-8 |    3.39e-8 |        2.43e-7 |
|  32 |   −1 |    4.41e-8 |    5.42e-8 |        4.63e-7 |
|  64 |   −1 |    8.00e-8 |    9.01e-8 |        5.65e-7 |
| 128 |   −1 |    7.99e-8 |    9.97e-8 |        5.65e-7 |

## Ordinary JAX results with x64 enabled

The complete records are in
[`jax-s2fft-float32-x64-on.json`](results/jax-s2fft-float32-x64-on.json) and
[`jax-s2fft-float32-x64-on-l128.json`](results/jax-s2fft-float32-x64-on-l128.json).
Across all three spins, the maximum errors by sampling and bandlimit were:

|   L | GL forward | GL inverse | GL round trip | MWSS forward | MWSS inverse | MWSS round trip |
| --: | ---------: | ---------: | ------------: | -----------: | -----------: | --------------: |
|   8 |    4.30e-8 |   5.84e-16 |       1.29e-7 |      7.91e-9 |     2.27e-16 |         3.48e-8 |
|  16 |    5.61e-8 |   4.63e-16 |       1.49e-7 |      6.56e-9 |     3.09e-16 |         4.02e-8 |
|  32 |    2.02e-8 |   8.36e-16 |       1.36e-7 |      3.84e-9 |     6.67e-16 |         5.47e-8 |
|  64 |    6.63e-8 |   9.48e-16 |       3.27e-7 |      2.50e-9 |     7.98e-16 |         5.62e-8 |
| 128 |    2.25e-8 |   1.20e-15 |       2.85e-7 |      1.26e-9 |     8.34e-16 |         6.70e-8 |

The MWSS pole and interior errors remain at approximately `10^-15` in this
configuration. The small forward-coefficient errors are consistent with using
float32 input fields against a float64 reference; the inverse transform uses
float64 intermediates.

## Compiled performance measurements

The records originally collected in this section used ordinary S2FFT JAX
calls with `precomps=None`. They are retained as historical diagnostic
records, not as measurements of the corrected production-supported setup path.
The separate benchmark script used for those records is
[`jax-s2fft-performance-study.py`](jax-s2fft-performance-study.py). It uses
`jax.jit`, separates `.lower(...).compile()` time from warmed calls, records
the median of seven blocking steady-state calls, and records cumulative Linux
process peak RSS together with field and coefficient byte counts. The complete
L=8 GL/MWSS and spin 0/±1 records are in
[`jax-s2fft-performance-x64-off.json`](results/jax-s2fft-performance-x64-off.json)
and
[`jax-s2fft-performance-x64-on.json`](results/jax-s2fft-performance-x64-on.json).
Both runs used Python 3.11.15, JAX/jaxlib 0.5.0, S2FFT 1.4.0, the CPU backend,
and the same Linux host as the precision study.

Representative scalar MWSS L=8 measurements were:

| x64 state | requested dtype | compile forward | compile inverse | warm forward | warm inverse | field bytes | coefficient bytes |
| :-------- | :-------------- | --------------: | --------------: | -----------: | -----------: | ----------: | ----------------: |
| off       | float32         |         0.476 s |         0.352 s |     0.113 ms |     0.101 ms |         576 |               960 |
| on        | float32         |         0.455 s |         0.372 s |     0.123 ms |     0.100 ms |         576 |               960 |
| on        | float64         |         0.314 s |         0.319 s |     0.096 ms |     0.064 ms |        1152 |              1920 |

For this CPU and problem size, the historical `precomps=None` float32 used half the array storage but did not
provide a consistent warmed speed advantage: the x64-enabled float64 case was
faster for both MWSS timings shown. The x64-enabled float32 MWSS errors were
`6.15e-9` forward, `4.98e-8` inverse, and `4.98e-8` round trip for this
diagnostic field; the float64 values were `1.83e-16`, `1.55e-16`, and
`6.18e-16`. These are measurements on this CPU at L=8, not general hardware
or resolution claims. The historical highest completed compiled benchmark was
L=8: an ordinary GL L=16 float32 forward compilation on the `precomps=None`
path did not finish within the approximately 300-second bound. This is not a
measurement of the corrected x64 production path. The bounded attempt and
observed process RSS are recorded in
[`jax-s2fft-performance-compile-limit.json`](results/jax-s2fft-performance-compile-limit.json).
The precision-only, non-JIT study remains practical through L=128.

### CUDA measurements

The same precision and historical `precomps=None` compiled-performance studies were run in a temporary
Python 3.11 environment with the official CUDA 13 JAX extra. The environment
reported Python 3.11.15, `jax==0.10.2`, `jaxlib==0.10.2`, `s2fft==1.4.0`,
NumPy 2.4.6, JAX backend `gpu`, and `[CudaDevice(id=0)]` for an NVIDIA
GeForce RTX 5070. The host NVIDIA driver was 615.71.09. The x64-off and x64-on
runs were separate processes, and the benchmark invoked `JAX_PLATFORMS=cuda`
and required the CUDA device check; a CPU fallback therefore failed rather
than being recorded as a GPU result.

The temporary environment used:

```bash
uv pip install -U "jax[cuda13]" "s2fft==1.4.0"
```

plus NumPy. It was not added to the project dependencies or lockfile.

The two precision processes used:

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 JAX_DISABLE_JIT=1 \
    python plans/jax-s2fft-float32-study.py --require-cuda \
    --output plans/results/jax-s2fft-float32-gpu-x64-off.json \
    --bandlimits 8 16 32 64

JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 JAX_DISABLE_JIT=1 \
    python plans/jax-s2fft-float32-study.py --require-cuda --dtype float64 \
    --output plans/results/jax-s2fft-float64-gpu-x64-on.json \
    --bandlimits 8 16 32 64
```

The precision records are
[`jax-s2fft-float32-gpu-x64-off.json`](results/jax-s2fft-float32-gpu-x64-off.json),
[`jax-s2fft-float32-gpu-x64-on.json`](results/jax-s2fft-float32-gpu-x64-on.json),
and
[`jax-s2fft-float64-gpu-x64-on.json`](results/jax-s2fft-float64-gpu-x64-on.json).
The float32 x64-off matrix and the float64 x64-on matrix cover GL and MWSS,
spins 0, +1, and −1, and L=8, 16, 32, and 64. For the float32 MWSS scalar
case, L=8 had forward, inverse/interior, and round-trip errors of
`9.36e-2`, `8.69e-2`, and `1.47e-1`; the north and south pole errors were
`8.29e-8` and `8.29e-9`. At L=64 the corresponding errors were `1.17e-2`,
`8.68e-2`, and `8.68e-2`, with pole errors `8.28e-8` and `8.28e-9`.
The float64 L=64 maximum errors across the three spins were `7.30e-15`
(GL forward), `1.47e-15` (GL inverse), `4.12e-14` (GL round trip),
`2.71e-15` (MWSS forward), `1.19e-15` (MWSS inverse), and `2.25e-14`
(MWSS round trip). These records retain the pole/interior breakdown for every
MWSS case.

The compiled performance records are
[`jax-s2fft-performance-gpu-x64-off.json`](results/jax-s2fft-performance-gpu-x64-off.json)
and
[`jax-s2fft-performance-gpu-x64-on.json`](results/jax-s2fft-performance-gpu-x64-on.json).
For scalar L=8 on this RTX 5070, the measured results were:

| sampling | x64 state | dtype   | compile forward | compile inverse | warm forward | warm inverse | forward error | inverse error | round trip |
| :------- | :-------- | :------ | --------------: | --------------: | -----------: | -----------: | ------------: | ------------: | ---------: |
| GL       | off       | float32 |         0.515 s |         0.393 s |     0.384 ms |     0.338 ms |       5.69e-7 |       1.94e-7 |    1.46e-6 |
| GL       | on        | float64 |         0.531 s |         0.407 s |     0.430 ms |     0.364 ms |      5.29e-16 |      3.09e-16 |   2.16e-15 |
| MWSS     | off       | float32 |         0.516 s |         0.422 s |     0.373 ms |     0.337 ms |       9.36e-2 |       8.69e-2 |    1.47e-1 |
| MWSS     | on        | float64 |         0.549 s |         0.530 s |     0.385 ms |     0.372 ms |      3.29e-16 |      3.09e-16 |   1.85e-15 |

At this tested L and device, float32 used half the field and coefficient
storage and was about 1.03× faster for the MWSS forward call and 1.10× faster
for its inverse call. The GL measurements were about 1.12× and 1.08× faster,
respectively. These are measured L=8 results, not a GPU or resolution-wide
speed claim. The recorded memory values are array byte counts and Linux host
process RSS; they do not measure the CUDA allocator's device memory. An L=16
historical `precomps=None` float32 attempt did not complete its first forward
compilation within approximately 300 seconds. Its explicit attempt record is
[`jax-s2fft-performance-gpu-x64-off-l16.json`](results/jax-s2fft-performance-gpu-x64-off-l16.json).

### Historical bounded full-matrix `precomps=None` CPU and CUDA runs

The first bounded matrix used the ordinary `s2fft.forward_jax` and
`s2fft.inverse_jax` calls with `precomps=None` for every case. It did not call
the O(L³) precomputed transform. Each case ran in a fresh worker with a
20-second wall time limit; the parent wrote the result after every attempt and
retained `complete`, `timeout`, or `error` status. The `--resume` option can
continue an incomplete result file without repeating completed cases. These
records are historical evidence for the unprecomputed path, not corrected
production performance measurements:

- [`jax-s2fft-performance-study.py`](jax-s2fft-performance-study.py)
- [`CPU, x64 disabled`](results/jax-s2fft-performance-cpu-x64-off-l16-128.json)
- [`CPU, x64 enabled`](results/jax-s2fft-performance-cpu-x64-on-l16-128.json)
- [`CUDA, x64 disabled`](results/jax-s2fft-performance-gpu-x64-off-l16-128.json)
- [`CUDA, x64 enabled`](results/jax-s2fft-performance-gpu-x64-on-l16-128.json)

Each matrix covers GL and MWSS, spins 0, +1, and −1, and L=16, 32, 64, and
128\. The requested dtype is float32 with x64 disabled and float64 with x64
enabled. The CPU runs used the exact minimum combination Python 3.11.15,
JAX/jaxlib 0.5.0, and S2FFT 1.4.0. The CUDA environment used Python 3.11.15,
JAX/jaxlib 0.10.2 from the official CUDA 13 extra, and S2FFT 1.4.0. The
different JAX versions are recorded in the JSON metadata; CPU/GPU timing ratios
must not be interpreted as same-version comparisons. The CUDA process reported
`CudaDevice(id=0)` for an NVIDIA GeForce RTX 5070 and failed rather than
accepting CPU fallback.

The bounded completion matrix is:

| process       | GL L=16             | GL L=32             | GL L=64             | GL L=128            | MWSS L=16–128       |
| :------------ | :------------------ | :------------------ | :------------------ | :------------------ | :------------------ |
| CPU, float32  | timeout, all spins  | timeout, all spins  | timeout, all spins  | timeout, all spins  | complete, all spins |
| CPU, float64  | complete, all spins | complete, all spins | complete, all spins | complete, all spins | complete, all spins |
| CUDA, float32 | timeout, all spins  | complete, all spins | timeout, all spins  | timeout, all spins  | complete, all spins |
| CUDA, float64 | complete, all spins | complete, all spins | complete, all spins | complete, all spins | complete, all spins |

The timeout records preserve the case identifiers, timeout, worker wall time,
and captured output. A timeout is not treated as a numerical result. Every
completed case records forward and inverse compile time, median warmed time
over seven blocking calls, forward coefficient error, inverse spatial error,
round-trip error, requested and effective dtype, field and coefficient byte
counts, process peak RSS, and JAX device memory counters when the backend
provides them.

The following scalar MWSS rows show the complete performance and accuracy
records at the four requested resolutions. Times are seconds except for the
warm columns, which are milliseconds; errors are relative maximum errors.

| backend / state |   L | compile fwd | compile inv | warm fwd | warm inv |  fwd err |  inv err | round trip | field / coeff bytes |
| :-------------- | --: | ----------: | ----------: | -------: | -------: | -------: | -------: | ---------: | ------------------: |
| CPU / float32   |  16 |       0.751 |       0.447 |    0.133 |    0.089 |  4.68e-2 |  8.69e-2 |    9.55e-2 |       2,176 / 3,968 |
| CPU / float32   |  32 |       0.770 |       0.418 |    0.625 |    0.276 |  2.34e-2 |  8.68e-2 |    8.68e-2 |      8,448 / 16,128 |
| CPU / float32   |  64 |       1.157 |       0.398 |    4.885 |    2.059 |  1.17e-2 |  8.68e-2 |    8.68e-2 |     33,280 / 65,024 |
| CPU / float32   | 128 |       1.813 |       0.441 |   37.614 |   16.830 |  5.85e-3 |  8.68e-2 |    8.68e-2 |   132,096 / 261,120 |
| CPU / float64   |  16 |       0.664 |       0.431 |    0.264 |    0.207 | 2.19e-16 | 3.09e-16 |   1.70e-15 |       4,352 / 7,936 |
| CPU / float64   |  32 |       0.836 |       0.458 |    1.759 |    1.272 | 4.25e-16 | 3.09e-16 |   4.62e-15 |     16,896 / 32,256 |
| CPU / float64   |  64 |       1.165 |       0.447 |   14.592 |    7.110 | 4.51e-16 | 4.63e-16 |   3.31e-15 |    66,560 / 130,048 |
| CPU / float64   | 128 |       1.868 |       0.478 |  148.620 |   55.355 | 9.40e-16 | 6.17e-16 |   1.74e-14 |   264,192 / 522,240 |
| CUDA / float32  |  16 |       0.648 |       0.488 |    0.808 |    0.628 |  4.68e-2 |  8.69e-2 |    9.55e-2 |       2,176 / 3,968 |
| CUDA / float32  |  32 |       0.810 |       0.489 |    1.239 |    1.217 |  2.34e-2 |  8.68e-2 |    8.68e-2 |      8,448 / 16,128 |
| CUDA / float32  |  64 |       0.905 |       0.437 |    2.360 |    2.367 |  1.17e-2 |  8.68e-2 |    8.68e-2 |     33,280 / 65,024 |
| CUDA / float32  | 128 |       1.484 |       0.460 |    4.665 |    4.590 |  5.85e-3 |  8.68e-2 |    8.68e-2 |   132,096 / 261,120 |
| CUDA / float64  |  16 |       0.618 |       0.451 |    0.692 |    0.725 | 3.24e-16 | 3.09e-16 |   2.47e-15 |       4,352 / 7,936 |
| CUDA / float64  |  32 |       0.763 |       0.451 |    1.299 |    1.404 | 3.03e-16 | 3.09e-16 |   4.93e-15 |     16,896 / 32,256 |
| CUDA / float64  |  64 |       0.983 |       0.466 |    3.366 |    3.010 | 4.56e-16 | 4.65e-16 |   1.17e-14 |    66,560 / 130,048 |
| CUDA / float64  | 128 |       1.620 |       0.453 |    9.023 |    6.754 | 6.31e-16 | 4.64e-16 |   1.42e-14 |   264,192 / 522,240 |

For the measured CUDA scalar MWSS L=128 case, float32 was 1.94 times faster
forward and 1.47 times faster inverse than float64 in the same JAX 0.10.2
environment, while using half the field and coefficient storage. This is a
measurement for the RTX 5070 and this problem size, not a general GPU or
resolution-wide speed claim. The corresponding CPU 0.5.0 environment measured
3.95 and 3.29 times faster warmed float32 forward and inverse calls, but those
float32 transforms have the documented MWSS errors and are not a supported
spharmgrid path.

CUDA `memory_stats()` was available and is recorded per case, including peak
bytes in use and allocator limits. For example, the scalar MWSS L=128 CUDA
cases recorded peak device bytes in use of 17,695,744 for float32 and
18,356,224 for float64; the process peak RSS was approximately 1.16 GB in
both cases. CPU device memory statistics were unavailable; the JSON records
the per-worker Linux peak RSS instead. These counters describe the observed
JAX allocator state and should not be extrapolated to other devices.

## Isolated precomputation decision

The isolated comparison in
[`s2fft-precomp-matrix-cpu-jax050-x64-on-l16-64-128.json`](results/s2fft-precomp-matrix-cpu-jax050-x64-on-l16-64-128.json)
and
[`s2fft-precomp-matrix-gpu-jax0102-x64-on-l16-64-128.json`](results/s2fft-precomp-matrix-gpu-jax0102-x64-on-l16-64-128.json)
contains 48 completed cases per environment. Each matrix covers GL and MWSS,
L=16, 64, and 128, scalar `reality=False` and `reality=True`, spin `+1` and
`−1` with `reality=False`, and `precomps=None` versus externally generated
O(L²) precomputations. The harness records setup, trace, lowering, compile,
first execution, warmed forward and inverse execution, errors, and array byte
counts.

The decision-relevant comparison holds the scalar `reality=True` optimization
constant while comparing `precomps=None` with external precomputations. For
spin `+1` and `−1`, both paths use `reality=False`. On CUDA, external
precomputation improved warmed runtime by approximately 26–57% for scalar
`reality=True` and 26–41% for spin ±1. External setup took approximately
0.26–0.30 s on CPU and 0.41–0.45 s on CUDA. Compile plus first-use cost was
broadly similar after setup was included. Numerical behavior was unchanged.

| environment | forward maximum | inverse maximum | round-trip maximum |
| :---------- | --------------: | --------------: | -----------------: |
| CPU         |        2.29e-15 |        1.11e-15 |           7.43e-14 |
| CUDA        |        2.09e-15 |        1.08e-15 |           1.25e-13 |

The warmed CUDA improvement justifies external O(L²) precomputation for
repeated static transform configurations.

### Cache implementation evidence

#### Array placement

The placement experiment used L=64, both samplings, spin 0 and spin 1, and
both transform directions. Each precompute tuple contains five arrays. The
arrays were uncommitted (`Array.committed == False`) in both tested
environments: Python 3.11 with JAX/jaxlib 0.5.0 on CPU and JAX/jaxlib 0.10.2
on the RTX 5070 CUDA environment. Direct S2FFT generation had the same
placement behavior as the spharmgrid helper. No device key is therefore used
in the cache. Only one CUDA device was available, so multi-GPU behavior was
not tested.

#### JAX staging

Eager precompute generation succeeds without `jax.ensure_compile_time_eval()`.
A first cache miss inside an outer `jax.jit` does not: without the context,
S2FFT precomputations remain `DynamicJaxprTracer` values and
`block_until_ready()` fails. The failure reproduced on JAX/jaxlib 0.5.0 CPU
and JAX/jaxlib 0.10.2 CUDA. With `jax.ensure_compile_time_eval()` restored,
the outer-JIT check passed under `jax.checking_leaks()` and the cached values
were concrete arrays.

`jax.ensure_compile_time_eval()` is retained to materialize precomputations
before they enter the Python LRU cache when the first request occurs during
JAX tracing.

### Post-cleanup operation benchmark

The post-cleanup operation records are
[`spharmgrid-jax-performance-cpu-x64-on-cache-lru32-l64.json`](results/spharmgrid-jax-performance-cpu-x64-on-cache-lru32-l64.json)
and
[`spharmgrid-jax-performance-gpu-x64-on-cache-lru32-l64-128.json`](results/spharmgrid-jax-performance-gpu-x64-on-cache-lru32-l64-128.json).
They cover scalar `filter`, scalar `regrid`, and vector `kinematics` using the
simplified bounded cache. Across the recorded CUDA L=64 and L=128 cases, the
spharmgrid/direct-S2FFT warmed-time ratio was approximately 0.936–1.030.
This supports the conclusion that the simplified LRU cache did not introduce
systematic operation-level overhead in these cases; individual run-to-run
timing changes are not generalized.

## Earlier direct S2FFT precomputation diagnosis

The earlier standalone matrix harness is
[`s2fft-precomp-study.py`](s2fft-precomp-study.py). Its records compare
`precomps=None`, externally generated `generate_precomputes_jax` arrays with
`reality=False`, and the same external arrays with scalar `reality=True`. It
separates precompute setup, JAX tracing, lowering, XLA compilation, first
execution, and warmed execution. It records all GL/MWSS, spin 0/±1, and
L=8, 16, 32, and 64 cases:

- [Python 3.11.15, JAX/jaxlib 0.5.0, S2FFT 1.4.0, CPU](results/s2fft-precomp-matrix-cpu-jax050-x64-on.json)
- [Python 3.11.15, JAX/jaxlib 0.10.2, S2FFT 1.4.0, CUDA RTX 5070](results/s2fft-precomp-matrix-gpu-jax0102-x64-on.json)

All 56 cases in each matrix completed under a 60-second per-case bound. With
x64 enabled, the `precomps=None` path did not reproduce the earlier hundreds
of seconds at L=16; that historical result was from the x64-disabled
unprecomputed harness. External O(L²) arrays still reduced transform-stage
compile and lowering time on the tested GPU, while setup was measured
separately. This setup-versus-transform distinction is also discussed in
S2FFT issue [#317](https://github.com/astro-informatics/s2fft/issues/317).
The maximum completed-matrix forward, inverse, and round-trip
relative errors were `2.29e-15`, `1.11e-15`, and `7.33e-14` on the CPU and
`1.48e-15`, `1.08e-15`, and `1.75e-14` on the CUDA run. The scalar
`reality=True` cases retained the full `(L, 2L - 1)` centered coefficient
shape and returned real spatial arrays. Spin ±1 cases used `reality=False`.

Representative scalar MWSS L=64 records are below. Setup and compile columns
are seconds; warmed columns are milliseconds; errors are relative maximum
errors.

| environment     | path   | reality | precompute setup | compile fwd | compile inv | warm fwd | warm inv |  fwd err | inverse err | round trip |
| :-------------- | :----- | :------ | ---------------: | ----------: | ----------: | -------: | -------: | -------: | ----------: | ---------: |
| CPU JAX 0.5.0   | `None` | false   |            0.000 |       0.327 |       0.162 |   13.521 |    6.950 | 4.51e-16 |    4.63e-16 |   3.31e-15 |
| CPU JAX 0.5.0   | O(L²)  | false   |            0.272 |       0.259 |       0.118 |   12.746 |    6.136 | 4.51e-16 |    4.63e-16 |   3.31e-15 |
| CPU JAX 0.5.0   | O(L²)  | true    |            0.264 |       0.235 |       0.080 |    6.806 |    4.577 | 4.51e-16 |    4.62e-16 |   3.31e-15 |
| CUDA JAX 0.10.2 | `None` | false   |            0.000 |       0.572 |       0.333 |    3.374 |    3.046 | 4.56e-16 |    4.65e-16 |   1.17e-14 |
| CUDA JAX 0.10.2 | O(L²)  | false   |            0.428 |       0.478 |       0.253 |    2.196 |    1.919 | 4.56e-16 |    4.65e-16 |   1.17e-14 |
| CUDA JAX 0.10.2 | O(L²)  | true    |            0.439 |       0.420 |       0.180 |    1.156 |    0.997 | 4.56e-16 |    4.62e-16 |   1.17e-14 |

The two environments use different JAX versions because the CUDA installation
used the available official CUDA 13 build; the version distinction is explicit
in the result metadata. These are direct S2FFT measurements, not spharmgrid
parity validation.

## Corrected transform and operation performance

The corrected transform benchmark uses external public O(L²) precomputations
outside transform compilation, scalar `reality=True`, and spin `reality=False`.
The RTX 5070 x64-enabled run used JAX/jaxlib 0.10.2 and S2FFT 1.4.0. All 30
cases for GL/MWSS, spins 0/±1, and L=16, 32, 64, 128, and 256 completed under
a 120-second per-case bound. The complete records are in
[`jax-s2fft-performance-gpu-x64-on-precomps-l16-256.json`](results/jax-s2fft-performance-gpu-x64-on-precomps-l16-256.json).

For scalar MWSS, the corrected GPU records were:

|   L | precompute setup | compile fwd | compile inv |  warm fwd | warm inv |  fwd err | inverse err | round trip |   field / coeff bytes |
| --: | ---------------: | ----------: | ----------: | --------: | -------: | -------: | ----------: | ---------: | --------------------: |
|  16 |          0.421 s |     0.241 s |     0.192 s |  0.263 ms | 0.271 ms | 3.24e-16 |    3.09e-16 |   2.16e-15 |         4,352 / 7,936 |
|  32 |          0.428 s |     0.322 s |     0.176 s |  0.432 ms | 0.495 ms | 3.03e-16 |    3.08e-16 |   4.93e-15 |       16,896 / 32,256 |
|  64 |          0.435 s |     0.428 s |     0.186 s |  1.167 ms | 1.010 ms | 4.56e-16 |    4.62e-16 |   1.17e-14 |      66,560 / 130,048 |
| 128 |          0.449 s |     0.722 s |     0.190 s |  3.429 ms | 2.223 ms | 6.26e-16 |    6.17e-16 |   1.40e-14 |     264,192 / 522,240 |
| 256 |          0.466 s |     1.288 s |     0.215 s | 17.810 ms | 9.139 ms | 1.00e-15 |    6.17e-16 |   6.01e-14 | 1,052,672 / 2,093,056 |

At L=256 the scalar MWSS forward and inverse precomputations occupied
5,238,784 and 3,678,208 bytes in the worker, which is O(L²) storage. The
worker recorded 35,252,224 peak CUDA bytes in use and approximately 1.30 GB
Linux peak RSS for that scalar case. These are measurements of the RTX 5070
and tested problem sizes, not extrapolated hardware limits.

Representative full-operation records are in
[`spharmgrid-jax-performance-cpu-jax050-x64-on-l16-128.json`](results/spharmgrid-jax-performance-cpu-jax050-x64-on-l16-128.json)
and
[`spharmgrid-jax-performance-gpu-x64-on-l16-128.json`](results/spharmgrid-jax-performance-gpu-x64-on-l16-128.json),
generated by [`spharmgrid-jax-performance-study.py`](spharmgrid-jax-performance-study.py).
They cover scalar `filter`, GL-to-CC and CC-to-GL `regrid`, and vector
`kinematics` on both GL and MWSS grids at L=16, 32, 64, and 128. Each case
records recursion setup, trace/lowering/compile, warmed batched operation,
host/device transfer, array sizes, process RSS, device memory when available,
and a direct batched S2FFT transform baseline. At GPU L=128, the
spharmgrid/direct warmed-operation ratios ranged from 0.98 to 1.02 across the
six operation/grid combinations; this is evidence against a pathological
batching overhead for the tested cases, not a general performance claim.

## O(L²) recursion precomputations

The S2FFT-generated `generate_precomputes_jax` arrays are float32 when
`JAX_ENABLE_X64=0`. Passing them to ordinary `s2fft.forward_jax` and
`inverse_jax` produced the same measurements as omitting `precomps` for every
GL/MWSS and spin case in the L=8–64 study. This shows that the error is in the
float32 Price–McEwen recurrence, not merely in constructing the recursion
constants on demand. The complete run is in
[`jax-s2fft-precomps-jax-float32.json`](results/jax-s2fft-precomps-jax-float32.json).
S2FFT [issue #320](https://github.com/astro-informatics/s2fft/issues/320)
provides upstream context on lower-precision precomputed transforms; the
measurements here do not establish a scalable float32 remedy.

Host-generated `generate_precomputes` arrays had dtypes
`[float64, float64, float64, float64, int64]`. Passing those NumPy arrays
directly to the JAX transform failed with:

```text
AttributeError: 'numpy.ndarray' object has no attribute .at
```

Converting them to JAX arrays allowed the call, but x64-disabled JAX truncated
the arrays to float32. The scalar MWSS L=8 forward and inverse errors were
`9.359e-2` and `8.695e-2`, and the inverse/interior error remained about
`8.678e-2` at L=32–64. Host precomputations improved some spin ±1 cases but
did not establish a general float32 path. The complete host-cast run is in
[`jax-s2fft-precomps-host-float32.json`](results/jax-s2fft-precomps-host-float32.json).

## Public O(L³) diagnostic path

The optional diagnostic run used S2FFT's public
`construct.spin_spherical_kernel`, `precompute_transforms.spherical.forward`,
and `inverse` functions. Kernels were constructed in NumPy float64 and cast to
float32 before the JAX call. It produced approximately `10^-7` relative errors
for MWSS scalar and spin fields through L=64; the complete records are in
[`jax-s2fft-precomputed-float32-diagnostic.json`](results/jax-s2fft-precomputed-float32-diagnostic.json).

This path requires a full Wigner-d kernel with O(L³) storage. It is evidence
that higher-precision kernel construction fixes the numerical result, not a
supported scalable spharmgrid solution. At L=128, a single float64 MWSS
forward kernel has approximately `(2L, L, 2L - 1) = (256, 128, 255)` entries,
or about 64 MiB before additional temporaries; the inverse kernel has roughly
half that size. The S2FFT documentation and [issue #319](https://github.com/astro-informatics/s2fft/issues/319)
identify the same O(L³) memory limitation.

The public precomputed API was also checked at L=8 with float32 kernels:

- eager forward and inverse returned complex64 arrays;
- `jax.jit` and `jax.vmap` returned the expected shapes;
- a JAX gradient through the public forward operation returned finite values.

The diagnostic script uses only S2FFT's public `generate_precomputes_jax`,
`generate_precomputes`, `forward_jax`, and `inverse_jax` APIs for these O(L²)
comparisons. It does not reproduce any S2FFT recurrence or resampling logic.

## Current S2FFT `main`

The minimal MWSS scalar reproducer was run once against S2FFT commit
`86a4dff303539202ebbfa28661d85331e0e2bff4` (`1.4.1.dev21`) with
`jax==0.5.0`, `jaxlib==0.5.0`, and x64 disabled. At L=8 it produced:

```text
forward_relative_max 0.09359483334211248
inverse_relative_max 0.0892857105820178
```

The unreleased main branch therefore does not alter the measured failure. The
project does not depend on this unreleased version.

The release reproducer is saved as
[`s2fft-mwss-float32-reproducer.py`](s2fft-mwss-float32-reproducer.py). It
uses the ordinary JAX API without spharmgrid code, and its S2FFT 1.4.0 run
reported forward coefficient error `9.359483e-2`, inverse spatial error
`8.694812e-2`, north-pole error `8.292019e-8`, south-pole error `8.292018e-9`,
and interior error `8.694812e-2`.

## Exact minimum dependency check

PyPI metadata and wheels were checked for the candidate floor:

- S2FFT 1.4.0: `Requires-Python >=3.11`, `jax>=0.5.0`, and a CPython 3.11
  manylinux x86-64 wheel;
- JAX 0.5.0: Python 3.11 support and a requirement for `jaxlib==0.5.0`;
- jaxlib 0.5.0: a CPython 3.11 manylinux x86-64 wheel.

The isolated exact environment reported Python 3.11.15, JAX 0.5.0, jaxlib
0.5.0, S2FFT 1.4.0, and x64 enabled. After the x64-only policy change, the
JAX optional and DUCC parity tests passed: `46 passed, 1 skipped in 28.30s`.
The separate x64-disabled precision-capability process passed its rejection
check: `1 passed, 2 skipped in 0.03s`. The declared project requirements
remain unchanged.

The first x64-disabled run of the existing suite, before the later policy
change, reported `34 passed, 18 skipped, 2 failed`; both failures were the
pre-existing float32 vector-regrid parity cases, with maximum errors
`5.22e-7` and `5.55e-7` against `5e-7` tolerance. This result is retained as
context; no tolerance change was made from that run.

## Implementation decision and validation

The measured ordinary MWSS float32 error is an interior Price–McEwen
recurrence failure, not a pole error. O(L²) precomputations do not repair it,
and the O(L³) public precomputed transform is retained only as diagnostic
evidence. `spharmgrid.jax` therefore calls ordinary S2FFT JAX transforms and
requires JAX x64 mode with `float64` input arrays. It rejects single-precision
inputs before entering the transform. No spharmgrid pole correction or
resolution-dependent transform switch was added.

The x64-enabled suite retained the existing `jit`, `vmap`, and finite-gradient
smoke tests. The new near-bandlimit tests cover GL and CC/MWSS upsampling and
downsampling with scalar `l=m=11` at source L=12, removal of a source
`l=m=15` mode when downsampling from L=16 to L=12 while retaining `l=m=3`,
and vector regridding of the analytic gradient of the near-edge mode. The
float64 assertions use `rtol=2e-10` and `atol=2e-11`.

The finite-difference tests use x64-enabled float64 fields, fixed spatial
weights, and ε=`1e-4`. The measured scalar filter directional-derivative
relative error was `4.5e-13`; the two-input vector kinematics error was
`6.7e-11`. The tests use `rtol=2e-10` and `atol=2e-11`.
