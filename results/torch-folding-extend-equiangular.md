# Equiangular SHT analysis extension benchmark

This report compares the resampled equiangular analysis in `torch-harmonics`
with the dense Clenshaw--Curtis reference from the immutable parent commit. The
implementation applies the Appendix-A periodic extension, half-sample Fourier
shift, and folded-ring adjoint at runtime. The benchmark covers scalar and
vector analysis, random fields, representative spectral modes, both supported
floating-point precisions on CPU, and float32 CUDA performance.

## Provenance

| Field | Value |
| --- | --- |
| Benchmark date | `2026-09-16` |
| Optimized source | `torch-harmonics` commit `f4cc65515d79d66ff7ab1f02d53518e0b063fc99` |
| Dense reference | `torch-harmonics` commit `3278fb669483d04537aa60c87fb0754d989e2e70` |
| PyTorch | `2.14.0+cu130` |
| Python | `3.14.7` |
| CPU | AMD Ryzen 9 5950X 16-Core Processor |
| GPU | NVIDIA GeForce RTX 5070, 12 GB |
| Torch CPU threads | `1` |
| Timing | two warmups, five CPU repetitions; two warmups, three CUDA repetitions; median |

The [benchmark script](../scripts/torch_folding_benchmark.py) loads the dense
reference directly from the pinned commit and imports the optimized source
checkout. The [CPU JSON](torch-folding-extend-equiangular-v1.json) and
[CPU CSV](torch-folding-extend-equiangular-v1.csv) contain the full accuracy
and timing records. The corresponding [CUDA JSON](torch-folding-extend-equiangular-cuda-f32-delta.json)
and [CUDA CSV](torch-folding-extend-equiangular-cuda-f32-delta.csv) contain
CUDA timing and allocation-delta measurements.

## Workloads and accuracy

The CPU run includes `73x144` cases at `lmax=mmax=38` (first case beyond the
direct quadrature limit), `71` (near the supported limit), and `72` (the
supported limit), plus a `129x256`, `lmax=mmax=128` scaling case. Every case
checks scalar and vector transforms. Correctness also exercises random
triangular spectra, representative degrees/orders including `(70, 0)`,
`(70, 1)`, `(70, 2)`, `(70, 69)`, `(70, 70)`, `(71, 0)`, `(71, 70)`, and
`(71, 71)`, both vector channels, all four norms, and both Condon--Shortley
phase settings.

Maximum optimized-versus-dense errors from the CPU result are:

| Dtype | Transform | Random-field relative L2 | Mode relative L2 | Mode maximum absolute | Convention relative L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| float32 | scalar | `1.99e-7` | `3.10e-7` | `1.35e-7` | `1.48e-7` |
| float32 | vector | `1.98e-7` | `1.03e-5` | `5.30e-6` | `1.68e-7` |
| float64 | scalar | `4.53e-15` | `6.85e-14` | `1.51e-14` | `6.59e-16` |
| float64 | vector | `2.07e-15` | `9.80e-14` | `1.47e-14` | `5.01e-16` |

The larger vector relative error for an isolated mode is caused by a small
reference norm; its corresponding absolute error remains `5.30e-6` in
float32. The random-field and round-trip checks remain within the focused test
tolerances.

## CPU performance and storage

The entries below are median eager forward times as
`resampled / dense`; values below `1` favor the resampled implementation.

| Dtype | Transform | 73x144, lmax 38 | 73x144, lmax 71 | 73x144, lmax 72 | 129x256, lmax 128 |
| --- | --- | ---: | ---: | ---: | ---: |
| float32 | scalar | `1.15` | `0.99` | `0.96` | `0.45` |
| float32 | vector | `0.95` | `0.79` | `0.70` | `0.48` |
| float64 | scalar | `1.06` | `0.95` | `0.93` | `0.59` |
| float64 | vector | `0.96` | `0.70` | `0.58` | `0.69` |

At `129x256`, registered-buffer storage is approximately half the dense
reference:

| Dtype | Transform | Resampled | Dense |
| --- | --- | ---: | ---: |
| float32 | scalar | `8.07 MiB` | `16.06 MiB` |
| float32 | vector | `16.13 MiB` | `32.13 MiB` |
| float64 | scalar | `16.13 MiB` | `32.13 MiB` |
| float64 | vector | `32.25 MiB` | `64.25 MiB` |

The `73x144` first-limit case has small-transform FFT overhead, while the
larger cases show the intended latency benefit. Forward/backward timings are
also retained in the JSON records.

## CUDA performance and storage

The CUDA run uses float32 on the RTX 5070. Forward-time ratios
`resampled / dense` are:

| Transform | 73x144, lmax 72 | 129x256, lmax 128 | 257x512, lmax 256 |
| --- | ---: | ---: | ---: |
| scalar | `1.34` | `1.33` | `0.86` |
| vector | `1.16` | `1.17` | `0.64` |

At `257x512`, registered buffers are `64.26 / 128.25 MiB` for scalar and
`128.51 / 256.50 MiB` for vector (resampled / dense). CUDA peak constructor
and transform allocation deltas are retained in the CUDA result file.

## Runtime folding versus hoisted weights

The benchmark also constructs the equivalent folded projection at construction
time. The hoisted projection is twice the size of the resampled projection and
is materially complex because the even-length Nyquist coefficient is retained
at its negative-frequency representative. At `129x256`, the hoisted setup and
forward-time ratios relative to the runtime fold were:

| Dtype | Transform | Hoisted setup / runtime | Hoisted forward / runtime | Projection storage |
| --- | --- | ---: | ---: | ---: |
| float32 | scalar | `5.77x` | `1.45x` | `2.0x` |
| float32 | vector | `2.79x` | `1.87x` | `2.0x` |
| float64 | scalar | `14.3x` | `7.2x` | `2.0x` |
| float64 | vector | `6.79x` | `6.7x` | `2.0x` |

These measurements support retaining runtime folding: it preserves the memory
reduction and avoids the construction-time complex projection cost.

## Validation commands

```bash
PYTHONPATH=/home/albert/sht_bench/src \
  /home/albert/torch-harmonics/.venv/bin/python \
  scripts/torch_folding_benchmark.py \
  --cases 73x144x38,73x144x71,73x144x72,129x256x128 \
  --threads 1 --warmup 2 --repeat 5 \
  --output results/torch-folding-extend-equiangular-v1
```

The CUDA command used the same script and cases `73x144x72,129x256x128,257x512x256`
with `--device cuda --dtypes float32 --warmup 2 --repeat 3`.

The focused SHT suite passed with `308 passed`. The full CPU-visible upstream
command passed with `788 passed, 54 skipped`. A CUDA-visible full-suite attempt
is not treated as a product result because this CPython 3.14 editable build
has no compiled CUDA custom kernels; it reported 206 environment-dependent
failures. The CUDA SHT benchmark itself uses the PyTorch FFT path and completed
successfully.

## Method reference

M. Reinecke, S. Belkner, J. Carron, “Improved cosmic microwave background
(de-)lensing using general spherical harmonic transforms,” *Astronomy &
Astrophysics* 678, A165 (2023), Appendix A.

DOI: <https://doi.org/10.1051/0004-6361/202346717>
