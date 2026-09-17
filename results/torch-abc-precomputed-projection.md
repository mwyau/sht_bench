# PyTorch A/B/C precomputed-projection benchmark

This report records the final memory-safe matrix for the PyTorch equiangular
SHT projection experiment. It uses 0.5° resolution steps and stops at 0.5°:
2.5°, 2.0°, 1.5°, 1.0°, and 0.5°. A later 0.25° B-only attempt was killed by
the host/container memory limit before producing a result file, so no 0.25° or
finer case is included.

The comparison includes the three Torch implementations requested for the
experiment and a scalar CPU cross-backend comparison against DUCC:

| Label | Implementation |
| --- | --- |
| A | `dense` projection |
| B | `runtime_fold` projection folding |
| C | `precomputed_projection` effective projection |

## Executive summary

- C is numerically equivalent to B at float64 precision in the focused
  validation: randomized triangular spectra reached relative errors of about
  `2.60e-15` for scalar and `2.35e-15` for vector transforms.
- On CUDA at the maximum tested 0.5° grid (`361×720`), batch-1 forward time
  was `1.549 / 1.063 / 0.687 ms/frame` for scalar A/B/C and
  `5.109 / 2.889 / 2.479 ms/frame` for vector A/B/C.
- On one-thread CPU scalar analysis at 0.5°, DUCC/A/B/C measured
  `24.994 / 39.287 / 25.654 / 21.396 ms/frame` at batch 1 and
  `23.044 / 17.355 / 11.778 / 8.284 ms/frame` at batch 4.
- At batch 64 on the 0.5° grid, Torch CPU-to-GPU speedups were
  `114.3 / 120.4 / 95.3×` for scalar A/B/C and
  `100.3 / 109.6 / 92.1×` for vector A/B/C. Relative to DUCC CPU, the
  corresponding scalar GPU ratios were `149.2 / 154.3 / 552.0×`.
- C requires a larger persistent projection than B and has a material
  construction cost. At 0.5° float32, the C projection occupies
  `374,284,800` bytes for scalar and `748,569,600` bytes for vector.
- The isolated CUDA forward/backward run deliberately stops at 1.0° to avoid
  the OOM path. The 0.5° case is forward-only in the final CUDA matrix.

## Provenance and scope

| Field | Value |
| --- | --- |
| Benchmark date | `2026-09-17` |
| Torch source | `f4cc65515d79d66ff7ab1f02d53518e0b063fc99` |
| Dense reference base | `3278fb669483d04537aa60c87fb0754d989e2e70` |
| PyTorch | `2.14.0+cu130` |
| CUDA device | NVIDIA GeForce RTX 5070, 12,227 MiB |
| DUCC | `0.41.0` |
| CPU | AMD Ryzen 9 5950X 16-Core Processor |
| CUDA dtype | `float32` |
| CPU Torch comparisons | `float32`, 1, 4, and 16 Torch threads |
| CPU DUCC comparisons | `float32`, 1, 4, and 16 DUCC threads, scalar analysis |
| Timing statistic | median of warmed repetitions |

The Torch grid limits use the exclusive convention `lmax=mmax=nlat-1`.
For the DUCC comparison, DUCC uses the common inclusive limit `nlat-2`
because its CC path rejects the highest Torch band; the Torch A/B/C outputs
are still configured with the requested exclusive limit.

## Resolution matrix

| Angular spacing | Grid (`nlat×nlon`) | Torch exclusive `lmax=mmax` |
| ---: | ---: | ---: |
| 2.5° | `73×144` | 72 |
| 2.0° | `91×180` | 90 |
| 1.5° | `121×240` | 120 |
| 1.0° | `181×360` | 180 |
| 0.5° | `361×720` | 360 |

## CUDA forward performance

Values are median milliseconds per frame. Each cell is `A / B / C`. The
batch-16/64 cells use the memory-safe `--skip-contractions` mode described
below.

| Spacing | Grid | Scalar b1 | Scalar b4 | Scalar b16 | Scalar b64 | Vector b1 | Vector b4 | Vector b16 | Vector b64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5° | `73×144` | `0.397 / 0.516 / 0.111` | `0.101 / 0.130 / 0.028` | `0.025 / 0.034 / 0.007` | `0.007 / 0.009 / 0.002` | `0.737 / 0.872 / 0.327` | `0.197 / 0.221 / 0.085` | `0.047 / 0.056 / 0.022` | `0.012 / 0.014 / 0.005` |
| 2.0° | `91×180` | `0.394 / 0.506 / 0.110` | `0.093 / 0.127 / 0.029` | `0.023 / 0.032 / 0.008` | `0.006 / 0.008 / 0.002` | `0.731 / 0.849 / 0.327` | `0.184 / 0.217 / 0.085` | `0.045 / 0.053 / 0.021` | `0.014 / 0.013 / 0.006` |
| 1.5° | `121×240` | `0.388 / 0.521 / 0.109` | `0.093 / 0.125 / 0.034` | `0.027 / 0.032 / 0.009` | `0.009 / 0.008 / 0.003` | `0.732 / 0.850 / 0.328` | `0.179 / 0.210 / 0.085` | `0.050 / 0.055 / 0.022` | `0.028 / 0.020 / 0.009` |
| 1.0° | `181×360` | `0.401 / 0.506 / 0.145` | `0.117 / 0.122 / 0.056` | `0.042 / 0.043 / 0.015` | `0.030 / 0.026 / 0.006` | `0.883 / 0.855 / 0.376` | `0.229 / 0.227 / 0.163` | `0.118 / 0.093 / 0.046` | `0.087 / 0.080 / 0.029` |
| 0.5° | `361×720` | `1.549 / 1.063 / 0.687` | `0.396 / 0.274 / 0.246` | `0.233 / 0.169 / 0.066` | `0.167 / 0.161 / 0.045` | `5.109 / 2.889 / 2.479` | `1.371 / 0.764 / 0.946` | `0.797 / 0.575 / 0.280` | `0.456 / 0.392 / 0.197` |

At batch 1, C is the fastest Torch implementation across the tested scalar
and vector CUDA cells. At 0.5°, C remains fastest for scalar batches 1, 4, 16,
and 64; B is fastest for vector batches 4, 16, and 64, while C is fastest for
vector batch 1.

## Isolated CUDA forward/backward performance

This run uses batch 1 and stops at 1.0°. Values are milliseconds per frame;
each cell is `A / B / C`.

| Spacing | Grid | Scalar forward/backward | Vector forward/backward |
| ---: | ---: | ---: | ---: |
| 2.5° | `73×144` | `1.315 / 1.566 / 0.577` | `2.715 / 2.950 / 1.184` |
| 2.0° | `91×180` | `1.270 / 1.562 / 0.485` | `2.584 / 2.977 / 1.173` |
| 1.5° | `121×240` | `1.119 / 1.489 / 0.522` | `2.471 / 2.890 / 1.084` |
| 1.0° | `181×360` | `1.088 / 1.605 / 0.526` | `2.522 / 2.595 / 1.378` |

## CPU DUCC versus Torch A/B/C

This is scalar CC analysis with one CPU thread. Values are median
milliseconds per frame; each cell is `DUCC / A / B / C`.

| Spacing | Grid | Batch 1 | Batch 4 | Batch 16 | Batch 64 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5° | `73×144` | `0.911 / 0.403 / 0.374 / 0.265` | `0.939 / 0.179 / 0.175 / 0.094` | `0.953 / 0.131 / 0.129 / 0.051` | `0.955 / 0.416 / 0.340 / 0.041` |
| 2.0° | `91×180` | `1.388 / 0.576 / 0.505 / 0.387` | `1.422 / 0.293 / 0.269 / 0.161` | `1.469 / 0.283 / 0.235 / 0.091` | `1.498 / 0.576 / 0.601 / 0.081` |
| 1.5° | `121×240` | `2.483 / 1.853 / 0.911 / 0.985` | `2.485 / 0.869 / 0.544 / 0.332` | `2.632 / 0.568 / 0.464 / 0.205` | `2.673 / 1.186 / 0.983 / 0.160` |
| 1.0° | `181×360` | `5.573 / 6.664 / 3.307 / 3.643` | `5.608 / 3.061 / 1.940 / 1.308` | `6.037 / 1.696 / 1.468 / 0.663` | `6.020 / 2.905 / 2.483 / 0.511` |
| 0.5° | `361×720` | `24.994 / 39.287 / 25.654 / 21.396` | `23.043 / 17.355 / 11.778 / 8.284` | `24.657 / 15.928 / 11.795 / 4.392` | `24.903 / 18.943 / 19.191 / 4.378` |

At 0.5°, C is the fastest one-thread Torch CPU path for batches 4, 16, and
64; B is fastest at batch 1. The one-thread CPU-to-GPU ratios below use the
same batch size and transform on both sides.

## Batch 1/4/16/64 device comparison

The complete batch comparison uses float32 inference-only forward timing for
all three Torch implementations on both CPU and CUDA. The DUCC comparison
remains scalar-only because this DUCC benchmark path is scalar CC analysis.
Values are median milliseconds per frame; each A/B/C cell is ordered as A, B,
C.

At the maximum currently included 0.5° grid:

| Transform | Batch | Torch CPU 1T A/B/C | Torch GPU A/B/C | DUCC CPU 1T |
| --- | ---: | ---: | ---: | ---: |
| Scalar | 1 | `39.407 / 25.879 / 19.919` | `1.549 / 1.063 / 0.687` | `24.994` |
| Scalar | 4 | `16.737 / 12.137 / 8.120` | `0.396 / 0.274 / 0.246` | `23.043` |
| Scalar | 16 | `16.096 / 16.362 / 4.433` | `0.233 / 0.169 / 0.066` | `24.657` |
| Scalar | 64 | `19.075 / 19.434 / 4.298` | `0.167 / 0.161 / 0.045` | `24.903` |
| Vector | 1 | `145.874 / 92.379 / 80.021` | `5.109 / 2.889 / 2.479` | — |
| Vector | 4 | `67.687 / 47.881 / 32.313` | `1.371 / 0.764 / 0.946` | — |
| Vector | 16 | `49.716 / 36.790 / 17.581` | `0.797 / 0.575 / 0.280` | — |
| Vector | 64 | `45.743 / 42.934 / 18.103` | `0.456 / 0.392 / 0.197` | — |

### Torch CPU-to-GPU speedup

Speedup is one-thread Torch CPU time divided by Torch GPU time; values above
`1×` mean that CUDA is faster. Each cell is A/B/C.

#### Scalar

| Spacing | Grid | Batch 1 | Batch 4 | Batch 16 | Batch 64 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5° | `73×144` | `1.06 / 0.79 / 2.26×` | `1.94 / 1.42 / 3.36×` | `7.07 / 4.02 / 6.91×` | `66.37 / 38.03 / 20.29×` |
| 2.0° | `91×180` | `1.46 / 1.06 / 3.82×` | `3.27 / 2.20 / 5.23×` | `12.20 / 6.71 / 11.49×` | `104.13 / 76.88 / 34.87×` |
| 1.5° | `121×240` | `3.10 / 1.58 / 8.53×` | `8.86 / 3.63 / 10.12×` | `24.16 / 15.60 / 22.57×` | `115.88 / 138.57 / 60.01×` |
| 1.0° | `181×360` | `14.97 / 8.48 / 24.41×` | `24.24 / 16.37 / 21.91×` | `41.09 / 35.12 / 44.43×` | `117.85 / 136.05 / 82.63×` |
| 0.5° | `361×720` | `25.43 / 24.35 / 28.98×` | `42.23 / 44.36 / 32.96×` | `68.96 / 96.89 / 66.70×` | `114.29 / 120.42 / 95.27×` |

#### Vector

| Spacing | Grid | Batch 1 | Batch 4 | Batch 16 | Batch 64 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5° | `73×144` | `1.38 / 0.96 / 3.00×` | `2.47 / 1.84 / 4.17×` | `10.08 / 5.69 / 8.79×` | `72.98 / 53.05 / 31.68×` |
| 2.0° | `91×180` | `2.96 / 1.47 / 4.80×` | `6.60 / 3.15 / 7.33×` | `19.30 / 10.15 / 18.85×` | `107.25 / 99.65 / 61.03×` |
| 1.5° | `121×240` | `10.50 / 3.22 / 13.66×` | `18.26 / 7.30 / 19.04×` | `38.12 / 25.30 / 39.08×` | `122.00 / 163.27 / 83.84×` |
| 1.0° | `181×360` | `26.63 / 19.04 / 37.54×` | `42.26 / 26.93 / 31.16×` | `63.55 / 44.31 / 59.36×` | `107.41 / 87.92 / 73.20×` |
| 0.5° | `361×720` | `28.55 / 31.98 / 32.28×` | `49.39 / 62.69 / 34.16×` | `62.34 / 63.97 / 62.77×` | `100.30 / 109.58 / 92.11×` |

### DUCC CPU-to-Torch-GPU ratio

This is a cross-backend comparison, not a DUCC GPU measurement: DUCC is the
CPU numerator and Torch A/B/C is the CUDA denominator. It is reported only for
scalar analysis. Each cell is DUCC/A, DUCC/B, DUCC/C.

| Spacing | Grid | Batch 1 | Batch 4 | Batch 16 | Batch 64 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5° | `73×144` | `2.29 / 1.77 / 8.18×` | `9.32 / 7.21 / 33.86×` | `38.30 / 27.96 / 131.00×` | `145.59 / 106.66 / 482.78×` |
| 2.0° | `91×180` | `3.53 / 2.74 / 12.63×` | `15.21 / 11.19 / 49.60×` | `62.63 / 46.28 / 192.48×` | `245.32 / 191.68 / 675.86×` |
| 1.5° | `121×240` | `6.39 / 4.77 / 22.81×` | `26.82 / 19.84 / 72.47×` | `98.36 / 81.72 / 305.37×` | `281.48 / 321.69 / 954.63×` |
| 1.0° | `181×360` | `13.88 / 11.00 / 38.43×` | `48.00 / 46.08 / 100.86×` | `143.72 / 141.42 / 415.62×` | `198.13 / 231.49 / 971.06×` |
| 0.5° | `361×720` | `16.13 / 23.52 / 36.36×` | `58.15 / 84.23 / 93.54×` | `105.63 / 146.02 / 370.97×` | `149.20 / 154.31 / 552.02×` |

### CPU thread scaling at 0.5°

These tables use the same five-case matrix's maximum 0.5° grid. Torch cells
are A/B/C milliseconds per frame; DUCC cells are scalar milliseconds per
frame. All values are inference-only medians.

#### Scalar

| Batch | Torch 1T A/B/C | Torch 4T A/B/C | Torch 16T A/B/C | DUCC 1T / 4T / 16T |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `39.407 / 25.879 / 19.919` | `22.390 / 12.583 / 19.140` | `20.758 / 10.977 / 20.403` | `24.994 / 20.079 / 18.624` |
| 4 | `16.737 / 12.137 / 8.120` | `8.054 / 5.631 / 4.460` | `7.605 / 4.203 / 4.531` | `23.043 / 20.049 / 19.263` |
| 16 | `16.096 / 16.362 / 4.433` | `6.102 / 6.545 / 2.113` | `5.287 / 5.365 / 2.147` | `24.657 / 20.006 / 19.482` |
| 64 | `19.075 / 19.434 / 4.298` | `6.990 / 7.462 / 2.126` | `5.616 / 5.724 / 1.532` | `24.903 / 22.788 / 21.265` |

#### Vector

| Batch | Torch 1T A/B/C | Torch 4T A/B/C | Torch 16T A/B/C |
| ---: | ---: | ---: | ---: |
| 1 | `145.874 / 92.379 / 80.021` | `83.733 / 44.008 / 73.015` | `84.992 / 39.788 / 84.317` |
| 4 | `67.687 / 47.881 / 32.313` | `30.238 / 19.416 / 18.884` | `26.359 / 15.725 / 19.175` |
| 16 | `49.716 / 36.790 / 17.581` | `18.928 / 14.290 / 9.678` | `15.607 / 11.158 / 7.679` |
| 64 | `45.743 / 42.934 / 18.103` | `15.934 / 16.043 / 9.178` | `12.378 / 12.047 / 6.024` |

## 0.5° CUDA batch scaling

This scaling probe keeps the maximum tested grid at `361×720` and doubles the
batch from 64 through 1024. It uses `--skip-contractions`. Values are median
milliseconds per frame; `OOM` is an observed CUDA allocator failure.

| Transform | Implementation | b64 | b128 | b256 | b512 | b1024 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Scalar | A dense | `0.167` | `0.167` | `0.170` | `0.171` | OOM |
| Scalar | B runtime fold | `0.161` | `0.162` | `0.165` | `0.166` | OOM |
| Scalar | C precomputed | `0.045` | `0.045` | `0.046` | `0.046` | `0.053` |
| Vector | A dense | `0.456` | `0.460` | `0.464` | OOM | — |
| Vector | B runtime fold | `0.392` | `0.395` | `0.399` | OOM | — |
| Vector | C precomputed | `0.197` | `0.198` | `0.200` | `0.201` | `0.231` |

At this grid, C is the only tested implementation that reaches batch 1024 for
both scalar and vector transforms. A and B fail at scalar batch 1024; A and B
fail at vector batch 512. The vector C batch-512/1024 runs completed despite
the corresponding A/B failures. A 0.25° B-only attempt was started after this
scaling probe, but the container's host-memory OOM counter incremented and no
CUDA allocator traceback was emitted; no JSON was written. It is intentionally
skipped, leaving 0.5° as the maximum completed resolution.

The batch-16/64 Torch JSONs set `contraction_diagnostics` to `false`. This
only skips the optional dense BMM diagnostic, whose intentional batch
expansion is not part of the SHT forward path and attempted a 22.3 GiB
allocation at batch 64. The A/B/C transform timings and memory measurements
remain enabled.

## Accuracy and construction trade-offs

The C effective projection is complex-valued. That is expected from the
folding construction and is not treated as a numerical failure. Direct
float64 validation gave:

| Check | Scalar | Vector |
| --- | ---: | ---: |
| Maximum absolute C-versus-B error on the 73×144 canonical mode sweep | `1.87e-14` | `3.99e-16` |
| Randomized triangular-spectrum C-versus-B relative L2 error | `2.60e-15` | `2.35e-15` |

At 0.5° float32, the persistent module-buffer sizes were:

| Transform | A bytes | B bytes | C bytes |
| --- | ---: | ---: | ---: |
| Scalar | `373,766,760` | `187,151,404` | `374,284,800` |
| Vector | `747,533,160` | `374,293,804` | `748,569,600` |

C construction at 0.5° took approximately `4.589 s` for scalar and
`12.406 s` for vector in the CUDA harness. Separate row-chunked CPU probes
reported peak RSS of approximately `3.42 GiB` for C-only scalar,
`6.59 GiB` for C-only vector, and `6.91 GiB` for the complete scalar A/B/C
run. These probes are diagnostic memory measurements, not steady-state
latency measurements.

The small CUDA `torch.compile` contraction check at `73×144` compiled all
three A/B/C implementations for scalar and vector forward/backward execution;
the recorded compiled-versus-eager relative errors were zero.

## Reproduction and archived data

The research scripts are:

- [Torch A/B/C benchmark](../scripts/torch_abc_benchmark.py)
- [Torch/DUCC CPU comparison](../scripts/torch_ducc_abc_cpu_benchmark.py)
- [Precomputed-projection prototype](../scripts/torch_precomputed_projection_prototype.py)

The complete raw JSON outputs are archived beside this report:

- [CUDA scalar forward](torch-abc-half-degree-grid-cuda-scalar-final.json)
- [CUDA vector forward](torch-abc-half-degree-grid-cuda-vector-final.json)
- [CUDA scalar isolated training](torch-abc-half-degree-grid-cuda-scalar-train-isolated.json)
- [CUDA vector isolated training](torch-abc-half-degree-grid-cuda-vector-train-isolated.json)
- [CUDA A/B/C batch 16/64](torch-abc-half-degree-grid-cuda-b16-b64.json)
- [CUDA 0.5° scalar batch 128](torch-abc-05-degree-cuda-scalar-b128.json)
- [CUDA 0.5° scalar batch 256](torch-abc-05-degree-cuda-scalar-b256.json)
- [CUDA 0.5° scalar batch 512](torch-abc-05-degree-cuda-scalar-b512.json)
- [CUDA 0.5° scalar C batch 1024](torch-abc-05-degree-cuda-scalar-b1024-c.json)
- [CUDA 0.5° vector batch 128](torch-abc-05-degree-cuda-vector-b128.json)
- [CUDA 0.5° vector batch 256](torch-abc-05-degree-cuda-vector-b256.json)
- [CUDA 0.5° vector C batch 512](torch-abc-05-degree-cuda-vector-b512-c.json)
- [CUDA 0.5° vector C batch 1024](torch-abc-05-degree-cuda-vector-b1024-c.json)
- [CUDA 0.5° batch-scaling status manifest](torch-abc-05-degree-cuda-batch-scaling.json)
- [CPU A/B/C one thread, batch 1/4](torch-abc-half-degree-grid-cpu-t1-b1-b4.json)
- [CPU A/B/C one thread, batch 16/64](torch-abc-half-degree-grid-cpu-b16-b64.json)
- [CPU A/B/C four threads, batch 1/4/16/64](torch-abc-half-degree-grid-cpu-t4-b1-b4-b16-b64.json)
- [CPU A/B/C sixteen threads, batch 1/4/16/64](torch-abc-half-degree-grid-cpu-t16-b1-b4-b16-b64.json)
- [CUDA compiled contractions](torch-abc-73-cuda-compile-contractions-final.json)
- [CPU DUCC/A/B/C batch 1](torch-ducc-abc-half-degree-grid-cpu-final.json)
- [CPU DUCC/A/B/C batch 4](torch-ducc-abc-half-degree-grid-cpu-b4-final.json)
- [CPU DUCC/A/B/C batch 16](torch-ducc-abc-half-degree-grid-cpu-b16.json)
- [CPU DUCC/A/B/C batch 64](torch-ducc-abc-half-degree-grid-cpu-b64.json)
- [CPU DUCC/A/B/C four threads, batch 1](torch-ducc-abc-half-degree-grid-cpu-t4-b1.json)
- [CPU DUCC/A/B/C four threads, batch 4](torch-ducc-abc-half-degree-grid-cpu-t4-b4.json)
- [CPU DUCC/A/B/C four threads, batch 16](torch-ducc-abc-half-degree-grid-cpu-t4-b16.json)
- [CPU DUCC/A/B/C four threads, batch 64](torch-ducc-abc-half-degree-grid-cpu-t4-b64.json)
- [CPU DUCC/A/B/C sixteen threads, batch 1](torch-ducc-abc-half-degree-grid-cpu-t16-b1.json)
- [CPU DUCC/A/B/C sixteen threads, batch 4](torch-ducc-abc-half-degree-grid-cpu-t16-b4.json)
- [CPU DUCC/A/B/C sixteen threads, batch 16](torch-ducc-abc-half-degree-grid-cpu-t16-b16.json)
- [CPU DUCC/A/B/C sixteen threads, batch 64](torch-ducc-abc-half-degree-grid-cpu-t16-b64.json)
- [CPU float64 73×144 check](torch-ducc-abc-73-cpu-f64-final.json)
- [Derived batch 1/4/16/64 speedups](torch-abc-batch1-4-16-64-speedups.json)

Representative commands, using the exact five-case matrix, are:

```bash
uv run python scripts/torch_abc_benchmark.py \
  --device cuda \
  --cases 73x144x72,91x180x90,121x240x120,181x360x180,361x720x360 \
  --dtypes float32 --transforms scalar,vector --batches 1,4 \
  --warmup 2 --repeat 5 \
  --output results/torch-abc-half-degree-grid-cuda.json

uv run python scripts/torch_abc_benchmark.py \
  --device cuda --cases 73x144x72,91x180x90,121x240x120,181x360x180,361x720x360 \
  --dtypes float32 --transforms scalar,vector --batches 16,64 \
  --warmup 2 --repeat 5 --skip-contractions \
  --output results/torch-abc-half-degree-grid-cuda-b16-b64.json

uv run python scripts/torch_abc_benchmark.py \
  --device cpu --cases 73x144x72,91x180x90,121x240x120,181x360x180,361x720x360 \
  --dtypes float32 --transforms scalar,vector --batches 16,64 \
  --threads 1 --warmup 2 --repeat 5 --skip-contractions \
  --output results/torch-abc-half-degree-grid-cpu-b16-b64.json

uv run python scripts/torch_ducc_abc_cpu_benchmark.py \
  --cases 73x144x72,91x180x90,121x240x120,181x360x180,361x720x360 \
  --dtypes float32 --batch 1 --threads 1 --warmup 2 --repeat 5 \
  --output results/torch-ducc-abc-half-degree-grid-cpu.json
```

The C prototype remains research-only; no production `torch-harmonics` source
files were changed by this benchmark run. The report intentionally omits the
older finer-resolution matrix rather than presenting it as part of this
memory-safe result set.
