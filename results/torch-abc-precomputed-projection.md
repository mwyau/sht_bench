# PyTorch A/B/C precomputed-projection benchmark

This report records the final memory-safe matrix for the PyTorch equiangular
SHT projection experiment. It uses 0.5° resolution steps and stops at 0.5°:
2.5°, 2.0°, 1.5°, 1.0°, and 0.5°. No 0.25° or finer case is included.

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
| CPU DUCC comparison | `float32`, one thread, scalar analysis |
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

Values are median milliseconds per frame. Each cell is `A / B / C`.

| Spacing | Grid | Scalar batch 1 | Scalar batch 4 | Vector batch 1 | Vector batch 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5° | `73×144` | `0.397 / 0.516 / 0.111` | `0.101 / 0.130 / 0.028` | `0.737 / 0.872 / 0.327` | `0.197 / 0.221 / 0.085` |
| 2.0° | `91×180` | `0.394 / 0.506 / 0.110` | `0.093 / 0.127 / 0.029` | `0.731 / 0.849 / 0.327` | `0.184 / 0.217 / 0.085` |
| 1.5° | `121×240` | `0.388 / 0.521 / 0.109` | `0.093 / 0.125 / 0.034` | `0.732 / 0.850 / 0.328` | `0.179 / 0.210 / 0.085` |
| 1.0° | `181×360` | `0.401 / 0.506 / 0.145` | `0.117 / 0.122 / 0.056` | `0.883 / 0.855 / 0.376` | `0.229 / 0.227 / 0.163` |
| 0.5° | `361×720` | `1.549 / 1.063 / 0.687` | `0.396 / 0.274 / 0.246` | `5.109 / 2.889 / 2.479` | `1.371 / 0.764 / 0.946` |

At batch 1, C is the fastest Torch implementation across the tested scalar
and vector CUDA cells. At batch 4, B is faster for the 0.5° vector case,
while C remains faster for the 0.5° scalar case.

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

| Spacing | Grid | Batch 1 | Batch 4 |
| ---: | ---: | ---: | ---: |
| 2.5° | `73×144` | `0.911 / 0.403 / 0.374 / 0.265` | `0.939 / 0.179 / 0.175 / 0.094` |
| 2.0° | `91×180` | `1.388 / 0.576 / 0.505 / 0.387` | `1.422 / 0.293 / 0.269 / 0.161` |
| 1.5° | `121×240` | `2.483 / 1.853 / 0.911 / 0.985` | `2.485 / 0.869 / 0.544 / 0.332` |
| 1.0° | `181×360` | `5.573 / 6.664 / 3.307 / 3.643` | `5.608 / 3.061 / 1.940 / 1.308` |
| 0.5° | `361×720` | `24.994 / 39.287 / 25.654 / 21.396` | `23.044 / 17.355 / 11.778 / 8.284` |

At 0.5°, C is the fastest of the four batch-1 and batch-4 CPU paths. The
batch-1 1.0° result is the exception where B is slightly faster than C.

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
- [CUDA compiled contractions](torch-abc-73-cuda-compile-contractions-final.json)
- [CPU DUCC/A/B/C batch 1](torch-ducc-abc-half-degree-grid-cpu-final.json)
- [CPU DUCC/A/B/C batch 4](torch-ducc-abc-half-degree-grid-cpu-b4-final.json)
- [CPU float64 73×144 check](torch-ducc-abc-73-cpu-f64-final.json)

Representative commands, using the exact five-case matrix, are:

```bash
uv run python scripts/torch_abc_benchmark.py \
  --device cuda \
  --cases 73x144x72,91x180x90,121x240x120,181x360x180,361x720x360 \
  --dtypes float32 --transforms scalar,vector --batches 1,4 \
  --warmup 2 --repeat 5 \
  --output results/torch-abc-half-degree-grid-cuda.json

uv run python scripts/torch_ducc_abc_cpu_benchmark.py \
  --cases 73x144x72,91x180x90,121x240x120,181x360x180,361x720x360 \
  --dtypes float32 --batch 1 --threads 1 --warmup 2 --repeat 5 \
  --output results/torch-ducc-abc-half-degree-grid-cpu.json
```

The C prototype remains research-only; no production `torch-harmonics` source
files were changed by this benchmark run. The report intentionally omits the
older finer-resolution matrix rather than presenting it as part of this
memory-safe result set.
