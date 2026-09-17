# CPU DUCC/Torch equiangular analysis benchmark

This report compares the resampled equiangular analysis in the current
`torch-harmonics` checkout with direct CPU DUCC on the two requested
Clenshaw--Curtis grids. Each measurement uses one scalar frame, one CPU
thread, and the largest recoverable triangular degree range for that grid.

## Provenance

| Field | Value |
| --- | --- |
| Benchmark date | `2026-09-16` |
| Torch source | `f4cc65515d79d66ff7ab1f02d53518e0b063fc99` |
| DUCC | `0.41.0` |
| PyTorch | `2.14.0+cu130` |
| Device | CPU |
| Threads | `1` for DUCC and Torch |
| Batch | `1` frame |
| Timing | two warmups, five repetitions, `0.2 s` minimum calibration time; median |

The `2.5°` case is `73×144` with maximum degree `71` (`lmax=72` in
Torch's exclusive convention). The `0.25°` case is `721×1440` with maximum
degree `719` (`lmax=720` in Torch's exclusive convention). The latter uses
analysis beyond the direct quadrature limit through resampling/folding.

## Median transform time

The ratio is Torch divided by DUCC; values above `1` mean Torch took longer
for that operation. Timed DUCC analysis returns its native packed coefficient
array, while Torch returns its native rectangular coefficient array; conversion
is excluded from both timed kernels.

| Dtype | Grid | Operation | DUCC (ms) | Torch (ms) | Torch / DUCC |
| --- | --- | --- | ---: | ---: | ---: |
| float32 | 73×144 (2.5°) | analysis | `0.257` | `0.375` | `1.46×` |
| float32 | 73×144 (2.5°) | synthesis | `0.090` | `0.219` | `2.44×` |
| float32 | 721×1440 (0.25°) | analysis | `35.251` | `178.487` | `5.06×` |
| float32 | 721×1440 (0.25°) | synthesis | `19.165` | `143.198` | `7.47×` |
| float64 | 73×144 (2.5°) | analysis | `0.261` | `0.535` | `2.05×` |
| float64 | 73×144 (2.5°) | synthesis | `0.092` | `0.297` | `3.23×` |
| float64 | 721×1440 (0.25°) | analysis | `37.470` | `325.571` | `8.69×` |
| float64 | 721×1440 (0.25°) | synthesis | `19.816` | `225.816` | `11.40×` |

Torch setup and registered-buffer storage were:

| Dtype | Grid | Setup (s) | Analysis buffers | Synthesis buffers |
| --- | --- | ---: | ---: | ---: |
| float32 | 73×144 | `0.026` | `1.45 MiB` | `1.44 MiB` |
| float32 | 721×1440 | `11.234` | `1,425.8 MiB` | `1,425.8 MiB` |
| float64 | 73×144 | `0.020` | `2.89 MiB` | `2.89 MiB` |
| float64 | 721×1440 | `6.903` | `2,851.6 MiB` | `2,851.6 MiB` |

## Cross-backend checks

Torch synthesis used the same canonical triangular coefficients as DUCC;
Torch analysis used the DUCC-synthesized map. The following are Torch-versus-
DUCC relative L2 and maximum absolute errors:

| Dtype | Grid | Operation | Relative L2 | Maximum absolute |
| --- | --- | --- | ---: | ---: |
| float32 | 73×144 | analysis | `1.96e-7` | `8.43e-7` |
| float32 | 73×144 | synthesis | `1.38e-7` | `1.91e-5` |
| float32 | 721×1440 | analysis | `2.94e-7` | `2.10e-6` |
| float32 | 721×1440 | synthesis | `2.44e-7` | `3.36e-4` |
| float64 | 73×144 | analysis | `7.57e-15` | `8.29e-14` |
| float64 | 73×144 | synthesis | `1.52e-14` | `5.60e-12` |
| float64 | 721×1440 | analysis | `6.97e-14` | `2.63e-12` |
| float64 | 721×1440 | synthesis | `2.89e-13` | `2.11e-9` |

## Reproduction

```bash
uv run python scripts/torch_ducc_resampled_benchmark.py \
  --dtype float32,float64 \
  --warmup 2 --repeat 5 --min-time 0.2 \
  --output results/torch-ducc-resampled-2p5-0p25
```

The [benchmark script](../scripts/torch_ducc_resampled_benchmark.py),
[JSON result](torch-ducc-resampled-2p5-0p25.json), and
[CSV result](torch-ducc-resampled-2p5-0p25.csv) contain the complete run.
No other resolution is included in this result set.

## Method reference

M. Reinecke, S. Belkner, J. Carron, “Improved cosmic microwave background
(de-)lensing using general spherical harmonic transforms,” *Astronomy &
Astrophysics* 678, A165 (2023), Appendix A.

DOI: [10.1051/0004-6361/202346717](https://doi.org/10.1051/0004-6361/202346717)
