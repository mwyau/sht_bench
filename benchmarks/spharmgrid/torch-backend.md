# ERA5 Torch backend benchmark

This repository-only benchmark compares the optional `spharmgrid.torch` backend
with the Xarray/DUCC CPU path on the real ERA5 2024 mean-sea-level-pressure
field. It measures eager, inference-mode throughput and does not introduce a
new spherical harmonic transform implementation: spharmgrid delegates the
numerical transforms to `ducc0` and `torch-harmonics`.

The measurements correspond to the spharmgrid Torch numerical implementation
finalized at `b4cbd9f`. Later spharmgrid commits changed CI, packaging,
capability diagnostics, documentation, and tests without changing the
benchmarked GL numerical kernels.

## Benchmark provenance

| Field | Value |
| --- | --- |
| Project | `spharmgrid` |
| Source repository | https://github.com/mwyau/spharmgrid |
| Feature | Optional `spharmgrid.torch` backend |
| Numerical implementation commit | `b4cbd9f88337f4e04f42a971ea06decd69297aee` (`b4cbd9f`) |
| Reviewed feature state | `0aaa7e35bde7758f41d2ad711577dfc59be34b42` (`0aaa7e3`) |
| Benchmark date | `2026-09-16` |

## Result summary

Float32 is the primary throughput comparison. At batch size 128 on the full
year, reusable CUDA compute measured 0.127 ms/frame for filtering, 0.063
ms/frame for regridding, and 0.014 ms/frame for the pressure-gradient path.
Including ordinary pageable host-to-device transfer raised those figures to
0.375, 0.310, and 0.015 ms/frame, respectively. All three operations
completed for one frame, January, and all 1,464 frames.

Float64 also completed the full year. Full-year reusable CUDA compute was
0.396, 0.199, and 0.022 ms/frame for filtering, regridding, and the gradient
path. Relative to float32, the consumer RTX 5070 was 3.1 times slower for
filtering, 3.1 times slower for regridding, and 1.5 times slower for the
gradient path. This reflects consumer-GPU float64 throughput; it is not a
spharmgrid limitation.

The native-grid float32 CUDA probe tested batch sizes 32, 64, 128, 256, 384,
512, 576, and 640. The final probe reached 576 frames successfully, while 640
frames failed with CUDA out-of-memory. Batch 128 remains well below the
observed OOM boundary and was selected for the complete comparisons because it
was at the measured compute-time minimum while leaving substantial VRAM
headroom on this 12 GB GPU.

## Data and workloads

The inspected data directory was
`../PyStormTracker-Reference-Data/era5-2024/`. The benchmark used:

- `ERA5_mslp_6hr_2024_DET.nc`, variable `msl`, with CF standard name
  `air_pressure_at_mean_sea_level`;
- `ERA5_vo850_6hr_2024_DET.nc`, variable `vo`, with CF standard name
  `atmosphere_relative_vorticity`, which was inspected but not used as a
  substitute for wind components.

The mean-sea-level-pressure file contains 1,464 timestamps from
`2024-01-01 00:00` through `2024-12-31 18:00`, at six-hour intervals. January
contains 124 frames. Its native grid is descending-latitude full GL with 640
latitudes and 1,280 longitudes, from 0 to 359.718994 degrees east. The target
for regridding and the differential path is descending-latitude GL T42, with
43 latitudes and 86 longitudes.

No matching `u`/`v` files were present in the inspected ERA5 directory. The
third path therefore uses the real `msl` field and computes its eastward and
northward pressure gradients after preparing the T42 working field. It does
not fabricate a wind pair or report vector kinematics for unobserved data.

The exact workloads were:

| Workload     | Frames | Batch size | Batches | Last batch |
| ------------ | -----: | ---------: | ------: | ---------: |
| One frame    |      1 |          1 |       1 |          1 |
| January 2024 |    124 |        128 |       1 |        124 |
| Full 2024    |  1,464 |        128 |      12 |         56 |

January and full-year data were streamed in time batches. No complete month or
year was moved to CUDA as one tensor, and outputs were consumed after
synchronized execution without retaining the batch sequence.

## Environment and method

Measurements were made on 2026-09-16 with:

| Component       | Version or value                                             |
| --------------- | ------------------------------------------------------------ |
| CPU             | AMD Ryzen 9 5950X 16-Core Processor                          |
| GPU             | NVIDIA GeForce RTX 5070, 12,227 MiB reported by `nvidia-smi` |
| CUDA runtime    | 13.0                                                         |
| Python          | 3.12.13                                                      |
| PyTorch         | 2.11.0+cu130                                                 |
| torch-harmonics | 0.9.2+torch2.11.0.cpu                                        |
| ducc0           | 0.41.0                                                       |
| NumPy           | 2.5.3                                                        |
| SciPy           | 1.18.1                                                       |
| Xarray          | 2026.7.0                                                     |

The runner is [torch_backend_era5.py](torch_backend_era5.py). Run it from the
`sht_bench` repository root with a checked-out spharmgrid source tree available
on `PYTHONPATH`; the runner imports that checkout and does not vendor or install
spharmgrid. The main
comparison used one Torch intra-op thread, one Torch inter-op thread, and
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. The controlled DUCC column used
`sht_threads=1`, labelled `DUCC CPU 1T`. A separate `DUCC CPU 16T` comparison
used `sht_threads=16`. This setting controls DUCC's internal threads for each
individual 2-D spherical harmonic transform. Xarray leading-dimension
vectorization still applies that transform independently to each frame, so
this is not a frame-level parallelism measurement. It is reported separately
because it represents this machine's native multithreaded behavior, not a
general maximum-performance claim. Torch CPU thread settings were unchanged
between the comparisons.

Reusable Torch modules were constructed once per operation and reused across
all batches. The primary benchmark ran under `torch.inference_mode()` and did
not use `torch.compile`.

The environment reports the package suffix `torch-harmonics 0.9.2+torch2.11.0.cpu`. The exercised `RealSHT`/`InverseRealSHT` path still
runs through PyTorch tensor kernels on CUDA; the benchmark assertions verify
that CUDA inputs and outputs remain on CUDA.

### Dtype and device contract

Both requested dtypes were used as public input dtypes through the complete CPU
and CUDA paths:

| Requested dtype | Torch CPU input/output | Torch CUDA input/output | Vector outputs |
| --------------- | ---------------------- | ----------------------- | -------------- |
| `float32`       | `float32` / `cpu`      | `float32` / `cuda:0`    | Both checked   |
| `float64`       | `float64` / `cpu`      | `float64` / `cuda:0`    | Both checked   |

The Torch paths preserve the requested dtype. The Xarray/DUCC adapter currently
converts spherical harmonic transform inputs to NumPy `float64`, computes
`complex128` coefficients, and returns `float64`. Thus the float32 tables
compare requested-float32 Torch arithmetic with the `DUCC CPU (float64 internal)` path; the float64 tables are the precision-matched comparison.

For every Torch CUDA call, the runner asserts `input.is_cuda`, `output.is_cuda`,
the requested input dtype, and the requested output dtype. For the gradient
path, both returned components are checked. The permanent CUDA autograd test
also asserts output and backward-gradient dtype preservation for both
`torch.float32` and `torch.float64`; it passed on the actual GPU (`2 passed`).
The float64 CUDA path therefore does not silently convert the transform to
float32.

Compute-only timing transfers the prepared host batch to CUDA before the
timer, then synchronizes before and after the reusable module call. The
separate `CUDA pageable H2D+compute` timing uses ordinary NumPy pageable host
memory through `torch.from_numpy(values).to("cuda")` and includes the copy and
transform. Pinned memory, asynchronous copies, and pipelining were not
benchmarked. The post-timing scalar checksum copy back to the host is excluded.
Reported stream-wall time includes file loading and host preparation and is
therefore not used as the backend latency comparison. GPU memory values are
`torch.cuda.max_memory_allocated()` and
`torch.cuda.max_memory_reserved()`; peak statistics are reset for each
measured CUDA case.

The repetition policy was explicit:

| Workload or case                   | Warmup | Complete repetitions | Reported value |
| ---------------------------------- | ------ | -------------------: | -------------- |
| One frame                          | Yes    |                    5 | Median         |
| January CUDA, compute-only         | Yes    |                    5 | Median         |
| January CUDA, pageable H2D+compute | Yes    |                    5 | Median         |
| January DUCC/Torch CPU             | Yes    |                    1 | Measured pass  |
| Full year, all paths               | Yes    |                    1 | Measured pass  |

January repetitions are complete streamed January executions. The benchmark
does not concatenate fragments or average individual batch timings. The JSON
runner stores each repetition total plus the minimum, maximum, and count.

## CUDA batch-size probe

This probe used one float32 filter batch on the native 640 x 1,280 GL grid.
Compute time excludes host-to-device transfer. The GPU reported 12,378,112,000
bytes of total memory to PyTorch.

| Batch size | Status | Compute ms/frame | Peak allocated MiB | Peak reserved MiB |
| ---------: | ------ | ---------------: | -----------------: | ----------------: |
|         32 | ok     |         0.131372 |                539 |               626 |
|         64 | ok     |         0.126735 |              1,062 |             1,108 |
|        128 | ok     |         0.125587 |              2,106 |             2,508 |
|        256 | ok     |         0.131325 |              4,197 |             4,906 |
|        384 | ok     |         0.131721 |              6,285 |             7,306 |
|        512 | ok     |         0.131682 |              8,374 |             9,710 |
|        576 | ok     |         0.132635 |              9,418 |            10,910 |
|        640 | OOM    |                — |                  — |                 — |

The complete comparison uses batch 128. A supporting full-workload CUDA
scaling comparison for batch sizes 32, 64, and 128 is shown as
`compute ms/frame / pageable H2D+compute ms/frame`:

| Path     | Workload  |      Batch 32 |      Batch 64 |     Batch 128 |
| -------- | --------- | ------------: | ------------: | ------------: |
| Filter   | January   | 0.139 / 0.370 | 0.136 / 0.370 | 0.126 / 0.369 |
| Filter   | Full year | 0.132 / 0.376 | 0.129 / 0.377 | 0.127 / 0.375 |
| Regrid   | January   | 0.079 / 0.314 | 0.072 / 0.307 | 0.063 / 0.308 |
| Regrid   | Full year | 0.073 / 0.315 | 0.066 / 0.306 | 0.063 / 0.310 |
| Gradient | January   | 0.061 / 0.052 | 0.037 / 0.027 | 0.013 / 0.015 |
| Gradient | Full year | 0.056 / 0.059 | 0.024 / 0.026 | 0.014 / 0.015 |

## Float32 primary comparison

Each cell reports `total seconds / frames per second / milliseconds per frame`.
The Torch CPU and CUDA columns use reusable module state. The final column is
CUDA with ordinary pageable host-to-device transfer included. The DUCC CPU
column is the controlled `DUCC CPU (float64 internal) 1T` path with float64
internal transform arithmetic and float64 output, despite the float32 public
input.

| Path     | Workload  |                DUCC CPU 1T |         Torch CPU reusable |         Torch CUDA reusable |   CUDA pageable H2D+compute |
| -------- | --------- | -------------------------: | -------------------------: | --------------------------: | --------------------------: |
| Filter   | One frame |   0.011651 / 85.8 / 11.651 |   0.006350 / 157.5 / 6.350 |  0.000688 / 1,454.0 / 0.688 |  0.000982 / 1,018.7 / 0.982 |
| Filter   | January   |   4.774224 / 26.0 / 38.502 |   3.536772 / 35.1 / 28.522 |  0.015600 / 7,948.7 / 0.126 |  0.045704 / 2,713.1 / 0.369 |
| Filter   | Full year |  53.160455 / 27.5 / 36.312 |  45.277763 / 32.3 / 30.927 |  0.185463 / 7,893.8 / 0.127 |  0.549675 / 2,663.4 / 0.375 |
| Regrid   | One frame |   0.006587 / 151.8 / 6.587 |   0.003446 / 290.2 / 3.446 |  0.000721 / 1,387.3 / 0.721 |    0.001008 / 992.5 / 1.008 |
| Regrid   | January   |   0.587796 / 211.0 / 4.740 |   1.716625 / 72.2 / 13.844 | 0.007763 / 15,972.2 / 0.063 |  0.038222 / 3,244.2 / 0.308 |
| Regrid   | Full year |   7.222403 / 202.7 / 4.933 |  23.423385 / 62.5 / 16.000 | 0.092906 / 15,757.8 / 0.063 |  0.453306 / 3,229.6 / 0.310 |
| Gradient | One frame |   0.003033 / 329.7 / 3.033 | 0.000868 / 1,151.4 / 0.868 |    0.001312 / 762.4 / 1.312 |    0.001375 / 727.1 / 1.375 |
| Gradient | January   | 0.040757 / 3,042.4 / 0.329 | 0.016977 / 7,304.1 / 0.137 | 0.001599 / 77,550.4 / 0.013 | 0.001861 / 66,623.6 / 0.015 |
| Gradient | Full year | 0.518288 / 2,824.7 / 0.354 | 0.200850 / 7,289.0 / 0.137 | 0.021167 / 69,164.5 / 0.014 | 0.022102 / 66,238.6 / 0.015 |

The additional one-frame functional API timings, which include constructing
transform state on each call, were:

| Path     | Torch CPU functional | Torch CUDA functional |
| -------- | -------------------: | --------------------: |
| Filter   |            76.969 ms |             70.785 ms |
| Regrid   |            38.339 ms |             36.495 ms |
| Gradient |            19.081 ms |             19.933 ms |

The reusable and functional columns measure different usage modes. Reusable
modules are the appropriate comparison for streamed month/year throughput;
the functional path pays state-construction cost on every call.

## Float64 comparison

The same batch 128 and workload definitions were run deliberately with
float64. Each cell again reports `total seconds / frames per second / milliseconds per frame`. The `DUCC CPU (float64 internal) 1T` path is
internally precision-matched here: its float64 transform input and output use
float64 arithmetic with complex128 coefficients.

| Path     | Workload  |                DUCC CPU 1T |         Torch CPU reusable |         Torch CUDA reusable |   CUDA pageable H2D+compute |
| -------- | --------- | -------------------------: | -------------------------: | --------------------------: | --------------------------: |
| Filter   | One frame |   0.011743 / 85.2 / 11.743 |   0.012700 / 78.7 / 12.700 |  0.000807 / 1,239.0 / 0.807 |    0.001279 / 781.7 / 1.279 |
| Filter   | January   |   4.111251 / 30.2 / 33.155 |   5.234091 / 23.7 / 42.210 |  0.049206 / 2,520.0 / 0.397 |  0.112600 / 1,101.2 / 0.908 |
| Filter   | Full year |  61.698136 / 23.7 / 42.144 |  66.957507 / 21.9 / 45.736 |  0.579298 / 2,527.2 / 0.396 |  1.357949 / 1,078.1 / 0.928 |
| Regrid   | One frame |   0.006944 / 144.0 / 6.944 |   0.005138 / 194.6 / 5.138 |  0.000737 / 1,357.3 / 0.737 |    0.001274 / 785.2 / 1.274 |
| Regrid   | January   |   1.916445 / 64.7 / 15.455 |   2.541479 / 48.8 / 20.496 |  0.024365 / 5,089.2 / 0.196 |  0.083927 / 1,477.5 / 0.677 |
| Regrid   | Full year |  20.575736 / 71.2 / 14.054 |  32.959072 / 44.4 / 22.513 |  0.290857 / 5,033.4 / 0.199 |  1.211549 / 1,208.4 / 0.828 |
| Gradient | One frame |   0.003151 / 317.4 / 3.151 |   0.001418 / 705.1 / 1.418 |    0.001799 / 556.0 / 1.799 |    0.001631 / 613.3 / 1.631 |
| Gradient | January   | 0.042114 / 2,944.4 / 0.340 | 0.029215 / 4,244.4 / 0.236 | 0.002042 / 60,726.5 / 0.016 | 0.002502 / 49,558.5 / 0.020 |
| Gradient | Full year | 0.491534 / 2,978.4 / 0.336 | 0.376483 / 3,888.6 / 0.257 | 0.032734 / 44,724.8 / 0.022 | 0.034672 / 42,223.9 / 0.024 |

The one-frame float64 functional API timings were:

| Path     | Torch CPU functional | Torch CUDA functional |
| -------- | -------------------: | --------------------: |
| Filter   |            77.200 ms |             70.191 ms |
| Regrid   |            39.000 ms |             35.289 ms |
| Gradient |            20.126 ms |             21.617 ms |

## January CUDA repetition spread

Each row below is the median total over five complete January repetitions,
with the minimum and maximum complete-repetition totals shown for dispersion.
The DUCC and Torch CPU January rows used one measured pass and are not included
in this repetition table.

| Dtype   | Path     | Mode                 | Repetitions | Median s |    Min s |    Max s |
| ------- | -------- | -------------------- | ----------: | -------: | -------: | -------: |
| float32 | Filter   | Compute-only         |           5 | 0.015600 | 0.015576 | 0.015748 |
| float32 | Filter   | Pageable H2D+compute |           5 | 0.045704 | 0.045197 | 0.048817 |
| float32 | Regrid   | Compute-only         |           5 | 0.007763 | 0.007746 | 0.007918 |
| float32 | Regrid   | Pageable H2D+compute |           5 | 0.038222 | 0.037840 | 0.039200 |
| float32 | Gradient | Compute-only         |           5 | 0.001599 | 0.001565 | 0.002087 |
| float32 | Gradient | Pageable H2D+compute |           5 | 0.001861 | 0.001812 | 0.002355 |
| float64 | Filter   | Compute-only         |           5 | 0.049206 | 0.049119 | 0.049254 |
| float64 | Filter   | Pageable H2D+compute |           5 | 0.112600 | 0.109231 | 0.120125 |
| float64 | Regrid   | Compute-only         |           5 | 0.024365 | 0.024090 | 0.024536 |
| float64 | Regrid   | Pageable H2D+compute |           5 | 0.083927 | 0.083770 | 0.092172 |
| float64 | Gradient | Compute-only         |           5 | 0.002042 | 0.001965 | 0.002571 |
| float64 | Gradient | Pageable H2D+compute |           5 | 0.002502 | 0.002372 | 0.002808 |

## Multithreaded DUCC CPU comparison

The main tables intentionally retain the controlled `DUCC CPU 1T` baseline.
This separate section compares it with `DUCC CPU 16T` on the same machine.
Both DUCC columns use float64 internal transform arithmetic; only the DUCC
thread setting differs. Each timing cell is `total seconds / frames per second / milliseconds per frame`; speedup is `1T / 16T`. These measurements describe
this machine and do not imply ideal thread scaling.

### Float32

| Path     | Workload  |                DUCC CPU 1T |               DUCC CPU 16T | Speedup |
| -------- | --------- | -------------------------: | -------------------------: | ------: |
| Filter   | One frame |   0.011651 / 85.8 / 11.651 |   0.011676 / 85.6 / 11.676 |  0.998x |
| Filter   | January   |   4.774224 / 26.0 / 38.502 |   4.701756 / 26.4 / 37.917 |  1.015x |
| Filter   | Full year |  53.160455 / 27.5 / 36.312 |  53.118526 / 27.6 / 36.283 |  1.001x |
| Regrid   | One frame |   0.006587 / 151.8 / 6.587 |   0.006591 / 151.7 / 6.591 |  0.999x |
| Regrid   | January   |   0.587796 / 211.0 / 4.740 |   0.578436 / 214.4 / 4.665 |  1.016x |
| Regrid   | Full year |   7.222403 / 202.7 / 4.933 |   6.934824 / 211.1 / 4.737 |  1.041x |
| Gradient | One frame |   0.003033 / 329.7 / 3.033 |   0.003021 / 331.0 / 3.021 |  1.004x |
| Gradient | January   | 0.040757 / 3,042.4 / 0.329 | 0.040560 / 3,057.2 / 0.327 |  1.005x |
| Gradient | Full year | 0.518288 / 2,824.7 / 0.354 | 0.482853 / 3,032.0 / 0.330 |  1.073x |

### Float64

| Path     | Workload  |                DUCC CPU 1T |               DUCC CPU 16T | Speedup |
| -------- | --------- | -------------------------: | -------------------------: | ------: |
| Filter   | One frame |   0.011743 / 85.2 / 11.743 |   0.011714 / 85.4 / 11.714 |  1.002x |
| Filter   | January   |   4.111251 / 30.2 / 33.155 |   4.741690 / 26.2 / 38.239 |  0.867x |
| Filter   | Full year |  61.698136 / 23.7 / 42.144 |  53.964909 / 27.1 / 36.861 |  1.143x |
| Regrid   | One frame |   0.006944 / 144.0 / 6.944 |   0.006876 / 145.4 / 6.876 |  1.010x |
| Regrid   | January   |   1.916445 / 64.7 / 15.455 |   1.971765 / 62.9 / 15.901 |  0.972x |
| Regrid   | Full year |  20.575736 / 71.2 / 14.054 |  20.763434 / 70.5 / 14.183 |  0.991x |
| Gradient | One frame |   0.003151 / 317.4 / 3.151 |   0.003168 / 315.6 / 3.168 |  0.994x |
| Gradient | January   | 0.042114 / 2,944.4 / 0.340 | 0.041930 / 2,957.3 / 0.338 |  1.004x |
| Gradient | Full year | 0.491534 / 2,978.4 / 0.336 | 0.492042 / 2,975.4 / 0.336 |  0.999x |

The 16-thread path ranges from effectively unchanged to a 1.07x full-year
gradient improvement in float32. Float64 shows both modest gains and
slowdowns, including a 0.87x January filtering ratio. This is why the 1T and
16T results are kept as separate comparison contexts.

## Peak CUDA memory

The following are the largest measured reusable-module peaks in the complete
batch-128 runs, shown as allocated/reserved MiB:

| Path     |       Float32 |       Float64 |
| -------- | ------------: | ------------: |
| Filter   | 2,106 / 2,448 | 4,206 / 4,848 |
| Regrid   | 1,614 / 1,652 | 3,220 / 3,230 |
| Gradient |       57 / 72 |     107 / 128 |

The separate native-grid probe reached 576 frames and reserved 10,910 MiB;
640 frames failed with CUDA OOM. No OOM occurred in the reported batch-128
float32 or float64 workload runs.

## Numerical agreement

The representative frame was `2024-01-01 00:00`. All paths received the
requested public input dtype. The Xarray/DUCC adapter currently promotes its
spherical harmonic transform input to NumPy `float64`, computes `complex128`
coefficients, and returns `float64`; Torch preserves the requested dtype.
Therefore the float32 rows compare Torch float32 with DUCC float64 internal
arithmetic, while the float64 rows are precision-matched. The pressure-gradient
comparison contains both returned components.

### Float32

| Path     | Output    | Comparison              | Max absolute |       RMS | Relative RMS |
| -------- | --------- | ----------------------- | -----------: | --------: | -----------: |
| Filter   | Field     | DUCC CPU vs Torch CPU   |    6.994e-02 | 2.804e-02 |    2.779e-07 |
| Filter   | Field     | DUCC CPU vs Torch CUDA  |    4.741e-02 | 1.232e-02 |    1.221e-07 |
| Filter   | Field     | Torch CPU vs Torch CUDA |    8.594e-02 | 2.720e-02 |    2.696e-07 |
| Regrid   | Field     | DUCC CPU vs Torch CPU   |    7.008e-02 | 2.881e-02 |    2.855e-07 |
| Regrid   | Field     | DUCC CPU vs Torch CUDA  |    4.844e-02 | 1.429e-02 |    1.416e-07 |
| Regrid   | Field     | Torch CPU vs Torch CUDA |    7.031e-02 | 2.714e-02 |    2.690e-07 |
| Gradient | Eastward  | DUCC CPU vs Torch CPU   |    6.846e-08 | 1.704e-08 |    1.831e-05 |
| Gradient | Eastward  | DUCC CPU vs Torch CUDA  |    4.087e-08 | 9.696e-09 |    1.042e-05 |
| Gradient | Eastward  | Torch CPU vs Torch CUDA |    6.155e-08 | 1.599e-08 |    1.719e-05 |
| Gradient | Northward | DUCC CPU vs Torch CPU   |    1.409e-07 | 5.071e-08 |    4.218e-05 |
| Gradient | Northward | DUCC CPU vs Torch CUDA  |    1.578e-07 | 4.924e-08 |    4.097e-05 |
| Gradient | Northward | Torch CPU vs Torch CUDA |    2.400e-07 | 6.292e-08 |    5.233e-05 |

### Float64

| Path     | Output    | Comparison              | Max absolute |       RMS | Relative RMS |
| -------- | --------- | ----------------------- | -----------: | --------: | -----------: |
| Filter   | Field     | DUCC CPU vs Torch CPU   |    1.144e-06 | 1.825e-07 |    1.809e-12 |
| Filter   | Field     | DUCC CPU vs Torch CUDA  |    1.144e-06 | 1.825e-07 |    1.809e-12 |
| Filter   | Field     | Torch CPU vs Torch CUDA |    2.619e-10 | 8.648e-11 |    8.572e-16 |
| Regrid   | Field     | DUCC CPU vs Torch CPU   |    5.815e-07 | 1.257e-07 |    1.245e-12 |
| Regrid   | Field     | DUCC CPU vs Torch CUDA  |    5.815e-07 | 1.257e-07 |    1.246e-12 |
| Regrid   | Field     | Torch CPU vs Torch CUDA |    1.746e-10 | 7.684e-11 |    7.615e-16 |
| Gradient | Eastward  | DUCC CPU vs Torch CPU   |    2.463e-15 | 2.845e-16 |    3.057e-13 |
| Gradient | Eastward  | DUCC CPU vs Torch CUDA  |    2.506e-15 | 2.811e-16 |    3.021e-13 |
| Gradient | Eastward  | Torch CPU vs Torch CUDA |    1.340e-16 | 2.605e-17 |    2.799e-14 |
| Gradient | Northward | DUCC CPU vs Torch CPU   |    2.051e-13 | 4.890e-14 |    4.067e-11 |
| Gradient | Northward | DUCC CPU vs Torch CUDA  |    2.052e-13 | 4.891e-14 |    4.068e-11 |
| Gradient | Northward | Torch CPU vs Torch CUDA |    2.090e-16 | 6.173e-17 |    5.134e-14 |

The separate DUCC thread validation on the representative frame found exact
agreement between the controlled 1-thread and 16-thread calls for both
dtypes. For filter field, regrid field, and both eastward and northward
gradient components, max absolute error, RMS error, and relative RMS error
were all zero.

## Interpretation and limitations

On this machine, full-year float32 reusable CUDA compute was approximately
286.6x, 77.7x, and 24.5x faster than DUCC CPU 1T for filtering, regridding,
and the gradient path, respectively. Against DUCC CPU 16T, the corresponding
ratios were 286.4x, 74.6x, and 22.8x. Including pageable H2D transfer, CUDA
was approximately 96.7x, 15.9x, and 23.4x faster than DUCC CPU 1T for those
three float32 full-year paths; against DUCC CPU 16T the ratios were 96.6x,
15.3x, and 21.8x. These are descriptive results for the measured machine and
workload, not a claim that the controlled 1T DUCC path represents maximum CPU
throughput. Transfer methodology is explicitly limited to ordinary pageable
memory.

For float64 full-year compute, the corresponding CUDA speedups were 106.5x,
70.7x, and 15.0x versus DUCC CPU 1T, and 93.2x, 71.4x, and 15.0x versus DUCC
CPU 16T for filtering, regridding, and the gradient path. With pageable H2D,
the float64 ratios versus DUCC CPU 1T were 45.4x, 17.0x, and 14.2x.

The practical conclusions are:

- batch 128 is a sensible working point for this RTX 5070: it is near the
  measured filter minimum and well below the observed OOM boundary; 576 frames
  succeeded while 640 frames failed on this 12 GB GPU;
- month and year processing should remain streamed rather than assembled as a
  single device tensor;
- CUDA float64 is functional, preserves dtype, and completed the full year,
  while consumer-GPU float64 throughput is slower than float32;
- reusable modules are much faster than reconstructing functional transform
  state for every frame;
- representative float32 and float64 errors are small for all requested
  DUCC/Torch and CPU/CUDA comparisons; and
- no synthetic wind field was used because matching `u`/`v` inputs were not
  available.

The primary measurements are eager inference throughput, not training
throughput. `torch.compile` was not part of the primary benchmark. The
existing real CUDA compile test remains an expected xfail for the Torch 2.11
CUDA Inductor `complex64` lowering issue in the upstream `RealSHT` graph;
eager CUDA execution completed normally.

Host preparation, especially the T42 preparation for the gradient path, is
reported separately through `host_prepare_seconds` and
`stream_wall_seconds`. It includes file access and is sensitive to filesystem
cache state. The benchmark records machine-readable JSON when passed
`--output-json`.

## Reproduction commands

Run these commands from the `sht_bench` repository root. They assume that the
spharmgrid checkout is the sibling directory `../spharmgrid`; replace that
`PYTHONPATH` value when the checkout is elsewhere. The primary float32
invocation was:

```bash
PYTHONPATH=../spharmgrid/src python benchmarks/spharmgrid/torch_backend_era5.py \
    --dtype float32 --batch-size 128 --threads 1 --ducc-threads 16 \
    --one-frame-repetitions 5 --month-repetitions 5 --year-repetitions 1 \
    --output-json /tmp/era5-torch-benchmark-f32.json
```

The float64 invocation changed only `--dtype` and the output path. The CUDA
probe was:

```bash
PYTHONPATH=../spharmgrid/src python benchmarks/spharmgrid/torch_backend_era5.py \
    --dtype float32 --threads 1 --probe-only \
    --probe-batches 32,64,128,256,384,512,576,640 \
    --output-json /tmp/era5-batch-probe.json
```
