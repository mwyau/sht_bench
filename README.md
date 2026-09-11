# sht_bench

`sht_bench` measures CPU performance of spherical harmonic transform (SHT) implementations exposed through Python. It benchmarks repeated scalar analysis and synthesis on atmospheric Gaussian grids and endpoint-including regular/Clenshaw–Curtis grids, with build and timing metadata retained for reproducibility.

The default comparison includes DUCC, SHTns, and Spherepack/pyspharm where their grid interfaces are compatible. Native SHTOOLS/pyshtools remains available for compact GLQ experiments but is not part of the default atmospheric full-Gaussian-grid comparison.

## Results

Median transform time is shown in milliseconds; lower is better. Backend colors are fixed across figures. Wheel/installed builds use solid lines and source builds use dashed lines. SHTns is source-built and shown once as a dashed series.

### AMD64 — AMD Ryzen 9 5950X, 16 threads

[Full AMD64 plot matrix](results/matrix/plots/README.md)

| Grid | Analysis | Synthesis |
| --- | --- | --- |
| CC | <a href="results/matrix/plots/by-lmax/cc/analysis/threads-16.png"><img src="results/matrix/plots/by-lmax/cc/analysis/threads-16.png" alt="AMD64 CC analysis, 16 threads" width="430"></a> | <a href="results/matrix/plots/by-lmax/cc/synthesis/threads-16.png"><img src="results/matrix/plots/by-lmax/cc/synthesis/threads-16.png" alt="AMD64 CC synthesis, 16 threads" width="430"></a> |
| GL | <a href="results/matrix/plots/by-lmax/gl/analysis/threads-16.png"><img src="results/matrix/plots/by-lmax/gl/analysis/threads-16.png" alt="AMD64 GL analysis, 16 threads" width="430"></a> | <a href="results/matrix/plots/by-lmax/gl/synthesis/threads-16.png"><img src="results/matrix/plots/by-lmax/gl/synthesis/threads-16.png" alt="AMD64 GL synthesis, 16 threads" width="430"></a> |

### ARM64 — Raspberry Pi 5 8GB, 4 threads

[Full ARM64 plot matrix](results/matrix-arm64/plots/README.md)

| Grid | Analysis | Synthesis |
| --- | --- | --- |
| CC | <a href="results/matrix-arm64/plots/by-lmax/cc/analysis/threads-4.png"><img src="results/matrix-arm64/plots/by-lmax/cc/analysis/threads-4.png" alt="ARM64 CC analysis, 4 threads" width="430"></a> | <a href="results/matrix-arm64/plots/by-lmax/cc/synthesis/threads-4.png"><img src="results/matrix-arm64/plots/by-lmax/cc/synthesis/threads-4.png" alt="ARM64 CC synthesis, 4 threads" width="430"></a> |
| GL | <a href="results/matrix-arm64/plots/by-lmax/gl/analysis/threads-4.png"><img src="results/matrix-arm64/plots/by-lmax/gl/analysis/threads-4.png" alt="ARM64 GL analysis, 4 threads" width="430"></a> | <a href="results/matrix-arm64/plots/by-lmax/gl/synthesis/threads-4.png"><img src="results/matrix-arm64/plots/by-lmax/gl/synthesis/threads-4.png" alt="ARM64 GL synthesis, 4 threads" width="430"></a> |

The two systems are separate benchmark environments; these panels are not intended as a direct cross-architecture speed comparison.

## Benchmark grids

The default GL sweep uses ten full atmospheric Gaussian grids. `F` is half the latitude count, so the full grid is `2F × 4F`. Gaussian latitudes are nonuniform; the spacing column is the nominal mean latitude spacing.

| GL label | Spectral truncation | `lmax` | Grid | Nominal spacing |
| --- | --- | ---: | ---: | ---: |
| F32 | T42 | 42 | 64 × 128 | 2.8125° |
| F48 | T63 | 63 | 96 × 192 | 1.875° |
| F64 | T85 | 85 | 128 × 256 | 1.40625° |
| F80 | TL159 | 159 | 160 × 320 | 1.125° |
| F128 | TL255 | 255 | 256 × 512 | 0.703125° |
| F160 | TL319 | 319 | 320 × 640 | 0.5625° |
| F256 | TL511 | 511 | 512 × 1024 | 0.3515625° |
| F320 | TL639 | 639 | 640 × 1280 | 0.28125° |
| F512 | TL1023 | 1023 | 1024 × 2048 | 0.17578125° |
| F640 | TL1279 | 1279 | 1280 × 2560 | 0.140625° |

The default CC sweep uses common equal-angle resolutions with both poles included and no duplicate 360° longitude column.

| CC resolution | `lmax` | Grid |
| ---: | ---: | ---: |
| 2.5° | 36 | 73 × 144 |
| 1.5° | 60 | 121 × 240 |
| 1.25° | 72 | 145 × 288 |
| 1.0° | 90 | 181 × 360 |
| 0.75° | 120 | 241 × 480 |
| 0.5° | 180 | 361 × 720 |
| 0.25° | 360 | 721 × 1440 |
| 0.125° | 720 | 1441 × 2880 |

For CC, `nlat = 2L + 1`, `nlon = 4L`, and the angular spacing is `90/L` degrees. A custom GL `lmax` outside the atmospheric table uses the compact quadrature geometry `nlat = L + 1`, `nlon = 2L + 1`.

## Run the benchmark

On Debian/Ubuntu, SHTns requires FFTW and Python development headers in addition to the compiler toolchain:

```bash
sudo apt-get install build-essential gfortran libfftw3-dev python3-dev
uv sync --all-extras
```

Run the full wheel/source matrix and generate plots:

```bash
uv run --extra plot sht-bench matrix \
  --build both \
  --threads 1,2,4,8,16 \
  --output results/matrix \
  --plot
```

For ARM64, a focused DUCC/SHTns run is:

```bash
uv run --extra plot sht-bench matrix \
  --build both \
  --backend ducc,shtns \
  --threads 1,2,4 \
  --output results/matrix-arm64 \
  --plot
```

## Method

The benchmark uses triangular truncation with `mmax = lmax`. Backend construction, grid setup, and reusable precomputation occur before timing. Each transform is warmed up and repeatedly timed with `time.perf_counter_ns`; raw samples and summary statistics are retained. Thread-count cells run in separate Python processes so thread-related environment variables are set before numerical libraries are imported.

DUCC and SHTns use double-precision spatial and spectral data. The public pyspharm interface uses `float32` spatial data and `complex64` coefficients, so its timings are a mixed-precision comparison. pyshtools and pyspharm do not expose the same explicit per-call thread control used by DUCC and SHTns. Passing tests or completing a timing run is not a cross-backend numerical validation of normalization, phase conventions, or transform accuracy.

See [BUILD.md](BUILD.md) for package/build provenance and [docs/MATRIX.md](docs/MATRIX.md) for matrix dimensions, incremental reruns, output files, and plotting details.
