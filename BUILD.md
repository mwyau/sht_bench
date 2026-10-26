# Builds for the current SHT scaling benchmark

Use the local source checkouts with isolated environments, and the separate conda-forge pyspharm environment.

| Backend | Source / environment | Precision |
| --- | --- | --- |
| DUCC | `../ducc` native source | float32, float64 |
| SHTns | `../shtns` source with FFTW | float64 |
| RustSHT | `../RustSHT` source, `main` | float32, float64 |
| pyspharm | conda-forge `pyspharm` in Python 3.14 conda env | float32 / complex64 |
| pyspharm-cpp (optional) | `../pyspharm-cpp`, its DUCC submodule | DUCC-derived |

The archived, reproducible Linux pyspharm environment pins **pyspharm 1.0.9**, build `np2py314h2766e9b_1017`, Python 3.14.9 and NumPy 2.5.4. See [environment spec](environments/pyspharm-conda-forge-linux-64.yml). Do not install the old PyPI `pyspharm-syl`.

Prerequisites on Debian/Ubuntu:

```bash
sudo apt-get install build-essential gfortran libfftw3-dev python3-dev
uv sync --extra plot
```

The benchmark script is `scripts/run_source_resolution_sweep.py`. Use `--build` with an empty output directory for a new set of source builds, or **`--build-manifest PATH`** to reuse already-built binaries. The script's `--help` shows source paths and Git ref overrides. No Git source checkout should be modified for benchmarking.

Save the exact source SHAs, compiler/build configuration, linked native dependencies, conda package identity, Python/NumPy versions and import paths with results. Setup and plan construction are **not** warm-call timings. Current scenarios and execution requirements are in [docs/MATRIX.md](docs/MATRIX.md). Archived build studies are indexed in [archive/README.md](archive/README.md).
