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

The low-overhead scaling runner is `scripts/run_fast_scaling.py`; it reuses installed adapters and accepts **`--build-manifest PATH`**. The full-audit source runner remains available for numerical studies, but is not used by this timing-only experiment. The completed sweep's manifest records the exact source refs, SHAs, Python/NumPy builds, installed module paths, and native binary hashes. No Git source checkout is modified by the benchmark.

Reproduce the complete sweep using the saved build manifest and a fresh output directory:

```bash
uv run --extra plot python scripts/run_fast_scaling.py \
  --build-manifest results/gl-cc-resolution-scaling/build-manifest.json \
  --output results/gl-cc-resolution-scaling-reproduction \
  --backend ducc,shtns,rustsht,pyspharm \
  --grid GL,CC --threads 1,2,4,8,16 \
  --precision float32,float64 --min-time 0.005 --timeout 180
```

Setup and plan construction are **not** warm-call timings. Current scenarios and execution requirements are in [docs/MATRIX.md](docs/MATRIX.md). Archived build studies are indexed in [archive/README.md](archive/README.md).
