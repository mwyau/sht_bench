# Builds for the current SHT scaling benchmark

Use the local source checkouts with isolated environments, and the separate conda-forge pyspharm environment.

| Backend | Source / environment | Precision |
| --- | --- | --- |
| DUCC | `../ducc` native source | float32, float64 |
| SHTns | `../shtns` source with FFTW | float64 |
| RustSHT | `../RustSHT` source, current refresh at `dev` commit `3194dc095c71343931b2ba0cfa0fc4f72dcab4cf` | float32, float64 |
| pyspharm | conda-forge `pyspharm` in Python 3.14 conda env | float32 / complex64 |
| pyspharm-cpp (optional) | `../pyspharm-cpp`, its DUCC submodule | DUCC-derived |

The archived, reproducible Linux pyspharm environment pins **pyspharm 1.0.9**, build `np2py314h2766e9b_1017`, Python 3.14.9 and NumPy 2.5.4. See [environment spec](environments/pyspharm-conda-forge-linux-64.yml). Do not install the old PyPI `pyspharm-syl`.

Prerequisites on Debian/Ubuntu:

```bash
sudo apt-get install build-essential gfortran libfftw3-dev python3-dev
uv sync --extra plot
```

The low-overhead scaling runner is `scripts/run_fast_scaling.py`; it reuses installed adapters and accepts **`--build-manifest PATH`**. The full-audit source runner remains available for numerical studies, but is not used by this timing-only experiment. The original dense L1024 baseline is in `results/gl-cc-resolution-scaling-l1024/`; the current merged refresh is in `results/gl-cc-resolution-scaling-dev-refresh/`. RustSHT `main` (`579c09e`) remains in the baseline, while the refreshed comparison uses a detached `dev` build in a separate Python 3.14.9/NumPy 2.5.4 environment. Its extension SHA256 and rustc/cargo 1.99.0 build details are recorded in the new build manifest. The active RustSHT development worktree was not modified.

Reproduce the RustSHT-only refresh and merge it into the saved baseline. These commands leave the other backends' measurements untouched:

```bash
env -u OMP_WAIT_POLICY uv run --extra plot python scripts/run_fast_scaling.py \
  --build-manifest results/gl-cc-resolution-scaling-rustsht-dev-build-3194dc09/build-manifest.json \
  --output results/gl-cc-resolution-scaling-rustsht-dev-3194dc09-reproduction \
  --profile dense-to-1024 --backend rustsht \
  --grid GL,CC --threads 1,2,4,8,16 \
  --precision float64,float32 --min-time 0.005 --timeout 180

uv run --extra plot python scripts/merge_scaling_results.py \
  --baseline results/gl-cc-resolution-scaling-l1024 \
  --rustsht-refresh results/gl-cc-resolution-scaling-rustsht-dev-3194dc09-reproduction \
  --output results/gl-cc-resolution-scaling-dev-refresh-reproduction
```

Setup and plan construction are **not** warm-call timings. Current scenarios and execution requirements are in [docs/MATRIX.md](docs/MATRIX.md). Archived build studies are indexed in [archive/README.md](archive/README.md).
