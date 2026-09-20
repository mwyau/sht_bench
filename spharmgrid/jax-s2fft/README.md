# spharmgrid JAX/S2FFT archive

This archive contains the reproducibility and benchmark material extracted from
the `mwyau/spharmgrid` repository for PR #44, `feat: JAX S2FFT backend`.

- Source branch: `feat/jax-s2fft`
- Source commit at extraction: `fd02eeac17afcd323e4efacfcb6dac9879f506c4`
- S2FFT version primarily studied: `1.4.0`
- CPU minimum environment: Python 3.11 with JAX/JAXLIB 0.5.0
- CUDA environment: RTX 5070 with JAX/JAXLIB 0.10.2

The subdirectories contain:

- `studies/`: the written investigation record;
- `scripts/`: reproducible benchmark and reproducer programs;
- `results/`: raw JSON measurements.

Durable conclusions from the investigation:

- `spharmgrid.jax` requires x64/float64: the tested ordinary MWSS float32
  path had field-scale interior error of about 9%.
- Explicit O(L²) precomputations materially reduced warmed CUDA transform time
  in the tested cases, so production uses cached public S2FFT precomputations.
- Cached arrays were uncommitted in the tested CPU and single-GPU environments.
- `jax.ensure_compile_time_eval()` is required when the first precompute-cache
  miss occurs inside an outer `jax.jit`.
