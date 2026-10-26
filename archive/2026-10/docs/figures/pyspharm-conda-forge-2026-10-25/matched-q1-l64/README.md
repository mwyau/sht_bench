# Source benchmark results

576 recorded outcomes of 576; 0 pending. 252 timed after numerical validation.

Measurements are provisional on a shared host. Ratios pair the same grid, bandwidth, precision, allocation, operation, thread request, and process round.

The Spherepack series uses conda-forge pyspharm 1.0.9-np2py314h2766e9b_1017; its timing environment, package hash, native extension, and linked libraries are in `build-provenance.json`.

| Outcome | Cells |
| --- | ---: |
| numerical_validation_failed | 24 |
| timed_successfully | 252 |
| unsupported | 300 |

`outcomes.csv` identifies every recorded status and its raw cell. `build-provenance.json` preserves the referenced build manifest. Raw cells contain every timing sample, numerical metrics, CPU controls, and build-manifest reference. `matched-ratios.csv` records pyspharm/DUCC, pyspharm/RustSHT, and pyspharm/SHTns ratios when a complete workload and precision match exists, along with RustSHT/DUCC and SHTns/DUCC ratios. Absolute latency excess is in milliseconds; `paired-summary.csv` summarizes those process pairs. `thread-scaling.csv` pairs each thread request with its q1 process round. `accuracy.json` includes successful and failed validation fixtures. Peak RSS includes setup and validation buffers.

Run dimensions are in [run-config.json](run-config.json) and [workload-plan.json](workload-plan.json). The [audit](audit.json) and [evidence audit](evidence-audit.json) retain coverage checks. Raw cells are in [raw-cells.tar.gz](raw-cells.tar.gz) under `cells/<id>.json`; extract that archive here before rerunning the audit or report.

Figures are limited to workload groups containing `pyspharm`.

For secondary workloads, accuracy entries distinguish base A/S gates from requested derivative synthesis and adjoint identity checks. Adjoint scaled residuals are separate from relative L2 synthesis errors.

## Figures

100 figures; each page contains at most 48 previews.

- [Figures 1–48](figures-0001.md)
- [Figures 49–96](figures-0002.md)
- [Figures 97–100](figures-0003.md)
