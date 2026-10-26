# Results status

**The requested all-backend, multi-resolution GL and CC scaling comparison is not yet published on this branch.**

The deliverable is 16 figures (GL/CC × float32/float64 × spin0/spin1 × analysis/synthesis), each with q1/q2/q4/q8/q16 thread panels containing validated-as-executed, *exploratory* warm-transform measurements across the shared resolution ladder. One compact CSV/JSON must retain timing samples, exact workload identity, source/build identity and failed/unsupported records.

Historical L64 and L360 controls and the conda-forge pyspharm-only degree sweep are preserved in [archive/README.md](../archive/README.md). They do **not** constitute the requested matched multi-resolution comparison.

Do not fill absent points by interpolation or present old precision-mismatched timing cells as a combined benchmark.
