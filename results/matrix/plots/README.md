# Benchmark plots

Each thumbnail links to the full-size figure.
Colors identify backends consistently across figures: DUCC is blue, SHTns orange, SHTOOLS/pyshtools green, and Spherepack/pyspharm red. Wheel and installed builds use solid lines; source builds use dashed lines. SHTns is source-built in both benchmark environments and is shown once as a dashed source series. Lower transform time is better.

## Transform time versus spectral scale

Each row fixes the requested thread count. Atmospheric GL figures use Gaussian-grid/spectral labels such as `N32/T42` and `N320/TL639`; CC figures use regular-grid spacing in degrees, from coarse to fine resolution.

### CC

| Threads | Analysis | Synthesis |
| ---: | --- | --- |
| 1 | <a href="by-lmax/cc/analysis/threads-1.png"><img src="by-lmax/cc/analysis/threads-1.png" alt="CC analysis, 1 thread(s)" width="420"></a> | <a href="by-lmax/cc/synthesis/threads-1.png"><img src="by-lmax/cc/synthesis/threads-1.png" alt="CC synthesis, 1 thread(s)" width="420"></a> |
| 2 | <a href="by-lmax/cc/analysis/threads-2.png"><img src="by-lmax/cc/analysis/threads-2.png" alt="CC analysis, 2 thread(s)" width="420"></a> | <a href="by-lmax/cc/synthesis/threads-2.png"><img src="by-lmax/cc/synthesis/threads-2.png" alt="CC synthesis, 2 thread(s)" width="420"></a> |
| 4 | <a href="by-lmax/cc/analysis/threads-4.png"><img src="by-lmax/cc/analysis/threads-4.png" alt="CC analysis, 4 thread(s)" width="420"></a> | <a href="by-lmax/cc/synthesis/threads-4.png"><img src="by-lmax/cc/synthesis/threads-4.png" alt="CC synthesis, 4 thread(s)" width="420"></a> |
| 8 | <a href="by-lmax/cc/analysis/threads-8.png"><img src="by-lmax/cc/analysis/threads-8.png" alt="CC analysis, 8 thread(s)" width="420"></a> | <a href="by-lmax/cc/synthesis/threads-8.png"><img src="by-lmax/cc/synthesis/threads-8.png" alt="CC synthesis, 8 thread(s)" width="420"></a> |
| 16 | <a href="by-lmax/cc/analysis/threads-16.png"><img src="by-lmax/cc/analysis/threads-16.png" alt="CC analysis, 16 thread(s)" width="420"></a> | <a href="by-lmax/cc/synthesis/threads-16.png"><img src="by-lmax/cc/synthesis/threads-16.png" alt="CC synthesis, 16 thread(s)" width="420"></a> |

### GL

| Threads | Analysis | Synthesis |
| ---: | --- | --- |
| 1 | <a href="by-lmax/gl/analysis/threads-1.png"><img src="by-lmax/gl/analysis/threads-1.png" alt="GL analysis, 1 thread(s)" width="420"></a> | <a href="by-lmax/gl/synthesis/threads-1.png"><img src="by-lmax/gl/synthesis/threads-1.png" alt="GL synthesis, 1 thread(s)" width="420"></a> |
| 2 | <a href="by-lmax/gl/analysis/threads-2.png"><img src="by-lmax/gl/analysis/threads-2.png" alt="GL analysis, 2 thread(s)" width="420"></a> | <a href="by-lmax/gl/synthesis/threads-2.png"><img src="by-lmax/gl/synthesis/threads-2.png" alt="GL synthesis, 2 thread(s)" width="420"></a> |
| 4 | <a href="by-lmax/gl/analysis/threads-4.png"><img src="by-lmax/gl/analysis/threads-4.png" alt="GL analysis, 4 thread(s)" width="420"></a> | <a href="by-lmax/gl/synthesis/threads-4.png"><img src="by-lmax/gl/synthesis/threads-4.png" alt="GL synthesis, 4 thread(s)" width="420"></a> |
| 8 | <a href="by-lmax/gl/analysis/threads-8.png"><img src="by-lmax/gl/analysis/threads-8.png" alt="GL analysis, 8 thread(s)" width="420"></a> | <a href="by-lmax/gl/synthesis/threads-8.png"><img src="by-lmax/gl/synthesis/threads-8.png" alt="GL synthesis, 8 thread(s)" width="420"></a> |
| 16 | <a href="by-lmax/gl/analysis/threads-16.png"><img src="by-lmax/gl/analysis/threads-16.png" alt="GL analysis, 16 thread(s)" width="420"></a> | <a href="by-lmax/gl/synthesis/threads-16.png"><img src="by-lmax/gl/synthesis/threads-16.png" alt="GL synthesis, 16 thread(s)" width="420"></a> |

## Transform time versus threads

Each row fixes the spectral scale; each figure compares backend/build series across requested thread counts.

### CC

| Resolution | Analysis | Synthesis |
| ---: | --- | --- |
| 2.5° (L=36) | <a href="by-threads/cc/analysis/lmax-36.png"><img src="by-threads/cc/analysis/lmax-36.png" alt="CC analysis, lmax=36" width="420"></a> | <a href="by-threads/cc/synthesis/lmax-36.png"><img src="by-threads/cc/synthesis/lmax-36.png" alt="CC synthesis, lmax=36" width="420"></a> |
| 1.5° (L=60) | <a href="by-threads/cc/analysis/lmax-60.png"><img src="by-threads/cc/analysis/lmax-60.png" alt="CC analysis, lmax=60" width="420"></a> | <a href="by-threads/cc/synthesis/lmax-60.png"><img src="by-threads/cc/synthesis/lmax-60.png" alt="CC synthesis, lmax=60" width="420"></a> |
| 1.25° (L=72) | <a href="by-threads/cc/analysis/lmax-72.png"><img src="by-threads/cc/analysis/lmax-72.png" alt="CC analysis, lmax=72" width="420"></a> | <a href="by-threads/cc/synthesis/lmax-72.png"><img src="by-threads/cc/synthesis/lmax-72.png" alt="CC synthesis, lmax=72" width="420"></a> |
| 1° (L=90) | <a href="by-threads/cc/analysis/lmax-90.png"><img src="by-threads/cc/analysis/lmax-90.png" alt="CC analysis, lmax=90" width="420"></a> | <a href="by-threads/cc/synthesis/lmax-90.png"><img src="by-threads/cc/synthesis/lmax-90.png" alt="CC synthesis, lmax=90" width="420"></a> |
| 0.75° (L=120) | <a href="by-threads/cc/analysis/lmax-120.png"><img src="by-threads/cc/analysis/lmax-120.png" alt="CC analysis, lmax=120" width="420"></a> | <a href="by-threads/cc/synthesis/lmax-120.png"><img src="by-threads/cc/synthesis/lmax-120.png" alt="CC synthesis, lmax=120" width="420"></a> |
| 0.5° (L=180) | <a href="by-threads/cc/analysis/lmax-180.png"><img src="by-threads/cc/analysis/lmax-180.png" alt="CC analysis, lmax=180" width="420"></a> | <a href="by-threads/cc/synthesis/lmax-180.png"><img src="by-threads/cc/synthesis/lmax-180.png" alt="CC synthesis, lmax=180" width="420"></a> |
| 0.25° (L=360) | <a href="by-threads/cc/analysis/lmax-360.png"><img src="by-threads/cc/analysis/lmax-360.png" alt="CC analysis, lmax=360" width="420"></a> | <a href="by-threads/cc/synthesis/lmax-360.png"><img src="by-threads/cc/synthesis/lmax-360.png" alt="CC synthesis, lmax=360" width="420"></a> |
| 0.125° (L=720) | <a href="by-threads/cc/analysis/lmax-720.png"><img src="by-threads/cc/analysis/lmax-720.png" alt="CC analysis, lmax=720" width="420"></a> | <a href="by-threads/cc/synthesis/lmax-720.png"><img src="by-threads/cc/synthesis/lmax-720.png" alt="CC synthesis, lmax=720" width="420"></a> |

### GL

| Gaussian grid / truncation | Analysis | Synthesis |
| ---: | --- | --- |
| F32 | <a href="by-threads/gl/analysis/lmax-42.png"><img src="by-threads/gl/analysis/lmax-42.png" alt="GL analysis, lmax=42" width="420"></a> | <a href="by-threads/gl/synthesis/lmax-42.png"><img src="by-threads/gl/synthesis/lmax-42.png" alt="GL synthesis, lmax=42" width="420"></a> |
| F48 | <a href="by-threads/gl/analysis/lmax-63.png"><img src="by-threads/gl/analysis/lmax-63.png" alt="GL analysis, lmax=63" width="420"></a> | <a href="by-threads/gl/synthesis/lmax-63.png"><img src="by-threads/gl/synthesis/lmax-63.png" alt="GL synthesis, lmax=63" width="420"></a> |
| F64 | <a href="by-threads/gl/analysis/lmax-85.png"><img src="by-threads/gl/analysis/lmax-85.png" alt="GL analysis, lmax=85" width="420"></a> | <a href="by-threads/gl/synthesis/lmax-85.png"><img src="by-threads/gl/synthesis/lmax-85.png" alt="GL synthesis, lmax=85" width="420"></a> |
| F80 | <a href="by-threads/gl/analysis/lmax-159.png"><img src="by-threads/gl/analysis/lmax-159.png" alt="GL analysis, lmax=159" width="420"></a> | <a href="by-threads/gl/synthesis/lmax-159.png"><img src="by-threads/gl/synthesis/lmax-159.png" alt="GL synthesis, lmax=159" width="420"></a> |
| F128 | <a href="by-threads/gl/analysis/lmax-255.png"><img src="by-threads/gl/analysis/lmax-255.png" alt="GL analysis, lmax=255" width="420"></a> | <a href="by-threads/gl/synthesis/lmax-255.png"><img src="by-threads/gl/synthesis/lmax-255.png" alt="GL synthesis, lmax=255" width="420"></a> |
| F160 | <a href="by-threads/gl/analysis/lmax-319.png"><img src="by-threads/gl/analysis/lmax-319.png" alt="GL analysis, lmax=319" width="420"></a> | <a href="by-threads/gl/synthesis/lmax-319.png"><img src="by-threads/gl/synthesis/lmax-319.png" alt="GL synthesis, lmax=319" width="420"></a> |
| F256 | <a href="by-threads/gl/analysis/lmax-511.png"><img src="by-threads/gl/analysis/lmax-511.png" alt="GL analysis, lmax=511" width="420"></a> | <a href="by-threads/gl/synthesis/lmax-511.png"><img src="by-threads/gl/synthesis/lmax-511.png" alt="GL synthesis, lmax=511" width="420"></a> |
| F320 | <a href="by-threads/gl/analysis/lmax-639.png"><img src="by-threads/gl/analysis/lmax-639.png" alt="GL analysis, lmax=639" width="420"></a> | <a href="by-threads/gl/synthesis/lmax-639.png"><img src="by-threads/gl/synthesis/lmax-639.png" alt="GL synthesis, lmax=639" width="420"></a> |
| F512 | <a href="by-threads/gl/analysis/lmax-1023.png"><img src="by-threads/gl/analysis/lmax-1023.png" alt="GL analysis, lmax=1023" width="420"></a> | <a href="by-threads/gl/synthesis/lmax-1023.png"><img src="by-threads/gl/synthesis/lmax-1023.png" alt="GL synthesis, lmax=1023" width="420"></a> |
| F640 | <a href="by-threads/gl/analysis/lmax-1279.png"><img src="by-threads/gl/analysis/lmax-1279.png" alt="GL analysis, lmax=1279" width="420"></a> | <a href="by-threads/gl/synthesis/lmax-1279.png"><img src="by-threads/gl/synthesis/lmax-1279.png" alt="GL synthesis, lmax=1279" width="420"></a> |

