# DUCC FFT benchmark

This report contains forward one-dimensional DUCC FFT timings. Backend setup, input generation, output allocation, dtype validation, and warm-up are outside the measured samples; `out=` is preallocated for every timed transform.

Timing alone is not a numerical-accuracy comparison.

## Runtime environment

| Field | Detected value |
| --- | --- |
| OS | Linux |
| Architecture | x86_64 |
| CPU | AMD EPYC 7763 64-Core Processor |
| Python | 3.13.15 |
| NumPy | 2.5.3 |
| DUCC | 0.41.0 |
| DUCC binding | pybind11 |
| Build mode | source |
| DUCC source | ducc0==0.41.0 source distribution |
| DUCC optimization | portable |
| Long-double representation | x87 extended precision (64-bit significand, 16-byte storage) |

**Detected longdouble:** `x87 extended precision (64-bit significand, 16-byte storage)`.

NumPy `finfo.nmant` is reported separately below as the mantissa-bit count; for the usual normalized binary formats the effective significand precision is `nmant + 1`.

| longdouble field | Value |
| --- | ---: |
| itemsize (bytes) | 16 |
| finfo.bits | 128 |
| finfo.nmant | 63 |
| finfo.eps | 1.084202172485504434e-19 |

## Median transform time

### r2c

| FFT size | Threads | float32 (ms) | float64 (ms) | longdouble (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 0.002 | 0.002 | 0.007 |
| 1024 | 1 | 0.004 | 0.004 | 0.036 |
| 4096 | 1 | 0.012 | 0.014 | 0.164 |
| 16384 | 1 | 0.047 | 0.057 | 0.757 |
| 65536 | 1 | 0.199 | 0.269 | 3.71 |
| 262144 | 1 | 1.24 | 1.89 | 16.19 |
| 1048576 | 1 | 5.27 | 8.06 | 88.90 |
| 256 | 2 | 0.002 | 0.002 | 0.007 |
| 1024 | 2 | 0.004 | 0.004 | 0.036 |
| 4096 | 2 | 0.012 | 0.014 | 0.167 |
| 16384 | 2 | 0.047 | 0.056 | 0.747 |
| 65536 | 2 | 0.196 | 0.268 | 3.70 |
| 262144 | 2 | 0.857 | 1.20 | 16.32 |
| 1048576 | 2 | 3.36 | 4.95 | 88.47 |
| 256 | 4 | 0.002 | 0.002 | 0.007 |
| 1024 | 4 | 0.004 | 0.004 | 0.036 |
| 4096 | 4 | 0.012 | 0.015 | 0.162 |
| 16384 | 4 | 0.047 | 0.057 | 0.753 |
| 65536 | 4 | 0.197 | 0.265 | 3.68 |
| 262144 | 4 | 0.823 | 1.09 | 16.19 |
| 1048576 | 4 | 3.26 | 4.71 | 88.40 |

### c2c

| FFT size | Threads | float32 (ms) | float64 (ms) | longdouble (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 0.003 | 0.003 | 0.012 |
| 1024 | 1 | 0.004 | 0.005 | 0.058 |
| 4096 | 1 | 0.014 | 0.018 | 0.263 |
| 16384 | 1 | 0.058 | 0.082 | 1.45 |
| 65536 | 1 | 0.354 | 0.436 | 6.45 |
| 262144 | 1 | 1.85 | 2.45 | 29.25 |
| 1048576 | 1 | 8.07 | 12.29 | 154.1 |
| 256 | 2 | 0.003 | 0.003 | 0.012 |
| 1024 | 2 | 0.004 | 0.005 | 0.056 |
| 4096 | 2 | 0.014 | 0.017 | 0.263 |
| 16384 | 2 | 0.057 | 0.082 | 1.46 |
| 65536 | 2 | 0.350 | 0.436 | 6.47 |
| 262144 | 2 | 0.994 | 1.33 | 14.75 |
| 1048576 | 2 | 4.09 | 6.71 | 83.33 |
| 256 | 4 | 0.003 | 0.003 | 0.012 |
| 1024 | 4 | 0.004 | 0.005 | 0.056 |
| 4096 | 4 | 0.014 | 0.018 | 0.263 |
| 16384 | 4 | 0.058 | 0.083 | 1.46 |
| 65536 | 4 | 0.351 | 0.437 | 6.46 |
| 262144 | 4 | 1.00 | 1.23 | 14.99 |
| 1048576 | 4 | 3.77 | 5.93 | 83.13 |

## Long-double slowdown

Ratios use matching FFT size and requested thread count. A missing value means that the corresponding precision was unsupported or not run.

### r2c

| FFT size | Threads | longdouble / float64 | float32 / float64 |
| ---: | ---: | ---: | ---: |
| 256 | 1 | 2.89× | 1× |
| 256 | 2 | 2.94× | 1.01× |
| 256 | 4 | 2.9× | 0.994× |
| 1024 | 1 | 8.1× | 0.987× |
| 1024 | 2 | 8.18× | 0.984× |
| 1024 | 4 | 8.03× | 0.967× |
| 4096 | 1 | 11.4× | 0.849× |
| 4096 | 2 | 11.6× | 0.844× |
| 4096 | 4 | 11.1× | 0.836× |
| 16384 | 1 | 13.3× | 0.83× |
| 16384 | 2 | 13.3× | 0.835× |
| 16384 | 4 | 13.1× | 0.816× |
| 65536 | 1 | 13.8× | 0.741× |
| 65536 | 2 | 13.8× | 0.733× |
| 65536 | 4 | 13.9× | 0.743× |
| 262144 | 1 | 8.57× | 0.657× |
| 262144 | 2 | 13.6× | 0.717× |
| 262144 | 4 | 14.8× | 0.754× |
| 1048576 | 1 | 11× | 0.654× |
| 1048576 | 2 | 17.9× | 0.679× |
| 1048576 | 4 | 18.8× | 0.692× |

### c2c

| FFT size | Threads | longdouble / float64 | float32 / float64 |
| ---: | ---: | ---: | ---: |
| 256 | 1 | 4.52× | 1× |
| 256 | 2 | 4.61× | 0.993× |
| 256 | 4 | 4.59× | 1.01× |
| 1024 | 1 | 11.3× | 0.871× |
| 1024 | 2 | 10.5× | 0.836× |
| 1024 | 4 | 10.5× | 0.845× |
| 4096 | 1 | 14.6× | 0.751× |
| 4096 | 2 | 15.2× | 0.786× |
| 4096 | 4 | 14.7× | 0.766× |
| 16384 | 1 | 17.8× | 0.709× |
| 16384 | 2 | 17.8× | 0.699× |
| 16384 | 4 | 17.6× | 0.705× |
| 65536 | 1 | 14.8× | 0.812× |
| 65536 | 2 | 14.8× | 0.803× |
| 65536 | 4 | 14.8× | 0.804× |
| 262144 | 1 | 11.9× | 0.755× |
| 262144 | 2 | 11.1× | 0.75× |
| 262144 | 4 | 12.2× | 0.815× |
| 1048576 | 1 | 12.5× | 0.657× |
| 1048576 | 2 | 12.4× | 0.61× |
| 1048576 | 4 | 14× | 0.635× |

## Figures

- [r2c-threads-1.png](r2c-threads-1.png)
- [r2c-threads-2.png](r2c-threads-2.png)
- [r2c-threads-4.png](r2c-threads-4.png)
- [c2c-threads-1.png](c2c-threads-1.png)
- [c2c-threads-2.png](c2c-threads-2.png)
- [c2c-threads-4.png](c2c-threads-4.png)

## Method notes

DUCC is invoked through `ducc0.fft.r2c` or `ducc0.fft.c2c` with `forward=True`, `inorm=0`, explicit `nthreads`, and a preallocated output. Matrix thread-count cells are process-isolated. The longdouble label is portable; the detected ABI above is the scientific interpretation of that label.
