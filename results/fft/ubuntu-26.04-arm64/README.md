# DUCC FFT benchmark

This report contains forward one-dimensional DUCC FFT timings. Backend setup, input generation, output allocation, dtype validation, and warm-up are outside the measured samples; `out=` is preallocated for every timed transform.

Timing alone is not a numerical-accuracy comparison.

## Runtime environment

| Field | Detected value |
| --- | --- |
| OS | Linux |
| Architecture | aarch64 |
| CPU | 0 |
| Python | 3.13.15 |
| NumPy | 2.5.3 |
| DUCC | 0.41.0 |
| DUCC binding | pybind11 |
| Build mode | source |
| DUCC source | ducc0==0.41.0 source distribution |
| DUCC optimization | portable |
| Long-double representation | IEEE binary128 |

**Detected longdouble:** `IEEE binary128`.

NumPy `finfo.nmant` is reported separately below as the mantissa-bit count; for the usual normalized binary formats the effective significand precision is `nmant + 1`.

| longdouble field | Value |
| --- | ---: |
| itemsize (bytes) | 16 |
| finfo.bits | 128 |
| finfo.nmant | 112 |
| finfo.eps | 1.9259299443872358530559779425849273e-34 |

## Median transform time

### r2c

| FFT size | Threads | float32 (ms) | float64 (ms) | longdouble (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 0.002 | 0.002 | 0.043 |
| 1024 | 1 | 0.004 | 0.006 | 0.404 |
| 4096 | 1 | 0.014 | 0.021 | 2.02 |
| 16384 | 1 | 0.056 | 0.098 | 9.40 |
| 65536 | 1 | 0.250 | 0.456 | 45.93 |
| 262144 | 1 | 1.46 | 1.95 | 203.7 |
| 1048576 | 1 | 6.13 | 8.65 | 904.8 |
| 256 | 2 | 0.002 | 0.002 | 0.044 |
| 1024 | 2 | 0.004 | 0.006 | 0.404 |
| 4096 | 2 | 0.014 | 0.021 | 2.02 |
| 16384 | 2 | 0.118 | 0.097 | 9.40 |
| 65536 | 2 | 0.248 | 0.356 | 45.90 |
| 262144 | 2 | 1.01 | 1.27 | 203.6 |
| 1048576 | 2 | 3.96 | 5.34 | 904.8 |
| 256 | 4 | 0.002 | 0.002 | 0.043 |
| 1024 | 4 | 0.004 | 0.006 | 0.404 |
| 4096 | 4 | 0.014 | 0.021 | 2.02 |
| 16384 | 4 | 0.056 | 0.097 | 9.39 |
| 65536 | 4 | 0.313 | 0.257 | 45.90 |
| 262144 | 4 | 0.743 | 0.897 | 203.5 |
| 1048576 | 4 | 2.81 | 3.51 | 904.1 |

### c2c

| FFT size | Threads | float32 (ms) | float64 (ms) | longdouble (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 0.002 | 0.003 | 0.109 |
| 1024 | 1 | 0.005 | 0.008 | 0.691 |
| 4096 | 1 | 0.017 | 0.036 | 3.42 |
| 16384 | 1 | 0.071 | 0.177 | 18.16 |
| 65536 | 1 | 0.413 | 0.800 | 82.40 |
| 262144 | 1 | 2.19 | 3.25 | 372.3 |
| 1048576 | 1 | 10.27 | 15.66 | 1681.5 |
| 256 | 2 | 0.002 | 0.003 | 0.109 |
| 1024 | 2 | 0.005 | 0.008 | 0.691 |
| 4096 | 2 | 0.017 | 0.035 | 3.42 |
| 16384 | 2 | 0.071 | 0.152 | 18.14 |
| 65536 | 2 | 0.419 | 0.505 | 82.24 |
| 262144 | 2 | 1.28 | 1.83 | 186.7 |
| 1048576 | 2 | 5.43 | 8.24 | 843.4 |
| 256 | 4 | 0.002 | 0.003 | 0.109 |
| 1024 | 4 | 0.005 | 0.008 | 0.689 |
| 4096 | 4 | 0.017 | 0.035 | 3.42 |
| 16384 | 4 | 0.071 | 0.124 | 18.14 |
| 65536 | 4 | 0.420 | 0.349 | 82.25 |
| 262144 | 4 | 0.729 | 1.04 | 93.48 |
| 1048576 | 4 | 2.92 | 4.43 | 424.1 |

## Long-double slowdown

Ratios use matching FFT size and requested thread count. A missing value means that the corresponding precision was unsupported or not run.

### r2c

| FFT size | Threads | longdouble / float64 | float32 / float64 |
| ---: | ---: | ---: | ---: |
| 256 | 1 | 21.4× | 0.982× |
| 256 | 2 | 21.1× | 0.97× |
| 256 | 4 | 20.9× | 0.97× |
| 1024 | 1 | 71.9× | 0.791× |
| 1024 | 2 | 71.2× | 0.786× |
| 1024 | 4 | 71.1× | 0.786× |
| 4096 | 1 | 96× | 0.679× |
| 4096 | 2 | 95.8× | 0.678× |
| 4096 | 4 | 96.2× | 0.68× |
| 16384 | 1 | 96.2× | 0.572× |
| 16384 | 2 | 97× | 1.22× |
| 16384 | 4 | 96.9× | 0.576× |
| 65536 | 1 | 101× | 0.548× |
| 65536 | 2 | 129× | 0.697× |
| 65536 | 4 | 179× | 1.22× |
| 262144 | 1 | 105× | 0.748× |
| 262144 | 2 | 161× | 0.798× |
| 262144 | 4 | 227× | 0.828× |
| 1048576 | 1 | 105× | 0.708× |
| 1048576 | 2 | 169× | 0.741× |
| 1048576 | 4 | 258× | 0.8× |

### c2c

| FFT size | Threads | longdouble / float64 | float32 / float64 |
| ---: | ---: | ---: | ---: |
| 256 | 1 | 43.3× | 0.962× |
| 256 | 2 | 42.8× | 0.955× |
| 256 | 4 | 43.2× | 0.966× |
| 1024 | 1 | 84.6× | 0.588× |
| 1024 | 2 | 84.2× | 0.586× |
| 1024 | 4 | 84.1× | 0.586× |
| 4096 | 1 | 96.3× | 0.481× |
| 4096 | 2 | 96.6× | 0.482× |
| 4096 | 4 | 97.1× | 0.486× |
| 16384 | 1 | 103× | 0.404× |
| 16384 | 2 | 119× | 0.465× |
| 16384 | 4 | 146× | 0.573× |
| 65536 | 1 | 103× | 0.516× |
| 65536 | 2 | 163× | 0.83× |
| 65536 | 4 | 236× | 1.2× |
| 262144 | 1 | 114× | 0.674× |
| 262144 | 2 | 102× | 0.702× |
| 262144 | 4 | 89.9× | 0.701× |
| 1048576 | 1 | 107× | 0.656× |
| 1048576 | 2 | 102× | 0.659× |
| 1048576 | 4 | 95.7× | 0.658× |

## Figures

- [r2c-threads-1.png](r2c-threads-1.png)
- [r2c-threads-2.png](r2c-threads-2.png)
- [r2c-threads-4.png](r2c-threads-4.png)
- [c2c-threads-1.png](c2c-threads-1.png)
- [c2c-threads-2.png](c2c-threads-2.png)
- [c2c-threads-4.png](c2c-threads-4.png)

## Method notes

DUCC is invoked through `ducc0.fft.r2c` or `ducc0.fft.c2c` with `forward=True`, `inorm=0`, explicit `nthreads`, and a preallocated output. Matrix thread-count cells are process-isolated. The longdouble label is portable; the detected ABI above is the scientific interpretation of that label.
