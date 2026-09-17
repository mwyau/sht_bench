# B/C equiangular SHT resolution scaling

This report contains only B (`runtime_fold`) and C (`precomputed_projection`) forward measurements. Each native cell ran in a fresh subprocess; A/dense, backward, compilation, and DUCC were not part of this probe.

## Provenance and safety budgets

| Field | Value |
| --- | --- |
| starting `sht_bench` SHA | `8e139a6b7930691ffd6308786abbbfcafcbf4d8a` |
| result-generation `sht_bench` SHA | `7ad9ecffaa5431a47ff9e5211d6bf863f2c610ef` |
| measurement worker code SHA(s) | `['8e139a6b7930691ffd6308786abbbfcafcbf4d8a']` |
| torch-harmonics current source SHA | `f4cc65515d79d66ff7ab1f02d53518e0b063fc99` |
| historical report source SHA | `f4cc65515d79d66ff7ab1f02d53518e0b063fc99` |
| GPU | NVIDIA GeForce RTX 5070 |
| VRAM | 12227.0 MiB |
| host/cgroup memory limit | 98304.0 MiB (/proc/meminfo:MemTotal) |
| CUDA safety budget | 0.85 of VRAM |
| host safety budget | 0.8 of host/cgroup limit |
| selected finer batch grid | `721x1440x720` |
| record status counts | `{'skipped_predicted_capacity': 18, 'completed': 63}` |
| superseded pre-fix CUDA OOMs | `4` (retained in progress history) |

## Batch-1 resolution scaling

C/B is the C latency divided by B latency: below 1 means C is faster; above 1 means B is faster.

### Scalar

| Grid | N | B ms/frame | C ms/frame | C/B time | B module MiB | C module MiB | B runtime-extra MiB | C runtime-extra MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `361×720` | 360 | 1.005 | 0.686 | 0.682 | 178.5 | 356.9 | 43.9 | 36.0 |
| `481×960` | 480 | 1.849 | 1.502 | 0.812 | 422.8 | 845.5 | 53.8 | 39.0 |
| `601×1200` | 600 | 3.186 | 2.844 | 0.893 | 825.4 | 1650.7 | 65.0 | 43.0 |
| `721×1440` | 720 | 5.164 | 4.836 | 0.936 | 1425.8 | 2851.6 | 79.5 | 47.8 |

### Vector

| Grid | N | B ms/frame | C ms/frame | C/B time | B module MiB | C module MiB | B runtime-extra MiB | C runtime-extra MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `361×720` | 360 | 2.859 | 2.479 | 0.867 | 357.0 | 713.9 | 55.8 | 39.9 |
| `481×960` | 480 | 6.089 | 5.788 | 0.951 | 845.5 | 1691.0 | 74.2 | 46.1 |
| `601×1200` | 600 | 11.478 | 11.186 | 0.974 | 1650.7 | 3301.4 | 98.0 | 54.8 |
| `721×1440` | 720 | 19.494 | 19.102 | 0.980 | 2851.6 | 5703.2 | 127.7 | 63.9 |

## Memory decomposition

`module MiB` is persistent module-buffer storage. `runtime-extra MiB` is the measured forward peak allocated memory minus the CUDA allocation before input allocation. The fixed-plus-per-batch model below is fitted to measured peak-reserved bytes, so it is deliberately conservative for capacity planning.

### Anchor and selected finer grid (`361×720` and `721x1440x720`)

#### `361x720x360`

| Transform | Implementation | module MiB | fixed peak-reserved MiB | per-batch peak-reserved MiB | fixed runtime-extra MiB | per-batch runtime-extra MiB | measured points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| scalar | runtime_fold | 178.5 | 211.4 | 13.8 | 33.4 | 11.9 | b1, b4, b16, b64, b128, b256, b512 |
| scalar | precomputed_projection | 356.9 | 392.1 | 4.0 | 32.7 | 4.0 | b1, b4, b16, b64, b128, b256, b512, b1024 |
| vector | runtime_fold | 357.0 | 384.7 | 27.6 | 34.1 | 23.7 | b1, b4, b16, b64, b128, b256 |
| vector | precomputed_projection | 713.9 | 751.4 | 9.9 | 33.5 | 7.9 | b1, b4, b16, b64, b128, b256, b512, b1024 |

#### `721x1440x720`

| Transform | Implementation | module MiB | fixed peak-reserved MiB | per-batch peak-reserved MiB | fixed runtime-extra MiB | per-batch runtime-extra MiB | measured points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| scalar | runtime_fold | 1425.8 | 1455.7 | 55.2 | 33.2 | 47.5 | b1, b4, b16, b64, b128 |
| scalar | precomputed_projection | 2851.6 | 2886.5 | 15.8 | 33.2 | 15.8 | b1, b4, b16, b64, b128, b256 |
| vector | runtime_fold | 2851.6 | 2868.2 | 110.7 | 35.2 | 94.9 | b1, b4, b16, b64 |
| vector | precomputed_projection | 5703.2 | 5736.0 | 40.0 | 32.6 | 31.7 | b1, b4, b16 |

## Batch scaling

### Maximum safely measured batch

| Grid | Transform | B maximum safe batch | C maximum safe batch |
| --- | --- | ---: | ---: |
| `361×720` | scalar | 512 | 1024 |
| `361×720` | vector | 256 | 1024 |
| `721×1440` | scalar | 128 | 256 |
| `721×1440` | vector | 64 | 16 |

### `361x720x360` scalar

| Batch | B ms/frame | C ms/frame | B peak MiB | C peak MiB | B status | C status |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1.005 | 0.686 | 238.0 | 394.0 | completed | completed |
| 4 | 0.256 | 0.248 | 274.0 | 410.0 | completed | completed |
| 16 | 0.164 | 0.067 | 438.0 | 454.0 | completed | completed |
| 64 | 0.162 | 0.045 | 1078.0 | 646.0 | completed | completed |
| 128 | 0.163 | 0.045 | 1964.0 | 902.0 | completed | completed |
| 256 | 0.165 | 0.046 | 3740.0 | 1408.0 | completed | completed |
| 512 | 0.166 | 0.046 | 7282.0 | 2424.0 | completed | completed |
| 1024 | SKIP | 0.053 | — | 4456.0 | skipped_predicted_capacity | completed |

### `721x1440x720` scalar

| Batch | B ms/frame | C ms/frame | B peak MiB | C peak MiB | B status | C status |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 5.164 | 4.836 | 1520.0 | 2904.0 | completed | completed |
| 4 | 1.537 | 1.734 | 1684.0 | 2948.0 | completed | completed |
| 16 | 1.138 | 0.518 | 2324.0 | 3140.0 | completed | completed |
| 64 | 0.786 | 0.319 | 4984.0 | 3900.0 | completed | completed |
| 128 | 0.793 | 0.320 | 8528.0 | 4916.0 | completed | completed |
| 256 | SKIP | 0.321 | — | 6942.0 | skipped_predicted_capacity | completed |
| 512 | SKIP | SKIP | — | — | skipped_predicted_capacity | skipped_predicted_capacity |
| 1024 | SKIP | SKIP | — | — | skipped_predicted_capacity | skipped_predicted_capacity |

### `361x720x360` vector

| Batch | B ms/frame | C ms/frame | B peak MiB | C peak MiB | B status | C status |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2.859 | 2.479 | 438.0 | 770.0 | completed | completed |
| 4 | 0.764 | 0.943 | 496.0 | 786.0 | completed | completed |
| 16 | 0.576 | 0.281 | 808.0 | 906.0 | completed | completed |
| 64 | 0.392 | 0.197 | 2142.0 | 1386.0 | completed | completed |
| 128 | 0.395 | 0.199 | 3918.0 | 2022.0 | completed | completed |
| 256 | 0.399 | 0.200 | 7460.0 | 3288.0 | completed | completed |
| 512 | SKIP | 0.201 | — | 5828.0 | skipped_predicted_capacity | completed |
| 1024 | SKIP | 0.231 | — | 10904.0 | skipped_predicted_capacity | completed |

### `721x1440x720` vector

| Batch | B ms/frame | C ms/frame | B peak MiB | C peak MiB | B status | C status |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 19.494 | 19.102 | 2990.0 | 5776.0 | completed | completed |
| 4 | 5.833 | 6.998 | 3302.0 | 5896.0 | completed | completed |
| 16 | 3.494 | 2.113 | 4636.0 | 6376.0 | completed | completed |
| 64 | 2.077 | SKIP | 9954.0 | — | completed | skipped_predicted_capacity |
| 128 | SKIP | SKIP | — | — | skipped_predicted_capacity | skipped_predicted_capacity |
| 256 | SKIP | SKIP | — | — | skipped_predicted_capacity | skipped_predicted_capacity |
| 512 | SKIP | SKIP | — | — | skipped_predicted_capacity | skipped_predicted_capacity |
| 1024 | SKIP | SKIP | — | — | skipped_predicted_capacity | skipped_predicted_capacity |

## Float64 probe

The limited float64 probe covers 361×720 and 481×960 at batch 1. The 721×1440 vector C safety-gate record is explicit and was not launched.

- `361x720x360`: scalar/runtime_fold: 1.676; scalar/precomputed_projection: 2.190; vector/runtime_fold: 5.476; vector/precomputed_projection: 8.611
- `481x960x480`: scalar/runtime_fold: 3.429; scalar/precomputed_projection: 3.069; vector/runtime_fold: 12.351; vector/precomputed_projection: 12.037

## Findings

- **scalar C/B batch-1 ratios:** 361: 0.682, 481: 0.812, 601: 0.893, 721: 0.936
- **vector C/B batch-1 ratios:** 361: 0.867, 481: 0.951, 601: 0.974, 721: 0.980
- The batch-1 ratios rise toward 1 at the tested resolutions, so B shows evidence of catching C as resolution increases, but no completed pair crosses above 1 through 721×1440.
- The batch tables contain both C-faster and B-faster cells at the same resolution. Together with the persistent/module and runtime-extra columns, that rejects a resolution-only cutoff for this measured range; the comparison depends on both resolution and batch.
- At 721×1440 scalar, C first uses less total peak VRAM than B at batch 64 (3.9 versus 5.0 GiB); the vector pair has no completed common batch where C's larger projection is outweighed because C is capacity-skipped at batch 64.
- At the selected grid, the fitted runtime-extra slope is about 47.5 MiB per batch for scalar B versus 15.8 MiB for C, and 94.9 versus 31.7 MiB for vector B/C; C's runtime-memory advantage grows with batch even though its persistent module is larger.
- Maximum safe batch is the largest `completed` batch recorded separately for each implementation and transform; a skipped or OOM batch is not counted as safe.

## Resume and raw data

The durable progress manifest is `/home/albert/sht_bench/results/torch-bc-resolution-scaling-progress.json`. A stale `in_progress` entry is reported as `skipped_after_process_kill` and is never retried automatically. Raw JSON and CSV retain timing samples, allocator fields, host RSS fields, source SHA, and all explicit capacity/OOM statuses.

No production `torch-harmonics` files were modified.
