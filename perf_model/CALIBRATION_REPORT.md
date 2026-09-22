# u-ORCA performance-model calibration

## Scope and metric

- Compiled designs discovered by `libadf.a`: **154**.
- Designs with a valid measured first-iteration latency: **153**.
- Explicitly excluded: **1** (`generated/exp_20260907/nparticle/N032/fp32`), because its existing VCD ends before any complete graph-kernel invocation.
- No design was generated, compiled, or simulated during calibration.
- Error metric: `mean(abs(predicted_ns - actual_ns) / actual_ns) * 100` (MAPE).
- Every fitted hyperparameter is constrained to a nonnegative integer.
- `mac_cycles` is fixed at 1 for INT4, INT8, INT16, and BF16; it is removed
  from the optimization variables for those datatypes.
- The baseline is the working-tree model at the start of calibration. In particular, FP32 already used `mac_cycles = 38` (not the nested repository's committed value 9).

The integer result was searched directly rather than obtained by rounding the
continuous fit. The search used 15 deterministic starts and exact conditional
one- and two-parameter integer minimization to convergence. An additional
conditional three-parameter scan over conservative bounds found no improving
neighbor. Fixed parameters were included as constants in every prediction,
not overwritten after fitting.

Every measured layer has `bias=1`, so the original `L_epi`, `L_br`, and
`L_cas` terms cannot be identified separately. Their observable sum is
reported as `setup_cycles`:

- baseline phi setup = `2 + 1 + 14 = 17` cycles;
- baseline rho setup = `2 + 2 + 8 = 12` cycles.

`global_aggregation_ns` remains 150 ns and both phi/rho `O_cas` terms remain
0 cycles for every datatype.

The AIE-only traces do not observe host/PL/DMA communication, so the model's
AIE frequency (1250 MHz), PL frequency (300 MHz), DMA bandwidth (4 B/cycle),
DMA startup (30 cycles), and branch/ReLU communication term (2 cycles) remain
unchanged. Datatype tile sizes and bytes per element are hardware properties,
not fitted hyperparameters, and also remain unchanged.

## Hyperparameters: before -> after

| Datatype | `mac_cycles` | phi `setup_cycles` | phi `L_o` cycles | rho `setup_cycles` | rho `L_o` cycles |
|---|---:|---:|---:|---:|---:|
| INT4 | 1 -> 1 | 17 -> 30 | 0 -> 1 | 12 -> 1 | 59 -> 19 |
| INT8 | 1 -> 1 | 17 -> 15 | 0 -> 9 | 12 -> 18 | 59 -> 10 |
| INT16 | 1 -> 1 | 17 -> 18 | 0 -> 13 | 12 -> 42 | 59 -> 3 |
| BF16 | 1 -> 1 | 17 -> 28 | 0 -> 1 | 12 -> 2 | 59 -> 44 |
| FP32 (`fp32_safe`) | 38 -> 38 | 17 -> 12 | 0 -> 1 | 12 -> 0 | 59 -> 237 |

## Error: before -> after

| Datatype | Measured designs | Unique spatial configs | MAPE before | MAPE after |
|---|---:|---:|---:|---:|
| INT4 | 23 | 11 | 11.821185512% | 5.728878559% |
| INT8 | 87 | 71 | 14.017713329% | 2.676888133% |
| INT16 | 14 | 9 | 8.242127926% | 2.942649604% |
| BF16 | 17 | 8 | 14.420346420% | 1.818725445% |
| FP32 | 12 | 8 | 3.881335678% | 1.163552140% |
| **All designs (micro-average)** | **153** | **107** | **12.408758396%** | **2.945957694%** |

The machine-readable `calibration_report.json` contains the actual latency,
before/after prediction, and before/after absolute percentage error for every
included design.
