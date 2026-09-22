# u-ORCA performance model

`perf_model.py` estimates AIE latency for a u-ORCA spatial design. The
datatype-specific compute and stage-overhead parameters are calibrated from
the already-compiled projects in the repository.

Regenerate the calibration report without generating, compiling, or
simulating any design:

```bash
python perf_model/perf_model/calibrate.py
```

The script combines the successful measurements in
`generated/exp_20260907/summary.json` with the existing-VCD measurements in
`calibration_measurements.json`. It verifies that every directory containing
`libadf.a` is either measured or explicitly excluded, fits nonnegative
parameters independently for INT4, INT8, INT16, BF16, and FP32, then writes
`calibration_report.json` with per-design predictions and aggregate errors.

All calibration designs enable bias and ReLU. Consequently `L_epi`, `L_br`,
and `L_cas` are not independently identifiable; the calibrated model uses
their observable sum, `L_setup`, for each phi/rho datatype profile.
