#!/usr/bin/env python3
"""Calibrate datatype-specific u-ORCA latency parameters from existing runs.

The script is intentionally offline: it reads already-recorded summary values
and existing-VCD measurements from ``calibration_measurements.json``.  It does
not generate a project, compile a graph, or invoke the simulator.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY = REPO_ROOT / "generated" / "exp_20260907" / "summary.json"
DEFAULT_MANIFEST = MODEL_DIR / "calibration_measurements.json"
DEFAULT_REPORT = MODEL_DIR / "calibration_report.json"

DATATYPES = ("int4", "int8", "int16", "bf16", "fp32")
PERF_DATATYPE = {
    "int4": "int4",
    "int8": "int8",
    "int16": "int16",
    "bf16": "bf16",
    "fp32": "fp32_safe",
}
TILES = {
    "int4": (4, 16, 8),
    "int8": (4, 8, 8),
    "int16": (4, 4, 4),
    "bf16": (4, 8, 4),
    "fp32": (4, 8, 4),
}

# [mac_cycles, phi_setup_cycles, phi_layer_overhead_cycles,
#  rho_setup_cycles, rho_layer_overhead_cycles]
BEFORE = {
    "int4": np.array([1, 17, 0, 12, 59]),
    "int8": np.array([1, 17, 0, 12, 59]),
    "int16": np.array([1, 17, 0, 12, 59]),
    "bf16": np.array([1, 17, 0, 12, 59]),
    # The working tree already contained the trace-calibrated value 38.
    "fp32": np.array([38, 17, 0, 12, 59]),
}
PARAMETER_NAMES = (
    "mac_cycles",
    "phi_setup_cycles",
    "phi_layer_overhead_cycles",
    "rho_setup_cycles",
    "rho_layer_overhead_cycles",
)
FIXED_PARAMETERS_BY_DATATYPE = {
    "int4": {"mac_cycles": 1},
    "int8": {"mac_cycles": 1},
    "int16": {"mac_cycles": 1},
    "bf16": {"mac_cycles": 1},
    "fp32": {},
}
GLOBAL_AGGREGATION_NS = 150
AIE_CYCLES_PER_NS = 1.25
INTEGER_SEARCH_SEED = 42
INTEGER_SEARCH_PERTURBATION = np.array([5, 30, 60, 30, 60])
INTEGER_SEARCH_START_COUNT = 15
LOSS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Sample:
    name: str
    project: str
    datatype: str
    phi: list[list[int]]
    rho: list[list[int]]
    actual_latency_ns: float
    measurement_source: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_project(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def load_samples(summary_path: Path, manifest_path: Path) -> tuple[list[Sample], dict[str, Any]]:
    summary = _read_json(summary_path)
    records = {record["task_key"]: record for record in summary["records"]}
    samples: list[Sample] = []

    for record in summary["records"]:
        if record.get("status") != "success":
            continue
        spec = record["design_spec"]
        samples.append(
            Sample(
                name=record["task_key"],
                project=str(_resolve_project(record["project_dir"]).relative_to(REPO_ROOT)),
                datatype=record["datatype"],
                phi=spec["phi"],
                rho=spec["rho"],
                actual_latency_ns=float(record["latency_ns"]),
                measurement_source="exp_20260907_summary",
            )
        )

    manifest = _read_json(manifest_path)
    for measurement in manifest["measurements"]:
        if "task_key" in measurement:
            record = records[measurement["task_key"]]
            spec = record["design_spec"]
            datatype = record["datatype"]
        else:
            spec = _read_json(REPO_ROOT / measurement["spec_path"])
            datatype = spec["datatype"]
        samples.append(
            Sample(
                name=measurement["project"],
                project=measurement["project"],
                datatype=datatype,
                phi=spec["phi"],
                rho=spec["rho"],
                actual_latency_ns=float(measurement["latency_ns"]),
                measurement_source="existing_vcd",
            )
        )

    names = [sample.name for sample in samples]
    if len(names) != len(set(names)):
        raise ValueError("duplicate calibration sample name")
    return samples, manifest


def validate_compiled_coverage(samples: Sequence[Sample], manifest: dict[str, Any]) -> dict[str, Any]:
    compiled = {
        path.parent.resolve()
        for path in REPO_ROOT.rglob("libadf.a")
        if ".git" not in path.parts
    }
    measured = {_resolve_project(sample.project) for sample in samples}
    excluded = {_resolve_project(item["project"]) for item in manifest["excluded"]}

    missing_artifacts = sorted(str(path) for path in measured if path not in compiled)
    if missing_artifacts:
        raise ValueError(f"measurement has no libadf.a: {missing_artifacts}")
    uncovered = sorted(str(path.relative_to(REPO_ROOT)) for path in compiled - measured - excluded)
    if uncovered:
        raise ValueError(f"compiled projects are not covered by calibration: {uncovered}")
    unexpected_exclusions = sorted(
        str(path.relative_to(REPO_ROOT)) for path in excluded - compiled
    )
    if unexpected_exclusions:
        raise ValueError(f"excluded projects are not compiled: {unexpected_exclusions}")

    return {
        "compiled_design_count": len(compiled),
        "measured_design_count": len(measured),
        "excluded_design_count": len(excluded),
        "uncovered_compiled_design_count": 0,
    }


def feature_vector(sample: Sample) -> np.ndarray:
    bm, bk, bn = TILES[sample.datatype]
    stage_features: list[tuple[float, float, float]] = []
    for layers in (sample.phi, sample.rho):
        mac = 0.0
        setup = 0.0
        for m, k, n, a, b, c, _bias, _relu in layers:
            local_m = math.ceil(m / a)
            local_k = math.ceil(k / b)
            local_n = math.ceil(n / c)
            repetitions = local_m * local_n / (4 * bm * bn) + b - 1
            mac += repetitions * 4 * local_k / bk
            setup += repetitions
        stage_features.append(
            (
                mac / AIE_CYCLES_PER_NS,
                setup / AIE_CYCLES_PER_NS,
                len(layers) / AIE_CYCLES_PER_NS,
            )
        )
    phi, rho = stage_features
    return np.array([phi[0] + rho[0], phi[1], phi[2], rho[1], rho[2]])


def _relative_nnls_start(
    features: np.ndarray, actual: np.ndarray, offset: np.ndarray
) -> np.ndarray:
    """Solve the reduced nonnegative relative least-squares start exactly."""
    matrix = features / actual[:, None]
    target = (actual - offset) / actual
    best_loss = float("inf")
    best: np.ndarray | None = None
    parameter_count = features.shape[1]
    for size in range(1, parameter_count + 1):
        for active in itertools.combinations(range(parameter_count), size):
            coefficients = np.linalg.lstsq(matrix[:, active], target, rcond=None)[0]
            if np.any(coefficients < -1e-9):
                continue
            candidate = np.zeros(parameter_count)
            candidate[list(active)] = np.maximum(0.0, coefficients)
            loss = float(np.sum((matrix @ candidate - target) ** 2))
            if loss < best_loss:
                best_loss = loss
                best = candidate
    if best is None:
        raise RuntimeError("nonnegative calibration fit failed")
    return best


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, weights.sum() / 2, side="left"))
    return float(sorted_values[index])


def _mape(
    features: np.ndarray,
    actual: np.ndarray,
    params: np.ndarray,
    offset: np.ndarray,
) -> float:
    predicted = features @ params + offset
    return float(np.mean(np.abs(predicted - actual) / actual) * 100)


def _fit_continuous_mape(
    features: np.ndarray,
    actual: np.ndarray,
    offset: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """Find a continuous MAPE solution used only to seed integer search."""
    start = _relative_nnls_start(features, actual, offset)
    rng = np.random.default_rng(3)
    best = start.copy()
    best_loss = _mape(features, actual, best, offset)
    starts = (
        start,
        baseline.astype(float),
        np.zeros(features.shape[1]),
    )
    for initial in starts:
        params = initial.copy()
        for _ in range(10_000):
            previous = params.copy()
            order = np.arange(features.shape[1])
            rng.shuffle(order)
            for column in order:
                residual = actual - offset - (
                    features @ params - features[:, column] * params[column]
                )
                nonzero = features[:, column] > 0
                candidates = residual[nonzero] / features[nonzero, column]
                weights = features[nonzero, column] / actual[nonzero]
                params[column] = max(0.0, _weighted_median(candidates, weights))
            loss = _mape(features, actual, params, offset)
            if loss < best_loss:
                best_loss = loss
                best = params.copy()
            if float(np.max(np.abs(params - previous))) < 1e-10:
                break
    return best


def _is_better_integer_solution(
    candidate: np.ndarray,
    candidate_loss: float,
    incumbent: np.ndarray,
    incumbent_loss: float,
) -> bool:
    """Compare solutions by MAPE, then lexicographically for stable ties."""
    if candidate_loss < incumbent_loss - LOSS_TOLERANCE:
        return True
    return (
        abs(candidate_loss - incumbent_loss) <= LOSS_TOLERANCE
        and tuple(candidate.tolist()) < tuple(incumbent.tolist())
    )


def _safe_integer_upper_bounds(
    features: np.ndarray,
    actual: np.ndarray,
    offset: np.ndarray,
    incumbent_loss: float,
) -> np.ndarray:
    """Bound every parameter for any solution that can beat ``incumbent_loss``.

    All features and parameters are nonnegative.  If a candidate has lower
    MAPE than the incumbent, each sample's relative error must be smaller than
    the incumbent's total relative-error sum.  Applying that fact to the
    contribution of one parameter gives a finite, conservative integer bound.
    """
    relative_error_budget = len(actual) * incumbent_loss / 100
    bounds: list[int] = []
    for column in range(features.shape[1]):
        nonzero = features[:, column] > 0
        if not np.any(nonzero):
            bounds.append(0)
            continue
        limits = (
            actual[nonzero] * (1 + relative_error_budget)
            - offset[nonzero]
        ) / features[nonzero, column]
        bounds.append(max(0, math.floor(float(np.min(limits)))))
    return np.array(bounds, dtype=int)


def _best_integer_coordinate(
    features: np.ndarray,
    actual: np.ndarray,
    offset: np.ndarray,
    params: np.ndarray,
    column: int,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Exactly minimize one integer coordinate with all others fixed."""
    base = (
        features @ params
        - features[:, column] * params[column]
        + offset
    )
    nonzero = features[:, column] > 0
    values = (actual[nonzero] - base[nonzero]) / features[nonzero, column]
    weights = features[nonzero, column] / actual[nonzero]
    median = max(0.0, _weighted_median(values, weights))
    upper = int(upper_bounds[column])
    choices = {
        0,
        upper,
        int(params[column]),
        min(upper, max(0, math.floor(median))),
        min(upper, max(0, math.ceil(median))),
    }

    best = params.copy()
    best_loss = _mape(features, actual, best, offset)
    for value in sorted(choices):
        candidate = params.copy()
        candidate[column] = value
        candidate_loss = _mape(features, actual, candidate, offset)
        if _is_better_integer_solution(
            candidate, candidate_loss, best, best_loss
        ):
            best = candidate
            best_loss = candidate_loss
    return best, best_loss


def _best_integer_pair(
    features: np.ndarray,
    actual: np.ndarray,
    offset: np.ndarray,
    params: np.ndarray,
    first: int,
    second: int,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Exactly minimize two integer coordinates with all others fixed."""
    best = params.copy()
    best_loss = _mape(features, actual, best, offset)
    for value in range(int(upper_bounds[first]) + 1):
        candidate = params.copy()
        candidate[first] = value
        candidate, candidate_loss = _best_integer_coordinate(
            features, actual, offset, candidate, second, upper_bounds
        )
        if _is_better_integer_solution(
            candidate, candidate_loss, best, best_loss
        ):
            best = candidate
            best_loss = candidate_loss
    return best, best_loss


def _polish_integer_mape(
    features: np.ndarray,
    actual: np.ndarray,
    offset: np.ndarray,
    initial: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run exact one- and two-coordinate integer updates to convergence."""
    params = np.clip(np.asarray(initial, dtype=int), 0, upper_bounds)
    loss = _mape(features, actual, params, offset)
    for _ in range(100):
        changed = False
        for column in range(features.shape[1]):
            candidate, candidate_loss = _best_integer_coordinate(
                features, actual, offset, params, column, upper_bounds
            )
            if _is_better_integer_solution(
                candidate, candidate_loss, params, loss
            ):
                params = candidate
                loss = candidate_loss
                changed = True
        for first, second in itertools.combinations(
            range(features.shape[1]), 2
        ):
            candidate, candidate_loss = _best_integer_pair(
                features,
                actual,
                offset,
                params,
                first,
                second,
                upper_bounds,
            )
            if _is_better_integer_solution(
                candidate, candidate_loss, params, loss
            ):
                params = candidate
                loss = candidate_loss
                changed = True
        if not changed:
            return params, loss
    raise RuntimeError("integer MAPE search did not converge")


def _fit_free_mape(
    features: np.ndarray,
    actual: np.ndarray,
    offset: np.ndarray,
    baseline: np.ndarray,
    perturbation: np.ndarray,
) -> np.ndarray:
    """Search the free nonnegative integer parameters for minimum MAPE."""
    continuous = _fit_continuous_mape(
        features, actual, offset, baseline
    )
    starts = [
        np.rint(continuous),
        np.floor(continuous),
        np.ceil(continuous),
        baseline,
        np.zeros(features.shape[1]),
    ]
    rng = np.random.default_rng(INTEGER_SEARCH_SEED)
    for _ in range(INTEGER_SEARCH_START_COUNT - len(starts)):
        starts.append(
            np.maximum(
                0,
                np.rint(
                    continuous
                    + rng.normal(0, perturbation)
                ),
            )
        )

    start_losses = [
        _mape(features, actual, start, offset) for start in starts
    ]
    best_index = int(np.argmin(start_losses))
    best = np.asarray(starts[best_index], dtype=int)
    best_loss = start_losses[best_index]
    upper_bounds = _safe_integer_upper_bounds(
        features, actual, offset, best_loss
    )

    for start in starts:
        candidate, candidate_loss = _polish_integer_mape(
            features, actual, offset, start, upper_bounds
        )
        if _is_better_integer_solution(
            candidate, candidate_loss, best, best_loss
        ):
            best = candidate
            best_loss = candidate_loss
    return best


def _fixed_parameter_indices(
    fixed_parameters: dict[str, int],
) -> dict[int, int]:
    indices: dict[int, int] = {}
    for name, value in fixed_parameters.items():
        if name not in PARAMETER_NAMES:
            raise ValueError(f"unknown fixed calibration parameter: {name}")
        if type(value) is not int or value < 0:
            raise ValueError(
                f"fixed parameter {name} must be a nonnegative integer"
            )
        indices[PARAMETER_NAMES.index(name)] = value
    return indices


def fit_mape(
    features: np.ndarray,
    actual: np.ndarray,
    baseline: np.ndarray,
    fixed_parameters: dict[str, int],
) -> np.ndarray:
    """Search integer parameters while enforcing datatype-specific constants.

    The continuous fit is used only to construct deterministic starts. Every
    returned free value comes from an integer-domain search, not from rounding
    the continuous result. Fixed parameters are removed from the optimization
    variables and included in each sample's constant offset.
    """
    fixed = _fixed_parameter_indices(fixed_parameters)
    free_columns = [
        column
        for column in range(len(PARAMETER_NAMES))
        if column not in fixed
    ]
    result = np.zeros(len(PARAMETER_NAMES), dtype=int)
    offset = np.full(len(actual), GLOBAL_AGGREGATION_NS, dtype=float)
    for column, value in fixed.items():
        result[column] = value
        offset += features[:, column] * value
    if not free_columns:
        return result

    free_fit = _fit_free_mape(
        features[:, free_columns],
        actual,
        offset,
        baseline[free_columns],
        INTEGER_SEARCH_PERTURBATION[free_columns],
    )
    result[free_columns] = free_fit
    return result


def _constrained_safe_integer_upper_bounds(
    features: np.ndarray,
    actual: np.ndarray,
    params: np.ndarray,
    fixed_parameters: dict[str, int],
) -> np.ndarray:
    """Return safe bounds for free parameters and singleton fixed domains."""
    fixed = _fixed_parameter_indices(fixed_parameters)
    free_columns = [
        column
        for column in range(len(PARAMETER_NAMES))
        if column not in fixed
    ]
    bounds = np.zeros(len(PARAMETER_NAMES), dtype=int)
    offset = np.full(len(actual), GLOBAL_AGGREGATION_NS, dtype=float)
    for column, value in fixed.items():
        bounds[column] = value
        offset += features[:, column] * value
    if free_columns:
        incumbent_loss = _mape(
            features, actual, params, np.full(len(actual), GLOBAL_AGGREGATION_NS)
        )
        bounds[free_columns] = _safe_integer_upper_bounds(
            features[:, free_columns], actual, offset, incumbent_loss
        )
    return bounds


def _parameter_mapping(values: np.ndarray) -> dict[str, int]:
    result = {
        name: int(value)
        for name, value in zip(PARAMETER_NAMES, values)
    }
    result["global_aggregation_ns"] = GLOBAL_AGGREGATION_NS
    result["phi_cascade_inter_layer_cycles"] = 0
    result["rho_cascade_inter_layer_cycles"] = 0
    return result


def _error_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mean_absolute_percentage_error_percent": round(
            float(np.mean(np.abs(error) / actual) * 100), 9
        ),
        "mean_absolute_error_ns": round(float(np.mean(np.abs(error))), 9),
        "mean_signed_percentage_error_percent": round(
            float(np.mean(error / actual) * 100), 9
        ),
        "root_mean_squared_percentage_error_percent": round(
            float(np.sqrt(np.mean((error / actual) ** 2)) * 100), 9
        ),
    }


def build_report(samples: Sequence[Sample], inventory: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    fits: dict[str, np.ndarray] = {}
    search_bounds: dict[str, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    designs: list[dict[str, Any]] = []

    for datatype in DATATYPES:
        selected = [sample for sample in samples if sample.datatype == datatype]
        features = np.stack([feature_vector(sample) for sample in selected])
        actual = np.array([sample.actual_latency_ns for sample in selected])
        after = fit_mape(
            features,
            actual,
            BEFORE[datatype],
            FIXED_PARAMETERS_BY_DATATYPE[datatype],
        )
        fits[datatype] = after
        search_bounds[datatype] = _constrained_safe_integer_upper_bounds(
            features,
            actual,
            after,
            FIXED_PARAMETERS_BY_DATATYPE[datatype],
        )
        before_prediction = features @ BEFORE[datatype] + GLOBAL_AGGREGATION_NS
        after_prediction = features @ after + GLOBAL_AGGREGATION_NS
        signatures = {
            json.dumps([sample.phi, sample.rho], separators=(",", ":"))
            for sample in selected
        }
        summaries[datatype] = {
            "design_count": len(selected),
            "unique_spatial_config_count": len(signatures),
            "before": _error_summary(actual, before_prediction),
            "after": _error_summary(actual, after_prediction),
        }
        for sample, predicted_before, predicted_after in zip(
            selected, before_prediction, after_prediction
        ):
            designs.append(
                {
                    "name": sample.name,
                    "project": sample.project,
                    "datatype": datatype,
                    "measurement_source": sample.measurement_source,
                    "actual_latency_ns": sample.actual_latency_ns,
                    "before_predicted_latency_ns": round(float(predicted_before), 9),
                    "before_absolute_percentage_error_percent": round(
                        abs(float(predicted_before) - sample.actual_latency_ns)
                        / sample.actual_latency_ns
                        * 100,
                        9,
                    ),
                    "after_predicted_latency_ns": round(float(predicted_after), 9),
                    "after_absolute_percentage_error_percent": round(
                        abs(float(predicted_after) - sample.actual_latency_ns)
                        / sample.actual_latency_ns
                        * 100,
                        9,
                    ),
                }
            )

    designs.sort(key=lambda item: (item["datatype"], item["name"]))
    actual_all = np.array([item["actual_latency_ns"] for item in designs])
    before_all = np.array([item["before_predicted_latency_ns"] for item in designs])
    after_all = np.array([item["after_predicted_latency_ns"] for item in designs])
    inventory = dict(inventory)
    inventory["counts_by_datatype"] = {
        datatype: sum(sample.datatype == datatype for sample in samples)
        for datatype in DATATYPES
    }

    return {
        "methodology": {
            "target_metric": (
                "mean(abs(predicted_latency_ns - actual_latency_ns) / "
                "actual_latency_ns) * 100"
            ),
            "fit": "nonnegative integer datatype-specific MAPE search",
            "parameter_domain": "nonnegative integers",
            "fixed_parameters_by_datatype": FIXED_PARAMETERS_BY_DATATYPE,
            "integer_search": {
                "deterministic_start_count": INTEGER_SEARCH_START_COUNT,
                "continuous_fit_used_only_as_search_seed": True,
                "updates": (
                    "exact conditional one- and two-parameter integer "
                    "minimization to convergence"
                ),
                "better_solution_safe_upper_bounds_by_datatype": {
                    datatype: {
                        name: int(bound)
                        for name, bound in zip(
                            PARAMETER_NAMES,
                            search_bounds[datatype],
                        )
                    }
                    for datatype in DATATYPES
                },
            },
            "global_aggregation_ns_fixed": GLOBAL_AGGREGATION_NS,
            "all_layers_have_bias_and_relu": True,
            "identifiability_note": (
                "L_epi, L_br, and L_cas are reported and fitted as their "
                "identifiable sum, setup_cycles, because every measured layer "
                "has bias=1."
            ),
            "unchanged_non_aie_only_constants": {
                "aie_frequency_mhz": 1250,
                "pl_frequency_mhz": 300,
                "dma_bandwidth_bytes_per_cycle": 4,
                "dma_startup_cycles": 30,
                "branch_relu_latency_cycles": 2,
                "reason": (
                    "The measured VCD target is graph AIE-only latency, so "
                    "host/PL/DMA constants are not observable in this dataset."
                ),
            },
            "no_generation_compilation_or_simulation": True,
        },
        "inventory": inventory,
        "excluded": manifest["excluded"],
        "hyperparameters": {
            datatype: {
                "performance_datatype": PERF_DATATYPE[datatype],
                "before": _parameter_mapping(BEFORE[datatype]),
                "after": _parameter_mapping(fits[datatype]),
            }
            for datatype in DATATYPES
        },
        "errors_by_datatype": summaries,
        "all_designs": {
            "design_count": len(designs),
            "before": _error_summary(actual_all, before_all),
            "after": _error_summary(actual_all, after_all),
        },
        "designs": designs,
    }


def verify_model_matches_report(report: dict[str, Any]) -> None:
    from perf_model import DATATYPE_CONFIGS, get_stage_params

    for datatype in DATATYPES:
        perf_datatype = PERF_DATATYPE[datatype]
        phi = get_stage_params(perf_datatype, "phi")
        rho = get_stage_params(perf_datatype, "rho")
        model_values = np.array(
            [
                DATATYPE_CONFIGS[perf_datatype]["mac_cycles"],
                phi["L_setup"],
                phi["L_o"],
                rho["L_setup"],
                rho["L_o"],
            ]
        )
        report_values = np.array(
            [report["hyperparameters"][datatype]["after"][name] for name in PARAMETER_NAMES]
        )
        if not all(
            isinstance(value, (int, np.integer))
            for value in model_values.tolist()
        ):
            raise ValueError(
                f"{datatype} model parameters are not all integers: "
                f"{model_values.tolist()}"
            )
        if not np.array_equal(model_values, report_values):
            raise ValueError(
                f"{datatype} model parameters differ from calibration: "
                f"model={model_values.tolist()}, fit={report_values.tolist()}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    samples, manifest = load_samples(args.summary.resolve(), args.manifest.resolve())
    inventory = validate_compiled_coverage(samples, manifest)
    report = build_report(samples, inventory, manifest)
    if not args.skip_model_check:
        verify_model_matches_report(report)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.resolve().write_text(encoded, encoding="utf-8")
    if args.stdout:
        print(encoded, end="")
    else:
        print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
