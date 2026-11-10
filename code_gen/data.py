"""Create deterministic INT8 test data and check AIE simulator output."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .generator import load_spec
from .spec import ModelSpec

DEFAULT_RUN_ITERATIONS = 20


class DataMismatchError(AssertionError):
    """Raised when an AIE simulator output frame differs from golden data."""


@dataclass(frozen=True)
class QuantizedData:
    """Random quantized tensors plus the INT8 golden output."""

    input_act: np.ndarray
    weights: dict[str, np.ndarray]
    biases: dict[str, np.ndarray]
    golden_output: np.ndarray


def to_block_layout(matrix: np.ndarray, row_block: int, col_block: int) -> np.ndarray:
    """Convert a row-major matrix into tile-major blocked layout."""

    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {matrix.shape}")
    rows, cols = matrix.shape
    if rows % row_block != 0 or cols % col_block != 0:
        raise ValueError(
            f"shape {matrix.shape} is not divisible by block "
            f"({row_block}, {col_block})"
        )

    blocks: list[np.ndarray] = []
    for row in range(0, rows, row_block):
        for col in range(0, cols, col_block):
            blocks.append(matrix[row:row + row_block, col:col + col_block].reshape(-1))
    return np.concatenate(blocks).astype(matrix.dtype, copy=False)


def from_block_layout(
    data: np.ndarray,
    rows: int,
    cols: int,
    row_block: int,
    col_block: int,
) -> np.ndarray:
    """Convert tile-major blocked layout back to row-major layout."""

    flat = np.asarray(data).reshape(-1)
    if flat.size != rows * cols:
        raise ValueError(f"expected {rows * cols} values, got {flat.size}")
    if rows % row_block != 0 or cols % col_block != 0:
        raise ValueError(
            f"shape ({rows}, {cols}) is not divisible by block "
            f"({row_block}, {col_block})"
        )

    matrix = np.empty((rows, cols), dtype=flat.dtype)
    offset = 0
    for row in range(0, rows, row_block):
        for col in range(0, cols, col_block):
            size = row_block * col_block
            matrix[row:row + row_block, col:col + col_block] = flat[
                offset:offset + size
            ].reshape(row_block, col_block)
            offset += size
    return matrix


def write_txt(path: str | Path, data: np.ndarray, values_per_line: int) -> None:
    """Write integer data as whitespace-separated tokens."""

    flat = np.asarray(data).reshape(-1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        for offset in range(0, flat.size, values_per_line):
            line = flat[offset:offset + values_per_line]
            out_file.write(" ".join(str(int(v)) for v in line))
            out_file.write("\n")


def read_tokens(path: Path, *, simulator: bool = False) -> np.ndarray:
    values = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or (simulator and fields[0] in ("T", "TLAST")):
            continue
        values.extend(int(value) for value in fields)
    return np.asarray(values, dtype=np.int64)


def _round_shift_symmetric_inf(values: np.ndarray, shift: int) -> np.ndarray:
    """Round integer values right by shift, with halves away from zero."""

    values = np.asarray(values, dtype=np.int64)
    if shift == 0:
        return values
    offset = 1 << (shift - 1)
    magnitude = np.abs(values)
    rounded = (magnitude + offset) >> shift
    return np.where(values < 0, -rounded, rounded)


def _to_output_dtype(values: np.ndarray, spec: ModelSpec) -> np.ndarray:
    return np.clip(values, -128, 127).astype(np.int8)


def _aie_matmul(
    lhs: np.ndarray,
    rhs: np.ndarray,
    bias: np.ndarray | None,
    *,
    spec: ModelSpec,
    use_bias: bool,
    use_relu: bool,
    scale_shift: int,
) -> np.ndarray:
    """Emulate the current generated matmul tail epilogue in integer space."""

    acc = lhs.astype(np.int64) @ rhs.astype(np.int64)
    if use_bias:
        if bias is None:
            raise ValueError("bias is required for a biased layer")
        acc += bias.astype(np.int64).reshape(1, -1) << scale_shift
    quantized = _to_output_dtype(_round_shift_symmetric_inf(acc, scale_shift), spec)
    if use_relu:
        quantized = np.maximum(quantized, 0).astype(quantized.dtype)
    return quantized


def _aie_aggregation(phi_output: np.ndarray, spec: ModelSpec) -> np.ndarray:
    """Emulate the current generated global aggregation kernel chain."""

    reduced = phi_output.astype(np.int64).sum(axis=0)
    row0 = _to_output_dtype(_round_shift_symmetric_inf(reduced, spec.aggregation_shift), spec)
    rho_input = np.zeros((4, phi_output.shape[1]), dtype=row0.dtype)
    rho_input[0, :] = row0
    return rho_input


def generate_quantized_data(spec: ModelSpec, seed: int) -> QuantizedData:
    """Generate random integer tensors; no trained weights are required."""
    rng = np.random.default_rng(seed)
    first = spec.phi_layers[0]
    input_act = rng.integers(-96, 97, (first.M, first.K), dtype=np.int8)
    weights = {}
    biases = {}
    for name, _stage, _role, layer in spec.named_matmul_layers():
        weights[name] = rng.integers(-64, 65, (layer.K, layer.N), dtype=np.int8)
        biases[name] = rng.integers(-32, 33, layer.N, dtype=np.int8)
    activation = input_act
    for stage, layers in (("phi", spec.named_phi_layers()), ("rho", spec.named_rho_layers())):
        if stage == "rho":
            activation = _aie_aggregation(activation, spec)
        for name, layer in layers:
            activation = _aie_matmul(
                activation, weights[name], biases[name], spec=spec,
                use_bias=layer.use_bias, use_relu=layer.use_relu,
                scale_shift=spec.scale_shift,
            )
    return QuantizedData(input_act, weights, biases, activation)


def _write_first_phi_inputs(
    spec: ModelSpec,
    data: QuantizedData,
    data_dir: Path,
    run_iterations: int,
) -> list[dict[str, object]]:
    first_phi = spec.phi_layers[0]
    records: list[dict[str, object]] = []
    for a_pos in range(first_phi.A):
        row_slice = slice(a_pos * first_phi.rowA, (a_pos + 1) * first_phi.rowA)
        for b_pos in range(first_phi.B):
            col_slice = slice(
                b_pos * first_phi.colA,
                (b_pos + 1) * first_phi.colA,
            )
            partition = data.input_act[row_slice, col_slice]
            frame = to_block_layout(partition, first_phi.BM, first_phi.BK)
            stream = np.tile(frame, run_iterations)
            filename = f"input_{a_pos}_{b_pos}.txt"
            write_txt(data_dir / filename, stream, 16)
            records.append(
                {
                    "kind": "input",
                    "file": f"data/{filename}",
                    "shape": list(partition.shape),
                    "frame_values": int(frame.size),
                    "total_values": int(stream.size),
                }
            )
    return records


def _write_layer_params(
    spec: ModelSpec,
    data: QuantizedData,
    data_dir: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for layer_name, _stage, _role, layer in spec.named_matmul_layers():
        weight = data.weights[layer_name]
        for a_pos in range(layer.A):
            for b_pos in range(layer.B):
                row_slice = slice(b_pos * layer.colA, (b_pos + 1) * layer.colA)
                for c_pos in range(layer.C):
                    col_slice = slice(c_pos * layer.colB, (c_pos + 1) * layer.colB)
                    partition = weight[row_slice, col_slice]
                    blocked = to_block_layout(partition, layer.BK, layer.BN)
                    filename = f"{layer_name}_weight_{a_pos}_{b_pos}_{c_pos}.txt"
                    values_per_line = 16
                    write_txt(data_dir / filename, blocked, values_per_line)
                    records.append(
                        {
                            "kind": "weight",
                            "layer": layer_name,
                            "file": f"data/{filename}",
                            "shape": list(partition.shape),
                            "values": int(blocked.size),
                        }
                    )

        if layer.use_bias:
            bias = data.biases[layer_name].astype(np.int32) << spec.scale_shift
            for a_pos in range(layer.A):
                for c_pos in range(layer.C):
                    col_slice = slice(c_pos * layer.colB, (c_pos + 1) * layer.colB)
                    partition = bias[col_slice]
                    filename = f"{layer_name}_bias_{a_pos}_{c_pos}.txt"
                    write_txt(data_dir / filename, partition, 4)
                    records.append(
                        {
                            "kind": "bias",
                            "layer": layer_name,
                            "file": f"data/{filename}",
                            "shape": [int(partition.size)],
                            "values": int(partition.size),
                        }
                    )
    return records


def _write_golden_outputs(
    spec: ModelSpec,
    data: QuantizedData,
    data_dir: Path,
) -> list[dict[str, object]]:
    final_rho = spec.rho_layers[-1]
    records: list[dict[str, object]] = []
    for a_pos in range(final_rho.A):
        row_slice = slice(a_pos * final_rho.rowA, (a_pos + 1) * final_rho.rowA)
        for c_pos in range(final_rho.C):
            col_slice = slice(c_pos * final_rho.colB, (c_pos + 1) * final_rho.colB)
            partition = data.golden_output[row_slice, col_slice]
            blocked = to_block_layout(partition, final_rho.BM, final_rho.BN)
            filename = f"golden_{a_pos}_{c_pos}.txt"
            write_txt(data_dir / filename, blocked, 16)
            records.append(
                {
                    "kind": "golden",
                    "file": f"data/{filename}",
                    "shape": list(partition.shape),
                    "values": int(blocked.size),
                }
            )
    return records


def generate_data_files(
    spec: ModelSpec,
    project_dir: str | Path,
    *,
    seed: int = 0,
    run_iterations: int = DEFAULT_RUN_ITERATIONS,
) -> dict[str, object]:
    """Generate data files under a prepared AIE project directory."""

    if run_iterations <= 0:
        raise ValueError("run_iterations must be positive")
    project_path = Path(project_dir)
    data_dir = project_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    quantized = generate_quantized_data(spec, seed)
    files = []
    files.extend(_write_first_phi_inputs(spec, quantized, data_dir, run_iterations))
    files.extend(_write_layer_params(spec, quantized, data_dir))
    files.extend(_write_golden_outputs(spec, quantized, data_dir))

    manifest = {
        "graph_name": spec.graph_name,
        "seed": seed,
        "run_iterations": run_iterations,
        "datatype": spec.datatype,
        "block_sizes": {
            "BM": spec.datatype_profile.BM,
            "BK": spec.datatype_profile.BK,
            "BN": spec.datatype_profile.BN,
        },
        "scale_shift": spec.scale_shift,
        "aggregation_shift": spec.aggregation_shift,
        "files": files,
    }
    manifest_path = data_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def compare_outputs(spec: ModelSpec, project_dir: str | Path) -> int:
    """Require every simulated frame to match the integer reference exactly."""
    project = Path(project_dir)
    manifest = json.loads((project / "data/data_manifest.json").read_text())
    expected_frames = manifest["run_iterations"]
    checked = 0
    final = spec.rho_layers[-1]
    for a in range(final.A):
        for c in range(final.C):
            golden = read_tokens(project / "data" / f"golden_{a}_{c}.txt")
            output = read_tokens(project / "aiesimulator_output/data" / f"output_{a}_{c}.txt", simulator=True)
            if golden.size == 0 or output.size != golden.size * expected_frames:
                raise DataMismatchError(f"Expected {expected_frames} complete output frames")
            frames = output.reshape(expected_frames, golden.size)
            if not np.all(frames == golden):
                frame, position = np.argwhere(frames != golden)[0]
                raise DataMismatchError(
                    f"Frame {frame}, value {position}: expected {golden[position]}, got {frames[frame, position]}"
                )
            checked += expected_frames
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "compare"))
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        spec = load_spec(args.project_dir / "uorca_spec.json")
        if args.command == "generate":
            generate_data_files(spec, args.project_dir, seed=args.seed)
            print(f"Wrote random INT8 test data and golden outputs to {args.project_dir / 'data'}")
        else:
            count = compare_outputs(spec, args.project_dir)
            print(f"Correctness check passed: {count} output frames match exactly")
    except (ValueError, KeyError, OSError, DataMismatchError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
