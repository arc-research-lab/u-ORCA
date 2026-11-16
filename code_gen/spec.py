"""DeepSets/MLP specifications and datatype-specific shape rules shared with DSE."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

Stage = Literal["phi", "rho", "mlp"]
Role = Literal["first", "body", "last", "single"]


class SpecError(ValueError):
    """Raised when a model cannot be represented by the generator."""


@dataclass(frozen=True)
class DatatypeProfile:
    name: str = "int8"
    input_cpp_type: str = "int8"
    weight_cpp_type: str = "int8"
    bias_cpp_type: str = "int32"
    accumulator_cpp_type: str = "acc32"
    output_cpp_type: str = "int8"
    BM: int = 4
    BK: int = 8
    BN: int = 8
    matmul_shift: int = 7
    aggregation_shift: int = 6

    @property
    def floating(self) -> bool:
        return self.name in ("bf16", "fp32")

    @property
    def activation_bytes(self) -> int:
        return {"int8": 1, "int4": 1, "int16": 2, "bf16": 2, "fp32": 4}[self.name]

    @property
    def cascade_k_multiple(self) -> int:
        # INT8/INT16 pack two K tiles into one cascade transfer.
        return 2 if self.name in ("int8", "int16") else 1


INT8_PROFILE = DatatypeProfile()
INT4_PROFILE = DatatypeProfile(name="int4", weight_cpp_type="int4", BK=16, matmul_shift=3)
INT16_PROFILE = DatatypeProfile(
    name="int16", input_cpp_type="int16", weight_cpp_type="int16",
    accumulator_cpp_type="acc64", output_cpp_type="int16", BK=4, BN=4,
    matmul_shift=15, aggregation_shift=14,
)
BF16_PROFILE = DatatypeProfile(
    name="bf16", input_cpp_type="bfloat16", weight_cpp_type="bfloat16",
    bias_cpp_type="float", accumulator_cpp_type="accfloat", output_cpp_type="bfloat16",
    BN=4, matmul_shift=0, aggregation_shift=0,
)
FP32_PROFILE = DatatypeProfile(
    name="fp32", input_cpp_type="float", weight_cpp_type="float",
    bias_cpp_type="float", accumulator_cpp_type="accfloat", output_cpp_type="float",
    BN=4, matmul_shift=0, aggregation_shift=0,
)
DATATYPE_PROFILES = {profile.name: profile for profile in (
    INT8_PROFILE, INT4_PROFILE, INT16_PROFILE, BF16_PROFILE, FP32_PROFILE,
)}


def resolve_datatype_profile(datatype: object) -> DatatypeProfile:
    if not isinstance(datatype, str) or datatype not in DATATYPE_PROFILES:
        raise SpecError(f"Unsupported datatype {datatype!r}; choose {', '.join(DATATYPE_PROFILES)}")
    return DATATYPE_PROFILES[datatype]


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SpecError(f"{name} must be a positive integer, got {value!r}")


def _require_binary_flag(name: str, value: int) -> None:
    if type(value) is not int or value not in (0, 1):
        raise SpecError(f"{name} must be 0 or 1, got {value!r}")


def _require_divisible(name: str, numerator: int, denominator: int) -> None:
    if numerator % denominator:
        raise SpecError(f"{name} must divide evenly: {numerator} / {denominator}")


def _require_even(name: str, value: int, formula: str) -> None:
    if value % 2:
        raise SpecError(f"{name}={value} must be even ({formula})")


@dataclass(frozen=True)
class LayerInput:
    """The unchanged eight-entry model-layer input."""

    M: int
    K: int
    N: int
    A: int
    B: int
    C: int
    bias: int
    relu: int

    @classmethod
    def from_sequence(cls, values: Sequence[int]) -> "LayerInput":
        if len(values) != 8:
            raise SpecError(
                "layer input must have 8 entries: [M, K, N, A, B, C, bias, relu]"
            )
        return cls(*values)

    def validate(self, label: str) -> None:
        for field_name in ("M", "K", "N", "A", "B", "C"):
            _require_positive_int(f"{label}.{field_name}", getattr(self, field_name))
        _require_binary_flag(f"{label}.bias", self.bias)
        _require_binary_flag(f"{label}.relu", self.relu)
        _require_divisible(f"{label}.A", self.M, self.A)
        _require_divisible(f"{label}.B", self.K, self.B)
        _require_divisible(f"{label}.C", self.N, self.C)


@dataclass(frozen=True)
class LayerSpec:
    """One logical layer and its datatype-selected MMUL tile."""

    input: LayerInput
    BM: int
    BK: int
    BN: int

    @property
    def M(self) -> int:
        return self.input.M

    @property
    def K(self) -> int:
        return self.input.K

    @property
    def N(self) -> int:
        return self.input.N

    @property
    def A(self) -> int:
        return self.input.A

    @property
    def B(self) -> int:
        return self.input.B

    @property
    def C(self) -> int:
        return self.input.C

    @property
    def use_bias(self) -> bool:
        return bool(self.input.bias)

    @property
    def use_relu(self) -> bool:
        return bool(self.input.relu)

    @property
    def rowA(self) -> int:
        return self.M // self.A

    @property
    def colA(self) -> int:
        return self.K // self.B

    @property
    def colB(self) -> int:
        return self.N // self.C

    @property
    def bound_i(self) -> int:
        return self.rowA // self.BM

    @property
    def bound_k(self) -> int:
        return self.colA // self.BK

    @property
    def bound_j(self) -> int:
        return self.colB // self.BN


@dataclass(frozen=True)
class AggregationSpec:
    """Sum aggregation derived from the final phi layer."""

    PM: int
    PN: int
    A: int
    BM: int
    BN: int

    @property
    def bound_i(self) -> int:
        return self.PM // self.A // self.BM

    @property
    def bound_j(self) -> int:
        return self.PN // self.BN


@dataclass(frozen=True)
class ModelSpec:
    """Complete persistent specification; render-only variance stays outside."""

    datatype_profile: DatatypeProfile
    phi_layers: tuple[LayerSpec, ...]
    aggregation: AggregationSpec | None
    rho_layers: tuple[LayerSpec, ...]
    graph_name: str = "UOrcaGraph"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def model_type(self) -> str:
        return "deepsets" if self.has_aggregation else "mlp"

    @property
    def has_aggregation(self) -> bool:
        return self.aggregation is not None

    @property
    def input_layer(self) -> LayerSpec:
        return self.matmul_layers[0]

    @property
    def output_layer(self) -> LayerSpec:
        return self.matmul_layers[-1]

    @property
    def datatype(self) -> str:
        return self.datatype_profile.name

    @property
    def scale_shift(self) -> int:
        return self.datatype_profile.matmul_shift

    @property
    def aggregation_shift(self) -> int:
        return self.datatype_profile.aggregation_shift

    @property
    def matmul_layers(self) -> tuple[LayerSpec, ...]:
        return self.phi_layers + self.rho_layers

    @property
    def aggregation_index(self) -> int:
        return len(self.phi_layers)

    def named_phi_layers(self) -> tuple[tuple[str, LayerSpec], ...]:
        return tuple((f"l{index}", layer) for index, layer in enumerate(self.phi_layers))

    def named_rho_layers(self) -> tuple[tuple[str, LayerSpec], ...]:
        start = self.aggregation_index + 1
        return tuple(
            (f"l{start + index}", layer) for index, layer in enumerate(self.rho_layers)
        )

    def named_matmul_layers(self) -> tuple[tuple[str, Stage, Role, LayerSpec], ...]:
        rendered: list[tuple[str, Stage, Role, LayerSpec]] = []
        for stage, named in (
            ("phi", self.named_phi_layers()),
            ("rho", self.named_rho_layers()),
        ):
            total = len(named)
            for index, (name, layer) in enumerate(named):
                rendered.append((name, stage, network_role(index, total), layer))
        return tuple(rendered)


def network_role(index: int, total: int) -> Role:
    if total == 1:
        return "single"
    if index == 0:
        return "first"
    if index == total - 1:
        return "last"
    return "body"


def _parse_inputs(values: Iterable[Sequence[int]], label: str) -> tuple[LayerInput, ...]:
    inputs = tuple(LayerInput.from_sequence(value) for value in values)
    minimum = 1 if label == "mlp" else 2
    if len(inputs) < minimum:
        raise SpecError(f"{label} must contain at least {minimum} layer(s)")
    for index, layer_input in enumerate(inputs):
        layer_input.validate(f"{label}[{index}]")
    return inputs


def validate_layer(layer: LayerSpec, stage: Stage, index: int, datatype: str = "int8") -> None:
    """Check one physical layer, including its cascade packet alignment."""
    if stage not in ("phi", "rho", "mlp"):
        raise SpecError(f"unknown stage: {stage}")
    profile = resolve_datatype_profile(datatype)
    label = f"{stage}[{index}]"
    layer.input.validate(label)
    for name, value, tile in (
        ("rowA", layer.rowA, layer.BM),
        ("colA", layer.colA, layer.BK),
        ("colB", layer.colB, layer.BN),
    ):
        if value % tile:
            raise SpecError(f"{label}.{name}={value} must be divisible by tile {tile}")
    if layer.C != 1:
        raise SpecError(f"{label}.C must be 1")
    _require_even(f"{label}.bound_j", layer.bound_j, "N/C/BN")
    if stage in ("phi", "mlp"):
        _require_even(f"{label}.bound_i", layer.bound_i, "M/A/BM")
    elif layer.M != 4 or layer.A != 1:
        raise SpecError(f"{label} must use M=4 and A=1")
    if profile.cascade_k_multiple == 2 and (stage == "rho" or index > 0):
        _require_even(f"{label}.bound_k", layer.bound_k, "K/B/BK; cascade packets")


def build_stage_specs(
    values: Iterable[Sequence[int]], stage: Stage, datatype: str = "int8"
) -> tuple[LayerSpec, ...]:
    """Validate a whole MLP, phi or rho chain; DSE uses these same checks."""
    profile = resolve_datatype_profile(datatype)
    layers = tuple(
        LayerSpec(item, profile.BM, profile.BK, profile.BN)
        for item in _parse_inputs(values, stage)
    )
    for index, layer in enumerate(layers):
        validate_layer(layer, stage, index, datatype)
        if index:
            previous = layers[index - 1]
            if previous.M != layer.M or previous.N != layer.K:
                raise SpecError(f"{stage}[{index}] dimensions must match the previous layer")
            if stage in ("phi", "mlp") and previous.A != layer.A:
                raise SpecError(f"{stage} layers must use the same A")
    if stage == "phi":
        if layers[-1].A < 2:
            raise SpecError("aggregation requires final phi A >= 2")
        if layers[-1].bound_i != 2:
            raise SpecError("final phi.bound_i must be 2 (M/A must be 8)")
    return layers


def _validate_graph_name(graph_name: str) -> None:
    if not isinstance(graph_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", graph_name):
        raise SpecError("graph_name must be a C++ identifier")


def build_mlp_spec(
    layers: Iterable[Sequence[int]],
    *,
    datatype: str = "int8",
    graph_name: str = "UOrcaGraph",
    metadata: dict[str, object] | None = None,
) -> ModelSpec:
    """Restore the standalone matmul chain introduced in commit 06558e7.

    MLP uses the same local-memory kernels as phi, with no aggregation or
    rho stage. Its common A is independent of DeepSets' aggregation layout.
    """
    profile = resolve_datatype_profile(datatype)
    _validate_graph_name(graph_name)
    return ModelSpec(
        datatype_profile=profile,
        phi_layers=build_stage_specs(layers, "mlp", datatype),
        aggregation=None,
        rho_layers=(),
        graph_name=graph_name,
        metadata=metadata or {},
    )


def build_model_spec(
    phi: Iterable[Sequence[int]],
    rho: Iterable[Sequence[int]],
    *,
    datatype: str = "int8",
    graph_name: str = "UOrcaGraph",
    metadata: dict[str, object] | None = None,
) -> ModelSpec:
    """Build the specification used by both search and generation."""
    profile = resolve_datatype_profile(datatype)
    _validate_graph_name(graph_name)
    phi_layers = build_stage_specs(phi, "phi", datatype)
    rho_layers = build_stage_specs(rho, "rho", datatype)
    final_phi = phi_layers[-1]
    if final_phi.N != rho_layers[0].K:
        raise SpecError("rho[0].K must match final phi N")
    return ModelSpec(
        datatype_profile=profile,
        phi_layers=phi_layers,
        aggregation=AggregationSpec(
            PM=final_phi.M, PN=final_phi.N, A=final_phi.A,
            BM=profile.BM, BN=profile.BN,
        ),
        rho_layers=rho_layers,
        graph_name=graph_name,
        metadata=metadata or {},
    )
