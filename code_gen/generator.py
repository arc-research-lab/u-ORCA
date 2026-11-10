"""uORCA source generation and datatype/stage/role routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .spec import LayerSpec, ModelSpec, Role, Stage, build_model_spec


TEMPLATE_DIR = Path(__file__).with_name("templates")
AGGREGATION_ROLES: tuple[Role, ...] = ("first", "body", "last")


def make_environment(template_dir: str | Path = TEMPLATE_DIR) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def build_spec_from_mapping(mapping: dict[str, Any]) -> ModelSpec:
    if "datatype" not in mapping:
        raise ValueError("model input requires a top-level datatype entry")
    return build_model_spec(
        mapping["phi"],
        mapping["rho"],
        datatype=mapping["datatype"],
        graph_name=mapping.get("graph_name", "UOrcaGraph"),
        metadata=mapping.get("metadata"),
    )


def load_spec(path: str | Path) -> ModelSpec:
    with Path(path).open("r", encoding="utf-8") as spec_file:
        return build_spec_from_mapping(json.load(spec_file))


def source_template_name(datatype: str, stage: str, role: Role) -> str:
    if stage not in ("phi", "rho", "aggregation"):
        raise ValueError(f"unknown kernel stage: {stage}")
    return f"{datatype}_{stage}_{role}.c.jinja"


def _layer_contexts(spec: ModelSpec) -> list[dict[str, object]]:
    contexts: list[dict[str, object]] = []
    for name, stage, role, layer in spec.named_matmul_layers():
        contexts.append(
            {
                "name": name,
                "stage": stage,
                "role": role,
                "layer": layer,
                "header_file": f"{name}_kernel.h",
                "M": layer.M,
                "K": layer.K,
                "N": layer.N,
                "A": layer.A,
                "B": layer.B,
                "C": layer.C,
                "use_bias": layer.use_bias,
                "use_relu": layer.use_relu,
                "source": "io_buf" if stage == "phi" and role == "first" else "cascade",
                "sink": "io_buf" if role == "last" else "cascade",
            }
        )
    return contexts


def _kernel_context(
    spec: ModelSpec,
    layer_context: dict[str, object],
    B_pos: int,
) -> dict[str, object]:
    layer = layer_context["layer"]
    assert isinstance(layer, LayerSpec)
    return {
        **layer_context,
        "spec": spec,
        "profile": spec.datatype_profile,
        "B_pos": B_pos,
        "b_pos": B_pos,
        "is_B_first": B_pos == 0,
        "is_B_last": B_pos == layer.B - 1,
        "is_B_single": layer.B == 1,
        "needs_partial_acc": B_pos > 0,
        "needs_activation_forward": (
            layer_context["stage"] == "phi"
            and layer_context["role"] != "first"
            and B_pos < layer.B - 1
        ),
        "needs_bias_port": B_pos == layer.B - 1 and layer.use_bias,
        "emits_io_buffer": (
            B_pos == layer.B - 1 and layer_context["role"] == "last"
        ),
        "kernel_name": f"{layer_context['name']}_mmul_k{B_pos}",
        "source_file": f"{layer_context['name']}_mmul_k{B_pos}.cc",
    }


def _aggregation_context(spec: ModelSpec, role: Role) -> dict[str, object]:
    suffix = {"first": "nl1", "body": "nl2", "last": "nl3"}[role]
    name = f"l{spec.aggregation_index}"
    return {
        "spec": spec,
        "profile": spec.datatype_profile,
        "aggregation": spec.aggregation,
        "name": name,
        "role": role,
        "has_partial_acc_in": role != "first",
        "emits_quantized_cascade": role == "last",
        "header_file": f"{name}_kernel.h",
        "kernel_name": f"{name}_{suffix}",
        "source_file": f"{name}_{suffix}.cc",
    }


def format_manifest(spec: ModelSpec) -> str:
    rendered = render_manifest(spec)
    lines = [
        f"Graph: {rendered['graph_name']}",
        f"Datatype: {rendered['datatype']}",
        "Layer order:",
    ]
    lines.extend(f"  - {entry}" for entry in rendered["layer_order"])
    lines.append("Sources:")
    lines.extend(f"  - {entry}" for entry in rendered["sources"])
    return "\n".join(lines)


def render_manifest(spec: ModelSpec) -> dict[str, object]:
    layer_contexts = _layer_contexts(spec)
    aggregation_name = f"l{spec.aggregation_index}"
    headers = [str(context["header_file"]) for context in layer_contexts]
    headers.insert(spec.aggregation_index, f"{aggregation_name}_kernel.h")
    sources: list[str] = []
    for context in layer_contexts:
        layer = context["layer"]
        assert isinstance(layer, LayerSpec)
        sources.extend(
            str(_kernel_context(spec, context, B_pos)["source_file"])
            for B_pos in range(layer.B)
        )
    sources.extend(
        str(_aggregation_context(spec, role)["source_file"])
        for role in AGGREGATION_ROLES
    )
    sources.append("graph.cpp")
    order = [str(context["name"]) for context in layer_contexts]
    order.insert(spec.aggregation_index, f"{aggregation_name} (aggregation)")
    return {
        "graph_name": spec.graph_name,
        "datatype": spec.datatype,
        "headers": headers,
        "sources": sources,
        "layer_order": order,
    }


def render_header_strings(
    spec: ModelSpec, *, template_dir: str | Path = TEMPLATE_DIR
) -> dict[str, str]:
    env = make_environment(template_dir)
    layer_header = env.get_template("layer_kernel.h.jinja")
    aggregation_header = env.get_template("aggregation_kernel.h.jinja")
    rendered: dict[str, str] = {}
    for context in _layer_contexts(spec):
        layer = context["layer"]
        assert isinstance(layer, LayerSpec)
        kernel_contexts = [
            _kernel_context(spec, context, B_pos) for B_pos in range(layer.B)
        ]
        rendered[str(context["header_file"])] = layer_header.render(
            **context,
            spec=spec,
            profile=spec.datatype_profile,
            kernels=kernel_contexts,
        )
    aggregation_contexts = [
        _aggregation_context(spec, role) for role in AGGREGATION_ROLES
    ]
    aggregation_name = f"l{spec.aggregation_index}"
    rendered[f"{aggregation_name}_kernel.h"] = aggregation_header.render(
        spec=spec,
        profile=spec.datatype_profile,
        aggregation=spec.aggregation,
        name=aggregation_name,
        kernels=aggregation_contexts,
    )
    return rendered


def render_all_kernel_source_strings(
    spec: ModelSpec, *, template_dir: str | Path = TEMPLATE_DIR
) -> dict[str, str]:
    env = make_environment(template_dir)
    rendered: dict[str, str] = {}
    for context in _layer_contexts(spec):
        layer = context["layer"]
        assert isinstance(layer, LayerSpec)
        template = env.get_template(
            source_template_name(spec.datatype, str(context["stage"]), context["role"])
        )
        for B_pos in range(layer.B):
            kernel_context = _kernel_context(spec, context, B_pos)
            rendered[str(kernel_context["source_file"])] = template.render(
                **kernel_context
            )
    for role in AGGREGATION_ROLES:
        context = _aggregation_context(spec, role)
        template = env.get_template(
            source_template_name(spec.datatype, "aggregation", role)
        )
        rendered[str(context["source_file"])] = template.render(**context)
    return rendered


def render_graph_strings(
    spec: ModelSpec, *, template_dir: str | Path = TEMPLATE_DIR
) -> dict[str, str]:
    env = make_environment(template_dir)
    layers = _layer_contexts(spec)
    for layer_context in layers:
        layer = layer_context["layer"]
        assert isinstance(layer, LayerSpec)
        layer_context["kernels"] = [
            _kernel_context(spec, layer_context, B_pos)
            for B_pos in range(layer.B)
        ]
    phi_layers = [layer for layer in layers if layer["stage"] == "phi"]
    rho_layers = [layer for layer in layers if layer["stage"] == "rho"]
    aggregation_contexts = [
        _aggregation_context(spec, role) for role in AGGREGATION_ROLES
    ]
    aggregation_view = {
        "name": f"l{spec.aggregation_index}",
        "A": spec.aggregation.A,
        "kernels": aggregation_contexts,
    }
    manifest = render_manifest(spec)
    graph_view = {
        "graph_name": spec.graph_name,
        "header_files": manifest["headers"],
        "matmul_layers": layers,
        "phi_layers": phi_layers,
        "rho_layers": rho_layers,
        "aggregation": aggregation_view,
    }
    context = {
        "spec": graph_view,
        "profile": spec.datatype_profile,
        "layers": layers,
        "phi_layers": phi_layers,
        "rho_layers": rho_layers,
        "aggregation": spec.aggregation,
        "aggregation_name": f"l{spec.aggregation_index}",
        "aggregation_kernels": aggregation_contexts,
        "kernel_context": lambda layer_context, B_pos: _kernel_context(
            spec, layer_context, B_pos
        ),
    }
    return {
        "graph.h": env.get_template("graph.h.jinja").render(**context),
        "graph.cpp": env.get_template("graph.cpp.jinja").render(**context),
    }


def render_review_tree_strings(
    spec: ModelSpec, *, template_dir: str | Path = TEMPLATE_DIR
) -> dict[str, str]:
    return {
        **render_header_strings(spec, template_dir=template_dir),
        **render_all_kernel_source_strings(spec, template_dir=template_dir),
        **render_graph_strings(spec, template_dir=template_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="uORCA AIE generator")
    parser.add_argument("spec", type=Path)
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    print(format_manifest(spec))


if __name__ == "__main__":
    main()
