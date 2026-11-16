"""Write an AIE project from a validated design specification."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .generator import build_spec_from_mapping, render_review_tree_strings


def _write_project(mapping: dict, out_dir: Path) -> dict[str, Path]:
    """Validate and render before creating a new output directory."""
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}; choose a new directory")
    spec = build_spec_from_mapping(mapping)
    rendered = render_review_tree_strings(spec)
    source_dir = out_dir / "src/aie"
    source_dir.mkdir(parents=True)
    paths = {}
    for name, contents in rendered.items():
        paths[name] = source_dir / name
        paths[name].write_text(contents)
    shutil.copyfile(Path(__file__).with_name("project_template") / "Makefile", out_dir / "Makefile")
    (out_dir / "uorca_spec.json").write_text(json.dumps(mapping, indent=2) + "\n")
    manifest = {
        "datatype": spec.datatype,
        "model_type": spec.model_type,
        "graph_name": spec.graph_name,
        "sources": sorted(rendered),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def generate_project(spec_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Render first, then create a new project without replacing existing work."""
    return _write_project(json.loads(Path(spec_path).read_text()), Path(out_dir))


def prepare_project(
    input_path: str | Path, out_dir: str | Path, *,
    search: bool = False, A_limit: int = 8, B_limit: int = 37, seed: int = 0,
) -> dict:
    """Prepare source, design and simulation data from a shape or manual spec."""
    from .data import generate_data_files

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}; choose a new directory")
    model = json.loads(Path(input_path).read_text())
    if search:
        from perf_model.DSE_new import build_design_spec
        mapping = build_design_spec(model, A_limit=A_limit, B_limit=B_limit)
    else:
        mapping = model
    _write_project(mapping, out_dir)
    generate_data_files(build_spec_from_mapping(mapping), out_dir, seed=seed)
    if search:
        (out_dir / "model.json").write_text(json.dumps(model, indent=2) + "\n")
        performance = {key: mapping["metadata"][key] for key in (
            "estimated_latency_ns", "aie_only_latency_ns", "parameter_count",
        )}
        (out_dir / "performance.json").write_text(json.dumps(performance, indent=2) + "\n")
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--model", type=Path, help="model shapes; run DSE before generating")
    inputs.add_argument("--spec", type=Path, help="manual hardware design; skip DSE")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--a-limit", type=int, default=8, help="maximum row partitions for DSE")
    parser.add_argument("--b-limit", type=int, default=37, help="maximum sum of K partitions for DSE")
    parser.add_argument("--seed", type=int, default=0, help="random simulation data seed")
    args = parser.parse_args()
    try:
        mapping = prepare_project(
            args.model or args.spec, args.out_dir, search=args.model is not None,
            A_limit=args.a_limit, B_limit=args.b_limit, seed=args.seed,
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    if args.model is not None:
        print(f"Estimated latency: {mapping['metadata']['estimated_latency_ns']:.1f} ns")
        print(f"AIE-only estimate: {mapping['metadata']['aie_only_latency_ns']:.1f} ns")
        print(f"Parameters: {mapping['metadata']['parameter_count']}")
    print(f"AIE source: {args.out_dir / 'src/aie'}")
    print(f"Design: {args.out_dir / 'uorca_spec.json'}")
    print(f"Simulation data: {args.out_dir / 'data'}")
    print(f"Project: {args.out_dir}")


if __name__ == "__main__":
    main()
