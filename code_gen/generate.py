"""Write an INT8 AIE project from a validated design specification."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .generator import load_spec, render_review_tree_strings


def generate_project(spec_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Render first, then create a new project without replacing existing work."""
    spec_path, out_dir = Path(spec_path), Path(out_dir)
    spec = load_spec(spec_path)
    rendered = render_review_tree_strings(spec)
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}; choose a new directory")
    source_dir = out_dir / "src/aie"
    source_dir.mkdir(parents=True)
    paths = {}
    for name, contents in rendered.items():
        paths[name] = source_dir / name
        paths[name].write_text(contents)
    shutil.copyfile(Path(__file__).with_name("project_template") / "Makefile", out_dir / "Makefile")
    shutil.copyfile(spec_path, out_dir / "uorca_spec.json")
    manifest = {
        "datatype": spec.datatype,
        "graph_name": spec.graph_name,
        "sources": sorted(rendered),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = generate_project(args.spec, args.out_dir)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Generated {len(paths)} AIE source files in {args.out_dir / 'src/aie'}")
    print(f"Project: {args.out_dir}")


if __name__ == "__main__":
    main()
