"""Search a logical model and save the selected INT8 code-generation spec."""

import argparse
import json
from pathlib import Path

from .DSE_new import build_design_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("example.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--a-limit", type=int, default=8)
    parser.add_argument("--b-limit", type=int, default=37)
    args = parser.parse_args()
    try:
        model = json.loads(args.model.read_text())
        spec = build_design_spec(model, A_limit=args.a_limit, B_limit=args.b_limit)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Estimated latency: {spec['metadata']['estimated_latency_ns']:.1f} ns")
    print(f"AIE-only estimate: {spec['metadata']['aie_only_latency_ns']:.1f} ns")
    print(f"Parameters: {spec['metadata']['parameter_count']}")
    print(f"Design saved to {args.out}")


if __name__ == "__main__":
    main()
