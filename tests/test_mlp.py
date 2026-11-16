"""Check the restored MLP chain and its shape-to-project entry point."""

import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from code_gen.data import DataMismatchError, compare_outputs, generate_data_files, generate_quantized_data
from code_gen.generate import prepare_project
from code_gen.generator import build_spec_from_mapping, render_review_tree_strings
from code_gen.spec import SpecError, build_mlp_spec
from perf_model.DSE_new import build_design_spec, mlp_estimation_new
from perf_model.perf_model import estimate_model, get_stage_params

ROOT = Path(__file__).resolve().parents[1]


class MlpTests(unittest.TestCase):
    def test_historical_mlp_has_no_aggregation_and_preserves_all_rows(self):
        mapping = json.loads((ROOT / "code_gen/example_mlp_design.json").read_text())
        spec = build_spec_from_mapping(mapping)
        rendered = render_review_tree_strings(spec)
        self.assertFalse(spec.has_aggregation)
        self.assertEqual(spec.output_layer.rowA, 16)
        self.assertNotIn("l3_kernel.h", rendered)
        self.assertNotIn("global aggregation", rendered["graph.h"])
        data = generate_quantized_data(spec, 7)
        self.assertEqual(data.golden_output.shape, (64, 16))

    def test_search_matches_exhaustive_mlp_minimum(self):
        shapes = [[32, 32, 32], [32, 32, 16]]
        for budget in (2, 3, 6):
            best = float("inf")
            for A, b0, b1 in itertools.product((1, 2, 4), (1, 2, 4), (1, 2, 4)):
                if b0 + b1 > budget:
                    continue
                config = [shape + [A, b, 1, 1, 1] for shape, b in zip(shapes, (b0, b1))]
                try:
                    build_mlp_spec(config)
                except SpecError:
                    continue
                cost = estimate_model(config, "cascade", get_stage_params("int8", "phi"))[0] / 1.25
                best = min(best, cost)
            result = mlp_estimation_new(shapes, A_limit=4, B_limit=budget)
            self.assertAlmostEqual(result["estimated_latency_ns"], best)
            self.assertLessEqual(sum(layer[4] for layer in result["layers_config"]), budget)

    def test_single_sample_and_internal_widths_are_padded_consistently(self):
        logical = {"model_type": "mlp", "layers": [[1, 5, 7], [1, 7, 3]]}
        design = build_design_spec(logical)
        spec = build_spec_from_mapping(design)
        self.assertEqual(spec.input_layer.M, 8)
        self.assertEqual(spec.input_layer.N, spec.output_layer.K)
        self.assertEqual(design["metadata"]["logical_layers"], logical["layers"])
        self.assertEqual(spec.output_layer.N, 16)

    def test_single_layer_routes_both_ends_to_plio(self):
        spec = build_mlp_spec([[8, 16, 16, 1, 1, 1, 0, 0]])
        rendered = render_review_tree_strings(spec)
        self.assertIn("adf::input_buffer<int8>", rendered["l0_mmul_k0.cc"])
        self.assertIn("adf::output_buffer<int8>", rendered["l0_mmul_k0.cc"])
        self.assertIn("adf::input_buffer<int8>", rendered["l0_kernel.h"])
        self.assertIn("adf::output_buffer<int8>", rendered["l0_kernel.h"])
        self.assertNotIn("cascade", rendered["l0_mmul_k0.cc"].split("// phi_single")[-1])

    def test_invalid_manual_mapping_and_disconnected_model_are_rejected(self):
        for config in (
            [[1, 16, 16, 1, 1, 1, 1, 1]],
            [[8, 16, 16, 1, 1, 2, 1, 1]],
            [[16, 16, 16, 1, 1, 1, 1, 1], [16, 16, 16, 2, 1, 1, 1, 1]],
        ):
            with self.assertRaises(SpecError):
                build_mlp_spec(config)
        with self.assertRaises(SpecError):
            build_design_spec({"layers": [[64, 32, 21], [64, 22, 16]]})
        self.assertIsNone(mlp_estimation_new([[8, 16, 16]] * 2, B_limit=1)["layers_config"])

    def test_every_output_partition_is_checked(self):
        spec = build_mlp_spec([[32, 16, 16, 4, 1, 1, 1, 1]])
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            generate_data_files(spec, project)
            out = project / "aiesimulator_output/data"
            out.mkdir(parents=True)
            for a in range(4):
                golden = np.loadtxt(project / f"data/golden_{a}_0.txt", dtype=np.int64).reshape(-1)
                np.savetxt(out / f"output_{a}_0.txt", np.tile(golden, 20), fmt="%d")
            self.assertEqual(compare_outputs(spec, project), 80)
            with (out / "output_3_0.txt").open("a") as stream:
                stream.write("1\n")
            with self.assertRaises(DataMismatchError):
                compare_outputs(spec, project)

    def test_quick_start_cli_runs_both_model_types(self):
        with tempfile.TemporaryDirectory() as directory:
            for filename in ("example.json", "example_mlp.json"):
                project = Path(directory) / filename.removesuffix(".json")
                result = subprocess.run(
                    [sys.executable, "-B", "-m", "code_gen.generate", "--model",
                     str(ROOT / "perf_model" / filename), "--out-dir", str(project)],
                    cwd=ROOT, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Estimated latency:", result.stdout)
                for relative in ("src/aie/graph.cpp", "data/data_manifest.json", "performance.json", "model.json"):
                    self.assertTrue((project / relative).is_file(), relative)
                before = (project / "uorca_spec.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    prepare_project(ROOT / "perf_model" / filename, project, search=True)
                self.assertEqual(before, (project / "uorca_spec.json").read_bytes())

    def test_invalid_model_creates_no_project(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "invalid.json"
            model.write_text(json.dumps({"model_type": "mlp", "layers": [[8, 16, 16]], "datatype": "fp32"}))
            project = Path(directory) / "project"
            with self.assertRaises(SpecError):
                prepare_project(model, project, search=True)
            self.assertFalse(project.exists())


if __name__ == "__main__":
    unittest.main()
