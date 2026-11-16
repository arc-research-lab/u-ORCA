from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from code_gen.data import (
    DataMismatchError, _round_shift_symmetric_inf, compare_outputs,
    from_block_layout, generate_data_files, to_block_layout,
)
from code_gen.generate import generate_project
from code_gen.generator import build_spec_from_mapping, render_review_tree_strings
from code_gen.spec import SpecError, build_model_spec
from perf_model.DSE_new import build_design_spec

ROOT = Path(__file__).resolve().parents[1]


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        model = json.loads((ROOT / "perf_model/example.json").read_text())
        self.mapping = build_design_spec(model)

    def test_search_to_project_and_data(self):
        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / "design.json"
            spec_path.write_text(json.dumps(self.mapping))
            project = Path(directory) / "project"
            paths = generate_project(spec_path, project)
            self.assertIn("graph.cpp", paths)
            self.assertIn("graph.h", paths)
            self.assertTrue((project / "Makefile").is_file())
            spec = build_spec_from_mapping(self.mapping)
            manifest = generate_data_files(spec, project)
            self.assertEqual(manifest["run_iterations"], 20)
            self.assertTrue(all((project / record["file"]).is_file() for record in manifest["files"]))
            golden = np.loadtxt(project / "data/golden_0_0.txt", dtype=np.int64).reshape(-1)
            output = project / "aiesimulator_output/data/output_0_0.txt"
            output.parent.mkdir(parents=True)
            output.write_text("T 0 ns\n" + " ".join(map(str, np.tile(golden, 20))) + "\nTLAST\n")
            self.assertEqual(compare_outputs(spec, project), 20)
            wrong = np.tile(golden, 20)
            wrong[-1] += 1
            output.write_text(" ".join(map(str, wrong)))
            with self.assertRaises(DataMismatchError):
                compare_outputs(spec, project)
            output.write_text(" ".join(map(str, golden)))
            with self.assertRaisesRegex(DataMismatchError, "complete output frames"):
                compare_outputs(spec, project)

    def test_invalid_input_creates_no_project(self):
        self.mapping["rho"][0][0] = 8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(self.mapping))
            output = Path(directory) / "project"
            with self.assertRaises(SpecError):
                generate_project(path, output)
            self.assertFalse(output.exists())

    def test_existing_project_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "design.json"
            path.write_text(json.dumps(self.mapping))
            output = Path(directory) / "project"
            output.mkdir()
            sentinel = output / "user.cpp"
            sentinel.write_text("user work")
            with self.assertRaises(FileExistsError):
                generate_project(path, output)
            self.assertEqual(sentinel.read_text(), "user work")

    def test_single_partition_without_bias_or_relu_renders(self):
        for layer in self.mapping["phi"] + self.mapping["rho"]:
            layer[4] = 1
            layer[6:] = [0, 0]
        rendered = render_review_tree_strings(build_spec_from_mapping(self.mapping))
        self.assertNotIn("input_cascade", rendered["l0_mmul_k0.cc"])
        self.assertNotIn("_bias", rendered["graph.h"])
        self.assertIn("USE_RELU = false", rendered["l0_kernel.h"])

    def test_shape_rules_are_enforced_for_manual_specs(self):
        cases = [
            ("phi", 1, 3, 4),  # changed A
            ("phi", 1, 4, 8),  # odd cascade bound_k
            ("rho", 0, 0, 8),  # wrong M
            ("rho", 0, 3, 2),  # wrong A
            ("rho", 0, 5, 2),  # C partitioning
            ("rho", -1, 2, 24),  # odd bound_j
        ]
        for stage, index, field, value in cases:
            mapping = copy.deepcopy(self.mapping)
            mapping[stage][index][field] = value
            with self.subTest(case=(stage, index, field)), self.assertRaises(SpecError):
                build_spec_from_mapping(mapping)

    def test_unsupported_datatypes_are_rejected_by_generator(self):
        for datatype in ("int32", "fp64", "fp32_safe", None):
            with self.subTest(datatype=datatype), self.assertRaisesRegex(SpecError, "Unsupported datatype"):
                build_model_spec(self.mapping["phi"], self.mapping["rho"], datatype=datatype)

    def test_block_layout_round_trip(self):
        matrix = np.arange(16 * 32, dtype=np.int32).reshape(16, 32)
        packed = to_block_layout(matrix, 4, 8)
        self.assertFalse(np.array_equal(packed, matrix.reshape(-1)))
        np.testing.assert_array_equal(from_block_layout(packed, 16, 32, 4, 8), matrix)

    def test_signed_rounding_ties_match_aie(self):
        values = np.array([-193, -192, -64, -63, 63, 64, 192, 193])
        np.testing.assert_array_equal(_round_shift_symmetric_inf(values, 7), [-2, -2, -1, 0, 0, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()
