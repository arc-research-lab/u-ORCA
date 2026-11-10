from __future__ import annotations

import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from code_gen.spec import SpecError, build_model_spec
from perf_model.DSE_new import build_design_spec, deepsets_estimation_new
from perf_model.perf_model import estimate_layer, estimate_model, get_stage_params

ROOT = Path(__file__).resolve().parents[1]


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.phi = [[64, 32, 32], [64, 32, 32]]
        self.rho = [[1, 32, 32], [1, 32, 16]]

    def test_default_example_can_be_generated(self):
        model = json.loads((ROOT / "perf_model/example.json").read_text())
        result = build_design_spec(model)
        spec = build_model_spec(result["phi"], result["rho"])
        self.assertEqual([layer.M for layer in spec.rho_layers], [4, 4])
        self.assertGreater(result["metadata"]["estimated_latency_ns"],
                           result["metadata"]["aie_only_latency_ns"])

    def test_invalid_cheaper_cascade_candidate_is_skipped(self):
        # Make larger B attractive, including illegal one-tile cascade inputs.
        with patch("perf_model.DSE_new.estimate_layer", side_effect=lambda layer, params: -100 * layer[4]):
            result = deepsets_estimation_new(self.phi, self.rho)
        self.assertEqual([layer[4] for layer in result["phi_config"]], [4, 2])
        self.assertEqual([layer[4] for layer in result["rho_config"]], [2, 2])
        build_model_spec(result["phi_config"], result["rho_config"])

    def test_tight_budget_finds_a_slower_legal_mapping(self):
        result = deepsets_estimation_new(self.phi, self.rho, B_limit=4)
        self.assertIsNotNone(result["phi_config"])
        self.assertEqual([layer[4] for layer in result["phi_config"] + result["rho_config"]], [1] * 4)

    def test_budget_below_minimum_returns_no_design(self):
        result = deepsets_estimation_new(self.phi, self.rho, B_limit=3)
        self.assertIsNone(result["phi_config"])
        self.assertIsNone(result["estimated_latency_ns"])

    def test_phi_partition_limit_does_not_relax_aggregation(self):
        result = deepsets_estimation_new(self.phi, self.rho, A_limit=4)
        self.assertIsNone(result["phi_config"])

    def test_single_partition_aggregation_is_skipped(self):
        self.phi = [[8, 32, 32], [8, 32, 32]]
        self.assertIsNone(deepsets_estimation_new(self.phi, self.rho)["phi_config"])

    def test_stage_with_one_layer_is_skipped(self):
        for phi, rho in ((self.phi[:1], self.rho), (self.phi, self.rho[:1])):
            self.assertIsNone(deepsets_estimation_new(phi, rho)["phi_config"])

    def test_incompatible_padded_dimensions_are_skipped(self):
        phi = [[64, 32, 8], [64, 8, 32]]
        self.assertIsNone(deepsets_estimation_new(phi, self.rho)["phi_config"])

    def test_logical_mismatch_cannot_be_hidden_by_padding(self):
        self.phi[0][2] = 21
        self.assertIsNone(deepsets_estimation_new(self.phi, self.rho)["phi_config"])

    def test_phi_rho_boundary_must_match(self):
        self.rho[0][1] = 64
        self.assertIsNone(deepsets_estimation_new(self.phi, self.rho)["phi_config"])

    def test_unsupported_precision_is_rejected(self):
        for datatype in ("int4", "int16", "int32", "bf16", "fp32", "fp32_safe"):
            with self.subTest(datatype=datatype), self.assertRaisesRegex(SpecError, "only int8"):
                deepsets_estimation_new(self.phi, self.rho, datatype=datatype)

    def test_bad_shape_values_are_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                deepsets_estimation_new([[value, 32, 32], [64, 32, 32]], self.rho)

    def test_search_matches_exhaustive_legal_minimum(self):
        # Independent exhaustive oracle: try every partition and validate the
        # complete graph, including cases close to the total B budget.
        for budget in (4, 5, 7, 10):
            best = float("inf")
            for partitions in itertools.product((1, 2, 4), repeat=4):
                if sum(partitions) > budget:
                    continue
                phi = [shape + [8, b, 1, 1, 1] for shape, b in zip(self.phi, partitions[:2])]
                rho = [[4, shape[1], shape[2], 1, b, 1, 1, 1]
                       for shape, b in zip(self.rho, partitions[2:])]
                try:
                    build_model_spec(phi, rho)
                except SpecError:
                    continue
                compute = sum(estimate_layer(layer, get_stage_params("int8", stage))
                              for stage, layers in (("phi", phi), ("rho", rho)) for layer in layers)
                M, K, _, A, B, C, _, _ = phi[0]
                inputs = 30 + (M // A) * (K // B) / 4 + 4 * (A * C + 2)
                outputs = 30 + rho[-1][0] * rho[-1][2] / 4
                best = min(best, (compute + inputs + outputs) / 1.25 + 150)
            result = deepsets_estimation_new(self.phi, self.rho, B_limit=budget)
            with self.subTest(budget=budget):
                self.assertEqual(result["estimated_latency_ns"], best)

    def test_no_design_cli_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "design.json"
            output.write_text("keep this design")
            result = subprocess.run(
                [sys.executable, "-B", "-m", "perf_model.main", "--out", str(output), "--b-limit", "1"],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No generator-compatible", result.stderr)
            self.assertEqual(output.read_text(), "keep this design")


class PublishedCostTests(unittest.TestCase):
    def test_calibrated_int8_reference_is_unchanged(self):
        phi = [[64, 32, 32, 8, 2, 1, 1, 1]] * 3
        rho = [[4, 32, 64, 1, 2, 1, 1, 1], [4, 64, 64, 1, 2, 1, 1, 1],
               [4, 64, 16, 1, 1, 1, 1, 1]]
        compute = sum(estimate_model(layers, "cascade", get_stage_params("int8", stage))[1]
                      for stage, layers in (("phi", phi), ("rho", rho)))
        self.assertAlmostEqual(compute / 1.25 + 150, 525.2)


if __name__ == "__main__":
    unittest.main()
