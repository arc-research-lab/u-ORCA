"""Check datatype boundaries, calibrated costs and simulator data contracts."""

import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from code_gen.data import (
    DataMismatchError, _aie_matmul, _to_bf16, compare_outputs,
    generate_data_files, generate_quantized_data, read_tokens,
)
from code_gen.generate import prepare_project
from code_gen.generator import build_spec_from_mapping, render_review_tree_strings
from code_gen.spec import DATATYPE_PROFILES, SpecError, build_mlp_spec, build_model_spec
from perf_model.DSE_new import build_design_spec, deepsets_estimation_new, mlp_estimation_new
from perf_model.perf_model import estimate_layer, estimate_model, get_stage_params


class DatatypeTests(unittest.TestCase):
    def test_both_model_types_generate_complete_projects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for datatype in DATATYPE_PROFILES:
                for kind in ("mlp", "deepsets"):
                    with self.subTest(datatype=datatype, model_type=kind):
                        model = {"datatype": datatype, "model_type": kind}
                        if kind == "mlp":
                            model["layers"] = [[16, 21, 32], [16, 32, 13]]
                        else:
                            model.update(phi=[[16, 21, 32], [16, 32, 32]],
                                         rho=[[1, 32, 32], [1, 32, 13]])
                        source = root / "model.json"
                        source.write_text(json.dumps(model))
                        project = root / f"{kind}_{datatype}"
                        design = prepare_project(source, project, search=True)
                        self.assertEqual(design["datatype"], datatype)
                        spec = build_spec_from_mapping(design)
                        self.assertEqual(spec.datatype, datatype)
                        data = generate_quantized_data(spec, 0)
                        expected_dtype = np.float32 if spec.datatype_profile.floating else (
                            np.int16 if datatype == "int16" else np.int8)
                        self.assertEqual(data.golden_output.dtype, expected_dtype)
                        if datatype == "int4":
                            for weights in data.weights.values():
                                self.assertGreaterEqual(weights.min(), -8)
                                self.assertLessEqual(weights.max(), 7)
                            self.assertIn("pack_int4", (project / "src/aie/graph.cpp").read_text())
                        manifest = json.loads((project / "data/data_manifest.json").read_text())
                        self.assertTrue(all((project / item["file"]).is_file() for item in manifest["files"]))
                        self.assertTrue((project / "performance.json").is_file())
                        # A manual spec must preserve the exact mapping selected by DSE.
                        manual = root / f"manual_{kind}_{datatype}"
                        self.assertEqual(prepare_project(project / "uorca_spec.json", manual), design)

    def test_cascade_alignment_depends_on_datatype(self):
        for datatype, profile in DATATYPE_PROFILES.items():
            with self.subTest(datatype=datatype):
                width = max(2 * profile.BN, profile.BK)
                layers = [[8, width, width, 1, width // profile.BK, 1, 1, 1]] * 2
                if datatype in ("int8", "int16"):
                    with self.assertRaisesRegex(SpecError, "bound_k"):
                        build_mlp_spec(layers, datatype=datatype)
                else:
                    build_mlp_spec(layers, datatype=datatype)

    def test_single_sample_padding_and_single_layer_ports(self):
        for datatype in DATATYPE_PROFILES:
            with self.subTest(datatype=datatype):
                logical = [[1, 3, 5], [1, 5, 3]]
                design = build_design_spec({"datatype": datatype, "layers": logical})
                spec = build_spec_from_mapping(design)
                self.assertEqual(design["metadata"]["logical_layers"], logical)
                self.assertEqual(spec.input_layer.M, 8)
                self.assertEqual(spec.input_layer.N, spec.output_layer.K)
                single = build_design_spec({"datatype": datatype, "layers": logical[:1]})
                single_spec = build_spec_from_mapping(single)
                rendered = render_review_tree_strings(single_spec)
                for filename in ("l0_kernel.h", "l0_mmul_k0.cc"):
                    self.assertIn(f"input_buffer<{single_spec.datatype_profile.input_cpp_type}>", rendered[filename])
                    self.assertIn(f"output_buffer<{single_spec.datatype_profile.output_cpp_type}>", rendered[filename])
                invalid = [layer.copy() for layer in design["layers"]]
                invalid[0][0] = 1
                with self.assertRaises(SpecError):
                    build_mlp_spec(invalid, datatype=datatype)

    def test_deepsets_search_matches_exhaustive_legal_costs(self):
        phi_shapes = [[16, 32, 32]] * 2
        rho_shapes = [[1, 32, 32], [1, 32, 16]]
        for datatype, profile in DATATYPE_PROFILES.items():
            for budget in (4, 6):
                best = float("inf")
                for bs in itertools.product((1, 2, 4, 8), repeat=4):
                    if sum(bs) > budget:
                        continue
                    phi = [shape + [2, b, 1, 1, 1] for shape, b in zip(phi_shapes, bs[:2])]
                    rho = [[4, *shape[1:], 1, b, 1, 1, 1] for shape, b in zip(rho_shapes, bs[2:])]
                    try:
                        build_model_spec(phi, rho, datatype=datatype)
                    except SpecError:
                        continue
                    compute = sum(estimate_layer(layer, get_stage_params(datatype, stage), datatype)
                                  for stage, layers in (("phi", phi), ("rho", rho)) for layer in layers)
                    inputs = 30 + 8 * (32 // bs[0]) * profile.activation_bytes / 4 + 16
                    outputs = 30 + 4 * 16 * profile.activation_bytes / 4
                    best = min(best, (compute + inputs + outputs) / 1.25 + 150)
                result = deepsets_estimation_new(phi_shapes, rho_shapes, datatype, B_limit=budget)
                with self.subTest(datatype=datatype, budget=budget):
                    self.assertAlmostEqual(result["estimated_latency_ns"], best)

    def test_mlp_search_matches_exhaustive_legal_costs(self):
        shapes = [[16, 32, 32], [16, 32, 16]]
        for datatype in DATATYPE_PROFILES:
            best = float("inf")
            for a, b0, b1 in itertools.product((1, 2), (1, 2, 4, 8), (1, 2, 4, 8)):
                if b0 + b1 > 5:
                    continue
                layers = [shape + [a, b, 1, 1, 1] for shape, b in zip(shapes, (b0, b1))]
                try:
                    build_mlp_spec(layers, datatype=datatype)
                except SpecError:
                    continue
                cost = estimate_model(layers, "cascade", get_stage_params(datatype, "phi"), datatype)[0]
                best = min(best, cost / 1.25)
            with self.subTest(datatype=datatype):
                result = mlp_estimation_new(shapes, datatype, B_limit=5)
                self.assertAlmostEqual(result["estimated_latency_ns"], best)

    def test_fp32_retains_calibrated_cost(self):
        phi = [[64, 32, 8, 8, 4, 1, 1, 1], [64, 8, 8, 8, 1, 1, 1, 1]]
        phi.append(phi[-1].copy())
        rho = [[4, 8, 8, 1, 1, 1, 1, 1], [4, 8, 16, 1, 1, 1, 1, 1]]
        cycles = sum(estimate_model(layers, "cascade", get_stage_params("fp32", stage), "fp32")[1]
                     for stage, layers in (("phi", phi), ("rho", rho)))
        self.assertAlmostEqual(cycles / 1.25 + 150, 1501.2)

    def test_integer_output_saturates_at_the_right_width(self):
        for datatype, maximum, minimum, k, n in (("int4", 127, -128, 16, 16), ("int16", 32767, -32768, 4, 8)):
            spec = build_mlp_spec([[8, k, n, 1, 1, 1, 1, 0]], datatype=datatype)
            lhs = np.array([[maximum, maximum], [minimum, minimum]])
            rhs = np.array([[7], [7]]) if datatype == "int4" else np.array([[32767], [32767]])
            output = _aie_matmul(lhs, rhs, np.array([0]), spec=spec, use_bias=True, use_relu=False, scale_shift=spec.scale_shift)
            np.testing.assert_array_equal(output, [[maximum], [minimum]])

    def test_bf16_rounds_ties_to_even(self):
        values = np.array([1 + 1 / 256, 1 + 3 / 256, -1 - 1 / 256, -1 - 3 / 256])
        np.testing.assert_array_equal(_to_bf16(values), [1, 1.015625, -1, -1.015625])

    def test_floating_output_checks_reject_corruption_and_nonfinite_values(self):
        for datatype in ("bf16", "fp32"):
            spec = build_mlp_spec([[8, 8, 8, 1, 1, 1, 1, 0]], datatype=datatype)
            with tempfile.TemporaryDirectory() as directory:
                project = Path(directory)
                generate_data_files(spec, project)
                golden = read_tokens(project / "data/golden_0_0.txt", floating=True)
                output_path = project / "aiesimulator_output/data/output_0_0.txt"
                output_path.parent.mkdir(parents=True)
                frames = np.tile(golden, 20)
                np.savetxt(output_path, frames)
                self.assertEqual(compare_outputs(spec, project), 20)
                for wrong in (float("nan"), float("inf"), float(golden[0]) + 1):
                    frames[0] = wrong
                    np.savetxt(output_path, frames)
                    with self.subTest(datatype=datatype, wrong=wrong), self.assertRaises(DataMismatchError):
                        compare_outputs(spec, project)


if __name__ == "__main__":
    unittest.main()
