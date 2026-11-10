# u-ORCA release 1

u-ORCA estimates how long a DeepSets network will take to run and generates
AMD AI Engine (AIE) code for it. This release supports **INT8 only**.

Use Python 3.10 or later. Run the commands below from the folder containing
this README.

## 1. How to use the performance model

**Describe the network.** The included file `perf_model/example.json` contains:

```json
{
  "graph_name": "ExampleDeepSetsGraph",
  "datatype": "int8",
  "phi": [[64, 21, 64], [64, 64, 64], [64, 64, 64]],
  "rho": [[1, 64, 64], [1, 64, 10]]
}
```

Each entry describes one layer as `[number of rows, input width, output width]`.
For example, `[64, 21, 64]` takes 64 rows with 21 values in each row and
produces 64 values per row.

The `phi` layers process each row. Their final outputs are combined into one
row, which is then passed through the `rho` layers. In this example, the
network produces 10 values. Use 1 as the row count in `rho`.

To try your own network, copy this file and change the layer sizes. Keep
at least two layers in each list. The search enables bias and ReLU in every
layer.

**Run the search.** Start with the included example:

```bash
python -m perf_model.main --model perf_model/example.json --out build/design.json
```

For your own network, replace `perf_model/example.json` with your file path.
No extra Python packages are needed for this step.

**Read the result.** The example prints:

```text
Estimated latency: 746.0 ns
AIE-only estimate: 627.6 ns
Parameters: 15632
Design saved to build/design.json
```

The first estimate includes input and output transfers. The AIE-only
estimate excludes those transfers. Both are in nanoseconds.
`Parameters` counts weights and biases after rounding layer sizes up to fit
the hardware.

The search saves its chosen hardware configuration to `build/design.json`.
Use that file in the next section. Configurations that the code generator
cannot support are skipped. If none fit, the search reports an error.

## 2. How to generate code

**Generate a project.** Install the required Python packages, then use the
design file from section 1:

```bash
python -m pip install -r requirements.txt
python -m code_gen.generate --spec build/design.json --out-dir build/project
```

The generated project is in `build/project`:

- `src/aie/` contains the C++ code for the layers and their connections.
- `Makefile` contains the commands to compile and simulate the code.
- `uorca_spec.json` is a copy of the hardware configuration.

If `build/project` already exists, choose a new path with `--out-dir`.
Code generation is now complete. Vitis is only needed for the optional
compilation and simulation below.

**Optional: compile and check the code.** This requires AMD Vitis 2024.1
and its VEK280 base platform. Replace `/path/to/Vitis` with your installation
path. If you chose a different project directory, use it in all commands below.

```bash
source /path/to/Vitis/2024.1/settings64.sh
python -m code_gen.data generate --project-dir build/project
make -C build/project graph
make -C build/project aiesim
python -m code_gen.data compare --project-dir build/project
```

This creates random inputs and weights, runs the AIE simulator, and compares
its output with a Python calculation. It does not use a trained model.
If all 20 test inputs pass, the last command prints:

```text
Correctness check passed: 20 output frames match exactly
```
