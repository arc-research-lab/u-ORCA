# u-ORCA release 1

u-ORCA searches for a DeepSets design and generates AMD AI Engine code.
This release supports **INT8 only**. `perf_model/` contains the performance
model and search. `code_gen/` contains the generator and its templates.

## 1. Use the performance model

Use Python 3.10 or newer. Run all commands from the repository root.
The performance model uses only the Python standard library.

Edit `perf_model/example.json` to describe your network. Each layer is
`[M, K, N]`: M is the number of input rows, K is the input width, and N is
the output width. `phi` processes the set elements. Aggregation reduces
them to one row, which `rho` processes. Use M=1 for the logical rho layers.
Each stage needs at least two layers.

Search for a design:

```bash
python -m perf_model.main --model perf_model/example.json --out build/design.json
```

The command prints estimated latency in nanoseconds and saves the chosen
design to `build/design.json`. It also prints an AIE-only estimate. The full
estimate includes input and output transfers; the AIE-only estimate does not.
Both use a 1.25 GHz AIE clock and the published INT8 calibration.

The search pads dimensions to tile sizes and powers of two. It skips mappings
that do not meet the generator's shape rules, including cascade alignment
and the fixed rho layout (M=4, A=1). It enables bias and ReLU for every layer.
If no mapping fits, it reports an error and does not write a design file.

You can limit the search with `--a-limit 8 --b-limit 37`. A limits the number
of phi row partitions. B limits the **sum** of K partitions across all phi
and rho layers. These checks cover the generator's shape rules; Vitis still
checks device placement and memory use when compiling.

## 2. Generate code

Install the Python packages, then generate a project from the saved design:

```bash
python -m pip install -r requirements.txt
python -m code_gen.generate --spec build/design.json --out-dir build/project
```

The project contains AIE kernels and graph files in `src/aie/`, a Makefile,
and a copy of the design. Choose a new output directory for each generation.
The generator does not overwrite an existing project.

To provide a design yourself, use the same JSON format as `build/design.json`.
Each layer is `[M, K, N, A, B, C, bias, relu]`. A, B, and C split the M, K,
and N dimensions. Bias and ReLU use 0 or 1. All layers require C=1. Phi layers
must use the same A and M/A=8, with A at least 2. Rho layers require M=4 and
A=1. N must be a multiple of 16; K/B must be a multiple of 8 in the first
phi layer and a multiple of 16 in later phi layers and all rho layers.
Adjacent layer widths must match. The generator reports invalid inputs.

For a simulation check, create random inputs, weights, biases, and expected
outputs. This checks the generated kernels; it does not load a trained model.

```bash
python -m code_gen.data generate --project-dir build/project
```

Compile and simulate with AMD Vitis 2024.1 and the VEK280 base platform.
Replace the setup path below with your Vitis installation path:

```bash
source /path/to/Vitis/2024.1/settings64.sh
make -C build/project graph
make -C build/project aiesim
python -m code_gen.data compare --project-dir build/project
```

The Makefile finds the VEK280 platform under `$XILINX_VITIS/base_platforms`.
If it is installed elsewhere, pass `PLATFORM=/path/to/platform.xpfm` to make.
The simulation runs 20 frames; the comparison checks every output value.
The generated project covers AIE compilation and simulation. Host and PL
applications are outside this release.

To run the Python checks without Vitis:

```bash
python -m unittest discover -s tests -v
```
