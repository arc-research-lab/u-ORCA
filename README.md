## 🚀 Thank You for Using u-ORCA!!

### Your support and growing engagement inspire us to continually improve and enhance u-ORCA

# u-ORCA

## Overview

u-ORCA estimates performance and generates accelerator code for microsecond-scale DNN inference on AMD ACAP devices. This release supports **MLP and DeepSets models** with INT8, INT4 weights, INT16, BF16, or FP32.

- The `perf_model` folder estimates latency and searches for a hardware configuration. This search is called design space exploration (DSE).
- The `code_gen` folder contains the code generation framework that generates the AIE code from shape specifications.
- The `demo` folder contains a complete VEK280 reference project with AIE code, an FPGA HLS controller, and a CPU host program for measuring latency.

## Quick Start

### Environment requirements

- Python 3.10 or later; tested with Python 3.10.19.
- AMD Vitis 2024.1, an AIE license, and the `xilinx_vek280_base_202410_1` platform for compilation and AIE simulation. Newer Vitis versions have not been tested.

Python alone is enough to run DSE and generate source code. The on-board demo also needs the Versal Linux image and SDK described below.

### Installation

Clone the release branch and create a Python virtual environment:

```shell
git clone --branch u-orca-release-1 https://github.com/arc-research-lab/u-ORCA.git
cd u-ORCA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the remaining commands from this repository folder, with the virtual environment active.

### Describe the input shape

Use a JSON file to describe the layers. Each layer is `[M, K, N]`: it takes `M` rows of `K` values and produces `M` rows of `N` values. The output width of one layer must equal the input width of the next.

An MLP example is provided in `perf_model/example_mlp.json`. It applies three layers to 64 input rows:

```json
{
  "graph_name": "ExampleMlpGraph",
  "model_type": "mlp",
  "datatype": "int8",
  "layers": [[64, 32, 32], [64, 32, 32], [64, 32, 16]]
}
```

A DeepSets example is provided in `perf_model/example.json`. The `phi` layers process 64 set elements. Aggregation combines their outputs into one row, which the `rho` layers process. Use `M=1` for the logical rho shapes:

```json
{
  "graph_name": "ExampleDeepSetsGraph",
  "model_type": "deepsets",
  "datatype": "int8",
  "phi": [[64, 21, 64], [64, 64, 64], [64, 64, 64]],
  "rho": [[1, 64, 64], [1, 64, 10]]
}
```

Copy either example and change the layer sizes for your model. MLP needs at least one layer; DeepSets needs at least two phi layers and two rho layers. DSE enables bias and ReLU in every layer.

Both examples use `"datatype": "int8"`. To change the datatype, edit that field in your JSON. Use one datatype for the whole model:

| `datatype` | Inputs and outputs | Weights |
| --- | --- | --- |
| `int8` | INT8 | INT8 |
| `int4` | INT8 | Signed INT4, from -8 to 7 |
| `int16` | INT16 | INT16 |
| `bf16` | BF16 | BF16 |
| `fp32` | FP32 | FP32 |

For example, use `"datatype": "bf16"` for a BF16 model. The same generation and simulation commands work for every datatype. Integer models use INT32 bias; floating-point models use FP32 bias. INT4 weights are packed automatically when the simulator loads them.

DSE rounds sizes up to fit the selected kernels and saves both the original shapes and the hardware sizes. For example, an MLP with `M=1` uses a hardware matrix with 8 rows. It skips unsupported configurations and chooses the one with the lowest estimated latency. It reports an error if no design fits.

### Generate the code

Run one command to search for a design, estimate its latency, and generate an AIE project:

```shell
python -m code_gen.generate --model perf_model/example.json --out-dir build/deepsets
```

For the MLP example, run:

```shell
python -m code_gen.generate --model perf_model/example_mlp.json --out-dir build/mlp
```

Each command prints the estimated latency and creates the chosen output folder. The folder contains source code, `uorca_spec.json` with the chosen hardware configuration, `performance.json` with the estimates, and random data for simulation. Choose a new output folder when generating another project.

The full latency estimate includes input and output transfers. The AIE-only estimate excludes them. Both are in nanoseconds. The parameter count includes weights and biases after padding. The random inputs and weights test the generated code; they are not trained model parameters.

FP32 uses accurate multiplication in the generated code. Its performance estimates use the earlier calibration, which has not been updated for this arithmetic change.

To run only the performance model and DSE, without generating a project:

```shell
python -m perf_model.main --model perf_model/example.json --out build/design.json
```

### Conduct AIEsim

Load your Vitis environment, then compile and simulate the generated project. Replace `/path/to/Vitis` with your installation path:

```shell
source /path/to/Vitis/2024.1/settings64.sh
make -C build/deepsets aiesim
python -m code_gen.data compare --project-dir build/deepsets
```

`make aiesim` compiles the graph before running the simulator. The last command compares every simulated output with the Python reference and reports whether they match. To test the MLP project, use `build/mlp` in both commands.

By default, simulation runs until all 20 iterations finish. To set a cycle limit, add `AIESIM_CYCLE_TIMEOUT=100000` to the make command. A limit that is too short leaves incomplete outputs.

Integer outputs must match exactly. Floating-point outputs allow small rounding differences: BF16 allows 0.1% relative error plus 0.001 absolute error; FP32 allows 0.01% relative error plus 0.00001 absolute error.

The Makefile uses the VEK280 platform under `$XILINX_VITIS/base_platforms`. If your platform is elsewhere, add `PLATFORM=/path/to/platform.xpfm` to the make command.

## Advanced usages

For the hardware architecture, see the [u-ORCA paper](https://dl.acm.org/doi/10.1145/3787109.3815309).

### Generate from manually written design specs

DSE normally chooses how to split each matrix multiplication across AIE tiles. To choose these splits yourself, provide a design JSON instead of a shape JSON.

Each layer is `[M, K, N, A, B, C, bias, relu]`:

| Field | Meaning |
| --- | --- |
| `M` | Number of input rows. |
| `K` | Input width. |
| `N` | Output width. |
| `A` | Number of partitions along the rows. |
| `B` | Number of partitions along the input width. |
| `C` | Number of partitions along the output width; this release requires `C=1`. |
| `bias` | Set to 1 to add bias, or 0 to disable it. |
| `relu` | Set to 1 to apply ReLU, or 0 to disable it. |

A layer computes `(M x K) @ (K x N)` using `A x B x C` AIE tiles. With `C=1`, this is an `A x B` array. Consecutive MLP layers share the same `A`; consecutive phi layers also share the same `A`. Rho uses `A=1`.

The following MLP design is saved as `code_gen/example_mlp_design.json`:

```json
{
  "graph_name": "MinimalMlpGraph",
  "model_type": "mlp",
  "datatype": "int8",
  "layers": [
    [64, 32, 32, 4, 2, 1, 1, 1],
    [64, 32, 32, 4, 2, 1, 1, 1],
    [64, 32, 16, 4, 2, 1, 1, 1]
  ]
}
```

For example, the first layer uses `4 x 2 = 8` AIE tiles. Each tile handles `M/A=16` rows and `K/B=16` input values, with an output width of 32.

For DeepSets, use `phi` and `rho` lists with the same eight fields per layer. You can also copy a generated `uorca_spec.json`, change the splits or bias/ReLU flags, and generate another project.

Manual specs use hardware sizes and are checked without padding:

- `M/A` must be a multiple of 8 for MLP and phi layers.
- DeepSets requires `M/A=8` in the final phi layer with `A>=2`; rho requires `M=4, A=1`.
- Adjacent layer widths must match, and all partition sizes must divide evenly.

The width requirements depend on the datatype:

| `datatype` | `N` must be a multiple of | First MLP or phi `K/B` | Later layers and rho `K/B` |
| --- | --- | --- | --- |
| `int8` | 16 | Multiple of 8 | Multiple of 16 |
| `int4` | 16 | Multiple of 16 | Multiple of 16 |
| `int16` | 8 | Multiple of 4 | Multiple of 8 |
| `bf16` | 8 | Multiple of 8 | Multiple of 8 |
| `fp32` | 8 | Multiple of 8 | Multiple of 8 |

Generate the project without running DSE:

```shell
python -m code_gen.generate --spec code_gen/example_mlp_design.json --out-dir build/manual_mlp
```

### The project file structure

- `perf_model/`: performance estimation and design search.
  - `perf_model.py`: estimates the cost of layers and their data transfers.
  - `DSE_new.py`: searches for a supported MLP or DeepSets configuration for the selected datatype.
  - `main.py`: runs DSE and saves a design JSON without generating code.
  - `example.json`: contains the DeepSets input shapes.
  - `example_mlp.json`: contains the MLP input shapes.
  - `__init__.py`: marks this folder as a Python package.
- `code_gen/`: AIE source generation and simulation helpers.
  - `generate.py`: creates a project from model shapes or a manual design spec.
  - `spec.py`: parses the design and checks the supported hardware sizes.
  - `generator.py`: renders the C++ files from the templates.
  - `data.py`: creates random test data and compares simulator outputs.
  - `__init__.py`: exports the design builders for Python callers.
  - `example_mlp_design.json`: contains the manual MLP design shown above.
  - `templates/`: contains the layer, aggregation, header, and graph templates for each datatype.
  - `project_template/Makefile`: compiles and simulates a generated AIE project.
- `demo/`: the complete VEK280 reference design.
  - `Makefile`: selects the platform and exposes the build commands.
  - `makefile_versal_ps.mk`: builds the AIE graph, PL controller, host program, and SD card image.
  - `utils.mk`: checks the build environment and creates the board launch script.
  - `system.cfg`: connects the PL controller to the AIE input and output ports.
  - `src/aie/`: contains the reference graph and INT8 kernels.
  - `src/pl/pl_controller.cpp`: sends inputs, receives outputs, and records cycle timestamps.
  - `src/host/host.cpp`: configures the graph, runs the PL controller, and prints latency statistics.
  - `src/host/l*_kernel.h`: provides layer sizes for the host program.
  - `data/`: contains the input streams used for AIE simulation.
- `tests/`: Python checks for DSE, code generation, and the restored MLP implementation.
  - `test_dse.py`: checks search limits, skipped configurations, and performance estimates.
  - `test_codegen.py`: checks design validation, generated files, and data comparison.
  - `test_mlp.py`: checks MLP search, output partitions, and the unified command.
  - `test_datatypes.py`: checks datatype constraints, calibrated costs, and numeric output comparison.
- `requirements.txt`: lists the Python dependencies.

### Generated AIE project file structure

For the DeepSets quick-start command, the output folder is `build/deepsets/`:

- `src/aie/graph.h`: defines the kernels, their connections, and the PLIO ports.
- `src/aie/graph.cpp`: loads simulation weights and runs the graph for 20 iterations.
- `src/aie/l*_kernel.h` and `l*.cc`: contain the layer declarations and kernel code.
- `data/`: contains input streams, weights, biases, reference outputs, and a data manifest.
- `Makefile`: provides `graph` and `aiesim` build targets.
- `uorca_spec.json`: records the hardware configuration used to generate the code.
- `manifest.json`: lists the generated source files.
- `model.json`: preserves the input shapes when generated with `--model`.
- `performance.json`: stores the latency and parameter estimates when generated with `--model`.

Compilation adds `libadf.a` and `Work/`. Simulation writes its outputs to `aiesimulator_output/`.

### On-board testing and demo

Generated AIE projects expose PLIO ports for inputs and outputs. Running them on a board also needs a PL controller and a CPU host program. The PL controller must match the graph's ports, data layout, and transfer sizes.

`demo/` contains the INT8 reference project from `20260502_FCCM_demo/demo_PL`. It uses three `64 x 64 x 64` phi layers, followed by aggregation and two rho layers (`4 x 64 x 64` and `4 x 64 x 16`). Its host program uses synthetic inputs and weights and prints latency statistics. A generated graph with another datatype needs matching changes to the PL controller and host program before it can run on the board.

In addition to Vitis 2024.1, install the matching Versal common image and its SDK. `EDGE_COMMON_SW` must point to the folder containing `Image` and `rootfs.ext4`; `SYSROOT` must point to the installed SDK's target sysroot, which contains the ARM libraries and XRT headers.

From the repository folder, build the complete demo with:

```shell
make -C demo all TARGET=hw \
  EDGE_COMMON_SW=/path/to/xilinx-versal-common-v2024.1 \
  SYSROOT=/path/to/sdk/sysroots/cortexa72-cortexa53-xilinx-linux
```

This builds the CPU host program, FPGA design, and AIE graph, then creates `demo/package.hw/sd_card.img`. Write that image to an SD card and boot the VEK280 from it. In the board terminal, go to the directory containing `aie_adder` and `krnl_adder.xclbin`, then run 1,000 measurements:

```shell
export XILINX_XRT=/usr
./aie_adder krnl_adder.xclbin 1000
```

For another model, adapt the PL controller, connectivity, and host program to its AIE graph. You can give a coding agent the following prompt:

```text
Use <the reference design> to build a VEK280 demo for <the new hardware spec>
and <the new hardware AIE graph>.

Keep the supplied AIE graph and kernels unchanged. Update the HLS PL controller
to match every input and output port, including stream width, block layout,
and values per iteration. Send each input once, receive each complete output,
and record start and end cycle timestamps.

Update system.cfg, the CPU host program, and the build files. Load the correct
weights and biases, size all buffers from the design spec, and convert cycle
counts using the actual PL clock frequency. Check outputs against a Python
reference and report the latency.

Use Vitis 2024.1 and the VEK280 base platform. Keep installation paths
configurable. Provide the changed files, build and run commands, and validation
results. State which checks require a physical board.
```

## Reference

Please cite u-ORCA if it helps your research.
```
@inbook{10.1145/3787109.3815309,
author = {Ji, Shixin and Zhuang, Jinming and Yang, Zhuoping and Chen, Xingzhen and Zhang, Wei and Zhou, Peipei},
title = {μ-ORCA: Optimizing Acceleration for Microsecond-Scale Deep Neural Network Inference on ACAP},
year = {2026},
isbn = {9798400724312},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3787109.3815309},
booktitle = {Proceedings of the Great Lakes Symposium on VLSI 2026},
pages = {898–905},
numpages = {8}
}
```
