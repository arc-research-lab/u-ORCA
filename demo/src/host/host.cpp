#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_graph.h"
#include "xrt/xrt_kernel.h"

#include "l0_kernel.h"
#include "l1_kernel.h"
#include "l2_kernel.h"
#include "l4_kernel.h"
#include "l5_kernel.h"

namespace {

constexpr const char* kDefaultXclbin = "krnl_adder.xclbin";
constexpr const char* kKernelName = "pl_controller";
constexpr const char* kGraphName = "graph1";
constexpr double kRuntimeFreqMHz = 330.0;
constexpr int kTraceTailCount = 20;

constexpr int kDefaultPlIter = 1;
constexpr int kInputElems = l0::M * l0::K;
constexpr int kOutputElems = l5::rowA * l5::colB;

void fail_if(bool cond, const char* msg)
{
    if (cond) {
        std::fprintf(stderr, "ERROR: %s\n", msg);
        std::exit(EXIT_FAILURE);
    }
}

void log_step(const char* msg)
{
    std::printf("[HOST] %s\n", msg);
    std::fflush(stdout);
}

void update_rtp(xrtGraphHandle graph, const std::string& name, const void* data, size_t bytes)
{
    fail_if(
        xrtGraphUpdateRTP(graph, name.c_str(), static_cast<const char*>(data), bytes) != 0,
        ("xrtGraphUpdateRTP failed for " + name).c_str());
}

void init_graph_rtps(xrtGraphHandle graph)
{
    {
        std::vector<int8_t> weight(l0::colA * l0::colB, 2);
        std::vector<int> bias(l0::colB, 3 * 128);
        for (unsigned a = 0; a < l0::A; ++a) {
            for (unsigned b = 0; b < l0::B; ++b) {
                for (unsigned c = 0; c < l0::C; ++c) {
                    char weight_name[128];
                    std::snprintf(weight_name, sizeof(weight_name), "graph1.l0_weight[%u][%u][%u]", a, b, c);
                    update_rtp(graph, weight_name, weight.data(), weight.size() * sizeof(int8_t));
                    if (b == l0::B - 1) {
                        char bias_name[128];
                        std::snprintf(bias_name, sizeof(bias_name), "graph1.l0_bias[%u][%u]", a, c);
                        update_rtp(graph, bias_name, bias.data(), bias.size() * sizeof(int));
                    }
                }
            }
        }
    }

    {
        std::vector<int8_t> weight(l1::colA * l1::colB, 2);
        std::vector<int> bias(l1::colB, 3 * 128);
        for (unsigned a = 0; a < l1::A; ++a) {
            for (unsigned b = 0; b < l1::B; ++b) {
                for (unsigned c = 0; c < l1::C; ++c) {
                    char weight_name[128];
                    std::snprintf(weight_name, sizeof(weight_name), "graph1.l1_weight[%u][%u][%u]", a, b, c);
                    update_rtp(graph, weight_name, weight.data(), weight.size() * sizeof(int8_t));
                    if (b == l1::B - 1) {
                        char bias_name[128];
                        std::snprintf(bias_name, sizeof(bias_name), "graph1.l1_bias[%u][%u]", a, c);
                        update_rtp(graph, bias_name, bias.data(), bias.size() * sizeof(int));
                    }
                }
            }
        }
    }

    {
        std::vector<int8_t> weight(l2::colA * l2::colB, 2);
        std::vector<int> bias(l2::colB, 3 * 128);
        for (unsigned a = 0; a < l2::A; ++a) {
            for (unsigned b = 0; b < l2::B; ++b) {
                for (unsigned c = 0; c < l2::C; ++c) {
                    char weight_name[128];
                    std::snprintf(weight_name, sizeof(weight_name), "graph1.l2_weight[%u][%u][%u]", a, b, c);
                    update_rtp(graph, weight_name, weight.data(), weight.size() * sizeof(int8_t));
                    if (b == l2::B - 1) {
                        char bias_name[128];
                        std::snprintf(bias_name, sizeof(bias_name), "graph1.l2_bias[%u][%u]", a, c);
                        update_rtp(graph, bias_name, bias.data(), bias.size() * sizeof(int));
                    }
                }
            }
        }
    }

    {
        std::vector<int8_t> weight(l4::colA * l4::colB, 2);
        std::vector<int> bias(l4::colB, 3 * 128);
        for (unsigned a = 0; a < l4::A; ++a) {
            for (unsigned b = 0; b < l4::B; ++b) {
                for (unsigned c = 0; c < l4::C; ++c) {
                    char weight_name[128];
                    std::snprintf(weight_name, sizeof(weight_name), "graph1.l4_weight[%u][%u][%u]", a, b, c);
                    update_rtp(graph, weight_name, weight.data(), weight.size() * sizeof(int8_t));
                    if (b == l4::B - 1) {
                        char bias_name[128];
                        std::snprintf(bias_name, sizeof(bias_name), "graph1.l4_bias[%u][%u]", a, c);
                        update_rtp(graph, bias_name, bias.data(), bias.size() * sizeof(int));
                    }
                }
            }
        }
    }

    {
        std::vector<int8_t> weight(l5::colA * l5::colB, 2);
        std::vector<int> bias(l5::colB, 3 * 128);
        for (unsigned a = 0; a < l5::A; ++a) {
            for (unsigned b = 0; b < l5::B; ++b) {
                for (unsigned c = 0; c < l5::C; ++c) {
                    char weight_name[128];
                    std::snprintf(weight_name, sizeof(weight_name), "graph1.l5_weight[%u][%u][%u]", a, b, c);
                    update_rtp(graph, weight_name, weight.data(), weight.size() * sizeof(int8_t));
                    if (b == l5::B - 1) {
                        char bias_name[128];
                        std::snprintf(bias_name, sizeof(bias_name), "graph1.l5_bias[%u][%u]", a, c);
                        update_rtp(graph, bias_name, bias.data(), bias.size() * sizeof(int));
                    }
                }
            }
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    const char* xclbin_filename = (argc > 1) ? argv[1] : kDefaultXclbin;
    const int iter = (argc > 2) ? std::atoi(argv[2]) : kDefaultPlIter;
    fail_if(iter <= 0, "PL kernel iter must be positive");

    std::vector<int8_t> input_act(kInputElems);
    std::vector<int8_t> output_act(kOutputElems, 0);
    std::vector<long> trace_cycle(2 * iter, -1);
    std::vector<int32_t> trace_event(2 * iter, -1);

    for (int i = 0; i < kInputElems; ++i) {
        input_act[i] = static_cast<int8_t>((i % 13) - 6);
    }

    log_step("Opening device and loading xclbin");
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbin_filename);
    auto dhdl = xrtDeviceOpenFromXcl(device);
    fail_if(dhdl == nullptr, "xrtDeviceOpenFromXcl failed");

    auto pl_kernel = xrt::kernel(device, uuid, kKernelName);
    auto graph = xrtGraphOpen(dhdl, uuid.get(), kGraphName);
    fail_if(graph == nullptr, "xrtGraphOpen failed");

    auto input_bo = xrt::bo(device, input_act.size() * sizeof(int8_t), pl_kernel.group_id(0));
    auto output_bo = xrt::bo(device, output_act.size() * sizeof(int8_t), pl_kernel.group_id(1));
    auto trace_cycle_bo = xrt::bo(device, trace_cycle.size() * sizeof(long), pl_kernel.group_id(2));
    auto trace_event_bo = xrt::bo(device, trace_event.size() * sizeof(int32_t), pl_kernel.group_id(3));

    auto input_map = input_bo.map<int8_t*>();
    auto output_map = output_bo.map<int8_t*>();
    auto trace_cycle_map = trace_cycle_bo.map<long*>();
    auto trace_event_map = trace_event_bo.map<int32_t*>();

    std::memcpy(input_map, input_act.data(), input_act.size() * sizeof(int8_t));
    std::memset(output_map, 0, output_act.size() * sizeof(int8_t));
    std::fill(trace_cycle_map, trace_cycle_map + trace_cycle.size(), -1);
    std::fill(trace_event_map, trace_event_map + trace_event.size(), -1);

    input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, input_act.size() * sizeof(int8_t), 0);
    output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, output_act.size() * sizeof(int8_t), 0);
    trace_cycle_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, trace_cycle.size() * sizeof(long), 0);
    trace_event_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, trace_event.size() * sizeof(int32_t), 0);

    log_step("Updating graph RTPs");
    init_graph_rtps(graph);
    log_step("Graph RTP update completed");

    log_step("Starting AI Engine graph in free-running mode");
    fail_if(xrtGraphRun(graph, -1) != 0, "xrtGraphRun(-1) failed");
    log_step("AI Engine graph started");

    log_step("Launching PL kernel");
    auto pl_run = pl_kernel(input_bo, output_bo, trace_cycle_bo, trace_event_bo, iter);
    pl_run.wait();
    log_step("PL kernel completed");

    log_step("Stopping AI Engine graph");
    fail_if(xrtGraphEnd(graph, 1) != 0, "xrtGraphEnd failed");
    log_step("AI Engine graph stopped");

    output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, output_act.size() * sizeof(int8_t), 0);
    trace_cycle_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, trace_cycle.size() * sizeof(long), 0);
    trace_event_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, trace_event.size() * sizeof(int32_t), 0);

    std::printf("Output (%d int8 values from last PL transaction):\n", kOutputElems);
    for (int i = 0; i < kOutputElems; ++i) {
        std::printf("%d%c", static_cast<int>(output_map[i]), (i + 1 == kOutputElems) ? '\n' : ' ');
    }

    const int total_events = 2 * iter;
    const int trace_begin = (total_events > kTraceTailCount) ? (total_events - kTraceTailCount) : 0;
    std::printf("Trace tail (%d/%d events):\n", total_events - trace_begin, total_events);
    for (int i = trace_begin; i < total_events; ++i) {
        std::printf("  [%d] cycle=%ld event=%d\n", i, trace_cycle_map[i], trace_event_map[i]);
    }

    std::vector<long> latency_cycles;
    latency_cycles.reserve(iter);
    int valid_pairs = 0;
    for (int i = 0; i + 1 < total_events; i += 2) {
        const long start_cycle = trace_cycle_map[i];
        const long end_cycle = trace_cycle_map[i + 1];
        const int start_event = trace_event_map[i];
        const int end_event = trace_event_map[i + 1];
        if (start_event == 0 && end_event == 1 && start_cycle >= 0 && end_cycle >= start_cycle) {
            latency_cycles.push_back(end_cycle - start_cycle);
            ++valid_pairs;
        }
    }

    fail_if(valid_pairs == 0, "No valid start/end trace pairs were recorded");
    if (valid_pairs != iter) {
        std::printf("[HOST] Warning: only %d/%d trace pairs are valid for latency statistics\n", valid_pairs, iter);
    }

    long min_cycle = latency_cycles[0];
    long max_cycle = latency_cycles[0];
    long long sum_cycle = 0;
    for (long v : latency_cycles) {
        if (v < min_cycle) min_cycle = v;
        if (v > max_cycle) max_cycle = v;
        sum_cycle += v;
    }
    const double avg_cycle = static_cast<double>(sum_cycle) / latency_cycles.size();
    const double ns_per_cycle = 1000.0 / kRuntimeFreqMHz;
    const double min_ns = min_cycle * ns_per_cycle;
    const double max_ns = max_cycle * ns_per_cycle;
    const double avg_ns = avg_cycle * ns_per_cycle;

    std::printf("Per-iteration latency statistics (cycles):\n");
    std::printf("  min=%ld max=%ld avg=%.3f\n", min_cycle, max_cycle, avg_cycle);

    std::printf("Per-iteration latency statistics at %.1f MHz:\n", kRuntimeFreqMHz);
    std::printf("  min=%.3f ns max=%.3f ns avg=%.3f ns\n", min_ns, max_ns, avg_ns);
    std::printf("  min=%.3f us max=%.3f us avg=%.3f us\n", min_ns / 1000.0, max_ns / 1000.0, avg_ns / 1000.0);

    std::printf("Latency histogram (cycles):\n");
    constexpr int kHistogramBins = 10;
    int bins[kHistogramBins] = {0};
    if (min_cycle == max_cycle) {
        bins[0] = static_cast<int>(latency_cycles.size());
    } else {
        const double span = static_cast<double>(max_cycle - min_cycle + 1);
        for (long v : latency_cycles) {
            int bin = static_cast<int>(((v - min_cycle) * kHistogramBins) / span);
            if (bin >= kHistogramBins) {
                bin = kHistogramBins - 1;
            }
            ++bins[bin];
        }
    }
    int max_bin_count = 0;
    for (int i = 0; i < kHistogramBins; ++i) {
        if (bins[i] > max_bin_count) {
            max_bin_count = bins[i];
        }
    }
    for (int i = 0; i < kHistogramBins; ++i) {
        long range_lo;
        long range_hi;
        if (min_cycle == max_cycle) {
            range_lo = min_cycle;
            range_hi = max_cycle;
        } else {
            range_lo = min_cycle + ((max_cycle - min_cycle + 1) * i) / kHistogramBins;
            range_hi = min_cycle + ((max_cycle - min_cycle + 1) * (i + 1)) / kHistogramBins - 1;
        }
        const int bar_len = (max_bin_count > 0) ? (bins[i] * 40) / max_bin_count : 0;
        std::printf("  [%4ld, %4ld] %4d |", range_lo, range_hi, bins[i]);
        for (int j = 0; j < bar_len; ++j) {
            std::putchar('#');
        }
        std::putchar('\n');
    }

    xrtGraphClose(graph);
    xrtDeviceClose(dhdl);
    return 0;
}
