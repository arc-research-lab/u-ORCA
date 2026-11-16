#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

typedef ap_int<8> int8;
typedef ap_axiu<128, 0, 0, 0> axis128_t;

/*
controller logic
arg list:
    bo: (1) input mem(INT8), (2) output mem(INT8), (3) trace_cycle(long), trace_event(int)
    scalar: iter:#iteration to run
logic:
    (1) load the whole input from mem to PL
    (2) format the input for each PLIO buffer
    (3) Use a compute func with dataflow
        - sender: call mm2s iter times, before each calling send a signal to receiver
            - mm2s: put all things from input buffer to the PLIO
        - receiver: call s2mm iter times, after each calling, send a signal to receiver
            - s2mm: read all results to on-chip output buffer
        - recorder: use a II=1 loop to record the #cycles
            - use read_nb to poll from the sender and receiver
            - every time get a number from sender, write cycle_num to trace_cycle, and 0 to event
            - every time get a number from receiver, write cycle_num to trace_cycle, and 1 to event
*/

//layer shape
constexpr unsigned M = 64;
constexpr unsigned K = 64;
constexpr unsigned N = 64;
//spatial unroll
constexpr unsigned A = 8;
constexpr unsigned B = 4;
constexpr unsigned C = 1;
//MM blk shape
constexpr unsigned BM = 4;
constexpr unsigned BK = 8;
constexpr unsigned BN = 8;
//kernel shape
constexpr unsigned rowA = M/A;//8
constexpr unsigned colA = K/B;//16
constexpr unsigned colB = N/C;//64
constexpr unsigned PLIO_WIDTH_BYTES = 16;
//hyperparam for layer 5 to store output
const int OM=4;
const int ON=16;

static void send_one_beat(
    ap_uint<128> input_word,
    hls::stream<axis128_t>& plio
) {
    #pragma HLS inline
    axis128_t pkt;
    pkt.data = input_word;
    pkt.keep = -1;
    pkt.strb = -1;
    pkt.last = 0;
    plio.write(pkt);
}

static void recv_one_plio(
    ap_uint<128> output_buf[OM],
    hls::stream<axis128_t>& plio
) {
    for (int om = 0; om < OM; ++om) {
        #pragma HLS pipeline II=1
        axis128_t pkt = plio.read();
        output_buf[om] = pkt.data;
    }
}


void sender_one_iter(
    ap_uint<128> input_buf[A][B][rowA],
    hls::stream<axis128_t>& pl_in_0_0,
    hls::stream<axis128_t>& pl_in_0_1,
    hls::stream<axis128_t>& pl_in_0_2,
    hls::stream<axis128_t>& pl_in_0_3,
    hls::stream<axis128_t>& pl_in_1_0,
    hls::stream<axis128_t>& pl_in_1_1,
    hls::stream<axis128_t>& pl_in_1_2,
    hls::stream<axis128_t>& pl_in_1_3,
    hls::stream<axis128_t>& pl_in_2_0,
    hls::stream<axis128_t>& pl_in_2_1,
    hls::stream<axis128_t>& pl_in_2_2,
    hls::stream<axis128_t>& pl_in_2_3,
    hls::stream<axis128_t>& pl_in_3_0,
    hls::stream<axis128_t>& pl_in_3_1,
    hls::stream<axis128_t>& pl_in_3_2,
    hls::stream<axis128_t>& pl_in_3_3,
    hls::stream<axis128_t>& pl_in_4_0,
    hls::stream<axis128_t>& pl_in_4_1,
    hls::stream<axis128_t>& pl_in_4_2,
    hls::stream<axis128_t>& pl_in_4_3,
    hls::stream<axis128_t>& pl_in_5_0,
    hls::stream<axis128_t>& pl_in_5_1,
    hls::stream<axis128_t>& pl_in_5_2,
    hls::stream<axis128_t>& pl_in_5_3,
    hls::stream<axis128_t>& pl_in_6_0,
    hls::stream<axis128_t>& pl_in_6_1,
    hls::stream<axis128_t>& pl_in_6_2,
    hls::stream<axis128_t>& pl_in_6_3,
    hls::stream<axis128_t>& pl_in_7_0,
    hls::stream<axis128_t>& pl_in_7_1,
    hls::stream<axis128_t>& pl_in_7_2,
    hls::stream<axis128_t>& pl_in_7_3
) {
    for (int ra = 0; ra < rowA; ++ra) {
        #pragma HLS pipeline II=1
        send_one_beat(input_buf[0][0][ra], pl_in_0_0);
        send_one_beat(input_buf[0][1][ra], pl_in_0_1);
        send_one_beat(input_buf[0][2][ra], pl_in_0_2);
        send_one_beat(input_buf[0][3][ra], pl_in_0_3);
        send_one_beat(input_buf[1][0][ra], pl_in_1_0);
        send_one_beat(input_buf[1][1][ra], pl_in_1_1);
        send_one_beat(input_buf[1][2][ra], pl_in_1_2);
        send_one_beat(input_buf[1][3][ra], pl_in_1_3);
        send_one_beat(input_buf[2][0][ra], pl_in_2_0);
        send_one_beat(input_buf[2][1][ra], pl_in_2_1);
        send_one_beat(input_buf[2][2][ra], pl_in_2_2);
        send_one_beat(input_buf[2][3][ra], pl_in_2_3);
        send_one_beat(input_buf[3][0][ra], pl_in_3_0);
        send_one_beat(input_buf[3][1][ra], pl_in_3_1);
        send_one_beat(input_buf[3][2][ra], pl_in_3_2);
        send_one_beat(input_buf[3][3][ra], pl_in_3_3);
        send_one_beat(input_buf[4][0][ra], pl_in_4_0);
        send_one_beat(input_buf[4][1][ra], pl_in_4_1);
        send_one_beat(input_buf[4][2][ra], pl_in_4_2);
        send_one_beat(input_buf[4][3][ra], pl_in_4_3);
        send_one_beat(input_buf[5][0][ra], pl_in_5_0);
        send_one_beat(input_buf[5][1][ra], pl_in_5_1);
        send_one_beat(input_buf[5][2][ra], pl_in_5_2);
        send_one_beat(input_buf[5][3][ra], pl_in_5_3);
        send_one_beat(input_buf[6][0][ra], pl_in_6_0);
        send_one_beat(input_buf[6][1][ra], pl_in_6_1);
        send_one_beat(input_buf[6][2][ra], pl_in_6_2);
        send_one_beat(input_buf[6][3][ra], pl_in_6_3);
        send_one_beat(input_buf[7][0][ra], pl_in_7_0);
        send_one_beat(input_buf[7][1][ra], pl_in_7_1);
        send_one_beat(input_buf[7][2][ra], pl_in_7_2);
        send_one_beat(input_buf[7][3][ra], pl_in_7_3);
    }
}

void receiver_one_iter(
    ap_uint<128> output_buf[OM],
    hls::stream<axis128_t>& pl_out_0_0
) {
    recv_one_plio(output_buf, pl_out_0_0);
}

void recorder(
    int iter,
    long* trace_cycle,
    int* trace_event,
    hls::stream<int>& trace_stream
) {
    long clk = 0;
    int idx = 0;
    while (idx < 2 * iter) {
        #pragma HLS pipeline II=1
        int event;
        if (trace_stream.read_nb(event)) {
            trace_cycle[idx] = clk;
            trace_event[idx] = event;
            ++idx;
        }
        ++clk;
    }
}

void aie_runner_body(
    ap_uint<128> input_buf[A][B][rowA],
    ap_uint<128> output_buf[OM],
    int iter,
    hls::stream<int>& trace_stream,

    hls::stream<axis128_t>& pl_in_0_0,
    hls::stream<axis128_t>& pl_in_0_1,
    hls::stream<axis128_t>& pl_in_0_2,
    hls::stream<axis128_t>& pl_in_0_3,
    hls::stream<axis128_t>& pl_in_1_0,
    hls::stream<axis128_t>& pl_in_1_1,
    hls::stream<axis128_t>& pl_in_1_2,
    hls::stream<axis128_t>& pl_in_1_3,
    hls::stream<axis128_t>& pl_in_2_0,
    hls::stream<axis128_t>& pl_in_2_1,
    hls::stream<axis128_t>& pl_in_2_2,
    hls::stream<axis128_t>& pl_in_2_3,
    hls::stream<axis128_t>& pl_in_3_0,
    hls::stream<axis128_t>& pl_in_3_1,
    hls::stream<axis128_t>& pl_in_3_2,
    hls::stream<axis128_t>& pl_in_3_3,
    hls::stream<axis128_t>& pl_in_4_0,
    hls::stream<axis128_t>& pl_in_4_1,
    hls::stream<axis128_t>& pl_in_4_2,
    hls::stream<axis128_t>& pl_in_4_3,
    hls::stream<axis128_t>& pl_in_5_0,
    hls::stream<axis128_t>& pl_in_5_1,
    hls::stream<axis128_t>& pl_in_5_2,
    hls::stream<axis128_t>& pl_in_5_3,
    hls::stream<axis128_t>& pl_in_6_0,
    hls::stream<axis128_t>& pl_in_6_1,
    hls::stream<axis128_t>& pl_in_6_2,
    hls::stream<axis128_t>& pl_in_6_3,
    hls::stream<axis128_t>& pl_in_7_0,
    hls::stream<axis128_t>& pl_in_7_1,
    hls::stream<axis128_t>& pl_in_7_2,
    hls::stream<axis128_t>& pl_in_7_3,
    hls::stream<axis128_t>& pl_out_0_0
) {
    for (int it = 0; it < iter; ++it) {
        trace_stream.write(0);
        sender_one_iter(
            input_buf,
            pl_in_0_0,
            pl_in_0_1,
            pl_in_0_2,
            pl_in_0_3,
            pl_in_1_0,
            pl_in_1_1,
            pl_in_1_2,
            pl_in_1_3,
            pl_in_2_0,
            pl_in_2_1,
            pl_in_2_2,
            pl_in_2_3,
            pl_in_3_0,
            pl_in_3_1,
            pl_in_3_2,
            pl_in_3_3,
            pl_in_4_0,
            pl_in_4_1,
            pl_in_4_2,
            pl_in_4_3,
            pl_in_5_0,
            pl_in_5_1,
            pl_in_5_2,
            pl_in_5_3,
            pl_in_6_0,
            pl_in_6_1,
            pl_in_6_2,
            pl_in_6_3,
            pl_in_7_0,
            pl_in_7_1,
            pl_in_7_2,
            pl_in_7_3
        );

        receiver_one_iter(
            output_buf,
            pl_out_0_0
        );
        trace_stream.write(1);
    }
}

void aie_runner(
    ap_uint<128> input_buf[A][B][rowA],
    ap_uint<128> output_buf[OM],
    int iter,
    long* trace_cycle,
    int* trace_event,

    hls::stream<axis128_t>& pl_in_0_0,
    hls::stream<axis128_t>& pl_in_0_1,
    hls::stream<axis128_t>& pl_in_0_2,
    hls::stream<axis128_t>& pl_in_0_3,
    hls::stream<axis128_t>& pl_in_1_0,
    hls::stream<axis128_t>& pl_in_1_1,
    hls::stream<axis128_t>& pl_in_1_2,
    hls::stream<axis128_t>& pl_in_1_3,
    hls::stream<axis128_t>& pl_in_2_0,
    hls::stream<axis128_t>& pl_in_2_1,
    hls::stream<axis128_t>& pl_in_2_2,
    hls::stream<axis128_t>& pl_in_2_3,
    hls::stream<axis128_t>& pl_in_3_0,
    hls::stream<axis128_t>& pl_in_3_1,
    hls::stream<axis128_t>& pl_in_3_2,
    hls::stream<axis128_t>& pl_in_3_3,
    hls::stream<axis128_t>& pl_in_4_0,
    hls::stream<axis128_t>& pl_in_4_1,
    hls::stream<axis128_t>& pl_in_4_2,
    hls::stream<axis128_t>& pl_in_4_3,
    hls::stream<axis128_t>& pl_in_5_0,
    hls::stream<axis128_t>& pl_in_5_1,
    hls::stream<axis128_t>& pl_in_5_2,
    hls::stream<axis128_t>& pl_in_5_3,
    hls::stream<axis128_t>& pl_in_6_0,
    hls::stream<axis128_t>& pl_in_6_1,
    hls::stream<axis128_t>& pl_in_6_2,
    hls::stream<axis128_t>& pl_in_6_3,
    hls::stream<axis128_t>& pl_in_7_0,
    hls::stream<axis128_t>& pl_in_7_1,
    hls::stream<axis128_t>& pl_in_7_2,
    hls::stream<axis128_t>& pl_in_7_3,
    hls::stream<axis128_t>& pl_out_0_0
) {
    hls::stream<int> trace_stream("trace_stream");
    #pragma HLS dataflow

    aie_runner_body(
        input_buf,
        output_buf,
        iter,
        trace_stream,
        pl_in_0_0,
        pl_in_0_1,
        pl_in_0_2,
        pl_in_0_3,
        pl_in_1_0,
        pl_in_1_1,
        pl_in_1_2,
        pl_in_1_3,
        pl_in_2_0,
        pl_in_2_1,
        pl_in_2_2,
        pl_in_2_3,
        pl_in_3_0,
        pl_in_3_1,
        pl_in_3_2,
        pl_in_3_3,
        pl_in_4_0,
        pl_in_4_1,
        pl_in_4_2,
        pl_in_4_3,
        pl_in_5_0,
        pl_in_5_1,
        pl_in_5_2,
        pl_in_5_3,
        pl_in_6_0,
        pl_in_6_1,
        pl_in_6_2,
        pl_in_6_3,
        pl_in_7_0,
        pl_in_7_1,
        pl_in_7_2,
        pl_in_7_3,
        pl_out_0_0
    );

    recorder(
        iter,
        trace_cycle,
        trace_event,
        trace_stream
    );
}
extern "C" {

// reference code
// void mm2s(ap_int<32>* mem, hls::stream<qdma_axis<32, 0, 0, 0> >& s, int size) {
// data_mover:
//     for (int i = 0; i < size; i++) {
//         qdma_axis<32, 0, 0, 0> x;
//         x.data = mem[i];
//         x.keep_all();
//         s.write(x);
//     }
// }
// }

    void pl_controller(
        int8* input_act,
        int8* output_act,
        long* trace_cycle,
        int* trace_event,
        int iter,

        hls::stream<axis128_t>& pl_in_0_0,
        hls::stream<axis128_t>& pl_in_0_1,
        hls::stream<axis128_t>& pl_in_0_2,
        hls::stream<axis128_t>& pl_in_0_3,
        hls::stream<axis128_t>& pl_in_1_0,
        hls::stream<axis128_t>& pl_in_1_1,
        hls::stream<axis128_t>& pl_in_1_2,
        hls::stream<axis128_t>& pl_in_1_3,
        hls::stream<axis128_t>& pl_in_2_0,
        hls::stream<axis128_t>& pl_in_2_1,
        hls::stream<axis128_t>& pl_in_2_2,
        hls::stream<axis128_t>& pl_in_2_3,
        hls::stream<axis128_t>& pl_in_3_0,
        hls::stream<axis128_t>& pl_in_3_1,
        hls::stream<axis128_t>& pl_in_3_2,
        hls::stream<axis128_t>& pl_in_3_3,
        hls::stream<axis128_t>& pl_in_4_0,
        hls::stream<axis128_t>& pl_in_4_1,
        hls::stream<axis128_t>& pl_in_4_2,
        hls::stream<axis128_t>& pl_in_4_3,
        hls::stream<axis128_t>& pl_in_5_0,
        hls::stream<axis128_t>& pl_in_5_1,
        hls::stream<axis128_t>& pl_in_5_2,
        hls::stream<axis128_t>& pl_in_5_3,
        hls::stream<axis128_t>& pl_in_6_0,
        hls::stream<axis128_t>& pl_in_6_1,
        hls::stream<axis128_t>& pl_in_6_2,
        hls::stream<axis128_t>& pl_in_6_3,
        hls::stream<axis128_t>& pl_in_7_0,
        hls::stream<axis128_t>& pl_in_7_1,
        hls::stream<axis128_t>& pl_in_7_2,
        hls::stream<axis128_t>& pl_in_7_3,
        hls::stream<axis128_t>& pl_out_0_0
    ) {
        #pragma HLS interface m_axi offset=slave bundle=gmem0 port=trace_cycle
        #pragma HLS interface m_axi offset=slave bundle=gmem1 port=trace_event
        ap_uint<128> input_buf[A][B][rowA];
        #pragma HLS ARRAY_PARTITION variable=input_buf complete dim=1
        #pragma HLS ARRAY_PARTITION variable=input_buf complete dim=2
        ap_uint<128> output_buf[OM];
        // load input data and pack each 8x16 tile row into one 128-bit PLIO word
        for (int a = 0; a < A; ++a) {
            for (int b = 0; b < B; ++b) {
                for (int ra = 0; ra < rowA; ++ra) {
                    ap_uint<128> packed_word = 0;
                    for (int ca = 0; ca < colA; ++ca) {
                        const int ddr_idx = (a * rowA + ra) * K + (b * colA + ca);
                        packed_word.range(8 * ca + 7, 8 * ca) =
                            static_cast<ap_uint<8> >(input_act[ddr_idx]);
                    }
                    input_buf[a][b][ra] = packed_word;
                }
            }
        }
        aie_runner(
            input_buf,
            output_buf,
            iter,
            trace_cycle,
            trace_event,

            pl_in_0_0,pl_in_0_1,pl_in_0_2,pl_in_0_3,
            pl_in_1_0,pl_in_1_1,pl_in_1_2,pl_in_1_3,
            pl_in_2_0,pl_in_2_1,pl_in_2_2,pl_in_2_3,
            pl_in_3_0,pl_in_3_1,pl_in_3_2,pl_in_3_3,
            pl_in_4_0,pl_in_4_1,pl_in_4_2,pl_in_4_3,
            pl_in_5_0,pl_in_5_1,pl_in_5_2, pl_in_5_3,
            pl_in_6_0,pl_in_6_1,pl_in_6_2,pl_in_6_3,
            pl_in_7_0,pl_in_7_1,pl_in_7_2,pl_in_7_3,
            pl_out_0_0
        );

        for (int om = 0; om < OM; ++om) {
            for (int on = 0; on < ON; ++on) {
                output_act[om * ON + on] =
                    static_cast<int8>(output_buf[om].range(8 * on + 7, 8 * on));
            }
        }
    }
}
