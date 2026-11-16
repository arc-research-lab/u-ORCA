/*
JINJA: input: layer:LayerSpec
*/
#pragma once

#include <adf/stream/types.h>

namespace l0 {
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
    constexpr unsigned rowA = M/A;
    constexpr unsigned colA = K/B;
    constexpr unsigned colB = N/C;
    //loop boundaries
    constexpr unsigned bound_i = rowA/BM;
    constexpr unsigned bound_k = colA/BK;
    constexpr unsigned bound_j = colB/BN;
    //quant, bias, and relu
    constexpr int SCALE_SHIFT = 7;//scale = 2^7 = 128
    constexpr bool USE_BIAS = true;
    constexpr bool USE_RELU = true;
}

void l0_mmul1(adf::input_buffer<int8>& __restrict input_act,
        const int8(&weight1)[l0::colA*l0::colB],
        output_cascade<acc32>* __restrict cascade_o);
void l0_mmul2(adf::input_buffer<int8>& __restrict input_act,
        input_cascade<acc32>* __restrict cascade_i,
        const int8(&weight1)[l0::colA*l0::colB],
        output_cascade<acc32>* __restrict cascade_o);
void l0_mmul3(adf::input_buffer<int8>& __restrict input_act,
        input_cascade<acc32>* __restrict cascade_i,
        const int8(&weight1)[l0::colA*l0::colB],
        const int(&bias) [l0::colB],
        output_cascade<acc32>* __restrict cascade_o);
