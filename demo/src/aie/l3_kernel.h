#pragma once

#include <adf/stream/types.h>

namespace l3 {
    constexpr unsigned PM = 64;
    constexpr unsigned PN = 64;

    constexpr unsigned A = 8;

    constexpr unsigned blk_M = 4;
    constexpr unsigned blk_N = 8;
    constexpr unsigned blk_size = blk_M*blk_N;

    constexpr unsigned bound_i = PM/A/blk_M;//8
    constexpr unsigned bound_j = PN/blk_N;//64
}
void l3_nl1(adf::input_buffer<int8>& __restrict input_act,
            output_cascade<acc32>* __restrict cascade_o);
void l3_nl2(adf::input_buffer<int8>& __restrict input_act,
            input_cascade<acc32>* __restrict cascade_i,
            output_cascade<acc32>* __restrict cascade_o);
void l3_nl3(adf::input_buffer<int8>& __restrict input_act,
            input_cascade<acc32>* __restrict cascade_i,
            output_cascade<acc32>* __restrict cascade_o);
