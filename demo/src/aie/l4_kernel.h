/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#pragma once

#include <adf/stream/types.h>

namespace l4 {
    constexpr unsigned M = 4;
    constexpr unsigned K = 64;
    constexpr unsigned N = 64;

    constexpr unsigned A = 1;
    constexpr unsigned B = 2;
    constexpr unsigned C = 1;

    constexpr unsigned rowA = M/A;//4
    constexpr unsigned colA = K/B;//32
    constexpr unsigned colB = N/C;//64

    constexpr unsigned BM = 4;
    constexpr unsigned BK = 8;
    constexpr unsigned BN = 8;

    constexpr unsigned bound_i = rowA/BM;//1
    constexpr unsigned bound_k = colA/BK;//4
    constexpr unsigned bound_j = colB/BN;//8

    //quant, bias, and relu
    constexpr int SCALE_SHIFT = 7;//scale = 2^7 = 128
    constexpr bool USE_BIAS = true;
    constexpr bool USE_RELU = true;
}


void l4_mmul_k0(input_cascade<acc32>* __restrict cascade_i,
          const int8(&weight1)[l4::colA*l4::colB],
          output_cascade<acc32>* __restrict cascade_o);
void l4_mmul_k1(input_cascade<acc32>* __restrict cascade_i,
          const int8(&weight1)[l4::colA*l4::colB],
          const int(&bias) [l4::colB],
          output_cascade<acc32>* __restrict cascade_o);
