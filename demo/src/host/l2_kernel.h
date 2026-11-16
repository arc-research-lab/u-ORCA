#pragma once

namespace l2 {
constexpr unsigned M = 64;
constexpr unsigned K = 64;
constexpr unsigned N = 64;

constexpr unsigned A = 8;
constexpr unsigned B = 4;
constexpr unsigned C = 1;

constexpr unsigned BM = 4;
constexpr unsigned BK = 8;
constexpr unsigned BN = 8;

constexpr unsigned rowA = M / A;
constexpr unsigned colA = K / B;
constexpr unsigned colB = N / C;

constexpr unsigned bound_i = rowA / BM;
constexpr unsigned bound_k = colA / BK;
constexpr unsigned bound_j = colB / BN;

constexpr int SCALE_SHIFT = 7;
constexpr bool USE_BIAS = true;
constexpr bool USE_RELU = true;
}
