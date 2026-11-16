/*
JINJA: input: layer:LayerSpec
*/
#include <aie_api/aie.hpp>
#include "l0_kernel.h"

using namespace l0;
void l0_mmul3(adf::input_buffer<int8>& __restrict input_act,
          input_cascade<acc32>* __restrict cascade_i,
          const int8(&weight1)[colA*colB],
          const int(&bias) [colB],
          output_cascade<acc32>* __restrict cascade_o)
{
    aie::set_saturation(aie::saturation_mode::saturate);
    aie::set_rounding(aie::rounding_mode::symmetric_inf);
    using MMUL = aie::mmul<BM, BK, BN, int8, int8>;
    int8 __aie_dm_resource_a * __restrict pA = (int8 __aie_dm_resource_a * __restrict)input_act.data();
    int8 __aie_dm_resource_b * __restrict pB = (int8 __aie_dm_resource_b * __restrict)&weight1[0];
    int __aie_dm_resource_d * __restrict pBias = (int __aie_dm_resource_d * __restrict)&bias[0];

    for (unsigned i = 0; i < bound_i; i += 2) chess_loop_range(bound_i/2,bound_i/2) {
        int * __restrict pBias1 = pBias;

        for (unsigned j = 0; j < bound_j; j += 2) chess_prepare_for_pipelining chess_loop_range(bound_j/2,bound_j/2) {
            const int8 * __restrict pA1 = pA + (      i * bound_k +       0) * MMUL::size_A;
            const int8 * __restrict pA2 = pA + ((i + 1) * bound_k +       0) * MMUL::size_A;
            const int8 * __restrict pB1 = pB + (                (j)  ) * MMUL::size_B;//0*bound_j+j
            const int8 * __restrict pB2 = pB + (              (j+1)  ) * MMUL::size_B;

            //first iter use Mac
            //iterate along k dim, add p accordingly, each one only add 1
            aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
            aie::vector<int8, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
            aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B*bound_j;
            aie::vector<int8, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B*bound_j;
            MMUL C00(get_scd_v32acc32()); C00.mac(A0, B0);
            MMUL C01(get_scd_v32acc32()); C01.mac(A0, B1);
            MMUL C10(get_scd_v32acc32()); C10.mac(A1, B0);
            MMUL C11(get_scd_v32acc32()); C11.mac(A1, B1);
            //other iters use MAC
            for (unsigned k = 1; k < bound_k; k++) chess_flatten_loop chess_loop_range(bound_k-1,bound_k-1) {
                A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
                A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
                B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B*bound_j;
                B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B*bound_j;
                C00.mac(A0, B0);
                C01.mac(A0, B1);
                C10.mac(A1, B0);
                C11.mac(A1, B1);
            }
            //epilogue
            //apply bias
            if constexpr(USE_BIAS){
                aie::vector<int,BN> Bias0_no_dup = aie::load_v<BN>(pBias1);pBias1 += BN;
                auto Bias0 = aie::concat(Bias0_no_dup,Bias0_no_dup,Bias0_no_dup,Bias0_no_dup);
                aie::vector<int,BN> Bias1_no_dup = aie::load_v<BN>(pBias1);pBias1 += BN;
                auto Bias1 = aie::concat(Bias1_no_dup,Bias1_no_dup,Bias1_no_dup,Bias1_no_dup);
                C00 = aie::add(C00.to_accum(), Bias0);
                C01 = aie::add(C01.to_accum(), Bias1);
                C10 = aie::add(C10.to_accum(), Bias0);
                C11 = aie::add(C11.to_accum(), Bias1);
            }
            if constexpr(USE_RELU){
                //apply relu and store
                put_mcd(aie::concat(
                        aie::max(C00.template to_vector<int8>(SCALE_SHIFT), int8(0)),
                        aie::max(C01.template to_vector<int8>(SCALE_SHIFT), int8(0))));
                put_mcd(aie::concat(
                        aie::max(C10.template to_vector<int8>(SCALE_SHIFT), int8(0)),
                        aie::max(C11.template to_vector<int8>(SCALE_SHIFT), int8(0))));
            }
            else{
                //cascade directly
                put_mcd(aie::concat(
                        C00.template to_vector<int8>(SCALE_SHIFT),
                        C01.template to_vector<int8>(SCALE_SHIFT)));
                put_mcd(aie::concat(
                        C10.template to_vector<int8>(SCALE_SHIFT),
                        C11.template to_vector<int8>(SCALE_SHIFT)));
            }
       }
   }
}
