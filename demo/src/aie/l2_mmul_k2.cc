/*
JINJA: input:
(1) layer:LayerSpec
(2) B_pos:int
*/
#include <aie_api/aie.hpp>
#include "l2_kernel.h"

using namespace l2;
static int8 A_buf[rowA*colA];

constexpr int B_pos = 2;
void l2_mmul_k2(input_cascade<acc32>* __restrict cascade_i,
          const int8(&weight1)[colA*colB],
          output_cascade<acc32>* __restrict cascade_o)
{
    using MMUL = aie::mmul<BM, BK, BN, int8, int8>;
    int8 __aie_dm_resource_a * __restrict pA = (int8 __aie_dm_resource_a * __restrict)A_buf;
    int8 __aie_dm_resource_b * __restrict pB = (int8 __aie_dm_resource_b * __restrict)&weight1[0];

    //get and pass the input data
    //assume that all boundaries are even
    //there will not be cases that 00 is in this kernel but 01 is not
    for (unsigned i = 0; i < bound_i; i += 2)chess_loop_range(bound_i/2,bound_i/2){
        int8 * __restrict pA1 = pA + (      i * bound_k +       0) * MMUL::size_A;
        int8 * __restrict pA2 = pA + ((i + 1) * bound_k +       0) * MMUL::size_A;
        for (unsigned b = 0; b < B; b++)
            chess_prepare_for_pipelining chess_loop_range(B,B){
            bool write = (b == B_pos);
            for (unsigned k = 0; k < bound_k; k += 2)
                chess_flatten_loop chess_loop_range(bound_k/2,bound_k/2){
                aie::vector<int8, 2*MMUL::size_A> A00_01 = get_scd_v64int8();
                aie::vector<int8, 2*MMUL::size_A> A10_11 = get_scd_v64int8();
                put_mcd(A00_01);
                put_mcd(A10_11);
                if(write){
                    aie::store_v(pA1,A00_01); pA1 += 2*MMUL::size_A;
                    aie::store_v(pA2,A10_11); pA2 += 2*MMUL::size_A;
                }
            }
        }
    }


    for (unsigned i = 0; i < bound_i; i += 2) chess_loop_range(bound_i/2,bound_i/2) {
        for (unsigned j = 0; j < bound_j; j += 2) chess_prepare_for_pipelining chess_loop_range(bound_j/2,bound_j/2) {
            const int8 * __restrict pA1 = pA + (      i * bound_k +       0) * MMUL::size_A;
            const int8 * __restrict pA2 = pA + ((i + 1) * bound_k +       0) * MMUL::size_A;
            const int8 * __restrict pB1 = pB + (                (j)  ) * MMUL::size_B;//0*bound_j+j
            const int8 * __restrict pB2 = pB + (              (j+1)  ) * MMUL::size_B;

            //first iter use MUL
            //iterate along k dim, add p accordingly, each one only add 1
            aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
            aie::vector<int8, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
            aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B*bound_j;
            aie::vector<int8, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B*bound_j;
            MMUL C00;
            MMUL C01;
            MMUL C10;
            MMUL C11;
            if constexpr(B_pos==0){
                C00.mul(A0, B0);
                C01.mul(A0, B1);
                C10.mul(A1, B0);
                C11.mul(A1, B1);
            }
            else{
                C00=get_scd_v32acc32(); C00.mac(A0, B0);
                C01=get_scd_v32acc32(); C01.mac(A0, B1);
                C10=get_scd_v32acc32(); C10.mac(A1, B0);
                C11=get_scd_v32acc32(); C11.mac(A1, B1);
            }
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
            //C iters along j, every time increase by 1
            //4096-> 1 0000 0000 0000 shift 6 bits  100 0000 ->64, shift 5 bits 1000 0000 -> -128
            put_mcd(C00.to_accum());
            put_mcd(C01.to_accum());
            put_mcd(C10.to_accum());
            put_mcd(C11.to_accum());
       }
   }
}
