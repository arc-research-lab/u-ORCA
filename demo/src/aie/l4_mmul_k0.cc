#include <aie_api/aie.hpp>
#include "l4_kernel.h"
// #include <aie_api/aie_adf.hpp>
// #include "aie_api/utils.hpp"


/*use mmul instead of intrinsic for the loops
The mem layout is as follows:
- The matrix is tile-based lay out
- original dims (by element) MM is rowA, colA, colB
- The AIE compute the MM in tiles of M,K,N, thus the dims of tiles is rowA/M, rowB/K, rowC/N --> bound_i(or j, k)
- The lay out is as follows
- For elements in each tile, store close with each other, we won't address single element in MMUL
- For tiles, each tile is M*K(sizeA) or K*N(sizeB) orM*N(sizeC) elements
- Tilewise, we store the tiles of A in row major, B in row major, C in row major
- thus to address one tile in A,B,C using i,j,k loops of tile, we get
    - A: (i*bound_k + k)*sizeA
    - B: (k*bound_j + j)*sizeB
    - C: (i*bound_j + j)*sizeC
- Use output stationary dataflow, software pipeline 4 tiles in 2*1*2 dim
*/
//in this example, no partition to weights --> keep in heap directly
//All AIEs using the same shapes

using namespace l4;
static int8 A_buf[rowA*colA];

constexpr int B_pos = 0;
void l4_mmul_k0(input_cascade<acc32>* __restrict cascade_i,
          const int8(&weight1)[colA*colB],
          output_cascade<acc32>* __restrict cascade_o)
{
    using MMUL = aie::mmul<BM, BK, BN, int8, int8>;
    // int8 __aie_dm_resource_a * __restrict pA = (int8 __aie_dm_resource_a * __restrict)A_buf;
    int8 __aie_dm_resource_b * __restrict pB = (int8 __aie_dm_resource_b * __restrict)&weight1[0];
    //here the i is only 1:
    const unsigned i = 0;
    //first: get 4 A reg
    aie::vector<int8, MMUL::size_A*2> A0_1 = get_scd_v64int8();
    aie::vector<int8, MMUL::size_A> A0 = A0_1.extract<32>(0);
    aie::vector<int8, MMUL::size_A> A1 = A0_1.extract<32>(1);
    aie::vector<int8, MMUL::size_A*2> A2_3 = get_scd_v64int8();
    aie::vector<int8, MMUL::size_A> A2 = A2_3.extract<32>(0);
    aie::vector<int8, MMUL::size_A> A3 = A2_3.extract<32>(1);
    //then: send the next 4 regs to next kernel
    put_mcd(get_scd_v64int8());
    put_mcd(get_scd_v64int8());

    //main loop: i is always 0:
    for (unsigned j = 0; j < bound_j; j += 2)
    chess_prepare_for_pipelining chess_loop_range(bound_j/2,bound_j/2){
        const int8 * __restrict pB1 = pB + (0*bound_j+      (j)  ) * MMUL::size_B;//0*bound_j+j
        const int8 * __restrict pB2 = pB + (0*bound_j+    (j+1)  ) * MMUL::size_B;
        aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B*bound_j;
        aie::vector<int8, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B*bound_j;
        MMUL C00; C00.mul(A0, B0);
        MMUL C01; C01.mul(A0, B1);

        //flatten k loop
        B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B*bound_j;
        B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B*bound_j;
        C00.mac(A1, B0);
        C01.mac(A1, B1);

        B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B*bound_j;
        B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B*bound_j;
        C00.mac(A2, B0);
        C01.mac(A2, B1);

        B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B*bound_j;
        B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B*bound_j;
        C00.mac(A3, B0);
        C01.mac(A3, B1);

        put_mcd(C00.to_accum());
        put_mcd(C01.to_accum());
    }

}
