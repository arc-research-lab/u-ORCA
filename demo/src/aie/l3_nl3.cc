#include <aie_api/aie.hpp>
#include "l3_kernel.h"

//This non linear kernel
using namespace l3;
void l3_nl3(adf::input_buffer<int8>& __restrict input_act,
            input_cascade<acc32>* __restrict cascade_i,
            output_cascade<acc32>* __restrict cascade_o){
    int8 __aie_dm_resource_a * __restrict pA = (int8 __aie_dm_resource_a * __restrict)input_act.data();
    // int8 __aie_dm_resource_b * __restrict pB = (int8 __aie_dm_resource_b * __restrict)output_act.data();
    //
    using MMUL = aie::mmul<4, 8, 8, int8, int8>;
    //aggregate layer
    const int8 aggt_weights_mem[32] = {1,1,1,1,1,1,1,1,
                                        0,0,0,0,0,0,0,0,
                                        0,0,0,0,0,0,0,0,
                                        0,0,0,0,0,0,0,0};
    aie::vector<int8, MMUL::size_A> aggt_weights =aie::load_v<32>(aggt_weights_mem);
    //group j in pack of 2
    //as a result, the output of 2 v32acc32 --> v32 int 8 can be concat as 1 v64int8 and sent via cascade
    for(unsigned j=0; j<bound_j;j+=2)
        chess_prepare_for_pipelining chess_loop_range(bound_j,bound_j){
        int8 * __restrict pA00 = pA + (      0 * bound_j +       j) * blk_size;
        int8 * __restrict pA10 = pA + (      (0+1) * bound_j +       j) * blk_size;

        int8 * __restrict pA01 = pA + (      0 * bound_j +       j+1) * blk_size;
        int8 * __restrict pA11 = pA + (      (0+1) * bound_j +       j+1) * blk_size;

        aie::vector<int8, MMUL::size_B> act0 = aie::concat(aie::load_v<4*8>(pA00),aie::load_v<4*8>(pA10));
        aie::vector<int8, MMUL::size_B> act1 = aie::concat(aie::load_v<4*8>(pA01),aie::load_v<4*8>(pA11));
        MMUL C0(get_scd_v32acc32()); C0.mac(aggt_weights,act0);
        MMUL C1(get_scd_v32acc32()); C1.mac(aggt_weights,act1);

        // omit as i is only 2
        // for(unsigned i=0; i<bound_i; i+=2)
        //     chess_prepare_for_pipelining chess_loop_range(bound_i-1,bound_i-1){
        // }
        // aie::store_v(pB, C00.template to_vector<int8>(6)); pB+=32;
        put_mcd(aie::concat(C0.template to_vector<int8>(6), C1.template to_vector<int8>(6)));
    }

}
