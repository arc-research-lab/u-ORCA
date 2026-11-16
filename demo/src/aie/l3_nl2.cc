#include <aie_api/aie.hpp>
#include "l3_kernel.h"

//This non linear kernel
using namespace l3;
void l3_nl2(adf::input_buffer<int8>& __restrict input_act,
            input_cascade<acc32>* __restrict cascade_i,
            output_cascade<acc32>* __restrict cascade_o){
    int8 __aie_dm_resource_a * __restrict pA = (int8 __aie_dm_resource_a * __restrict)input_act.data();
    using MMUL = aie::mmul<4, 8, 8, int8, int8>;
    //aggregate layer
    const int8 aggt_weights_mem[32] = {1,1,1,1,1,1,1,1,
                                        0,0,0,0,0,0,0,0,
                                        0,0,0,0,0,0,0,0,
                                        0,0,0,0,0,0,0,0};
    aie::vector<int8, MMUL::size_A> aggt_weights =aie::load_v<32>(aggt_weights_mem);
    for(unsigned j=0; j<bound_j;j++)
        chess_prepare_for_pipelining chess_loop_range(bound_j,bound_j){
        int8 * __restrict pA1 = pA + (      0 * bound_j +       j) * blk_size;
        int8 * __restrict pA2 = pA + (      (0+1) * bound_j +       j) * blk_size;

        aie::vector<int8, MMUL::size_B> act = aie::concat(aie::load_v<4*8>(pA1),aie::load_v<4*8>(pA2));
        MMUL C00(get_scd_v32acc32()); C00.mac(aggt_weights,act);

        // omit as i is only 2
        // for(unsigned i=0; i<bound_i; i+=2)
        //     chess_prepare_for_pipelining chess_loop_range(bound_i-1,bound_i-1){
        // }
        put_mcd(C00.to_accum());
    }
}
