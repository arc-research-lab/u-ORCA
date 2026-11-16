/*
JINJA: input:
- name:str
- flayer:LayerSpec
- blayers:List[LayerSpec]
- llayer:LayerSpec
- all_layers:List[LayerSpec]
*/
#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
//--!!!--
#include "l0_kernel.h"
#include "l1_kernel.h"
#include "l2_kernel.h"
#include "l3_kernel.h"
#include "l4_kernel.h"
#include "l5_kernel.h"

using namespace adf;

class uCharmGraph : public graph {
   private:
    //kernels
    kernel l0[l0::A][l0::B][l0::C];
    kernel l1[l1::A][l1::B][l1::C];
    kernel l2[l2::A][l2::B][l2::C];
    kernel l3[l3::A];
    kernel l4[l4::A][l4::B][l4::C];
    kernel l5[l5::A][l5::B][l5::C];

   public:
    //weight and bias
    input_port l0_weight[l0::A][l0::B][l0::C];
    input_port l0_bias[l0::A][l0::C];
    input_port l1_weight[l1::A][l1::B][l1::C];
    input_port l1_bias[l1::A][l1::C];
    input_port l2_weight[l2::A][l2::B][l2::C];
    input_port l2_bias[l2::A][l2::C];
    //l3 is GA layer, no param
    input_port l4_weight[l4::A][l4::B][l4::C];
    input_port l4_bias[l4::A][l4::C];
    input_port l5_weight[l5::A][l5::B][l5::C];
    input_port l5_bias[l5::A][l5::C];
    //input and output
    input_plio pl_in[l0::A][l0::B];
    output_plio pl_out[l5::A][l5::C];

    uCharmGraph() {
        //create kernel,dim and source
        //create flayer:l0
        {
            using namespace l0;
            //create alias
            auto & layer = l0;
            auto file= std::array{
                "l0_mmul1.cc",
                "l0_mmul2.cc",
                "l0_mmul3.cc",
            };
            for(unsigned a =0; a<A; a++){
                for(unsigned c=0;c<C;c++){
                    layer[a][0][c] = kernel::create(l0_mmul1);
                    source(layer[a][0][c]) = file[0];
                    dimensions(layer[a][0][c].in[0]) = {rowA*colA};
                    runtime<ratio>(layer[a][0][c]) = 1;
                    for(unsigned b=1;b<B-1;b++){
                        layer[a][b][c] = kernel::create(l0_mmul2);
                        source(layer[a][b][c]) = file[1];
                        dimensions(layer[a][b][c].in[0]) = {rowA*colA};
                        runtime<ratio>(layer[a][b][c]) = 1;
                    }
                    layer[a][B-1][c] = kernel::create(l0_mmul3);
                    source(layer[a][B-1][c]) = file[2];
                    dimensions(layer[a][B-1][c].in[0]) = {rowA*colA};
                    runtime<ratio>(layer[a][B-1][c]) = 1;
                }
            }
        }
        //create blayers
        {
            using namespace l1;
            //create alias
            auto & layer = l1;
            for(unsigned a =0; a<A; a++){
                for(unsigned c=0;c<C;c++){
                    //different B dim need different src, fully unroll them
                    layer[a][0][c] = kernel::create(l1_mmul_k0);
                    source(layer[a][0][c]) = "l1_mmul_k0.cc";
                    runtime<ratio>(layer[a][0][c]) = 1;
                    layer[a][1][c] = kernel::create(l1_mmul_k1);
                    source(layer[a][1][c]) = "l1_mmul_k1.cc";
                    runtime<ratio>(layer[a][1][c]) = 1;
                    layer[a][2][c] = kernel::create(l1_mmul_k2);
                    source(layer[a][2][c]) = "l1_mmul_k2.cc";
                    runtime<ratio>(layer[a][2][c]) = 1;
                    layer[a][3][c] = kernel::create(l1_mmul_k3);
                    source(layer[a][3][c]) = "l1_mmul_k3.cc";
                    runtime<ratio>(layer[a][3][c]) = 1;
                }
            }
        }
        //create llayer
        {
            using namespace l2;
            //create alias
            auto & layer = l2;
            for(unsigned a =0; a<A; a++){
                for(unsigned c=0;c<C;c++){
                    //different B dim need different src, fully unroll them
                    layer[a][0][c] = kernel::create(l2_mmul_k0);
                    source(layer[a][0][c]) = "l2_mmul_k0.cc";
                    runtime<ratio>(layer[a][0][c]) = 1;
                    layer[a][1][c] = kernel::create(l2_mmul_k1);
                    source(layer[a][1][c]) = "l2_mmul_k1.cc";
                    runtime<ratio>(layer[a][1][c]) = 1;
                    layer[a][2][c] = kernel::create(l2_mmul_k2);
                    source(layer[a][2][c]) = "l2_mmul_k2.cc";
                    runtime<ratio>(layer[a][2][c]) = 1;
                    layer[a][3][c] = kernel::create(l2_mmul_k3);
                    source(layer[a][3][c]) = "l2_mmul_k3.cc";
                    runtime<ratio>(layer[a][3][c]) = 1;
                    dimensions(layer[a][B-1][c].out[0]) = {rowA*colB};
                }
            }
        }
        //create rho layers
        //create l3
        {
            using namespace l3;
            for(unsigned a=0; a<A; a++){
                if(a==0){
                    l3[a]=kernel::create(l3_nl1);
                    source(l3[a]) = "l3_nl1.cc";
                    runtime<ratio>(l3[a]) = 1;
                    dimensions(l3[a].in[0]) = {8*64};
                }
                else if(a<A-1){
                    l3[a]=kernel::create(l3_nl2);
                    source(l3[a]) = "l3_nl2.cc";
                    runtime<ratio>(l3[a]) = 1;
                    dimensions(l3[a].in[0]) = {8*64};
                }
                else if(a==A-1){
                    l3[a]=kernel::create(l3_nl3);
                    source(l3[a]) = "l3_nl3.cc";
                    runtime<ratio>(l3[a]) = 1;
                    dimensions(l3[a].in[0]) = {8*64};
                    // dimensions(l3[a].out[0]) = {8*4*8};
                }

            }
        }
        //create l4
        {
            using namespace l4;
            for(unsigned a =0; a<A; a++){
                for(unsigned c=0;c<C;c++){
                    l4[a][0][c] = kernel::create(l4_mmul_k0);
                    source(l4[a][0][c]) = "l4_mmul_k0.cc";
                    l4[a][1][c] = kernel::create(l4_mmul_k1);
                    source(l4[a][1][c]) = "l4_mmul_k1.cc";
                    //no io buf
                    for(unsigned b=0;b<B;b++){
                        runtime<ratio>(l4[a][b][c]) = 1;
                    }
                }
            }
        }
        //create l5
        {
            using namespace l5;
            for(unsigned a =0; a<A; a++){
                for(unsigned c=0;c<C;c++){
                    l5[a][0][c] = kernel::create(l5_mmul_k0);
                    source(l5[a][0][c]) = "l5_mmul_k0.cc";
                    //no io buf
                    for(unsigned b=0;b<B;b++){
                        runtime<ratio>(l5[a][b][c]) = 1;
                    }
                    dimensions(l5[a][B-1][c].out[0]) = {rowA*colB};
                }
            }
        }

        //connect within layers
        //connect flayer l0:out[0]->in[1]
        {
            using namespace l0;
            auto & layer = l0;
            for(unsigned a=0;a<A;a++){
                for(unsigned c=0;c<C;c++){
                    for(unsigned b=0;b<B-1;b++)
                    connect<cascade>(layer[a][b][c].out[0],layer[a][b+1][c].in[1]);
                }
            }
        }
        //connect blayers and llayer:out[0]->in[0]
        //connect l1
        {
            using namespace l1;
            auto & layer = l1;
            for(unsigned a=0;a<A;a++){
                for(unsigned c=0;c<C;c++){
                    for(unsigned b=0;b<B-1;b++)
                    connect<cascade>(layer[a][b][c].out[0],layer[a][b+1][c].in[0]);
                }
            }
        }
        //connect l2
        {
            using namespace l2;
            auto & layer = l2;
            for(unsigned a=0;a<A;a++){
                for(unsigned c=0;c<C;c++){
                    for(unsigned b=0;b<B-1;b++)
                    connect<cascade>(layer[a][b][c].out[0],layer[a][b+1][c].in[0]);
                }
            }
        }
        //connect l3
        {
            using namespace l3;
            for(unsigned a=0; a<A-1; a++){
                connect<cascade>(l3[a].out[0],l3[a+1].in[1]);
            }
        }
        //connect l4
        {
            using namespace l4;
            for(unsigned a=0;a<A;a++){
                for(unsigned c=0;c<C;c++){
                    for(unsigned b=0;b<B-1;b++)
                    connect<cascade>(l4[a][b][c].out[0],l4[a][b+1][c].in[0]);
                }
            }
        }
        //no connection in l5



        //connect between layers
        //assume for all layers, A is the same, C is the same and is 1
        // connect l0->l1
        for(unsigned a=0; a< l0::A; a++){
            for(unsigned c=0; c< l0::C; c++){
                connect<cascade>(
                    l0[a][l0::B-1][c].out[0],
                    l1[a][0][c].in[0]
                );
            }
        }
        // connect l1->l2
        for(unsigned a=0; a< l1::A; a++){
            for(unsigned c=0; c< l1::C; c++){
                connect<cascade>(
                    l1[a][l1::B-1][c].out[0],
                    l2[a][0][c].in[0]
                );
            }
        }
        //connect l2 -> l3
        for(unsigned a=0; a<l2::A;a++){
            //l2: bottom is 0, l3: top row is 0
            //l2[0]--> l3[7]
            connect(l2[a][l2::B-1][0].out[0],l3[l2::A-a-1].in[0]);
        }
        //connect l3 -> l4
        connect(l3[l3::A-1].out[0],l4[0][0][0].in[0]);
        //connect l4->l5
        for(unsigned a=0; a< l4::A; a++){
            for(unsigned c =0; c< l4::C; c++){
                connect<cascade> (l4[a][l4::B-1][c].out[0],l5[a][0][c].in[0]);
            }
        }

        //create and connect input PLIO
        {
            using namespace l0;
            auto & layer = l0;
            for(unsigned a=0;a<A;a++){
                for(unsigned b=0;b<B;b++){
                    std::string name = "DataIn" + std::to_string(a) + "_" + std::to_string(b);
                    std::string file = "data/input_" + std::to_string(a) + "_" + std::to_string(b) + ".txt";
                    pl_in[a][b] = input_plio::create(name.c_str(), adf::plio_128_bits, file.c_str(),300);
                    for(unsigned c=0;c<C;c++){
                        connect(pl_in[a][b].out[0], layer[a][b][c].in[0]);
                    }
                }
            }
        }
        //create and connect output PLIO
        {
            using namespace l5;
            auto & layer = l5;
            for(unsigned a=0;a<A;a++){
                for(unsigned c=0;c<C;c++){
                    std::string name = "DataOut" + std::to_string(a) + "_" + std::to_string(c);
                    std::string file = "data/output_" + std::to_string(a) + "_" + std::to_string(c) + ".txt";
                    pl_out[a][c] = output_plio::create(name.c_str(), adf::plio_128_bits, file.c_str(),300);
                    connect(layer[a][B-1][c].out[0], pl_out[a][c].in[0]);
                }
            }
        }

        //connect parameter
        //connect flayer:l0, weight->in[1]/in[2], bias->in[3]
        {
            using namespace l0;
            auto & layer = l0;
            auto & weight = l0_weight;
            auto & bias = l0_bias;
            for(unsigned a=0; a<A; a++){
                for(unsigned b=0;b<B;b++){
                    for(unsigned c=0; c<C; c++){
                        //connect weight
                        if(b==0){
                            connect<parameter>(weight[a][b][c],async(layer[a][b][c].in[1]));
                        }
                        else{
                            connect<parameter>(weight[a][b][c],async(layer[a][b][c].in[2]));
                        }
                        //connect bias
                        if(b==B-1){
                            connect<parameter>(bias[a][c],async(layer[a][b][c].in[3]));
                        }
                    }
                }
            }
        }
        //connect blayers and llayer:weight->in[1], bias->in[2]
        //connect l1
        {
            using namespace l1;
            auto & layer = l1;
            auto & weight = l1_weight;
            auto & bias = l1_bias;
            for(unsigned a=0; a<A; a++){
                for(unsigned b=0;b<B;b++){
                    for(unsigned c=0; c<C; c++){
                        //connect weight
                        connect<parameter>(weight[a][b][c],async(layer[a][b][c].in[1]));
                        //connect bias
                        if(b==B-1){
                            connect<parameter>(bias[a][c],async(layer[a][b][c].in[2]));
                        }
                    }
                }
            }
        }
        //connect l2
        {
            using namespace l2;
            auto & layer = l2;
            auto & weight = l2_weight;
            auto & bias = l2_bias;
            for(unsigned a=0; a<A; a++){
                for(unsigned b=0;b<B;b++){
                    for(unsigned c=0; c<C; c++){
                        //connect weight
                        connect<parameter>(weight[a][b][c],async(layer[a][b][c].in[1]));
                        //connect bias
                        if(b==B-1){
                            connect<parameter>(bias[a][c],async(layer[a][b][c].in[2]));
                        }
                    }
                }
            }
        }
        //no param for l3
        //connect l4
        {
            using namespace l4;
            auto & layer = l4;
            auto & weight = l4_weight;
            auto & bias = l4_bias;
            for(unsigned a=0; a<A; a++){
                for(unsigned b=0;b<B;b++){
                    for(unsigned c=0; c<C; c++){
                        //connect weight
                        connect<parameter>(weight[a][b][c],async(layer[a][b][c].in[1]));
                        //connect bias
                        if(b==B-1){
                            connect<parameter>(bias[a][c],async(layer[a][b][c].in[2]));
                        }
                    }
                }
            }
        }
        //connect l5
        {
            using namespace l5;
            auto & layer = l5;
            auto & weight = l5_weight;
            auto & bias = l5_bias;
            for(unsigned a=0; a<A; a++){
                for(unsigned b=0;b<B;b++){
                    for(unsigned c=0; c<C; c++){
                        //connect weight
                        connect<parameter>(weight[a][b][c],async(layer[a][b][c].in[1]));
                        //connect bias
                        if(b==B-1){
                            connect<parameter>(bias[a][c],async(layer[a][b][c].in[2]));
                        }
                    }
                }
            }
        }

        //placement and bank allocation
        //place l0
        {
            using namespace l0;
            auto & layer = l0;
            unsigned base_row = 0;
            unsigned base_col = 6;
            for(unsigned a=0;a<A;a++){
                    for(unsigned b=0;b<B;b++){
                        for(unsigned c=0;c<C;c++){
                            adf::location<adf::kernel>(layer[a][b][c]) = adf::tile(base_col+b, base_row+a*C+c);
                            //body layers: input_act(1)+weight(2)+stack(3)
                            adf::location<adf::buffer>(layer[a][b][c].in[0])=
                                {   adf::address(base_col+b, base_row+a*C+c, 0x0000),
                                    adf::address(base_col+b, base_row+a*C+c, 0xE000) };
                            if(b==0){
                                adf::location<adf::buffer>(layer[a][b][c].in[1])=
                                    {   adf::address(base_col+b, base_row+a*C+c, 0x4000),
                                        adf::address(base_col+b, base_row+a*C+c, 0x2000) };
                            }
                            else{
                                adf::location<adf::buffer>(layer[a][b][c].in[2])=
                                    {   adf::address(base_col+b, base_row+a*C+c, 0x4000),
                                        adf::address(base_col+b, base_row+a*C+c, 0x2000) };
                            }
                            adf::location<adf::stack>(layer[a][b][c])={ adf::address(base_col+b,base_row+a*C+c,0x8000) };
                            //tail layers: input_act(1)+weight(2)+stack(3)+bias(4)
                            if(b==B-1){
                                adf::location<adf::buffer>(layer[a][b][c].in[3])=
                                    {   adf::address(base_col+b, base_row+a*C+c, 0xC000),
                                        adf::address(base_col+b, base_row+a*C+c, 0x6000) };
                            }
                        }
                    }
            }
        }
        //place l1
        {
            using namespace l1;
            auto & layer = l1;
            unsigned base_row = 0;
            unsigned base_col = 10;
            for(unsigned a=0;a<A;a++){
                    for(unsigned b=0;b<B;b++){
                        for(unsigned c=0;c<C;c++){
                            adf::location<adf::kernel>(layer[a][b][c]) = adf::tile(base_col+b, base_row+a*C+c);
                            //body layers: weight(1)+stack(2)
                            adf::location<adf::buffer>(layer[a][b][c].in[1])=
                                {   adf::address(base_col+b, base_row+a*C+c, 0x0000),
                                    adf::address(base_col+b, base_row+a*C+c, 0xA000) };
                            adf::location<adf::stack>(layer[a][b][c])={ adf::address(base_col+b,base_row+a*C+c,0x4000) };
                            //tail layers: weight(1)+stack(2)+bias(3)
                            if(b==B-1){
                                adf::location<adf::buffer>(layer[a][b][c].in[2])=
                                    {   adf::address(base_col+b, base_row+a*C+c, 0x8000),
                                        adf::address(base_col+b, base_row+a*C+c, 0x2000) };
                            }
                        }
                    }
            }
        }
        //place l2
        {
            using namespace l2;
            auto & layer = l2;
            unsigned base_row = 0;
            unsigned base_col = 14;
            for(unsigned a=0;a<A;a++){
                    for(unsigned b=0;b<B;b++){
                        for(unsigned c=0;c<C;c++){
                            adf::location<adf::kernel>(layer[a][b][c]) = adf::tile(base_col+b, base_row+a*C+c);
                            //body layers: weight(1)+stack(2)
                            adf::location<adf::buffer>(layer[a][b][c].in[1])=
                                {   adf::address(base_col+b, base_row+a*C+c, 0x0000),
                                    adf::address(base_col+b, base_row+a*C+c, 0xE000) };
                            adf::location<adf::stack>(layer[a][b][c])={ adf::address(base_col+b,base_row+a*C+c,0x4000) };
                            //tail layers: weight(1)+stack(2)+bias(3)+out(4)
                            if(b==B-1){
                                adf::location<adf::buffer>(layer[a][b][c].in[2])=
                                    {   adf::address(base_col+b, base_row+a*C+c, 0x8000),
                                        adf::address(base_col+b, base_row+a*C+c, 0x2000) };
                                adf::location<adf::buffer>(layer[a][b][c].out[0])=
                                    {   adf::address(base_col+b, base_row+a*C+c, 0xC000),
                                        adf::address(base_col+b, base_row+a*C+c, 0xA000) };
                            }
                        }
                    }
            }
        }
    };
};

#endif /**********__GRAPH_H__**********/
