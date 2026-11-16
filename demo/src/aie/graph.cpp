/*
JINJA: input:
- name:str
- all_layers:List[LayerSpec]
*/
#include "graph.h"
#include "l0_kernel.h"
#include "l1_kernel.h"
#include "l2_kernel.h"
#include "l3_kernel.h"
#include "l4_kernel.h"
#include "l5_kernel.h"
using namespace adf;

uCharmGraph graph1;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(int argc, char** argv) {
    graph1.init();
    //feed in weights
    {
        using namespace l0;
        //INIT synthetic weights and bias
        int8 weight[colA*colB];
        int bias[colB];
        for(int i=0;i<colA*colB;i++){
            weight[i]=2;
        }
        for(int i=0;i<colB;i++){
            bias[i]=3*128;
        }
        for(unsigned a=0;a<A;a++){
            for(unsigned b=0;b<B;b++){
                for(unsigned c=0;c<C;c++){
                    graph1.update(graph1.l0_weight[a][b][c],weight,colA * colB);
                    if(b==B-1){
                        graph1.update(graph1.l0_bias[a][c],bias,colB);
                    }
                }
            }
        }
    }
    {
        using namespace l1;
        //INIT synthetic weights and bias
        int8 weight[colA*colB];
        int bias[colB];
        for(int i=0;i<colA*colB;i++){
            weight[i]=2;
        }
        for(int i=0;i<colB;i++){
            bias[i]=3*128;
        }
        for(unsigned a=0;a<A;a++){
            for(unsigned b=0;b<B;b++){
                for(unsigned c=0;c<C;c++){
                    graph1.update(graph1.l1_weight[a][b][c],weight,colA * colB);
                    if(b==B-1){
                        graph1.update(graph1.l1_bias[a][c],bias,colB);
                    }
                }
            }
        }
    }
    {
        using namespace l2;
        //INIT synthetic weights and bias
        int8 weight[colA*colB];
        int bias[colB];
        for(int i=0;i<colA*colB;i++){
            weight[i]=2;
        }
        for(int i=0;i<colB;i++){
            bias[i]=3*128;
        }
        for(unsigned a=0;a<A;a++){
            for(unsigned b=0;b<B;b++){
                for(unsigned c=0;c<C;c++){
                    graph1.update(graph1.l2_weight[a][b][c],weight,colA * colB);
                    if(b==B-1){
                        graph1.update(graph1.l2_bias[a][c],bias,colB);
                    }
                }
            }
        }
    }
    {
        using namespace l4;
        //INIT synthetic weights and bias
        int8 weight[colA*colB];
        int bias[colB];
        for(int i=0;i<colA*colB;i++){
            weight[i]=2;
        }
        for(int i=0;i<colB;i++){
            bias[i]=3*128;
        }
        for(unsigned a=0;a<A;a++){
            for(unsigned b=0;b<B;b++){
                for(unsigned c=0;c<C;c++){
                    graph1.update(graph1.l4_weight[a][b][c],weight,colA * colB);
                    if(b==B-1){
                        graph1.update(graph1.l4_bias[a][c],bias,colB);
                    }
                }
            }
        }
    }
    {
        using namespace l5;
        //INIT synthetic weights and bias
        int8 weight[colA*colB];
        int bias[colB];
        for(int i=0;i<colA*colB;i++){
            weight[i]=2;
        }
        for(int i=0;i<colB;i++){
            bias[i]=3*128;
        }
        for(unsigned a=0;a<A;a++){
            for(unsigned b=0;b<B;b++){
                for(unsigned c=0;c<C;c++){
                    graph1.update(graph1.l5_weight[a][b][c],weight,colA * colB);
                    if(b==B-1){
                        graph1.update(graph1.l5_bias[a][c],bias,colB);
                    }
                }
            }
        }
    }
    graph1.run(20);
    graph1.end();
    return 0;
}
#endif
