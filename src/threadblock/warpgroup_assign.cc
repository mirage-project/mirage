#include "mirage/threadblock/graph.h"

namespace mirage {
namespace threadblock {

  void Graph::add_warpgroup_config(int pipeline_stage, int num_warp_groups){
    assert(num_warp_groups >= 2 && "at least a producer and a consumer is needed");
    assert(num_warp_groups <= config::MAX_NUM_WARP_GROUPS && "MAX NUM WARP GROUPS is 4");
    pipe_stage = pipeline_stage;

    printf("pipe_stage %d, pipeline_stage %d, %d\n", pipe_stage, pipeline_stage, this->pipe_stage);
    num_consumer_wgs = num_warp_groups - num_producer_wgs;
    assert(pipe_stage % num_consumer_wgs == 0 && "for now assume workloads are balanced acroos consumer wgs");
    assert(config::NUM_THREADS_PER_WARP_GROUP * num_warp_groups == block_dim.x);
  }

//   void Graph::assign_task(mirage::kernel::DTensor const &stensor, vector<int> warpgroup_ids) {
//    int ws_num = warpgroup_ids.size();
//    assert(ws_num <= config::MAX_NUM_WARP_GROUPS && "MAX NUM WARP GROUPS is 4");

//    TBOperator &op = stensor.owner_op;

//    vector<int> warp_ids = op.warp_ids;

//    //split the stensor to different warpgroups
//    if(stensor.owner_op.op_type == type::TB_INPUT_OP){
//     //assign warp groups to TMA
//     warp_ids.push_back(warpgroup_ids);
//     producer_wgs.insert(warpgroup_ids);

//    }else if(stensor.owner_op.op_type == type::TB_MATMUL_OP){
//     // assign warp groups to tensor core
//     // now assume partition along the M tensor
//     //exp. 128X64X64 -> 2 X 64X64X64
//     //for now we put p/c in different wgs
//     for(int i = 0; i < warp_ids.size(); i++){
//       assert(producer_wgs.find(warp_ids.at(i)) == producer_wgs.end() 
//       && "producer consumer can't be in the same wg");
//       consumer_wgs.insert(warp_ids.at(i));
//     }
//     tb::TBMatmulOp const *mma =
//         dynamic_cast<tb::TBMatmulOp const *>(stensor.owner_op);
//     warp_ids.push_back(warpgroup_ids);
    
//     if(warp_ids.size() >= 1){
//       //m tensor
//       STensor &A = mma.input_tensors.at(0);
//       assert(A.dims[0] % warp_ids.size() == 0 && "M shape should be divided by warpgroup nums");
//       // todo violate any check of total element nums?
//       A.dims[A.num_dims - 2] / warp_ids.size();
//       C.dims[A.num_dims - 2] / warp_ids.size();
//       A.warp_partition_dim = warp_ids.size();
//       C.warp_partition_dim = warp_ids.size();
//     }

//    }else{
//     //assgin warp groups to cuda core(ALU)
//     warp_ids.push_back(warpgroup_ids);
//    }
// }

}
}