#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"

using namespace cute;

namespace tb {

static int const kStages = 2;

template <typename MainloopPipeline, size_t N>
struct SharedStorage {
  struct PipelineStorage : cute::aligned_struct<16, _1> {
    using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;
    alignas(16) MainloopPipelineStorage mainloop;
  };

  PipelineStorage pipelines[N]; // Define pipelines as an array of size N
};

enum class WarpGroupRole {
  Producer = 1,
  Consumer = 0,
};

enum class ProducerWarpRole {
  MainloopEpilogue = 0,
  Warp1 = 1,
  Warp2 = 2,
  Warp3 = 3
};

} // namespace tb
