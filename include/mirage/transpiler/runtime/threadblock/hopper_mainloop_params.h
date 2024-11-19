#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cute/arch/cluster_sm90.hpp"

using namespace cute;

namespace tb {

// static int const NumThreadsPerWarp = 32;
// static int const NumThreadsPerWarpGroup = 128;
// static int const NumWarpsPerWarpGroup =
//     NumThreadsPerWarpGroup / NumThreadsPerWarp;

static int const kStages = 2;

template <typename MainloopPipeline, size_t N>
struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
        using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;

        alignas(16) MainloopPipelineStorage mainloop;
    };

    PipelineStorage pipelines[N];  // Define pipelines as an array of size N
};

enum class WarpGroupRole {
    Producer = 0,
    Consumer = 1,
};

}


