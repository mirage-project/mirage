#pragma once

#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"

namespace kn {

template <typename Config>
static __global__ void chunk_kernel_fwd(typename Config::T *__restrict__ out0,
                                        typename Config::T *__restrict__ out1,
                                        typename Config::T const *__restrict__ in) {
    using T = typename Config::T;
    using Numel = typename Config::Numel;

    auto src_layout = (typename Config::SrcLayout){};
    auto dst0_layout = (typename Config::Dst0Layout){};
    auto dst1_layout = (typename Config::Dst1Layout){};

    static constexpr int SRC_SIZE = get<Config::CHUNK_DIM>(shape(src_layout));
    static constexpr int DST0_SIZE = get<Config::CHUNK_DIM>(shape(dst0_layout));
    static constexpr int DST1_SIZE = get<Config::CHUNK_DIM>(shape(dst1_layout));
    static_assert(DST0_SIZE == SRC_SIZE / 2);
    static_assert(DST1_SIZE == SRC_SIZE / 2);

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < Numel{}) {
        auto src_coord = src_layout.get_flat_coord(idx);
        auto src_phy_pos = src_layout(idx);
        if (get<Config::CHUNK_DIM>(src_coord) < DST0_SIZE) {
            auto dst0_phy_pos = dst0_layout(src_coord);
            out0[dst0_phy_pos] = in[src_phy_pos];
        } else {
            replace<Config::CHUNK_DIM>(src_coord, get<Config::CHUNK_DIM>(src_coord) - DST0_SIZE);
            auto dst1_phy_pos = dst1_layout(src_coord);
            out1[dst1_phy_pos] = in[src_phy_pos];
        }
    }
}

template<typename T_,
         typename SrcLayout_,
         typename Dst0Layout_,
         typename Dst1Layout_,
         int CHUNK_SIZE_,
         int CHUNK_DIM_>
class ChunkKernel {
public:
    using T = T_;
    using SrcLayout = SrcLayout_;
    using Dst0Layout = Dst0Layout_;
    using Dst1Layout = Dst1Layout_;

    using Numel = decltype(cute::size(SrcLayout{}));
    static constexpr int CHUNK_SIZE = CHUNK_SIZE_;
    static constexpr int CHUNK_DIM = CHUNK_DIM_;

    static constexpr int BLOCK_SIZE = 512;
    static constexpr dim3 block_shape = {BLOCK_SIZE, 1, 1};
    static constexpr dim3 grid_shape = {ceil_div(Numel::value, BLOCK_SIZE), 1, 1};

    static void run(T *out0, T *out1, T const *in) {
        chunk_kernel_fwd<ChunkKernel<T, SrcLayout, Dst0Layout, Dst1Layout, CHUNK_SIZE, CHUNK_DIM>><<<grid_shape, block_shape>>>(out0, out1, in);
    }
};

}