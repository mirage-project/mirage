/* Copyright 2023-2024 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mirage/threadblock/element_unary.h"
#include "mirage/threadblock/forloop_accum.h"
#include "mirage/threadblock/reduction.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/transpiler/common.h"
#include "mirage/transpiler/structs.h"
#include "mirage/transpiler/transpiler.h"

#include <algorithm>
#include <unordered_set>

#include "mirage/threadblock/graph.h"
#include "mirage/transpiler/sched_tb_graph.h"
#include "mirage/transpiler/utils.h"
#include "mirage/type.h"

namespace mirage {
namespace transpiler {

using std::string;
namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

namespace get_layout_detail {

// Get a CuTe layout from dims and strides
//
// The reason why we reverse the vector is that in CuTe, when mapping from an
// integer to a logical coordinate, the first dimension is consider to be the
// "innermost" (here "innermost" has a different meaning from the innermost dim)
//
// For example, assume the tensor has a shape of (3, 2), then 1 will be mapped
// to (1, 0) instead of (0, 1), which is not the same as the C/C++ convention
static string get_cute_layout(vector<int> dims, vector<size_t> strides) {
  assert(dims.size() == strides.size());
  std::reverse(dims.begin(), dims.end());
  std::reverse(strides.begin(), strides.end());
  return fmt("Layout<Shape<$>, Stride<$>>",
             map_to_cute_int(dims),
             map_to_cute_int(strides));
}

static auto get_cute_layout_array(vector<int> dims,
                                  vector<size_t> strides,
                                  bool swap01 = true)
    -> std::pair<std::vector<int>, std::vector<size_t>> {
  assert(dims.size() == strides.size());

  if (swap01) {
    std::reverse(dims.begin(), dims.end());
    std::reverse(strides.begin(), strides.end());
  }

  return {dims, strides};
}

static string get_reversed_cute_layout(vector<int> dims,
                                       vector<size_t> strides) {
  assert(dims.size() == strides.size());
  return fmt("Layout<Shape<$>, Stride<$>>",
             map_to_cute_int(dims),
             map_to_cute_int(strides));
}

template <typename Tensor_T, typename Meta_T>
static string
    get_cute_layout(Tensor_T const &tensor, Meta_T const &meta, int start_dim) {
  return get_cute_layout(
      vector<int>(tensor.dim + start_dim, tensor.dim + tensor.num_dims),
      vector<size_t>(meta.strides + start_dim, meta.strides + tensor.num_dims));
}

template <typename Tensor_T, typename Meta_T>
static string get_swap_01_layout(Tensor_T const &tensor,
                                 Meta_T const &meta,
                                 int start_dim) {
  return get_reversed_cute_layout(
      vector<int>(tensor.dim + start_dim, tensor.dim + tensor.num_dims),
      vector<size_t>(meta.strides + start_dim, meta.strides + tensor.num_dims));
}

// A helper function
template <typename T>
static std::vector<T> mov_to_last(T const *vec, size_t numel, int idx) {
  std::vector<T> result;
  result.reserve(numel);
  result.insert(result.end(), vec, vec + idx);
  result.insert(result.end(), vec + idx + 1, vec + numel);
  result.push_back(vec[idx]);
  return result;
}
} // namespace get_layout_detail

// Get the layout of a STensor
static string get_stensor_layout(tb::STensor const &stensor,
                                 STensorMeta const &meta,
                                 int start_dim = 0,
                                 bool swap01 = true) {

  if (!swap01) {
    if (!meta.is_xor_swizzled) {
      // Do not need to swizzle
      // (Probably swizzled by SHIFT-based swizzling, but we do not care about
      // that)
      return get_layout_detail::get_swap_01_layout(stensor, meta, start_dim);
    } else {
      // XOR-based swizzling
      // Use the planner's computed swizzle. These were hardcoded to
      // Swizzle<3,3,4>, which made every generated Blackwell matmul return the
      // right values at wrong positions: the UMMA reads its operands through a
      // 128B swizzle over 16B chunks, which in raw element units is
      // Swizzle<3,3,3>, and 3,3,4 differs from it by one bit of chunk
      // granularity. Verified against a PyTorch reference by sweeping <B,M,S>:
      // only <3,3,3> gives an exact match at (M,N,K)=(128,64,64), and that is
      // precisely what get_threadblock_swizzle_plan_blackwell computes here.
      return fmt(
          "decltype(composition(Swizzle<$, $, $>{}, ${}))",
          meta.xor_swizzle_b,
          meta.xor_swizzle_m,
          meta.xor_swizzle_s,
          get_layout_detail::get_swap_01_layout(stensor, meta, start_dim));
    }
  }

  if (!meta.is_xor_swizzled) {
    // Do not need to swizzle
    // (Probably swizzled by SHIFT-based swizzling, but we do not care about
    // that)
    return get_layout_detail::get_cute_layout(stensor, meta, start_dim);
  } else {
    return fmt("decltype(composition(Swizzle<$, $, $>{}, ${}))",
               meta.xor_swizzle_b,
               meta.xor_swizzle_m,
               meta.xor_swizzle_s,
               get_layout_detail::get_cute_layout(stensor, meta, start_dim));
  }
}

// Move the innermost dim to the last dim, and format it as a CuTe layout
// string.
//
// Assume the tensor has N dimensions and the innermost dim is i, then the
// function is equivalent to torch.permute(tensor, [0, 1, ..., i-1, i+1, ..., N,
// i])
static string mov_last_get_stensor_layout(tb::STensor const &stensor,
                                          STensorMeta const &meta,
                                          int innermost_dim,
                                          bool swap01 = true) {
  tb::STensor new_stensor = stensor;
  STensorMeta new_meta = meta;
  new_meta.swizzled_dim = -1;
  for (int i = 0; i < stensor.num_dims; ++i) {
    int src_dim = i == stensor.num_dims - 1 ? innermost_dim
                  : i < innermost_dim       ? i
                                            : i + 1;
    new_stensor.dim[i] = stensor.dim[src_dim];
    new_meta.strides[i] = meta.strides[src_dim];
    if (src_dim == meta.swizzled_dim) {
      new_meta.swizzled_dim = i;
    }
  }
  new_meta.innermost_dim = stensor.num_dims - 1;

  return get_stensor_layout(new_stensor, new_meta, 0, swap01);
}

// Move the innermost dim to the last dim, and format it as a CuTe layout
// string.
//
// Assume the tensor has N dimensions and the innermost dim is i, then the
// function is equivalent to torch.permute(tensor, [0, 1, ..., i-1, i+1, ..., N,
// i])
static auto mov_last_get_stensor_shape_stride(tb::STensor const &stensor,
                                              STensorMeta const &meta,
                                              int innermost_dim,
                                              bool swap01 = false)
    -> std::pair<std::vector<int>, std::vector<size_t>> {
  tb::STensor new_stensor = stensor;
  STensorMeta new_meta = meta;
  new_meta.swizzled_dim = -1;
  for (int i = 0; i < stensor.num_dims; ++i) {
    int src_dim = i == stensor.num_dims - 1 ? innermost_dim
                  : i < innermost_dim       ? i
                                            : i + 1;
    new_stensor.dim[i] = stensor.dim[src_dim];
    new_meta.strides[i] = meta.strides[src_dim];
    if (src_dim == meta.swizzled_dim) {
      new_meta.swizzled_dim = i;
    }
  }
  new_meta.innermost_dim = stensor.num_dims - 1;

  return {vector<int>(new_stensor.dim, new_stensor.dim + new_stensor.num_dims),
          vector<size_t>(new_meta.strides,
                         new_meta.strides + new_stensor.num_dims)};
}

// Get the layout of a DTensor tile for input/output operators
static string get_dtensor_tile_layout(kn::DTensor const &dtensor,
                                      DTensorMeta const &d_meta,
                                      tb::STensor const &stensor,
                                      STensorMeta const &s_meta,
                                      int d_innermost_dim) {
  using namespace get_layout_detail;
  return get_cute_layout(
      mov_to_last(stensor.dim,
                  dtensor.num_dims,
                  d_innermost_dim), // Here we use stensor.dim
      mov_to_last(d_meta.strides, dtensor.num_dims, d_innermost_dim));
}

static string append_epilogue_scalars(
    std::vector<std::pair<tb::TBOperator const *, TBSchedOpMeta>> const
        &chain) {
  string res = "const float scalars[] = {";
  if (chain.size() == 1) {
    return res.append("0.0f};");
  }
  // Mirrors the Ampere version (transpiler_tb.cc). The old Blackwell copy
  // replaced the LAST chain element's scalar with 0.0f unconditionally, on the
  // assumption that the last op is always the store -- but when an op with a
  // real scalar is the final fused op (e.g. reduction with mul_scalar fused
  // into its epilogue), that dropped the scalar and the epilogue multiplied by
  // ZERO. A decomposed RMSNorm lost its 1/H exactly this way. Only a trailing
  // forloop-accum stands for the store epilogue; everything else contributes
  // its own scalar, with the store's 0.0f appended afterwards.
  bool store_last = true;
  for (size_t i = 1; i < chain.size(); i++) {
    if (i == chain.size() - 1 &&
        chain.at(i).first->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
      // last one is EpilogueStoreAccum
      res.append("0.0f};");
      store_last = false;
    } else if (is_threadblock_element_unary(chain.at(i).first->op_type)) {
      tb::TBElementUnaryOp const *tb_unary_op =
          dynamic_cast<tb::TBElementUnaryOp const *>(chain.at(i).first);
      res.append(fmt("$f, ", tb_unary_op->scalar));
    } else {
      res.append("0.0f, ");
    }
  }
  if (store_last) {
    res.append("0.0f};");
  }
  return res;
}

static string get_tb_op_str(type::TBOperatorType type) {
  auto toString = [](type::TBOperatorType type) -> string {
    switch (type) {
      case type::TB_EXP_OP:
        return "EXP";
      case type::TB_SQUARE_OP:
        return "SQUARE";
      case type::TB_SQRT_OP:
        return "SQRT";
      case type::TB_MUL_SCALAR_OP:
        return "MULSCALAR";
      case type::TB_SILU_OP:
        return "SILU";
      case type::TB_GELU_OP:
        return "GELU";
      case type::TB_RELU_OP:
        return "RELU";
      case type::TB_CLAMP_OP:
        return "CLAMP";
      default:
        assert(0);
    }
  };

  return toString(type);
}

static void add_cta_warp_selector_if_need(CodeKeeper &code,
                                          bool is_in_loop,
                                          bool select_one_cta,
                                          bool select_one_warp) {
  if (!is_in_loop) {
    return;
  }
  if (select_one_cta && select_one_warp) {
    code.e("if (elect_one_cta && elect_one_warp) {");
  } else if (select_one_cta) {
    code.e("if (elect_one_cta) {");
  } else if (select_one_warp) {
    code.e("if (elect_one_warp) {");
  }
}

static std::pair<bool, std::vector<int64_t>>
    add_loop_node_consumer_wait_if_need(
        tb::TBOperator const *op,
        CodeKeeper &code,
        bool is_in_loop,
        std::map<int64_t, tb::TBInputOp const *> &pipeline_inputs) {
  if (!is_in_loop) {
    return {false, {}};
  }
  std::vector<int64_t> input_ids_waited;
  for (int i = 0; i < op->input_tensors.size(); i++) {
    int64_t input_id = op->input_tensors.at(i).guid;
    if (pipeline_inputs.find(input_id) != pipeline_inputs.end()) {
      code.e("int read_idx_$ = blackwell_async_pipeline_$.consumer_wait();",
             input_id,
             input_id);
      // only wait once
      pipeline_inputs.erase(input_id);
      input_ids_waited.push_back(input_id);
    }
  }

  if (!input_ids_waited.empty()) {
    return {true, input_ids_waited};
  }

  return {false, {}};
}

static void generate_Tmem_mbarrier_init_code(CodeKeeper &code,
                                             int arrive_cnt,
                                             string &tmem_base_ptr_name,
                                             string &mbarrier_ptr_name) {
  code.e("// Tensory Memory Allocation");

  // 1-SM MMA must use the 1-SM TMEM allocator: Allocator2Sm emits
  // cta_group::2 TMEM instructions, and ptxas rejects a kernel that mixes
  // .cta_group::1 (the 1-SM tcgen05 MMA) with .cta_group::2.
  code.e("using TmemAllocator = cute::TMEM::Allocator1Sm;");
  code.e("TmemAllocator tmem_allocator{};");
  code.e("if (elect_one_warp) { ");
  code.e(
      "tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, $); ",
      tmem_base_ptr_name);
  code.e("}");
  code.e("");
  code.e("__syncthreads();");

  if (arrive_cnt > 0) {
    code.e("// MMA mbarrier Initialization");
    code.e("if (elect_one_warp && cute::elect_one_sync()) {");
    code.e("cute::initialize_barrier(*$, $);", mbarrier_ptr_name, arrive_cnt);
    code.e("}");
  }
  // The sync mask for the blocks within the same cluster
  code.e("uint16_t consumer_sync_mask = static_cast<uint16_t>((1u << "
         "size(cluster_shape)) - 1u);");
  // code.e("uint16_t consumer_sync_mask = (1 << blockIdx.y * gridDim.x +
  // (blockIdx.x / 2) * 2) | (1 << blockIdx.y * gridDim.x + (blockIdx.x / 2) * 2
  // + 1);");
}

// Transpile a custom KN operator (i.e. a custom block graph) into CUDA code
// Will return a CustomOPTranspileResult object. See comments in transpiler.h
// for more details
CustomOPTranspileResult
    Transpiler::transpile_kn_custom_op_blackwell(kn::KNCustomizedOp const *op) {
  bool profiling = config.profiling;

  tb::Graph const &g = op->bgraph;
  int num_threads = g.block_dim.x * g.block_dim.y * g.block_dim.z;

  size_t profiler_buf_size =
      profiling ? (g.grid_dim.x * g.grid_dim.y * g.grid_dim.z *
                   (config.num_consumer_wgs + config.num_producer_wgs)) *
                      1000
                : 0;

  // Allocate a kernel name
  static int custom_kernel_idx_counter = 0;
  int cur_custom_kernel_idx = custom_kernel_idx_counter++;
  string func_name = fmt("custom_kernel_$", cur_custom_kernel_idx);

  // A warp-specialized region needs at least one consumer warp group. With
  // num_consumer_wgs == 0 warpgroup 0 becomes the producer and nothing consumes
  // the pipeline, so the kernel deadlocks by construction -- and the thread
  // count check below still passes, so it used to compile and hang. Python maps
  // num_consumer_wgs = num_warp_groups - 1 (core.pyx), i.e. num_warp_groups must
  // be >= 2 whenever forloop_range > 1.
  if (GPU_CC::B200 != config.target_cc ||
      (config::MAX_NUM_WARP_GROUPS <
       config.num_consumer_wgs + config.num_producer_wgs) ||
      (config.num_consumer_wgs < 1 && g.forloop_range > 1) ||
      (num_threads !=
           (config.num_consumer_wgs + config.num_producer_wgs) * 128 &&
       (g.forloop_range > 1))) {
    assert(false && "compiler assertion failure");
    return CustomOPTranspileResult{
        CUDA_T_CONFIG_ERROR, func_name, 0, 0, "", {}};
  }

  // int barrier_size = 16 * config.pipeline_stages;
  int tma_barrier_size = 16 * config.pipeline_stages;

  int cluster_barrier_size = 8;

  // Get the schedule
  TBSched sched = get_threadblock_schedule(g);

  get_threadblock_swizzle_plan_blackwell(g, sched);

  // Get the memory allocation plan
  TBMemoryPlan mem_plan = get_threadblock_memory_plan(g, sched, true, true);

  // Wide (row pitch > 128B) NON-pipelined matmul operands -- e.g. attention's
  // Q at head_dim=128 -- cannot go through InputChunkedSyncCopy: the UMMA
  // reads a non-pipelined operand through DstPipeLayout_A/B and beyond a 128B
  // pitch no dense-stride solver layout matches its panel tiling. Route them
  // through InputWideOperandSyncCopy instead, which writes through the same
  // cutlass-derived layout the matmul reads. Conditions: loaded whole (no
  // forloop tiling), 2-D tile, consumed ONLY by matmuls (any other consumer
  // reads the solver layout, which the wide atom does not write).
  std::unordered_set<sguid_t> wide_matmul_operands;
  for (tb::TBOperator const *tb_op : g.operators) {
    if (tb_op->op_type != type::TB_INPUT_OP) {
      continue;
    }
    tb::TBInputOp const *in_op = dynamic_cast<tb::TBInputOp const *>(tb_op);
    if (in_op->forloop_dim != -1) {
      continue;
    }
    tb::STensor const &st = tb_op->output_tensors.at(0);
    if (!stensor_metas.count(st.guid)) {
      continue;
    }
    STensorMeta const &mt = stensor_metas.at(st.guid);
    if (mt.is_pipelined_input) {
      continue;
    }
    bool leading_ones = true;
    for (int i = 0; i < st.num_dims - 2; ++i) {
      leading_ones &= (st.dim[i] == 1);
    }
    if (st.num_dims < 2 || !leading_ones) {
      continue;
    }
    bool narrow_ok =
        mt.is_xor_swizzled && mt.swizzled_dim >= 0 &&
        (size_t)mt.strides[mt.swizzled_dim] *
                type::get_datatype_size(st.data_type) <=
            128;
    if (narrow_ok) {
      continue; // the chunked-copy path handles it
    }
    bool any_consumer = false, all_matmul = true;
    for (tb::TBOperator const *cons : g.operators) {
      for (tb::STensor const &it : cons->input_tensors) {
        if (it.guid == st.guid) {
          any_consumer = true;
          all_matmul &= (cons->op_type == type::TB_MATMUL_OP);
        }
      }
    }
    if (any_consumer && all_matmul) {
      wide_matmul_operands.insert(st.guid);
    }
    if (getenv("MIRAGE_DEBUG_WIDE")) {
      fprintf(stderr,
              "[wide] st=%lld fd=%d pipe=%d xor=%d swdim=%d cons=%d allmm=%d "
              "-> %d\n",
              (long long)st.guid,
              in_op->forloop_dim,
              (int)mt.is_pipelined_input,
              (int)mt.is_xor_swizzled,
              mt.swizzled_dim,
              (int)any_consumer,
              (int)all_matmul,
              (int)wide_matmul_operands.count(st.guid));
    }
  }

  std::vector<TMAParams> tmaParamsList;

  // Generate code prologue
  CodeKeeper code;
  // Mirror of the MMA declarations the TMA atoms are derived from. A task body
  // needs them a second time on the HOST, in the builder that constructs and
  // uploads the atoms (see the builder emission at the end of this function).
  // Only populated in device-body mode.
  CodeKeeper mma_setup;
  auto e_mma = [&](auto const &fmt_str, auto const &...args) {
    code.e(fmt_str, args...);
    if (config.emit_device_body) {
      mma_setup.e(fmt_str, args...);
    }
  };
  string thread_idx;
  if (g.block_dim.y > 1 || g.block_dim.z > 1) {
    thread_idx = fmt("threadIdx.x + threadIdx.y * $ + threadIdx.z * $",
                     g.block_dim.x,
                     g.block_dim.x * g.block_dim.y);
  } else {
    thread_idx = "threadIdx.x";
  }
  code.e("int thread_idx = $;", thread_idx);

  // Warp specialization (a dedicated producer warp group issuing TMA loads plus
  // consumer warp groups doing the compute) is only emitted when there are
  // pipelined inputs, which in turn requires a real for-loop to pipeline
  // across. With forloop_range == 1 no producer/consumer split is generated at
  // all, so *every* thread in the block runs the copy and compute kernels.
  //
  // Those kernels are templated on NUM_THREADS / CONSUMER_NUM_THREADS. Leaving
  // both pinned at one warp group while the block actually has more threads
  // meant the surplus threads ran work sized for a smaller group, racing on and
  // overrunning the tile. Size the constants to the threads that really execute
  // the region instead.
  // The producer/consumer split is only ever EMITTED for pipelined inputs
  // (the producer loop iterates pipelined_input_ops). A forloop graph whose
  // tiled inputs were all demoted to in-loop sync copies (no matmul
  // consumers) has no split: every thread runs the loop, so the barrier
  // constants must cover the whole block or wg_sync deadlocks.
  bool has_pipelined_input = false;
  for (tb::TBOperator const *tb_op : g.operators) {
    for (tb::STensor const &st : tb_op->output_tensors) {
      if (stensor_metas.count(st.guid) &&
          stensor_metas.at(st.guid).is_pipelined_input) {
        has_pipelined_input = true;
      }
    }
  }
  bool const is_warp_specialized = (g.forloop_range > 1) && has_pipelined_input;
  int const compute_num_threads =
      is_warp_specialized
          ? config::NUM_THREADS_PER_GROUP * config.num_consumer_wgs
          : num_threads;

  code.e("static constexpr int NUM_THREADS = $;",
         is_warp_specialized ? config::NUM_THREADS_PER_GROUP : num_threads);
  code.e("static constexpr int CONSUMER_NUM_THREADS = $;", compute_num_threads);

  code.e("auto cluster_shape = make_shape(Int<$>{}, Int<$>{}, Int<$>{});",
         g.cluster_dim.x,
         g.cluster_dim.y,
         g.cluster_dim.z);

  code.e("uint32_t elect_one_warp = (threadIdx.x / 32 == 0); ");
  code.e("int cta_rank = cute::block_rank_in_cluster();");

  // `elect_one_cta` selects the leader CTA of each 2-SM MMA pair. It MUST be
  // declared here, unconditionally: the forloop-accumulator and epilogue paths
  // emit `if (elect_one_cta && elect_one_warp)` (see
  // add_cta_warp_selector_if_need) for every graph, including graphs with no
  // matmul at all. Declaring it only inside the matmul branch left those graphs
  // referencing an undeclared identifier, so any pure-elementwise custom op
  // failed to compile.
  //
  // This is equivalent to the previous
  //   get<0>(cluster_layout_vmnk.get_flat_coord(cta_rank)) == 0
  // formulation: cluster_layout_vmnk is make_layout(cluster_shape) tiled_divided
  // by AtomThrID, so mode 0 of the flat coord is exactly cta_rank % |AtomThrID|.
  // Computing it from cta_rank directly removes the dependency on tiled_mma,
  // which does not exist when the graph has no matmul.
  // 1-SM MMA (see the tiled_mma emission below): AtomThrID == 1, so every CTA
  // is its own MMA leader and elect_one_cta is unconditionally true. Kept as a
  // named variable because the accumulator/epilogue paths guard on it.
  int const mma_atom_thr_size = 1;
  code.e("bool elect_one_cta = (cta_rank % $) == 0;", mma_atom_thr_size);

  code.e("// STensors");
  // Declared in BOTH modes. An MPK task body owns its shared-memory
  // declaration exactly like a handwritten .cuh task does (see e.g.
  // tasks/blackwell/mla_mtp_decode_tp8_sm100.cuh); there is no smem pointer to
  // receive from the megakernel.
  //
  // 1024B is the period of the Swizzle<3,3,3> the UMMA reads its operands
  // through, and what plan_stensor_memory aligns operand offsets to. Those
  // offsets are relative to `buf`, so the base must carry the alignment too. A
  // standalone kernel gets it for free (dynamic smem starts the window); a task
  // body does not, since the worker kernel has static __shared__ ahead of it.
  // Declaring the alignment is the right way to ask for it -- an earlier
  // attempt rounded the `buf` POINTER up at runtime instead, which is unsafe
  // because the shift is not reflected in smem_size and the body can then
  // overrun its allocation.
  //
  // For the record this alignment was NOT what made generated matmul tasks
  // wrong (that was an unpinned output layout, see resolve_tensor_layout.cc);
  // it is here because the planner's assumption should be stated, not assumed.
  code.e("extern __shared__ __align__(1024) char buf[];");

  code.e("");

  bool need_mbarrier = false;
  string tmem_base_ptr_name = "tmem_base_ptr";
  string mbarrier_ptr_name = "mma_barrier_ptr";

  bool has_tmem_base_ptr = false;
  size_t barrier_addr = mem_plan.smem_size;
  for (auto [guid, addr] : mem_plan.addrs) {
    if (guid == mem_plan.tmem_base_ptr_guid) {
      code.e("uint32_t *$ = (uint32_t*)(buf + $);", tmem_base_ptr_name, addr);
      has_tmem_base_ptr = true;
    } else if (guid == mem_plan.mbarrier_buf_guid_offset) {
      // The SHARED completion barrier for matmuls whose accumulators are
      // written back after the loop.
      code.e("uint64_t *$ = (uint64_t*)(buf + $);", mbarrier_ptr_name, addr);
      need_mbarrier = true;
    } else if (guid > mem_plan.mbarrier_buf_guid_offset) {
      // Per-matmul barrier, used only by a CHAINED matmul (one whose result is
      // consumed inside the loop and must therefore be waited on each
      // iteration). Declared for every matmul; the unused ones are dropped by
      // the compiler.
      code.e("uint64_t *$_$ = (uint64_t*)(buf + $);",
             mbarrier_ptr_name,
             guid - mem_plan.mbarrier_buf_guid_offset,
             addr);
    } else {
      code.e("$ *stensor$_ptr = ($*)(buf + $);",
             get_datatype_str(op->input_tensors[0].data_type),
             guid,
             get_datatype_str(op->input_tensors[0].data_type),
             addr);
    }
  }

  // Erase the lowest 16 bytes to 0 for GEMM. Must happen BEFORE the TMEM
  // allocation below, which stores the allocated TMEM base into buf+0.
  code.e("*((uint128_t*)buf) = 0ul;");
  code.e("");

  // A matmul whose result is consumed by ANOTHER op inside the loop (rather
  // than only by its forloop accumulator) is not supported yet, and used to
  // produce garbage silently. Only the final accumulators get a
  // write_tC_to_sC (see the in-register write-back at the end of this
  // function), so a chained matmul's TMEM result is never materialised into the
  // smem tile the consumer reads -- and any elementwise op fused into that
  // matmul's epilogue (num_exps_before_store) never runs either. Emitted for
  // Q@K^T -> exp -> @V it compiled, ran, and returned rel 1.0.
  //
  // Supporting this needs, per iteration: wait for that matmul's MMA, write its
  // accumulator (with the fused epilogue) to smem, sync, and reset the
  // accumulator to ScaleOut::Zero because it does NOT accumulate across the
  // forloop. That is the fused-attention shape; see the notes in DESIGN.md.
  // See CUDA_T_FL1_PIPELINED_DEADLOCK in error_types.h.
  if (g.forloop_range == 1) {
    for (auto const &[guid, meta] : stensor_metas) {
      if (meta.is_pipelined_input) {
        return CustomOPTranspileResult{
            CUDA_T_FL1_PIPELINED_DEADLOCK, func_name, 0, 0, "", {}};
      }
    }
  }

  std::unordered_set<sguid_t> chained;
  // For each chained intermediate, the OUTPUT guid of the matmul that consumes
  // it. Its per-matmul barrier doubles as the anti-dependency barrier: the
  // producer must not overwrite the tile while the consumer's async MMA is
  // still reading it, so the consumer arrives there each iteration and the
  // producer waits on the PREVIOUS iteration's arrival before storing.
  std::unordered_map<sguid_t, sguid_t> chained_consumer;
  // GENERIC producers of matmul operands need the same anti-dependency: the
  // consuming MMA of iteration i reads the tile through the ASYNC proxy, and
  // nothing else stops iteration i+1's elementwise producer from overwriting
  // it mid-read. (Q=0 made this invisible -- overwriting with identical
  // values -- which is how it survived: the online-softmax numerator read
  // torn E tiles while the uniform-E probe was exact.) Maps the producer
  // node's output guid to the consuming matmul's output guid, whose
  // per-matmul barrier the consumer already arrives on each iteration.
  std::unordered_map<sguid_t, sguid_t> generic_antidep;
  {
    for (TBSchedNode const &prod : sched.loop_nodes) {
      if (prod.type != tb_sched_node_t::OPERATOR ||
          prod.ops.front().first->op_type != type::TB_MATMUL_OP) {
        continue;
      }
      // The node's LAST op is what other nodes see: the scheduler fuses an
      // elementwise op (e.g. exp) into the matmul node, so the visible output
      // is the fused op's, not the raw matmul's.
      sguid_t const produced =
          prod.ops.back().first->output_tensors.at(0).guid;
      // IN-LOOP consumers only. A post-loop consumer reads the forloop
      // ACCUMULATOR, which is materialised by the in-register write-back at the
      // end of the kernel -- that is the ordinary fused-epilogue shape (SwiGLU
      // is silu/mul on two accumulators) and is fully supported. Scanning
      // post-loop nodes here rejected it.
      for (TBSchedNode const &cons : sched.loop_nodes) {
        if (cons.type != tb_sched_node_t::OPERATOR || &cons == &prod) {
          continue;
        }
        for (auto const &[cop, cmeta] : cons.ops) {
          if (cop->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
            continue; // the only legal in-loop consumer
          }
          for (auto const &in : cop->input_tensors) {
            if (in.guid == produced) {
              chained.insert(produced);
              if (cop->op_type == type::TB_MATMUL_OP) {
                chained_consumer[produced] =
                    cons.ops.back().first->output_tensors.at(0).guid;
              }
            }
          }
        }
      }
    }
    // forloop_range == 1 chained matmuls are DONE and on by default
    // (Q@K^T -> exp -> @V at rel 3.2e-3, pinned by
    // test_chained_matmul_exp_matmul).
    //
    // forloop_range > 1 chained matmuls (flash-attention iteration) are ON:
    // per-iteration accumulator reset, phase-alternating waits, anti-dep via
    // the consumer's barrier, both proxy fences, mixed pipelined operands
    // (two-stage run()), and N_LOOP tile advance (the last bug: the input atom
    // walked Tiles_K while attention tiles N, refetching K0 forever).
    // Verified: model-A match at rel 4e-3 (attn_localize.py).
    for (TBSchedNode const &prod : sched.loop_nodes) {
      if (prod.type != tb_sched_node_t::OPERATOR ||
          prod.ops.front().first->op_type == type::TB_MATMUL_OP) {
        continue; // matmul producers use the chained path
      }
      sguid_t const produced =
          prod.ops.back().first->output_tensors.at(0).guid;
      for (TBSchedNode const &cons : sched.loop_nodes) {
        if (cons.type != tb_sched_node_t::OPERATOR ||
            cons.ops.front().first->op_type != type::TB_MATMUL_OP) {
          continue;
        }
        for (auto const &in : cons.ops.front().first->input_tensors) {
          if (in.guid == produced) {
            generic_antidep[produced] =
                cons.ops.back().first->output_tensors.at(0).guid;
          }
        }
      }
    }

    // Anti-dependency needs per consumer type:
    //  * a MATMUL consumer reads through the ASYNC proxy after the elect warp
    //    issues it -- the producer's next-iteration store must wait on the
    //    consumer's per-matmul barrier (chained_consumer, emitted below).
    //  * a GENERIC consumer (reduction_max, rescale accums, elementwise) runs
    //    on all consumer threads between this iteration's wg_syncs; the next
    //    iteration's store sits behind the loop-top sync, so thread-level
    //    barriers already order it. No extra barrier needed -- and rejecting
    //    these blocked the online-softmax rewrite, whose Q@K^T result is
    //    consumed by reduction_max.
  }

  // Every matmul issues one umma_arrive per forloop iteration, and the epilogue
  // waits for phase 0 of a barrier expecting `arrive_cnt` of them. Counting only
  // the iterations assumed exactly one matmul: with two, phase 0 completed
  // halfway through the loop and the accumulator write-back raced the MMAs that
  // were still running.
  int num_matmuls = 0;
  for (TBSchedNode const &node :
       Combine(sched.loop_nodes, sched.post_loop_nodes)) {
    if (node.type == tb_sched_node_t::OPERATOR &&
        node.ops.front().first->op_type == type::TB_MATMUL_OP &&
        // A chained matmul arrives on its OWN barrier, not the shared one.
        !chained.count(node.ops.back().first->output_tensors.at(0).guid)) {
      num_matmuls += 1;
    }
  }
  if (need_mbarrier) {
    CodeKeeper res;
    // In 2sm mma, one one cta in each 2-CTA pair issues arrival
    // 1-SM MMA: every CTA issues its own MMA and arrives itself, so there is
    // no division by the 2-CTA pair size that the 2-SM path needed.
    //
    // The shared barrier exists only for matmuls whose accumulator is written
    // back AFTER the loop. When every matmul is chained (its result is
    // materialised in-loop -- the online-softmax rewrite makes BOTH matmuls
    // feed rescale accumulators, which read the result from smem each
    // iteration), nothing ever arrives on it: emitting
    // initialize_barrier(count = 0) is invalid mbarrier usage, and it drove
    // ptxas into an internal compiler error (C7907) at -O1+ on sm_100a.
    // Guard the init; the TMEM allocation part must still run.
    generate_Tmem_mbarrier_init_code(res,
                                     num_matmuls > 0
                                         ? g.forloop_range * num_matmuls *
                                               g.cluster_dim.x *
                                               g.cluster_dim.y * g.cluster_dim.z
                                         : -1 /* skip barrier init */,
                                     tmem_base_ptr_name,
                                     mbarrier_ptr_name);
    // Each chained matmul gets a count-1 barrier: exactly one MMA completes
    // against it before its result is read, within the same iteration. The
    // CONSUMER's per-matmul barrier (the anti-dependency wait) must be
    // initialized too -- it was declared and waited on but never initialized,
    // and mbarrier ops on raw shared memory died with "unspecified launch
    // failure".
    std::unordered_set<sguid_t> barriers_to_init(chained.begin(),
                                                 chained.end());
    for (auto const &[prod, cons] : chained_consumer) {
      barriers_to_init.insert(cons);
    }
    // Consumers of GENERIC-produced operands arrive on their barrier too (the
    // producer's anti-dependency wait targets it), so it must be initialized.
    for (auto const &[prod, cons] : generic_antidep) {
      barriers_to_init.insert(cons);
    }
    for (sguid_t const cg : barriers_to_init) {
      res.e("if (elect_one_warp && cute::elect_one_sync()) {");
      res.e("cute::initialize_barrier(*$_$, 1);", mbarrier_ptr_name, cg);
      res.e("}");
    }
    code << res;
  }

  // Running TMEM column allocation across the body's matmuls (see get_mma_tC).
  int tmem_col_offset = 0;
  // cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns
  int const TMEM_CAPACITY_COLUMNS = 512;

  // Initialize all max accumulators
  int num_init_max_accums = 0;
  for (TBSchedNode const &node : sched.loop_nodes) {
    if (node.type != tb_sched_node_t::OPERATOR) {
      continue;
    }
    auto [last_op, last_op_meta] = node.ops.back();
    if (last_op->op_type == type::TB_FORLOOP_ACCUM_MAX_OP &&
        !last_op_meta.is_accum_in_reg) {
      tb::TBForloopAccumOp const *accum_op =
          dynamic_cast<tb::TBForloopAccumOp const *>(last_op);
      tb::STensor const &accum = accum_op->output_tensors.at(0);
      STensorMeta const &accum_meta = stensor_metas.at(accum.guid);
      size_t num_elems = 0;
      for (int i = 0; i < accum.num_dims; ++i) {
        num_elems = std::max(num_elems, accum.dim[i] * accum_meta.strides[i]);
      }
      code.e("tb::InitMaxAccumulatorKernel<$, $, "
             "NUM_THREADS>::run(stensor$_ptr, thread_idx);",
             get_datatype_str(accum.data_type),
             num_elems,
             accum.guid);
      num_init_max_accums += 1;
    }
  }

  // Initialize all reduction max
  int num_init_reductions = 0;
  for (TBSchedNode const &node : sched.loop_nodes) {
    if (node.type != tb_sched_node_t::OPERATOR) {
      continue;
    }
    auto [last_op, last_op_meta] = node.ops.back();
    if (last_op->op_type >= type::TB_REDUCTION_0_MAX_OP &&
        last_op->op_type <= type::TB_REDUCTION_2_MAX_OP) {
      assert(node.ops.size() == 1); // Should not be fused
      tb::TBReductionOp const *updated_max_op =
          dynamic_cast<tb::TBReductionOp const *>(last_op);
      tb::STensor const &updated_max = updated_max_op->output_tensors.at(0);
      STensorMeta const &updated_max_meta = stensor_metas.at(updated_max.guid);
      size_t num_elems = 0;
      for (int i = 0; i < updated_max.num_dims; ++i) {
        num_elems = std::max(num_elems,
                             updated_max.dim[i] * updated_max_meta.strides[i]);
      }
      code.e("tb::InitReductionMaxKernel<$, $, "
             "NUM_THREADS>::run(stensor$_ptr, thread_idx);",
             get_datatype_str(updated_max.data_type),
             num_elems,
             updated_max.guid);
      num_init_reductions += 1;
    }
  }
  code.e("");

  // Pre-define all matmul ops and allocate accumulators (if needed)
  // Since we may want to place the accumulator of a matmul op in register
  // files, we may need to allocate the accumulator in advance, and that
  // requires us to define the kernel (`using Matmul$Kernel = ...`) in advance
  for (TBSchedNode const &node :
       Combine(sched.loop_nodes, sched.post_loop_nodes)) {
    if (node.type == tb_sched_node_t::OPERATOR &&
        node.ops.front().first->op_type == type::TB_MATMUL_OP) {
      tb::TBOperator const *op = node.ops.front().first;
      tb::TBOperator const *output_op = node.ops.back().first;
      tb::STensor const &input0 = op->input_tensors.at(0);
      tb::STensor const &input1 = op->input_tensors.at(1);
      tb::STensor const &output = output_op->output_tensors.at(0);
      STensorMeta meta0 = stensor_metas.at(input0.guid);
      STensorMeta meta1 = stensor_metas.at(input1.guid);
      STensorMeta meta2 = stensor_metas.at(output.guid);
      int num_dims = input0.num_dims;
      assert(input1.num_dims == num_dims && output.num_dims == num_dims);
      int m = output.dim[num_dims - 2];
      int n = output.dim[num_dims - 1];
      int k = input0.dim[num_dims - 1];
      assert(input0.dim[num_dims - 2] == m && input0.dim[num_dims - 1] == k);
      assert(input1.dim[num_dims - 2] == k && input1.dim[num_dims - 1] == n);

      if (GPU_CC::H100 == config.target_cc) {
        // Hopper wgmma
        assert(num_threads >= 128);
      } else if (config.target_cc == GPU_CC::B200) {
        // Blackwell umma
        assert(num_threads >= 32);
      } else {
        assert(0 && "Unsupported GPU Architecture");
      }

      bool is_ldmatrix_avail = config.target_cc >= GPU_CC::T4;
      bool is_stmatrix_avail = false;

      int num_exps_before_store = std::count_if(
          node.ops.begin(), node.ops.end(), [](auto &op_and_meta) {
            return op_and_meta.first->op_type == type::TB_EXP_OP;
          });
      // write_tC_to_sC applies only exp (NUM_EXPS_BEFORE_STORE) and the
      // accumulator store; any other op fused into a Blackwell matmul's
      // epilogue chain would silently vanish (a fused SQUARE did exactly
      // that). Reject instead of emitting wrong code.
      for (size_t fused_i = 1; fused_i < node.ops.size(); ++fused_i) {
        auto fused_type = node.ops[fused_i].first->op_type;
        if (fused_type != type::TB_EXP_OP &&
            fused_type != type::TB_FORLOOP_ACCUM_NO_RED_OP) {
          return CustomOPTranspileResult{
              CUDA_T_UNSUPPORTED_FUSED_EPILOGUE, func_name, 0, 0, "", {}};
        }
      }
      bool is_store_accum =
          node.ops.back().first->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP;
      bool is_accum_in_reg = node.ops.back().second.is_accum_in_reg;

      // For threadblock matmul, cute requires 2-d matrices as inputs / outputs,
      // we assert that all other leading dimensions are of size 1, and only use
      // the last two dimensions when generating layouts
      // swapAB: 1-SM tcgen05 requires an M-tile of 64 or 128, but MPK decode
      // issues M = 1..8 tokens. Computing C^T = B^T * A^T puts the token count
      // into N (which is fine-grained: any multiple of 8 up to 256) and the
      // weight's output dim into M.
      //
      // sm100_make_1sm_trivial_tiled_mma does NOT reject an illegal M -- at M=8
      // it happily builds SM100_MMA_F16BF16_SS<...,8,64,...> and the failure
      // only surfaces later inside get_mma_tC as a CUTLASS error cascade. So
      // validate the effective shape here, on our side, before emitting.
      bool const swap_ab = (m != 64 && m != 128);
      int const mma_m = swap_ab ? n : m;
      int const mma_n = swap_ab ? m : n;
      if (config.target_cc == GPU_CC::B200 &&
          ((mma_m != 64 && mma_m != 128) || mma_n % 8 != 0 || mma_n > 256 ||
           mma_n == 0)) {
        return CustomOPTranspileResult{
            CUDA_T_CONFIG_ERROR, func_name, 0, 0, "", {}};
      }

      if (config.target_cc == GPU_CC::B200) {

        // These are suffixed with the matmul's output (C-matrix) guid for two
        // reasons: an unsuffixed name is re-declared in the same scope by every
        // matmul op (breaking any graph with two or more matmuls, e.g. a gated
        // MLP), and the *host* side already emits the guid-suffixed forms for
        // TMA-atom construction (transpiler_kn.cc). TiledMMA::guid is set to
        // exactly this c_matrix_guid, so both sides now agree.
        // 1-SM MMA. The MPK runtime executes tasks as single CTAs and has no
        // multicast, so the 2-SM (2x1SM) atom is not usable: it pairs CTAs via
        // a cluster and requires the multicast TMA atoms. 1-SM also relaxes the
        // M-tile constraint from {128, 256} to {64, 128}, and gives
        // AtomThrID == 1 so the per-CTA shape divides are identities.
        // Under swapAB the roles invert: the A operand is the old B (its MN
        // dim, N, is contiguous -> Major::MN) and the B operand is the old A
        // (K contiguous -> Major::K), so the Major flags swap with the extents.
        e_mma("auto tiled_mma_$ = "
               "cutlass::gemm::collective::detail::sm100_make_1sm_trivial_"
               "tiled_mma<$, $, $, Shape<Int<$>, Int<$>>, "
               "decltype(cluster_shape), UMMA::Major::$, UMMA::Major::$>();",
               output.guid,
               get_datatype_str(swap_ab ? input1.data_type : input0.data_type),
               get_datatype_str(swap_ab ? input0.data_type : input1.data_type),
               // Third parameter is ElementAccumulator, not the output element
               // type. tcgen05 F16/BF16 MMA accumulates in fp32 (the result is
               // narrowed on the way out of TMEM), so passing the bf16 output
               // type here made CUTLASS reject it with "Unknown type for
               // CFormat".
               "float",
               mma_m,
               mma_n,
               swap_ab ? "MN" : "K",
               swap_ab ? "K" : "MN");

      // Operand tile-width limit, for NON-pipelined operands only (a pipelined
      // one is exempt -- see operand_ok below).
      //
      // The transpiler models smem as dense strides plus one XOR swizzle. That
      // matches what the UMMA reads as long as an operand's contiguous dim is at
      // most 128B (64 elements at 16-bit). At exactly 128B the planner yields
      // Swizzle<3,3,3>, the UMMA's 128B swizzle in element units; narrower dims
      // scale down consistently with the atom (K=32 -> Swizzle<2,3,3> against
      // the atom's Sw<2,4,3>, K=16 -> Swizzle<1,3,3>), and those are verified
      // correct.
      //
      // Wider operands diverge two ways. The planner switches branches
      // (num_chunks_in_inner_dim > num_chunks_in_128B) and emits Swizzle<3,3,4>,
      // and CUTLASS panel-tiles the layout atom rather than laying rows out
      // contiguously -- tile_to_shape at K=128 gives
      // ((_8,_16),(_64,_2)):((_64,_512),(_1,_8192)), so the second K block sits
      // at +8192 elements, not at a row pitch of 128. No dense-stride layout can
      // express that. Measured relative error on those shapes is ~1.6, i.e.
      // silently wrong output, so reject them here instead.
      {
        // Test exactly what the planner branches on: the swizzled dim's stride
        // in bytes, i.e. the row pitch. Above 128B it takes the
        // num_chunks_in_inner_dim > num_chunks_in_128B path and emits a swizzle
        // the UMMA does not read. Do NOT use dim[innermost_dim] here -- the
        // layout resolver does not always make the K/N dim innermost (at K=32 it
        // picks M), so that measures the wrong axis and rejects working shapes.
        auto row_pitch_bytes = [](tb::STensor const &t, STensorMeta const &mt) {
          return (size_t)mt.strides[mt.swizzled_dim] *
                 type::get_datatype_size(t.data_type);
        };
        // Operands only. The constraint comes from the UMMA reading A and B
        // through smem descriptors, so their layout must be one the hardware
        // agrees with. C is written by the TMEM->smem copy and read by the S->G
        // copy, both through SmemLayoutC, so it is self-consistent at any
        // swizzle -- and measured so: at K=32 only C exceeds 128B (pitch 256B)
        // and the result is correct (rel 2.3e-3). Including C here rejected that
        // working shape.
        //
        // Require the operand to be provably readable, not merely
        // not-provably-bad: XOR-swizzled AND pitch <= 128B. A pitch whose chunk
        // count is not a power of 2 (K=48 -> 6 chunks) sends the planner down
        // its shift-based branch, which leaves is_xor_swizzled false and
        // produces a layout the UMMA cannot read. Keying the check off
        // is_xor_swizzled alone skipped those tensors entirely and let K=48
        // through at rel 1.18 -- silently wrong.
        // The pitch limit applies only to the NON-pipelined path.
        //
        // A pipelined operand is never addressed through the dense-stride
        // model: InputTMAAsyncCopy_Blackwell writes
        // make_tensor(dst_smem, DstPipeLayout{}) and Blackwell_Matmul::run
        // reads make_tensor(a_ptr, DstPipeLayout_A{}), and both derive that
        // layout independently from the same cutlass sm100_smem_selector for
        // the same (major, element, tile). So CUTLASS panel-tiles both sides
        // identically and the >128B divergence cannot arise. The transpiler's
        // dense-stride layout is then used only for SIZING, which still agrees:
        // tile_to_mma_shape is compact, so its per-stage cosize equals the
        // element count the planner reserved and the pipeline's
        // transactionBytes expects.
        //
        // Measured: N=128 and Ktile=128 are correct at rel ~2.5e-3 on the
        // pipelined path, the same as N=64. The non-pipelined path keeps the
        // original guard -- InputChunkedSyncCopy really does index linearly
        // through the dense-stride layout, where a >128B tile is silently
        // wrong (~1.6 relative error).
        auto operand_ok = [&](tb::STensor const &t, STensorMeta const &mt) {
          if (mt.is_pipelined_input) {
            return true;
          }
          // Routed through InputWideOperandSyncCopy: written through the same
          // cutlass-derived DstPipeLayout the UMMA reads, so the dense-stride
          // pitch limit does not apply.
          if (wide_matmul_operands.count(t.guid)) {
            return true;
          }
          return mt.is_xor_swizzled && row_pitch_bytes(t, mt) <= 128;
        };
        bool unsupported =
            !operand_ok(input0, meta0) || !operand_ok(input1, meta1);
        if (unsupported) {
          if (getenv("MIRAGE_DEBUG_WIDE")) {
            fprintf(stderr,
                    "[wide] operand_ok REJECT in0=%lld(%d) in1=%lld(%d)\n",
                    (long long)input0.guid,
                    (int)operand_ok(input0, meta0),
                    (long long)input1.guid,
                    (int)operand_ok(input1, meta1));
          }
          return CustomOPTranspileResult{
              CUDA_T_LAYOUT_ERROR, func_name, 0, 0, "", {}};
        }
      }

      assert(k % 16 == 0);
        e_mma("auto mma_tiler_$ = make_shape(tile_size<0>(tiled_mma_$), "
               "tile_size<1>(tiled_mma_$), tile_size<2>(tiled_mma_$)*_${});",
               output.guid,
               output.guid,
               output.guid,
               output.guid,
               k / 16);

        e_mma("Layout cluster_layout_vmnk_$ = "
               "tiled_divide(make_layout(cluster_shape), make_tile(typename "
               "decltype(tiled_mma_$)::AtomThrID{}));",
               output.guid,
               output.guid);

        // NOTE: `elect_one_cta` (and the cta_in_cluster_coord_vmnk it was
        // derived from) is now declared once in the kernel prologue. Declaring
        // it here re-declared it in the same scope for every matmul op, which
        // broke any graph with two or more matmuls (e.g. a gated MLP).

        // Blackwell_Matmul takes A as [K,M], B as [N,K], C as [N,M].
        // Unswapped, each of those is the dim-swapped view of the stensor.
        // Under swapAB, A := old B needs [K,N], B := old A needs [M,K], and
        // C needs [M,N] -- all three are the *natural* stensor order, i.e.
        // swap01=false. The epilogue then writes accumulator element
        // (n_idx, m_idx) through the swapped C layout, which is exactly where
        // output element (m_idx, n_idx) lives, so the S->G copy needs no change.
        code.e("using Matmul$LayoutA = $;",
               output.guid,
               swap_ab ? get_stensor_layout(input1, meta1, num_dims - 2, false)
                       : get_stensor_layout(input0, meta0, num_dims - 2));
        code.e("using Matmul$LayoutB = $;",
               output.guid,
               swap_ab ? get_stensor_layout(input0, meta0, num_dims - 2, false)
                       : get_stensor_layout(input1, meta1, num_dims - 2));
        code.e("using Matmul$LayoutC = $;",
               output.guid,
               swap_ab ? get_stensor_layout(output, meta2, num_dims - 2, false)
                       : get_stensor_layout(output, meta2, num_dims - 2));

        // code.e("using Mma$Shape_A = decltype(partition_shape_A(tiled_mma,
        // make_shape(size<0>(mma_tiler), size<2>(mma_tiler))));",
        //         output.guid);
        // code.e("using Mma$Shape_B = decltype(partition_shape_B(tiled_mma,
        // make_shape(size<1>(mma_tiler), size<2>(mma_tiler))));",
        //         output.guid);
        // code.e("using Mma$Shape_C = decltype(partition_shape_C(tiled_mma,
        // make_shape(size<0>(mma_tiler), size<1>(mma_tiler))));",
        //         output.guid);

      } else {
        code.e("using Matmul$LayoutC = $;",
               output.guid,
               get_stensor_layout(output, meta2, num_dims - 2 /*start_dim*/));
      }
      code.e("using Matmul$Kernel = tb::Blackwell_Matmul<$, "
             "$, $, Matmul$LayoutA, Matmul$LayoutB, "
             "Matmul$LayoutC, NUM_THREADS, "
             "$, $, $, $, $, $, decltype(cluster_shape), "
             "decltype(tiled_mma_$), "
             "decltype(mma_tiler_$), $, $>;",
             output.guid,
             get_datatype_str(input0.data_type),
             is_ldmatrix_avail,
             is_stmatrix_avail,
             output.guid,
             output.guid,
             output.guid,
             num_exps_before_store,
             is_accum_in_reg ? false : is_store_accum,
             config.num_consumer_wgs > 1 ? true : false,
             // IS_PIPELINE_A/B follow the MMA ROLE, which swapAB flips: under
             // swap the A operand is input1. Passing meta0/meta1 positionally
             // gave pipelined V a 1-stage read layout whose stage coordinate
             // collapses to 0 -- the UMMA consumed stage 0 (KV tile 0) every
             // iteration while the producer dutifully filled both stages.
             (swap_ab ? meta1 : meta0).is_pipelined_input,
             (swap_ab ? meta0 : meta1).is_pipelined_input,
             config.pipeline_stages,
             output.guid,             // decltype(tiled_mma_$)
             output.guid,             // decltype(mma_tiler_$)
             swap_ab ? "true" : "false",
             // TASK_BODY: in a megakernel task blockIdx is the worker id, so
             // the MMA coordinate must not be derived from it.
             config.emit_device_body ? "true" : "false");
      // Give every matmul in this body its own TMEM columns. A 128-lane fp32
      // accumulator occupies mma_n columns, so hand out [offset, offset+mma_n)
      // and advance. All of them previously shared *tmem_base_ptr, which made
      // the second matmul clobber the first's accumulator.
      if (tmem_col_offset == 0) {
        code.e("auto matmul_$_accum = Matmul$Kernel::get_mma_tC(blockIdx.x, "
               "blockIdx.y, *tmem_base_ptr);",
               output.guid,
               output.guid);
      } else {
        code.e("auto matmul_$_accum = Matmul$Kernel::get_mma_tC(blockIdx.x, "
               "blockIdx.y, *tmem_base_ptr + $);",
               output.guid,
               output.guid,
               tmem_col_offset);
      }
      tmem_col_offset += mma_n;
      if (tmem_col_offset > TMEM_CAPACITY_COLUMNS) {
        // Allocator1Sm hands out Sm100TmemCapacityColumns columns in total.
        // Overflowing them silently aliases accumulators again, so reject.
        return CustomOPTranspileResult{
            CUDA_T_LAYOUT_ERROR, func_name, 0, 0, "", {}};
      }
      code.e("");
    }
  }
  code.e("__syncthreads();");

  // Get matmul stensor_guid2stensor
  std::map<sguid_t, tb::STensor> SGuid2STensor;
  // Operands of a matmul. A pipeline feeding one of these is driven by a single
  // elected warp (see use_cta_warp_selector), so its consumer count is 32 rather
  // than a full warp group -- getting that wrong deadlocks the producer.
  std::unordered_set<sguid_t> matmul_operand_guids;
  for (TBSchedNode const &node :
       Combine(Combine(sched.pre_loop_nodes, sched.loop_nodes),
               sched.post_loop_nodes)) {
    if (node.type == tb_sched_node_t::OPERATOR &&
        node.ops.front().first->op_type == type::TB_MATMUL_OP) {
      tb::TBOperator const *op = node.ops.front().first;
      tb::TBOperator const *output_op = node.ops.back().first;
      tb::STensor const &input0 = op->input_tensors.at(0);
      tb::STensor const &input1 = op->input_tensors.at(1);
      tb::STensor const &output = output_op->output_tensors.at(0);
      SGuid2STensor[input0.guid] = input0;
      SGuid2STensor[input1.guid] = input1;
      SGuid2STensor[output.guid] = output;
      matmul_operand_guids.insert(input0.guid);
      matmul_operand_guids.insert(input1.guid);
      // If an stensor feeds two matmuls, the first one wins. That is only
      // sound if both require the same layout; the shapes that reach here are
      // validated numerically, so a mismatch would show up as a wrong result
      // rather than silently.
    }
  }

  // Define G2SCopy for all input STensors
  code.e("// G->S copy atoms");
  std::unordered_set<tb::TBInputOp const *>
      pipelined_input_ops; // A list of input ops that are software pipelined
                           // (asynchronously G->S copied)

  std::map<int64_t, tb::TBInputOp const *> pipeline_inputs;

  // for release smem_read;
  std::vector<sguid_t> smem_read_output_guids;
  int pipe_index = 0;

  for (TBSchedNode const &node :
       Combine(Combine(sched.pre_loop_nodes, sched.loop_nodes),
               sched.post_loop_nodes)) {
    if (node.type == tb_sched_node_t::OPERATOR &&
        node.ops.front().first->op_type == type::TB_INPUT_OP) {
      auto [_op, op_meta] = node.ops.front();
      tb::TBInputOp const *cur_op = dynamic_cast<tb::TBInputOp const *>(_op);
      tb::TBOperator const *output_op = fusion_chain.at(cur_op).back();
      kn::DTensor const &dtensor = cur_op->dtensor;
      tb::STensor const &stensor = output_op->output_tensors.at(0);
      DTensorMeta const &dtensor_meta = dtensor_metas.at(dtensor.guid);
      STensorMeta const &stensor_meta = stensor_metas.at(stensor.guid);
      assert(dtensor.num_dims == stensor.num_dims);
      assert(dtensor.data_type == stensor.data_type);

      code.e("// Copy for G->S: dtensor $ -> stensor $",
             dtensor.guid,
             stensor.guid);

      // Get the starting address of my tile
      // For input tensor that does not have a forloop_dim, the shape of the
      // tile should be identical to the STensor. Otherwise, it should be the
      // shape of STensor * forloop_range
      string offset = "";
      int3 imap = cur_op->input_map;
      // In device-body (MPK task) mode the caller hands us pointers ALREADY
      // offset to this task's tile: runtime.cc computes each task's offset from
      // the map and its block id when building the TensorDesc
      // (`input$.base_ptr = base + offset`). Applying a blockIdx offset on top
      // double-counts it -- and in a persistent megakernel blockIdx.x is the
      // WORKER id (one block per SM), not a tile index, so it indexed far past
      // the tensor and faulted with "unspecified launch failure".
      for (int dim = 0; dim < 3 && !config.emit_device_body; ++dim) {
        int div_dim = dim == 0 ? imap.x : dim == 1 ? imap.y : imap.z;
        if (div_dim >= 0) {
          // Dim `div_dim` is divided along `dim`
          int num_tbs = dim == 0   ? g.grid_dim.x
                        : dim == 1 ? g.grid_dim.y
                                   : g.grid_dim.z;
          offset += fmt(" + blockIdx.$*$*$",
                        (char)"xyz"[dim],
                        dtensor.dim[div_dim] / num_tbs,
                        dtensor_meta.strides[div_dim]);
        }
      }

      bool use_chunked_copy = op_meta.is_chunked_input;
      int real_innermost_dim = op_meta.chunked_input_real_innermost_dim;
      bool use_async_copy = op_meta.is_pipelined_input;

      if (!(use_chunked_copy) || (!use_async_copy)) {
        code.e("const $ *dtensor$_tile_ptr = dtensor$_ptr $;",
               get_datatype_str(dtensor.data_type),
               dtensor.guid,
               dtensor.guid,
               offset);
      }

      // Wide (>128B pitch) non-pipelined matmul operand: copy through the
      // matmul's own cutlass-derived smem layout (see the routing set above).
      // Needs the consuming matmul's tiled_mma/mma_tiler and role. Applies on
      // both the chunked and non-chunked branches below -- operand_ok admits
      // these tensors solely on the promise that this atom loads them.
      auto emit_wide_operand_atom = [&]() {
        tb::TBOperator const *mm = nullptr;
        bool is_in0 = false;
        for (tb::TBOperator const *cons : g.operators) {
          if (cons->op_type == type::TB_MATMUL_OP) {
            if (cons->input_tensors.at(0).guid == stensor.guid) {
              mm = cons;
              is_in0 = true;
            } else if (cons->input_tensors.at(1).guid == stensor.guid) {
              mm = cons;
              is_in0 = false;
            }
          }
        }
        assert(mm != nullptr);
        // tiled_mma_$/mma_tiler_$ are named after the matmul's SCHED-NODE
        // output (the fused chain's last op -- e.g. a folded forloop_accum),
        // not the matmul's own output. Follow the fusion chain for the name.
        tb::STensor const &mm_out =
            fusion_chain.at(mm).back()->output_tensors.at(0);
        int const mm_nd = mm_out.num_dims;
        int const mm_m = mm_out.dim[mm_nd - 2];
        bool const mm_swap_ab = (mm_m != 64 && mm_m != 128);
        bool const role_a = (is_in0 != mm_swap_ab);
        int const nd = stensor.num_dims;
        // Natural role order: A wants (M, K), B wants (N, K). input0 is
        // already (m, k) / under swapAB (n_role, k); input1 is (k, n) and
        // must be permuted. Strides come from the gmem dtensor tile.
        int const da = stensor.dim[nd - 2], db = stensor.dim[nd - 1];
        size_t const sa = dtensor_meta.strides[nd - 2],
                     sb = dtensor_meta.strides[nd - 1];
        string src_layout =
            is_in0
                ? fmt("Layout<Shape<Int<$>, Int<$>>, Stride<Int<$>, Int<$>>>",
                      da,
                      db,
                      sa,
                      sb)
                : fmt("Layout<Shape<Int<$>, Int<$>>, Stride<Int<$>, Int<$>>>",
                      db,
                      da,
                      sb,
                      sa);
        code.e("using STensor$InputAtom = tb::InputWideOperandSyncCopy<$, "
               "$, decltype(tiled_mma_$), decltype(mma_tiler_$), $, $, "
               "NUM_THREADS>;",
               stensor.guid,
               get_datatype_str(stensor.data_type),
               src_layout,
               mm_out.guid,
               mm_out.guid,
               role_a ? "true" : "false",
               mm_swap_ab ? "true" : "false");
      };

      // assert(use_chunked_copy && use_async_copy);
      if (!use_chunked_copy) {
        int d_innermost_dim = dtensor_meta.innermost_dim;
        assert(!use_async_copy);
        if (wide_matmul_operands.count(stensor.guid)) {
          emit_wide_operand_atom();
        } else {
        string dtensor_tile_layout = get_dtensor_tile_layout(
            dtensor, dtensor_meta, stensor, stensor_meta, d_innermost_dim);
        code.e(
            "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);
        // Non-chunked, synchronous copy
        code.e(
            "using STensor$InputAtom = tb::InputNonChunkedSyncCopy<$, "
            "$, DTensor$TileLayout, NUM_THREADS>;",
            stensor.guid,
            get_datatype_str(stensor.data_type),
            mov_last_get_stensor_layout(stensor, stensor_meta, d_innermost_dim),
            dtensor.guid);
        }
      } else {
        string dtensor_tile_layout = get_dtensor_tile_layout(
            dtensor, dtensor_meta, stensor, stensor_meta, real_innermost_dim);
        code.e(
            "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);
        if (!use_async_copy && wide_matmul_operands.count(stensor.guid)) {
          emit_wide_operand_atom();
        } else if (!use_async_copy) {
          // Chunked, synchronous copy
          code.e("using STensor$InputAtom = tb::InputChunkedSyncCopy<$, "
                 "$, DTensor$TileLayout, NUM_THREADS>;",
                 stensor.guid,
                 get_datatype_str(stensor.data_type),
                 mov_last_get_stensor_layout(
                     stensor, stensor_meta, real_innermost_dim),
                 dtensor.guid);
        } else {
          pipelined_input_ops.insert(cur_op);
          assert(cur_op->output_tensors.size() == 1);
          // make tma

          // gmem tensor
          string gmem_layout = get_layout_detail::get_cute_layout(
              vector<int>(dtensor.dim, dtensor.dim + dtensor.num_dims),
              vector<size_t>(dtensor_meta.strides,
                             dtensor_meta.strides + dtensor.num_dims));

          // imap;
          int forloop_dim = cur_op->forloop_dim;
          // NOTE: this stays keyed off the stensor's own position, NOT the MMA
          // role, even under swapAB. The smem layout's majorness must match the
          // gmem tensor's -- TMA cannot transpose -- and that is a property of
          // how the tensor is laid out, not of which operand slot it feeds.
          // Deriving it from the MMA role instead tripped CUTLASS's
          // "Majorness of smem doesn't match majorness of gmem" assertion.
          // Only the descriptor derivation (partition_shape_A/B, Major, Step,
          // make_tma_atom_A/B) follows the role -- see transpiler_kn.cc.
          bool m_input = stensor_meta.m_input;
          string smem_layout = mov_last_get_stensor_layout(
              stensor, stensor_meta, real_innermost_dim, !m_input);

          auto [dims, strides] = get_layout_detail::get_cute_layout_array(
              vector<int>(dtensor.dim, dtensor.dim + dtensor.num_dims),
              vector<size_t>(dtensor_meta.strides,
                             dtensor_meta.strides + dtensor.num_dims),
              !m_input);

          std::vector<int> partition_logic = {
              imap.x >= 0 ? (dtensor.num_dims - 1 - imap.x) : -1,
              imap.y >= 0 ? (dtensor.num_dims - 1 - imap.y) : -1,
              imap.z >= 0 ? (dtensor.num_dims - 1 - imap.z) : -1};

          // string SrcMNKLayout = generate_partitioned_and_expanded_layout(
          //     dim3(g.grid_dim.x, g.grid_dim.y, g.grid_dim.z),
          //     dims,
          //     strides,
          //     partition_logic,
          //     g.forloop_range,
          //     m_input ? forloop_dim : (dtensor.num_dims - 1 - forloop_dim));

          string SrcMNKLayout = fmt("Layout<Shape<$>, Stride<$>>",
                                    map_to_cute_int(dims),
                                    map_to_cute_int(strides));

          // Consumer count must equal the threads that really call
          // consumer_wait/consumer_release for this pipeline. A matmul operand
          // is consumed under `if (elect_one_cta && elect_one_warp)` -- one warp
          // -- while an elementwise consumer runs on every consumer thread.
          // Passing the warp-group count for a matmul-fed pipeline left the
          // empty-arrive spread incomplete and hung the producer.
          int const pipeline_num_consumers =
              matmul_operand_guids.count(stensor.guid)
                  ? 32
                  : config::NUM_THREADS_PER_GROUP * config.num_consumer_wgs;
          code.e(
              "tb::BlackwellAsyncPipeline<$, decltype(cluster_shape)> "
              "blackwell_async_pipeline_$((void *) (buf + $), "
              "(tb::warpgroup_id() "
              "== $ && tb::warp_id() % mirage::config::NUM_WARPS_PER_GROUP == "
              "0), tb::warpgroup_id() < $, $, $, elect_one_cta);",
              config.pipeline_stages,
              stensor.guid,
              barrier_addr,
              config.num_consumer_wgs,
              config.num_consumer_wgs,
              // Exactly the bytes this CTA's TMA delivers. The old
              // `(m_input ? 2 : 1)` doubling was a 2-SM multicast artifact --
              // there the A tile arrives from a 2-CTA pair, so the mbarrier
              // expects two contributions. With 1-SM and no multicast only one
              // arrives, the transaction count is never satisfied, and
              // consumer_wait blocks forever: this was the K-loop hang.
              stensor_meta.num_phy_elems *
                  type::get_datatype_size(stensor.data_type),
              pipeline_num_consumers);

          // The MMA objects this input feeds belong to the matmul that consumes
          // it, identified by c_matrix_guid -- the same guid the host side uses
          // when it declares tiled_mma_$ / mma_tiler_$.
          int const atom_matmul_m =
              stensor_meta.m_input
                  ? stensor.dim[0]
                  : SGuid2STensor[stensor_meta.m_matrix_guid].dim[0];
          bool const atom_swap_ab =
              (atom_matmul_m != 64 && atom_matmul_m != 128);
          pipe_index++;
          barrier_addr += tma_barrier_size;

          pipeline_inputs[stensor.guid] = cur_op;

          tmaParamsList.push_back((TMAParams(
              dtensor_meta.input_idx,
              dtensor.guid,
              stensor.guid,
              SrcMNKLayout,
              smem_layout,
              stensor_meta.m_input,
              fmt("shape(${})", smem_layout),
              {1, 1, 1},
              dims,
              strides,
              partition_logic,
              g.forloop_range,
              m_input ? forloop_dim : (dtensor.num_dims - 1 - forloop_dim),
              // 1-SM MMA has no multicast: CUTLASS selects a plain
              // SM90_TMA_LOAD for AtomThrID == 1 (see
              // sm100_cluster_shape_to_tma_atom_A). The N_MODE/M_MODE
              // directions are 2-SM-only and are not emitted any more.
              "NOT_MULTICAST",
              stensor_meta.m_input
                  ? TiledMMA(get_datatype_str(stensor.data_type),
                             get_datatype_str(
                                 SGuid2STensor[stensor_meta.n_matrix_guid]
                                     .data_type),
                             // ElementAccumulator: tcgen05 accumulates in fp32
                             "float",
                             // 1-SM MMA: M tile is this CTA's tile, not the
                             // 2-CTA pair's, so no doubling.
                             stensor.dim[0],
                             SGuid2STensor[stensor_meta.n_matrix_guid].dim[1],
                             stensor.dim[1],
                             stensor_meta.c_matrix_guid,
                             stensor.dim[0] != 64 && stensor.dim[0] != 128)
                  : TiledMMA(get_datatype_str(stensor.data_type),
                             get_datatype_str(
                                 SGuid2STensor[stensor_meta.m_matrix_guid]
                                     .data_type),
                             // ElementAccumulator: tcgen05 accumulates in fp32
                             "float",
                             SGuid2STensor[stensor_meta.m_matrix_guid].dim[0],
                             stensor.dim[1],
                             stensor.dim[0],
                             stensor_meta.c_matrix_guid,
                             SGuid2STensor[stensor_meta.m_matrix_guid].dim[0] !=
                                     64 &&
                                 SGuid2STensor[stensor_meta.m_matrix_guid]
                                         .dim[0] != 128))));

          // Resolve this operand's position within the OP, which is what a task
          // indexes its pointers by (see TMAParams::operand_id).
          for (size_t k = 0; k < op->input_tensors.size(); k++) {
            if (op->input_tensors[k].guid == dtensor.guid) {
              tmaParamsList.back().operand_id = k;
              break;
            }
          }

          // A task body has no kernel parameters, so it cannot be templated on
          // the atom's type the way the standalone kernel is. Name the type
          // here instead -- everything it derives from (tiled_mma_$,
          // mma_tiler_$, cluster_layout_vmnk_$) is already declared above --
          // and bind a reference to the device-resident copy the host built.
          if (config.emit_device_body) {
            std::vector<TMAParams> just_pushed{tmaParamsList.back()};
            generate_tma_code_blackwell(
                code, just_pushed, op, config, /*types_only=*/true);
            code.e("TMA_$ const &tma_$ = *reinterpret_cast<TMA_$ const *>("
                   "tma_ptr_$);",
                   dtensor.guid, dtensor.guid, dtensor.guid, dtensor.guid);
          }

          code.e(
              "using STensor$InputAtom = tb::InputTMAAsyncCopy_Blackwell<$, $, "
              "$, decltype(tma_$), decltype(blackwell_async_pipeline_$), $, $, "
              "decltype(tiled_mma_$), decltype(mma_tiler_$), "
              "decltype(cluster_shape), $, $, $>;",
              stensor.guid,
              get_datatype_str(stensor.data_type),
              smem_layout,
              SrcMNKLayout,
              dtensor.guid,
              stensor.guid,
              // MInput here is the MMA ROLE, which swapAB inverts relative to
              // the stensor's source-graph position.
              stensor_meta.m_input != atom_swap_ab,
              g.forloop_range,
              stensor_meta.c_matrix_guid,  // decltype(tiled_mma_$)
              stensor_meta.c_matrix_guid,  // decltype(mma_tiler_$)
              atom_swap_ab ? "true" : "false",
              // TASK_BODY: the gmem tile coordinate must not come from
              // blockIdx in a task body -- see the flag's comment in input.h.
              config.emit_device_body ? "true" : "false",
              // N_LOOP: an original-B operand (K,N convention) whose forloop
              // dim is dim 1 loops over its N -- see input.h.
              (!stensor_meta.m_input && cur_op->forloop_dim == 1) ? "true"
                                                                  : "false");
        }
      }
    }
  }
  code.e("");
  code.e("__syncthreads();");

  if (GPU_CC::B200 == config.target_cc) {
    assert(g.cluster_dim.x > 0 && g.cluster_dim.y > 0 && g.cluster_dim.z > 0);
    string tma;
    string tmplt;
    for (size_t i = 0; i < tmaParamsList.size(); ++i) {
      if (i == 0) {
        tmplt.append("template <");
      }
      TMAParams &params = tmaParamsList.at(i);
      tmplt.append("class TMA_" + std::to_string(params.guid));
      // A standalone kernel takes the atom by value in the constant bank; a
      // task body takes a pointer to a device-resident copy the host builder
      // uploaded. Both are legal operand locations for cp.async.bulk.tensor,
      // and the pointer form is the only one a task can use -- it is called
      // from device code and has no kernel parameters of its own.
      tma.append(config.emit_device_body
                     ? ("void const *tma_ptr_" + std::to_string(params.guid))
                     : ("CUTE_GRID_CONSTANT TMA_" +
                        std::to_string(params.guid) + " const tma_" +
                        std::to_string(params.guid)));

      if (i != tmaParamsList.size() - 1) {
        tmplt.append(", ");
      } else {
        tmplt.append(">");
      }
      tma.append(", ");
    }

    // A megakernel task body: the caller owns the launch and the smem block, so
    // emit a __device__ function taking `buf` rather than a __global__ kernel
    // that declares `extern __shared__`. TMA atoms arrive as `void const *` to
    // host-built, device-resident copies -- a task has no kernel parameters and
    // cannot be templated by the megakernel's call site.
    if (config.emit_device_body) {
      code.e_front(
          "__device__ __forceinline__ void $($$, $) {",
          func_name,
          tma,
          map<kn::DTensor, string>(op->output_tensors,
                                   [](kn::DTensor const &dtensor) -> string {
                                     return fmt(
                                         "$* dtensor$_ptr",
                                         get_datatype_str(dtensor.data_type),
                                         dtensor.guid);
                                   }),
          map<kn::DTensor, string>(
              op->input_tensors, [](kn::DTensor const &dtensor) -> string {
                return fmt("$ const* dtensor$_ptr",
                           get_datatype_str(dtensor.data_type),
                           dtensor.guid);
              }));
    } else if (profiling) {
      code.e_front(
          "__global__ void  __launch_bounds__($) "
          "$($ $, $, uint64_t *profiler_buffer) {",
          num_threads,
          func_name,
          tma,
          map<kn::DTensor, string>(op->output_tensors,
                                   [](kn::DTensor const &dtensor) -> string {
                                     return fmt(
                                         "$* dtensor$_ptr",
                                         get_datatype_str(dtensor.data_type),
                                         dtensor.guid);
                                   }),
          map<kn::DTensor, string>(
              op->input_tensors, [](kn::DTensor const &dtensor) -> string {
                return fmt("$ const* dtensor$_ptr",
                           get_datatype_str(dtensor.data_type),
                           dtensor.guid);
              }));
    } else {
      code.e_front(
          "__global__ void  __launch_bounds__($) "
          "$($ $, $) {",
          num_threads,
          func_name,
          tma,
          map<kn::DTensor, string>(op->output_tensors,
                                   [](kn::DTensor const &dtensor) -> string {
                                     return fmt(
                                         "$* dtensor$_ptr",
                                         get_datatype_str(dtensor.data_type),
                                         dtensor.guid);
                                   }),
          map<kn::DTensor, string>(
              op->input_tensors, [](kn::DTensor const &dtensor) -> string {
                return fmt("$ const* dtensor$_ptr",
                           get_datatype_str(dtensor.data_type),
                           dtensor.guid);
              }));
    }

    // A task body is called from device code with concrete arguments, so it
    // must not be a template -- it derives each TMA_$ locally instead.
    if (!config.emit_device_body) {
      code.e_front(tmplt);
    }
    code.inc_indent();
    // code.inc_indent();
  }

  // add mem_size based on tma copies
  mem_plan.smem_size += tmaParamsList.size() * config.pipeline_stages * 16;

  // NOTE: the "erase the lowest 16 bytes" write that the Ampere backend emits
  // here has been moved up, before the TMEM allocation. On Blackwell the memory
  // plan places tmem_base_ptr at buf+0, so zeroing the low 16 bytes at this
  // point wiped the TMEM address that tmem_allocator.allocate() had just stored
  // -- the epilogue then called tmem_allocator.free() on address 0.
  code.e("");

  // Launch G->S copy atoms for all pre-loop-ops
  int num_pre_loop_copies = 0;
  for (TBSchedNode const &sched_node : sched.pre_loop_nodes) {
    // Currently only non-fused input ops are allowed to appear in
    // pre_loop_nodes check against this condition
    assert(sched_node.type == tb_sched_node_t::OPERATOR);
    assert(sched_node.ops.size() == 1); // Should not be fused
    tb::TBOperator const *op = sched_node.ops[0].first;
    assert(op->op_type == type::TB_INPUT_OP);
    tb::TBInputOp const *cur_op = dynamic_cast<tb::TBInputOp const *>(op);
    tb::STensor const &stensor = cur_op->output_tensors.at(0);
    assert(cur_op->forloop_dim == -1);
    assert(!pipelined_input_ops.count(
        cur_op)); // An input op in pre_loop_nodes should not be software
                  // pipelined since they do not have forloop_dim
    num_pre_loop_copies += 1;
    code.e("STensor$InputAtom::run(stensor$_ptr, "
           "dtensor$_tile_ptr, "
           "thread_idx);",
           stensor.guid,
           stensor.guid,
           cur_op->dtensor.guid);
  }
  code.e("");

  // Define S2GCopy for all output STensors
  code.e("// S->G copy atoms");
  for (TBSchedNode const &node :
       Combine(Combine(sched.pre_loop_nodes, sched.loop_nodes),
               sched.post_loop_nodes)) {
    if (node.type == tb_sched_node_t::OPERATOR &&
        node.ops.front().first->op_type == type::TB_OUTPUT_OP) {
      auto [_op, op_meta] = node.ops.front();
      tb::TBOutputOp const *cur_op = dynamic_cast<tb::TBOutputOp const *>(_op);
      tb::STensor const &stensor = cur_op->input_tensors.at(0);
      kn::DTensor const &dtensor = cur_op->dtensor;
      STensorMeta const &stensor_meta = stensor_metas.at(stensor.guid);
      DTensorMeta const &dtensor_meta = dtensor_metas.at(dtensor.guid);
      assert(dtensor.num_dims == stensor.num_dims);
      assert(dtensor.data_type == stensor.data_type);

      code.e("// Copy for S->G: stensor $ -> dtensor $",
             stensor.guid,
             dtensor.guid);

      // Get the starting address of my tile
      // For output tensor that does not have a forloop_dim, the shape of the
      // tile should be identical to the STensor. Otherwise, it should be the
      // shape of STensor * forloop_range
      string offset = "";
      int3 omap = cur_op->output_map;
      for (int dim = 0; dim < 3; ++dim) {
        int div_dim = dim == 0 ? omap.x : dim == 1 ? omap.y : omap.z;
        int num_tbs = dim == 0   ? g.grid_dim.x
                      : dim == 1 ? g.grid_dim.y
                                 : g.grid_dim.z;
        // See the input side above: an MPK task body receives pre-offset
        // pointers, so it must not add a blockIdx-derived offset of its own.
        if (num_tbs > 1 && !config.emit_device_body) {
          // The output tensor MUST be divided along this dimension, as stated
          // in the paper
          assert(div_dim >= 0);
          offset += fmt(" + blockIdx.$*$*$",
                        (char)"xyz"[dim],
                        dtensor.dim[div_dim] / num_tbs,
                        dtensor_meta.strides[div_dim]);
          // if directly write back to gmem and use 2sm mma
          // needs to have same addr for CTA in same pair
          // if (config.target_cc == GPU_CC::B200 && dim == 0) {
          //   offset += fmt("-(blockIdx.x%2)*$*$",
          //                 dtensor.dim[div_dim] / num_tbs,
          //                 dtensor_meta.strides[div_dim]);
          // }
        }
      }
      code.e("$ *dtensor$_tile_ptr = dtensor$_ptr $;",
             get_datatype_str(dtensor.data_type),
             dtensor.guid,
             dtensor.guid,
             offset);

      bool use_chunked_copy = op_meta.is_chunked_output;
      int real_innermost_dim = op_meta.chunked_output_real_innermost_dim;

      if (!use_chunked_copy) {
        int d_innermost_dim = dtensor_meta.innermost_dim;
        string dtensor_tile_layout = get_dtensor_tile_layout(
            dtensor, dtensor_meta, stensor, stensor_meta, d_innermost_dim);
        code.e(
            "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);
        code.e("using STensor$OutputAtom = tb::OutputNonChunkedSyncCopy<$, "
               "DTensor$TileLayout, $, NUM_THREADS>;",
               stensor.guid,
               get_datatype_str(dtensor.data_type),
               dtensor.guid,
               mov_last_get_stensor_layout(
                   stensor, stensor_meta, d_innermost_dim));
      } else {
        string dtensor_tile_layout = get_dtensor_tile_layout(
            dtensor, dtensor_meta, stensor, stensor_meta, real_innermost_dim);
        code.e(
            "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);
        code.e("using STensor$OutputAtom = tb::OutputChunkedSyncCopy<$, "
               "DTensor$TileLayout, $, NUM_THREADS>;",
               stensor.guid,
               get_datatype_str(dtensor.data_type),
               dtensor.guid,
               mov_last_get_stensor_layout(
                   stensor, stensor_meta, real_innermost_dim));
      }
    }
  }
  code.e("");

  // Clear all accumulators
  // get all pipeline stensors
  int num_clear_accums = 0;
  for (TBSchedNode const &node : sched.loop_nodes) {
    if (node.type != tb_sched_node_t::OPERATOR) {
      continue;
    }
    auto [last_op, last_op_meta] = node.ops.back();
    // RESCALE accums must be cleared too (Ampere does): they accumulate in
    // smem via EpilogueStoreAccum, and uncleared they start from whatever the
    // buffer held -- the online-softmax kernel returned nan from exactly this.
    if ((last_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP ||
         last_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP) &&
        !last_op_meta.is_accum_in_reg) {
      tb::TBForloopAccumOp const *accum_op =
          dynamic_cast<tb::TBForloopAccumOp const *>(last_op);
      tb::STensor const &accum = accum_op->output_tensors.at(0);
      STensorMeta const &accum_meta = stensor_metas.at(accum.guid);
      size_t num_elems = 0;
      for (int i = 0; i < accum.num_dims; ++i) {
        num_elems = std::max(num_elems, accum.dim[i] * accum_meta.strides[i]);
      }
      code.e("tb::ClearAccumlatorKernel<$, $, "
             "NUM_THREADS>::run(stensor$_ptr, thread_idx);",
             get_datatype_str(accum.data_type),
             num_elems,
             accum.guid);
      num_clear_accums += 1;
    }
  }
  code.e("");

  if (num_pre_loop_copies > 0 || num_clear_accums > 0) {
    code.e("__syncthreads();");
    code.e("");
  }

  bool pipe_tma = !pipeline_inputs.empty();
  // if there is asyc copy defined
  if (pipe_tma) {
    code.e("int warpgroup_id = tb::warpgroup_id();");
    if (profiling) {
      code.e("PROFILER_CLOSURE_PARAMS_DECL");
      code.e("PROFILER_INIT(profiler_buffer, warpgroup_id, $, (threadIdx.x % "
             "128 == 0));",
             config.num_consumer_wgs + config.num_producer_wgs);
    }

    // run producers
    code.e("if (warpgroup_id == $) {", config.num_consumer_wgs);
    // allocate tma register files
    uint32_t tma_reg = config.num_consumer_wgs == 1 ? 56 : 32;

    // code.e("tb::wg_decrease_regs<$>();", tma_reg);
    code.e("if (tb::warp_id_in_wg() == 0) {");

    code.e("for (uint32_t for_idx = 0; for_idx < $; for_idx++) {",
           g.forloop_range);
    for (auto const &[stensor_id, op] : pipeline_inputs) {
      if (profiling) {
        code.e("PROFILER_EVENT_START($, $);",
               (op->op_type - type::TB_UNKOWN),
               "static_cast<uint32_t>(for_idx)");
      }
      code.e(fmt("STensor$InputAtom::run(tma_$, stensor$_ptr, "
                 "tiled_mma_$, mma_tiler_$, for_idx, "
                 "blackwell_async_pipeline_$);",
                 stensor_id,
                 op->dtensor.guid,
                 stensor_id,
                 stensor_metas.at(stensor_id).c_matrix_guid,
                 stensor_metas.at(stensor_id).c_matrix_guid,
                 stensor_id));
      if (profiling) {
        code.e("PROFILER_EVENT_END($, $);",
               (op->op_type - type::TB_UNKOWN),
               "static_cast<uint32_t>(for_idx)");
      }
    }
    code.e("}");
    code.e("}");
    code.e("}");
  }

  // A lambda function that transpiles a chain of (fusable) operators to an
  // epilogue Will automatically ignore the first operator in the `chain`
  // argument
  auto transpile_fusion_epilogue =
      [&](std::vector<std::pair<tb::TBOperator const *, TBSchedOpMeta>> const
              &chain,
          string dtype) -> string {
    size_t chain_size = chain.size();
    if (chain_size == 1) {
      // Not fused with anything
      return fmt("tb::EpilogueStore<$>", dtype);
    }
    // Deal with the last operator
    string res = fmt("tb::EpilogueStore<$>", dtype);
    for (size_t i = chain_size - 1; i >= 1; --i) {
      tb::TBOperator const *cur_op = chain[i].first;
      if (cur_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
        // Can only occur as the last operator in the chain
        assert(i == chain_size - 1);
        res = fmt("tb::EpilogueStoreAccum<$>", dtype);
      } else if (cur_op->op_type == type::TB_EXP_OP) {
        res = fmt("tb::EpilogueExp<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_SILU_OP) {
        res = fmt("tb::EpilogueSILU<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_GELU_OP) {
        res = fmt("tb::EpilogueGELU<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_RELU_OP) {
        res = fmt("tb::EpilogueRELU<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_CLAMP_OP) {
        res = fmt("tb::EpilogueClamp<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_SQUARE_OP) {
        res = fmt("tb::EpilogueSquare<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_SQRT_OP) {
        res = fmt("tb::EpilogueSqrt<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_MUL_SCALAR_OP) {
        res = fmt("tb::EpilogueMulScalar<$, $>", dtype, res);
      } else {
        assert(0 && "Unknown operator type");
      }
    }
    return res;
  };

  // A lambda function that transpiles an TBSchedNode
  auto transpile_tb_sched_node = [&](TBSchedNode const &sched_node,
                                     CodeKeeper &code,
                                     std::map<int64_t, tb::TBInputOp const *>
                                         &pipeline_inputs,
                                     bool is_in_loop) {
    if (sched_node.type == tb_sched_node_t::SYNCTHREADS) {
      // Writer-side proxy fence at every dependency-level boundary: a level
      // may end with GENERIC-proxy smem writes (elementwise, reduction) whose
      // next-level reader is a tcgen05 MMA on the ASYNC proxy -- e.g. the
      // online-softmax exp(x-max) tile feeding E@V. The chained-matmul path
      // fences its own write_tC_to_sC stores, but generic producers had no
      // fence and the consuming MMA read stale data (numerator rel ~1.5 while
      // the generic-read denominator was correct). One fence per level
      // boundary is cheap and covers every such edge.
      code.e("cutlass::arch::fence_view_async_shared();");
      // The scheduler inserts one of these between dependency LEVELS. In-loop
      // ones were silently swallowed (an empty branch with a commented-out
      // sync), which was survivable while every in-loop consumer read only its
      // own thread's elements -- elementwise chains partition identically, and
      // matmuls order themselves through the MMA mbarrier. A REDUCTION reads
      // the whole row other threads wrote: square -> reduction -> ... returned
      // the reduction of stale zeros, and a decomposed RMSNorm came back at
      // rel ~861 (denominator collapsed to sqrt(eps)). All in-loop nodes
      // execute on every consumer thread, so the consumer-scope named barrier
      // is the right sync in both positions.
      code.e("tb::wg_sync<CONSUMER_NUM_THREADS>(8);");
      // tcgen05 ops must observe the thread sync's ordering before their
      // next smem operand reads -- the handwritten sm100 tasks pair the
      // writer-side fence.proxy.async with this fence after the sync. Its
      // absence tore MMA reads of STANDALONE-elementwise-written operands.
      code.e("tb::tcgen05_fence_after_thread_sync();");
    } else {
      auto [op, first_op_meta] = sched_node.ops.front();
      auto [output_op, output_op_meta] = sched_node.ops.back();
      assert(output_op == fusion_chain.at(op).back());
      std::string op_type_str;
      to_json(op_type_str, op->op_type);
      // Generic producer of a matmul operand: do not overwrite the tile while
      // the consumer's previous-iteration MMA may still be reading it (see
      // generic_antidep above). The consumer arrives on its per-matmul
      // barrier once per iteration; arrival i-1 has parity (i-1) & 1.
      if (is_in_loop && g.forloop_range > 1 &&
          op->op_type != type::TB_MATMUL_OP &&
          !output_op->output_tensors.empty() &&
          generic_antidep.count(output_op->output_tensors.at(0).guid)) {
        code.e("if (for_idx > 0) {");
        code.e("cute::wait_barrier(*$_$, (for_idx - 1) & 1);",
               mbarrier_ptr_name,
               generic_antidep.at(output_op->output_tensors.at(0).guid));
        code.e("}");
      }
      code.e("{");
      code.e("// OP type: $", op_type_str);

      // TODO(zy): support other cases such as 1sm mma
      // NOT a 2-SM switch despite its former name (use_2sm_mma). The tcgen05
      // MMA is issued by a single elected warp on the 1-SM path too, and this
      // is what wraps the matmul node in `if (elect_one_cta && elect_one_warp)`.
      // Setting it false while "removing 2-SM code" would silently drop that
      // selector and let all 128 consumer threads issue the MMA.
      bool mma_needs_single_warp = true;

      // The single-CTA / single-warp selector exists so that exactly one warp
      // of the leader CTA *issues* the tcgen05 MMA. It must only wrap ops that
      // are issued that way. Cooperative kernels -- forloop accumulate,
      // element-wise unary/binary, reductions -- are parameterized over
      // NUM_THREADS / CONSUMER_NUM_THREADS and require every one of those
      // threads to participate. Wrapping them in `elect_one_warp` left just 32
      // of the expected 128 threads running, so most of each tile was never
      // written and the kernel silently produced wrong results.
      bool node_has_matmul = false;
      for (auto const &node_entry : sched_node.ops) {
        if (node_entry.first->op_type == type::TB_MATMUL_OP) {
          node_has_matmul = true;
          break;
        }
      }
      bool use_cta_warp_selector =
          mma_needs_single_warp && is_in_loop && node_has_matmul;
      if (use_cta_warp_selector) {
        add_cta_warp_selector_if_need(code, is_in_loop, true, true);
      }
      auto [need_advance_pipeline, pipe_ids] =
          add_loop_node_consumer_wait_if_need(
              op, code, is_in_loop, pipeline_inputs);
      // define
      if (pipe_tma && profiling) {
        // 2000 - 2999
        code.e("PROFILER_EVENT_START($, $);",
               (op->op_type - type::TB_UNKOWN),
               is_in_loop ? "static_cast<uint32_t>(for_idx)"
                          : "static_cast<uint32_t>(0)");
      }

      switch (op->op_type) {
        case type::TB_OUTPUT_OP: {
          assert(sched_node.ops.size() == 1); // Should not be fused
          tb::TBOutputOp const *cur_op =
              dynamic_cast<tb::TBOutputOp const *>(op);
          // Currently in Mirage core, an output op must have forloop_dim = -1
          assert(!is_in_loop);
          assert(cur_op->forloop_dim == -1);
          if (cur_op->forloop_dim >= 0) {
            assert(0);
          } else {
            tb::STensor const &stensor = cur_op->input_tensors.at(0);
            kn::DTensor const &dtensor = cur_op->dtensor;
            code.e("STensor$OutputAtom::run(dtensor$_tile_ptr, stensor$_ptr, "
                   "thread_idx);",
                   stensor.guid,
                   dtensor.guid,
                   stensor.guid);
          }
          break;
        }
        case type::TB_MATMUL_OP: {
          tb::STensor const &input0 = op->input_tensors.at(0);
          tb::STensor const &input1 = op->input_tensors.at(1);
          tb::STensor const &output = output_op->output_tensors.at(0);
          sguid_t output_guid = output.guid;
          // Must match the swapAB decision made when Matmul$Kernel was defined:
          // under swapAB the kernel's A operand is this op's *second* input.
          int const mm_m = output.dim[output.num_dims - 2];
          bool const swap_ab = (mm_m != 64 && mm_m != 128);
          sguid_t const a_guid = swap_ab ? input1.guid : input0.guid;
          sguid_t const b_guid = swap_ab ? input0.guid : input1.guid;

          // always pipeline for MMA
          if (need_advance_pipeline) {
            smem_read_output_guids.push_back(output_guid);

            // Per-operand stage index: a pipelined operand advances with the
            // loop (read_idx_<guid> from its consumer_wait); a non-pipelined
            // one has a single stage and must stay at 0. Mixed matmuls (Q
            // constant, K pipelined -- the attention shape) referenced a
            // pipeline the non-pipelined operand never had.
            auto stage_of = [&](sguid_t guid) -> string {
              if (std::find(pipe_ids.begin(), pipe_ids.end(),
                            (int64_t)guid) != pipe_ids.end()) {
                return fmt("read_idx_$", guid);
              }
              // Pipelined, but waited by an EARLIER node in this iteration
              // (consumer_wait is issued once per pipeline; SwiGLU's two
              // matmuls share the x pipeline). Its read_idx is out of scope
              // here, but every pipeline advances once per iteration, so the
              // stage is for_idx modulo the stage count. Returning "0" here
              // instead made the second matmul read stage 0 forever -- every
              // SwiGLU test failed.
              if (stensor_metas.at(guid).is_pipelined_input) {
                return fmt("(for_idx % $)", config.pipeline_stages);
              }
              return string("0"); // genuinely non-pipelined: single stage
            };
            code.e("Matmul$Kernel::run(matmul_$_accum, stensor$_ptr, "
                   "stensor$_ptr, "
                   "$, tiled_mma_$, $, $);",
                   output_guid,
                   output_guid,
                   a_guid,
                   b_guid,
                   // k_iter drives the accumulator reset: run() zeroes at
                   // k_iter == 0 only. A K-LOOP matmul accumulates across
                   // iterations (pass for_idx); a CHAINED one computes a fresh
                   // tile every iteration and must reset every time (pass 0).
                   chained.count(output_guid) ? "0" : "for_idx",
                   output_guid,
                   stage_of(a_guid),
                   stage_of(b_guid));
            // One warp of each block arrives at the barrier. The 2x1SM
            // variant issues a .cta_group::2 arrive, which ptxas refuses to mix
            // with the .cta_group::1 tcgen05 MMA the 1-SM path emits; with no
            // cluster pairing there is also nothing to multicast to.
            // A CHAINED matmul (its result is read inside the loop) arrives on
            // its own barrier, then the consumer threads wait for the MMA and
            // materialise the accumulator into the smem tile the next matmul
            // reads. write_tC_to_sC also runs whatever elementwise op the
            // scheduler fused into this matmul's epilogue
            // (NUM_EXPS_BEFORE_STORE) -- for Q@K^T -> exp -> @V that exp lives
            // here and nowhere else.
            if (chained.count(output_guid)) {
              code.e("cutlass::arch::umma_arrive($_$);",
                     mbarrier_ptr_name,
                     output_guid);
              code.e("}"); // close the elect_one_cta && elect_one_warp block
              // Anti-dependency: the consumer's MMA of the PREVIOUS iteration
              // must have finished reading this tile before it is overwritten.
              // The consumer arrives on its own per-matmul barrier once per
              // iteration; before store i (i >= 1) wait for arrival i-1, whose
              // phase parity is (i-1) & 1.
              if (g.forloop_range > 1 && chained_consumer.count(output_guid)) {
                code.e("if (for_idx > 0) {");
                code.e("cute::wait_barrier(*$_$, (for_idx - 1) & 1);",
                       mbarrier_ptr_name,
                       chained_consumer.at(output_guid));
                code.e("}");
              }
              // This matmul arrives once per iteration, so the phase to wait
              // out alternates with for_idx.
              code.e("cute::wait_barrier(*$_$, for_idx & 1);",
                     mbarrier_ptr_name,
                     output_guid);
              code.e("Matmul$Kernel::write_tC_to_sC(stensor$_ptr, "
                     "matmul_$_accum, thread_idx);",
                     output_guid,
                     output_guid,
                     output_guid);
              // The store above is a GENERIC-proxy write; the consuming
              // tcgen05 MMA reads smem through the ASYNC proxy. Without
              // fence.proxy.async the MMA can read stale data -- intermittent,
              // and more likely the more iterations run (measured rel grew
              // 0.12 -> 0.16 from FL=2 to FL=8). Same fence the handwritten
              // sm100 tasks issue after their smem stores.
              code.e("cutlass::arch::fence_view_async_shared();");
              // Publish it to the warp that issues the consuming MMA.
              code.e("tb::wg_sync<CONSUMER_NUM_THREADS>(8);");
              // tcgen05 ops must observe the thread sync's ordering before their
              // next smem operand reads -- the handwritten sm100 tasks pair the
              // writer-side fence.proxy.async with this fence after the sync. Its
              // absence tore MMA reads of STANDALONE-elementwise-written operands.
              code.e("tb::tcgen05_fence_after_thread_sync();");
              code.e("if (elect_one_cta && elect_one_warp) {");
            } else {
              // ORDER MATTERS: tcgen05.commit binds only the UNCOMMITTED
              // MMA group. Arriving on the shared barrier first bound the MMA
              // there, and the second arrive committed an EMPTY group to the
              // per-matmul barrier -- which therefore fired IMMEDIATELY, and
              // every anti-dependency wait on it passed while the MMA was
              // still reading its operands. The generic-produced operand was
              // then overwritten mid-read: diffuse chunk-level tearing (~7%
              // mean error over 55% of elements, Q=0-invariant). The
              // per-matmul barrier must take the REAL commit; the shared
              // barrier's arrival is then the empty-group instant one, which
              // still counts arrivals correctly for the post-loop wait
              // because the write-back ALSO waits the per-matmul barrier when
              // it exists (see the write-back emission).
              if (g.forloop_range > 1 &&
                  (chained.count(a_guid) || chained.count(b_guid) ||
                   generic_antidep.count(a_guid) ||
                   generic_antidep.count(b_guid))) {
                code.e("cutlass::arch::umma_arrive($_$);",
                       mbarrier_ptr_name,
                       output_guid);
              }
              code.e("cutlass::arch::umma_arrive($);", mbarrier_ptr_name);
            }
          } else {
            // Non-pipelined operands (plain synchronous G->S copies). This used
            // to emit *nothing at all*: the MMA was skipped silently and the
            // epilogue then deadlocked on wait_barrier() for an arrival that
            // could never happen. Emit the non-pipelined Matmul::run overload
            // with read_stage 0 -- the single-tile, no-pipeline shape an MPK
            // task body has.
            smem_read_output_guids.push_back(output_guid);

            code.e("Matmul$Kernel::run(matmul_$_accum, stensor$_ptr, "
                   "stensor$_ptr, $, tiled_mma_$, 0);",
                   output_guid,
                   output_guid,
                   a_guid,
                   b_guid,
                   // See the pipelined form above: chained matmuls reset their
                   // accumulator every iteration.
                   chained.count(output_guid) ? "0" : "for_idx",
                   output_guid);
            // A CHAINED matmul (its result is read inside the loop) arrives on
            // its own barrier, then the consumer threads wait for the MMA and
            // materialise the accumulator into the smem tile the next matmul
            // reads. write_tC_to_sC also runs whatever elementwise op the
            // scheduler fused into this matmul's epilogue
            // (NUM_EXPS_BEFORE_STORE) -- for Q@K^T -> exp -> @V that exp lives
            // here and nowhere else.
            if (chained.count(output_guid)) {
              code.e("cutlass::arch::umma_arrive($_$);",
                     mbarrier_ptr_name,
                     output_guid);
              code.e("}"); // close the elect_one_cta && elect_one_warp block
              // Anti-dependency: the consumer's MMA of the PREVIOUS iteration
              // must have finished reading this tile before it is overwritten.
              // The consumer arrives on its own per-matmul barrier once per
              // iteration; before store i (i >= 1) wait for arrival i-1, whose
              // phase parity is (i-1) & 1.
              if (g.forloop_range > 1 && chained_consumer.count(output_guid)) {
                code.e("if (for_idx > 0) {");
                code.e("cute::wait_barrier(*$_$, (for_idx - 1) & 1);",
                       mbarrier_ptr_name,
                       chained_consumer.at(output_guid));
                code.e("}");
              }
              // This matmul arrives once per iteration, so the phase to wait
              // out alternates with for_idx.
              code.e("cute::wait_barrier(*$_$, for_idx & 1);",
                     mbarrier_ptr_name,
                     output_guid);
              code.e("Matmul$Kernel::write_tC_to_sC(stensor$_ptr, "
                     "matmul_$_accum, thread_idx);",
                     output_guid,
                     output_guid,
                     output_guid);
              // The store above is a GENERIC-proxy write; the consuming
              // tcgen05 MMA reads smem through the ASYNC proxy. Without
              // fence.proxy.async the MMA can read stale data -- intermittent,
              // and more likely the more iterations run (measured rel grew
              // 0.12 -> 0.16 from FL=2 to FL=8). Same fence the handwritten
              // sm100 tasks issue after their smem stores.
              code.e("cutlass::arch::fence_view_async_shared();");
              // Publish it to the warp that issues the consuming MMA.
              code.e("tb::wg_sync<CONSUMER_NUM_THREADS>(8);");
              // tcgen05 ops must observe the thread sync's ordering before their
              // next smem operand reads -- the handwritten sm100 tasks pair the
              // writer-side fence.proxy.async with this fence after the sync. Its
              // absence tore MMA reads of STANDALONE-elementwise-written operands.
              code.e("tb::tcgen05_fence_after_thread_sync();");
              code.e("if (elect_one_cta && elect_one_warp) {");
            } else {
              // ORDER MATTERS: tcgen05.commit binds only the UNCOMMITTED
              // MMA group. Arriving on the shared barrier first bound the MMA
              // there, and the second arrive committed an EMPTY group to the
              // per-matmul barrier -- which therefore fired IMMEDIATELY, and
              // every anti-dependency wait on it passed while the MMA was
              // still reading its operands. The generic-produced operand was
              // then overwritten mid-read: diffuse chunk-level tearing (~7%
              // mean error over 55% of elements, Q=0-invariant). The
              // per-matmul barrier must take the REAL commit; the shared
              // barrier's arrival is then the empty-group instant one, which
              // still counts arrivals correctly for the post-loop wait
              // because the write-back ALSO waits the per-matmul barrier when
              // it exists (see the write-back emission).
              if (g.forloop_range > 1 &&
                  (chained.count(a_guid) || chained.count(b_guid) ||
                   generic_antidep.count(a_guid) ||
                   generic_antidep.count(b_guid))) {
                code.e("cutlass::arch::umma_arrive($_$);",
                       mbarrier_ptr_name,
                       output_guid);
              }
              code.e("cutlass::arch::umma_arrive($);", mbarrier_ptr_name);
            }
          }

          break;
        }
        case type::TB_EXP_OP:
        case type::TB_SILU_OP:
        case type::TB_GELU_OP:
        case type::TB_RELU_OP:
        case type::TB_CLAMP_OP:
        case type::TB_SQUARE_OP:
        case type::TB_SQRT_OP:
        case type::TB_MUL_SCALAR_OP: {
          tb::TBElementUnaryOp const *cur_op =
              dynamic_cast<tb::TBElementUnaryOp const *>(op);
          tb::STensor const &input = cur_op->input_tensors.at(0);
          tb::STensor const &output = output_op->output_tensors.at(0);
          assert(input.num_dims == output.num_dims);
          int num_dims = input.num_dims;
          // Find the iteration dim
          int iter_dim = -1;

          // at least one dim exists that fullfill the requirement:
          // dim i in input&output tensor == meta.innermost_dim or
          // meta.swizzled_dim
          for (int i = 0; i < num_dims; ++i) {
            bool failed = false;
            for (tb::STensor const &stensor : {input, output}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              if (i != meta.innermost_dim && meta.swizzled_dim != i) {
                failed = true;
                break;
              }
            }
            if (!failed) {
              iter_dim = i;
              break;
            }
          }
          assert(iter_dim != -1);
          // Define layouts
          string in_layout = mov_last_get_stensor_layout(
              input, stensor_metas.at(input.guid), iter_dim);
          string final_out_layout = mov_last_get_stensor_layout(
              output, stensor_metas.at(output.guid), iter_dim);
          code.e("using InLayout = $;", in_layout);
          code.e("using OutLayout = $;", final_out_layout);
          // Get the epilogue
          string epilogue = transpile_fusion_epilogue(
              sched_node.ops, get_datatype_str(input.data_type));
          // Define and run the kernel
          code.e("using Kernel = tb::ElementUnaryKernel<$, "
                 "tb::ElementUnaryOpType::$, OutLayout, InLayout, "
                 "CONSUMER_NUM_THREADS, $>;",
                 get_datatype_str(input.data_type),
                 get_tb_op_str(cur_op->op_type),
                 epilogue);
          code.e(append_epilogue_scalars(sched_node.ops));
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, thread_idx, $, "
                 "scalars);",
                 output.guid,
                 input.guid,
                 cur_op->scalar);
          break;
        }
        case type::TB_ADD_OP:
        case type::TB_MUL_OP:
        case type::TB_SUB_OP:
        case type::TB_DIV_OP:
        case type::TB_POW_OP: {
          tb::STensor const &input0 = op->input_tensors.at(0);
          tb::STensor const &input1 = op->input_tensors.at(1);
          tb::STensor const &output = output_op->output_tensors.at(0);
          assert(input0.num_dims == input1.num_dims &&
                 input0.num_dims == output.num_dims);
          int num_dims = input0.num_dims;
          // Find the iteration dim
          int iter_dim = -1;
          for (int i = 0; i < num_dims; ++i) {
            bool failed = false;
            for (tb::STensor const &stensor : {input0, input1, output}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              // A size-1 (broadcast) dim imposes no iteration-order
              // constraint: the operand contributes one element regardless of
              // which dim iterates fastest. Requiring innermost/swizzled on it
              // made a (1,64) operand veto every candidate dim, iter_dim
              // stayed -1, the NDEBUG-elided assert below vanished, and
              // mov_last(-1) permuted all three layouts into deterministic
              // garbage -- only when a broadcast operand was present, which is
              // exactly the failure boundary the discriminator measured.
              if (stensor.dim[i] == 1) {
                continue;
              }
              if (i != meta.innermost_dim && meta.swizzled_dim != i) {
                failed = true;
                break;
              }
            }
            if (!failed) {
              iter_dim = i;
              break;
            }
          }
          if (iter_dim == -1) {
            throw std::runtime_error(
                "Blackwell elementwise binary: no common iteration dim across "
                "operands (layout conflict)");
          }
          // Define op type
          string op_type_str = op->op_type == type::TB_ADD_OP   ? "ADD"
                               : op->op_type == type::TB_MUL_OP ? "MUL"
                               : op->op_type == type::TB_SUB_OP ? "SUB"
                               : op->op_type == type::TB_DIV_OP ? "DIV"
                               : op->op_type == type::TB_POW_OP ? "POW"
                                                                : "";
          assert(op_type_str != "");
          // Define layouts
          string in0_layout = mov_last_get_stensor_layout(
              input0, stensor_metas.at(input0.guid), iter_dim);
          string in1_layout = mov_last_get_stensor_layout(
              input1, stensor_metas.at(input1.guid), iter_dim);
          string final_out_layout = mov_last_get_stensor_layout(
              output, stensor_metas.at(output.guid), iter_dim);
          code.e("using In0Layout = $;", in0_layout);
          code.e("using In1Layout = $;", in1_layout);
          code.e("using OutLayout = $;", final_out_layout);
          // Get the epilogue
          string epilogue = transpile_fusion_epilogue(
              sched_node.ops, get_datatype_str(input0.data_type));
          // Define and run the kernel
          code.e("using Kernel = tb::ElementBinaryKernel<$, "
                 "tb::ElementBinaryOpType::$, OutLayout, In0Layout, In1Layout, "
                 "CONSUMER_NUM_THREADS, $>;",
                 get_datatype_str(input0.data_type),
                 op_type_str,
                 epilogue);
          code.e(append_epilogue_scalars(sched_node.ops));
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, stensor$_ptr, "
                 "thread_idx, scalars);",
                 output.guid,
                 input0.guid,
                 input1.guid);
          break;
        }
        case type::TB_REDUCTION_0_OP:
        case type::TB_REDUCTION_1_OP:
        case type::TB_REDUCTION_2_OP:
        case type::TB_REDUCTION_0_TO_DIMX_OP:
        case type::TB_REDUCTION_1_TO_DIMX_OP:
        case type::TB_REDUCTION_2_TO_DIMX_OP: {
          tb::STensor const &input = op->input_tensors.at(0);
          tb::STensor const &output = output_op->output_tensors.at(0);
          STensorMeta input_meta = stensor_metas.at(input.guid);
          STensorMeta final_output_meta = stensor_metas.at(output.guid);
          assert(input.num_dims == output.num_dims);
          int num_dims = input.num_dims;
          int reduc_dim = op->op_type >= type::TB_REDUCTION_0_TO_DIMX_OP
                              ? op->op_type - type::TB_REDUCTION_0_TO_DIMX_OP
                              : op->op_type - type::TB_REDUCTION_0_OP;
          assert(0 <= reduc_dim && reduc_dim < num_dims);
          // Find the iteration dim
          int iter_dim = -1;
          for (int i = 0; i < num_dims; ++i) {
            if (i == reduc_dim) {
              continue;
            }
            bool failed = false;
            for (tb::STensor const &stensor : {input, output}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              if (i != meta.innermost_dim && meta.swizzled_dim != i) {
                failed = true;
                break;
              }
            }
            if (!failed) {
              iter_dim = i;
              break;
            }
          }
          assert(iter_dim != -1);
          assert(iter_dim != reduc_dim);
          // Define layouts
          string in_layout =
              mov_last_get_stensor_layout(input, input_meta, iter_dim);
          string final_out_layout =
              mov_last_get_stensor_layout(output, final_output_meta, iter_dim);
          int cute_reduc_dim = reduc_dim < iter_dim ? num_dims - 1 - reduc_dim
                                                    : num_dims - reduc_dim;
          code.e("using InLayout = $;", in_layout);
          code.e("using OutLayout = $;", final_out_layout);
          // Get the epilogue
          string epilogue = transpile_fusion_epilogue(
              sched_node.ops, get_datatype_str(input.data_type));
          // Define and run the kernel
          code.e("using Kernel = tb::ReductionKernel<$, "
                 "OutLayout, InLayout, $, CONSUMER_NUM_THREADS, $>;",
                 get_datatype_str(input.data_type),
                 cute_reduc_dim,
                 epilogue);
          code.e(append_epilogue_scalars(sched_node.ops));
          code.e(
              "Kernel::run(stensor$_ptr, stensor$_ptr, thread_idx, scalars);",
              output.guid,
              input.guid);
          break;
        }
        case type::TB_FORLOOP_ACCUM_NO_RED_OP: {
          assert(sched_node.ops.size() == 1); // Should not be fused
          assert(is_in_loop);
          tb::STensor const &input = op->input_tensors.at(0);
          tb::STensor const &accum = op->output_tensors.at(0);
          int num_dims = input.num_dims;
          // Find the iteration dim
          int iter_dim = -1;
          for (int i = 0; i < num_dims; ++i) {
            bool failed = false;
            for (tb::STensor const &stensor : {input, accum}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              if (i != meta.innermost_dim && meta.swizzled_dim != i) {
                failed = true;
                break;
              }
            }
            if (!failed) {
              iter_dim = i;
              break;
            }
          }
          assert(iter_dim != -1);
          // Define layouts
          string in_layout = mov_last_get_stensor_layout(
              input, stensor_metas.at(input.guid), iter_dim);
          string accum_layout = mov_last_get_stensor_layout(
              accum, stensor_metas.at(accum.guid), iter_dim);
          code.e("using Kernel = tb::ForloopAccumKernel<$, $, $, "
                 "NUM_THREADS>;",
                 get_datatype_str(input.data_type),
                 accum_layout,
                 in_layout);
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, thread_idx);",
                 accum.guid,
                 input.guid);
          break;
        }
        case type::TB_CONCAT_0_OP:
        case type::TB_CONCAT_1_OP:
        case type::TB_CONCAT_2_OP: {
          assert(0 && "Not implemented");
          break;
        }
        case type::TB_CONCAT_THEN_MATMUL_OP: {
          assert(0 && "Not implemented");
          break;
        }
        case type::TB_CUSTOMIZED_OP: {
          assert(0 && "Not implemented");
          break;
        }
        // ===== Online-softmax ops, ported from the Ampere backend
        // (transpiler_tb.cc); shared runtime kernels in reduction.h /
        // forloop_accum.h -- only the emission was missing here.
        case type::TB_REDUCTION_0_MAX_OP:
        case type::TB_REDUCTION_1_MAX_OP:
        case type::TB_REDUCTION_2_MAX_OP: {
          assert(sched_node.ops.size() == 1); // Should not be fused
          tb::STensor const &input = op->input_tensors.at(0);
          tb::STensor const &updated_max = output_op->output_tensors.at(0);
          tb::STensor const &diff = output_op->output_tensors.at(1);
          STensorMeta input_meta = stensor_metas.at(input.guid);
          STensorMeta updated_max_meta = stensor_metas.at(updated_max.guid);
          STensorMeta diff_meta = stensor_metas.at(diff.guid);
          assert(input.num_dims == updated_max.num_dims &&
                 input.num_dims == diff.num_dims);
          int num_dims = input.num_dims;
          int reduc_dim = op->op_type - type::TB_REDUCTION_0_MAX_OP;
          assert(0 <= reduc_dim && reduc_dim < num_dims);
          // Find the iteration dim
          int iter_dim = -1;
          for (int i = 0; i < num_dims; ++i) {
            if (i == reduc_dim) {
              continue;
            }
            bool failed = false;
            for (tb::STensor const &stensor : {input, updated_max, diff}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              if (i != meta.innermost_dim && meta.swizzled_dim != i) {
                failed = true;
                break;
              }
            }
            if (!failed) {
              iter_dim = i;
              break;
            }
          }
          assert(iter_dim != -1);
          assert(iter_dim != reduc_dim);
          // Define layouts
          string in_layout =
              mov_last_get_stensor_layout(input, input_meta, iter_dim);
          string updated_max_layout = mov_last_get_stensor_layout(
              updated_max, updated_max_meta, iter_dim);
          string diff_layout =
              mov_last_get_stensor_layout(diff, diff_meta, iter_dim);
          int cute_reduc_dim = reduc_dim < iter_dim ? num_dims - 1 - reduc_dim
                                                    : num_dims - reduc_dim;
          code.e("using InLayout = $;", in_layout);
          code.e("using UpdatedMaxLayout = $;", updated_max_layout);
          code.e("using DiffLayout = $;", diff_layout);
          // Should not have epilogue
          // Define and run the kernel
          code.e("using Kernel = tb::ReductionMaxKernel<$, "
                 "UpdatedMaxLayout, DiffLayout, InLayout, $, NUM_THREADS>;",
                 get_datatype_str(input.data_type),
                 cute_reduc_dim);
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, stensor$_ptr, "
                 "thread_idx);",
                 updated_max.guid,
                 diff.guid,
                 input.guid);
          break;
        }
        case type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP: {
          assert(sched_node.ops.size() == 1); // Should not be fused
          assert(is_in_loop);
          tb::STensor const &input = op->input_tensors.at(0);
          tb::STensor const &rescale = op->input_tensors.at(1);
          tb::STensor const &accum = op->output_tensors.at(0);
          int num_dims = input.num_dims;
          // Find the iteration dim
          int iter_dim = -1;
          for (int i = 0; i < num_dims; ++i) {
            bool failed = false;
            for (tb::STensor const &stensor : {input, rescale, accum}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              if (i != meta.innermost_dim && meta.swizzled_dim != i) {
                failed = true;
                break;
              }
            }
            if (!failed) {
              iter_dim = i;
              break;
            }
          }
          assert(iter_dim != -1);
          // Define layouts
          string in_layout = mov_last_get_stensor_layout(
              input, stensor_metas.at(input.guid), iter_dim);
          string rescale_layout = mov_last_get_stensor_layout(
              rescale, stensor_metas.at(rescale.guid), iter_dim);
          string accum_layout = mov_last_get_stensor_layout(
              accum, stensor_metas.at(accum.guid), iter_dim);
          code.e("using Kernel = tb::ForloopAccumRescaleKernel<$, $, $, $, "
                 "NUM_THREADS>;",
                 get_datatype_str(input.data_type),
                 accum_layout,
                 in_layout,
                 rescale_layout);
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, stensor$_ptr, "
                 "thread_idx);",
                 accum.guid,
                 input.guid,
                 rescale.guid);
          break;
        }
        case type::TB_FORLOOP_ACCUM_MAX_OP: {
          assert(sched_node.ops.size() == 1); // Should not be fused
          assert(is_in_loop);
          tb::STensor const &input = op->input_tensors.at(0);
          tb::STensor const &accum = op->output_tensors.at(0);
          int num_dims = input.num_dims;
          // Find the iteration dim
          int iter_dim = -1;
          for (int i = 0; i < num_dims; ++i) {
            bool failed = false;
            for (tb::STensor const &stensor : {input, accum}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              if (i != meta.innermost_dim && meta.swizzled_dim != i) {
                failed = true;
                break;
              }
            }
            if (!failed) {
              iter_dim = i;
              break;
            }
          }
          assert(iter_dim != -1);
          // Define layouts
          string in_layout = mov_last_get_stensor_layout(
              input, stensor_metas.at(input.guid), iter_dim);
          string accum_layout = mov_last_get_stensor_layout(
              accum, stensor_metas.at(accum.guid), iter_dim);
          code.e(
              "using Kernel = tb::ForloopAccumMaxKernel<$, $, $, NUM_THREADS>;",
              get_datatype_str(input.data_type),
              accum_layout,
              in_layout);
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, thread_idx);",
                 accum.guid,
                 input.guid);
          break;
        }
        case type::TB_INPUT_OP: {
          // A NON-pipelined fd=-1 input's copy is emitted pre-loop; its
          // scheduler node reaching this switch needs nothing further. A
          // NON-pipelined FORLOOP-TILED input (demoted in sched_tb_graph
          // because no matmul consumes it -- e.g. an attention mask feeding
          // an add) must be copied HERE, every iteration, with the gmem tile
          // advanced by for_idx. Visibility to the consuming op comes from
          // the generic inter-op fence+wg_sync the emitter places between op
          // blocks -- do NOT sync inside this case: the op body may be
          // wrapped in a warp/CTA selector, where a warpgroup-wide sync
          // deadlocks (test_pipelined_elementwise hung at grid=32).
          tb::TBInputOp const *in_op = dynamic_cast<tb::TBInputOp const *>(op);
          tb::STensor const &in_st = op->output_tensors.at(0);
          STensorMeta const &in_mt = stensor_metas.at(in_st.guid);
          if (is_in_loop && in_op->forloop_dim != -1 &&
              !in_mt.is_pipelined_input) {
            int const fd = in_op->forloop_dim;
            DTensorMeta const &in_dmt = dtensor_metas.at(in_op->dtensor.guid);
            size_t const tile_advance =
                (size_t)in_st.dim[fd] * in_dmt.strides[fd];
            code.e("STensor$InputAtom::run(stensor$_ptr, "
                   "dtensor$_tile_ptr + for_idx * $, thread_idx);",
                   in_st.guid,
                   in_st.guid,
                   in_op->dtensor.guid,
                   tile_advance);
          }
          break;
        }
        default: {
          // assert(fmt(...).c_str()) asserted a non-null POINTER -- always
          // true, so an unhandled op emitted NOTHING and its consumers read
          // garbage. TB_SUB_OP fell through here and the online-softmax
          // x - max simply vanished from the kernel.
          throw std::runtime_error(
              fmt("Blackwell TB emitter: unhandled op type $", op->op_type));
        }
      }
      if (use_cta_warp_selector) {
        code.e("}"); // end of cta_warp_selector
      }
      if (pipe_tma && profiling) {
        code.e("PROFILER_EVENT_END($, $);",
               (op->op_type - type::TB_UNKOWN),
               is_in_loop ? "static_cast<uint32_t>(for_idx)"
                          : "static_cast<uint32_t>(0)");
      }
      code.e("}");
    }

    return CUDA_T_SUCCESS;
  };

  // Declare the for loop
  if (pipe_tma) {
    code.e("else {");
    // allocate register files for wgmma
    uint32_t mma_reg = config.num_consumer_wgs == 1
                           ? 256
                           : (config.num_consumer_wgs == 2 ? 232 : 160);
    // code.e("tb::wg_increase_regs<$>();", mma_reg);
    code.e("// Consumer main loop");
  }

  std::map<int64_t, tb::TBInputOp const *> copy_of_inputs = pipeline_inputs;
  assert(g.forloop_range >= 1);

  code.e("for (uint32_t for_idx = 0; for_idx < $; for_idx++) {",
         g.forloop_range);

  // warpgroup_id
  for (TBSchedNode const &sched_node : sched.loop_nodes) {
    if (sched_node.type == tb_sched_node_t::OPERATOR &&
        sched_node.ops[0].first->op_type == type::TB_INPUT_OP &&
        pipelined_input_ops.count(
            dynamic_cast<tb::TBInputOp const *>(sched_node.ops[0].first))) {
      continue;
    }
    CodeKeeper res;

    TranspileErrorType err =
        transpile_tb_sched_node(sched_node, res, pipeline_inputs, true);

    code << res;
    if (err != CUDA_T_SUCCESS) {
      return CustomOPTranspileResult{err, func_name, 0, 0, "", {}};
    }
  }

  if (!copy_of_inputs.empty()) {
    code.e("if (elect_one_cta && elect_one_warp) {");
    for (auto const &[pipe_id, op] : copy_of_inputs) {
      code.e("blackwell_async_pipeline_$.consumer_release();", pipe_id);
    }
    code.e("}");
  }

  code.e("}"); // For loop

  // Write back in-register accumulators
  int num_in_reg_accums = 0;
  CodeKeeper in_reg_writeback;
  for (TBSchedNode const &node : sched.loop_nodes) {
    if (node.type != tb_sched_node_t::OPERATOR) {
      continue;
    }
    auto [last_op, last_op_meta] = node.ops.back();
    if (last_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP &&
        last_op_meta.is_accum_in_reg) {
      tb::TBForloopAccumOp const *accum_op =
          dynamic_cast<tb::TBForloopAccumOp const *>(last_op);
      tb::STensor const &accum = accum_op->output_tensors.at(0);
      in_reg_writeback.e("Matmul$Kernel::write_tC_to_sC(stensor$_ptr, "
                         "matmul_$_accum, thread_idx);",
                         accum.guid,
                         accum.guid,
                         accum.guid);

      num_in_reg_accums += 1;
    }
  }

  if (num_in_reg_accums > 0) {
    code.e("// Write back tensor memory accumulators");
    // sync all consumer threads across peer CTA to ensure 2sm mma is done
    code.e("cute::wait_barrier(*$, 0);", mbarrier_ptr_name);
    // Matmuls whose REAL commit went to their per-matmul barrier (see the
    // arrive-order note) only empty-group-arrive on the shared barrier, so
    // additionally wait their own barrier's final arrival.
    if (g.forloop_range > 1) {
      std::unordered_set<sguid_t> own_waited;
      for (auto const &m : {&chained_consumer, &generic_antidep}) {
        for (auto const &[prod, cons] : *m) {
          if (!chained.count(cons) && !own_waited.count(cons)) {
            own_waited.insert(cons);
            code.e("cute::wait_barrier(*$_$, $);",
                   mbarrier_ptr_name,
                   cons,
                   (g.forloop_range - 1) & 1);
          }
        }
      }
    }
    code << in_reg_writeback;
  }

  // Transpile the epilogue of the kernel
  if (!sched.post_loop_nodes.empty()) {
    code.e("// The epilogue (kernels outside the loop)");
    code.e("tb::wg_sync<CONSUMER_NUM_THREADS>(8);");
    // tcgen05 ops must observe the thread sync's ordering before their
    // next smem operand reads -- the handwritten sm100 tasks pair the
    // writer-side fence.proxy.async with this fence after the sync. Its
    // absence tore MMA reads of STANDALONE-elementwise-written operands.
    code.e("tb::tcgen05_fence_after_thread_sync();");
    for (TBSchedNode const &sched_node : sched.post_loop_nodes) {
      CodeKeeper res;
      TranspileErrorType err =
          transpile_tb_sched_node(sched_node, res, pipeline_inputs, false);
      code << res;
      if (err != CUDA_T_SUCCESS) {
        return CustomOPTranspileResult{err, func_name, 0, 0, "", {}};
      }
    }
  }
  if (pipe_tma) {
    code.e("}");
  }
  // Release tensor memory.
  //
  // Only emit this when TMEM was actually allocated: `tmem_allocator` is
  // declared by generate_Tmem_mbarrier_init_code (guarded by need_mbarrier) and
  // `tmem_base_ptr` only when the memory plan reserved a TMEM base. Emitting the
  // release unconditionally left graphs without a matmul referencing both
  // identifiers before they were declared.
  if (need_mbarrier && has_tmem_base_ptr) {
    code.e("__syncthreads();");
    code.e("if (elect_one_warp) { ");
    // release_allocation_lock() is tcgen05.relinquish_alloc_permit: it tells
    // the hardware this CTA will never allocate TMEM again. That is right for a
    // standalone kernel, whose CTA exits straight after -- but fatal for a
    // megakernel task body, which runs on a PERSISTENT worker CTA that must
    // allocate again for the next task. Relinquishing per task made the second
    // generated matmul on a given worker die with "unspecified launch failure";
    // it went unnoticed because every test had at least as many workers as
    // tasks. The dealloc below is a separate instruction (tcgen05.dealloc) and
    // is still issued every time, so the columns are returned either way.
    //
    // cute's Allocator1Sm also requires that repeated allocations come from the
    // SAME warp; elect_one_warp is warp 0 in every task, so that holds.
    if (!config.emit_device_body) {
      code.e("tmem_allocator.release_allocation_lock(); ");
    }
    code.e("tmem_allocator.free(*tmem_base_ptr, "
           "TmemAllocator::Sm100TmemCapacityColumns); ");
    code.e("}");
  }

  code.e("}"); // kernel

  // A task body's TMA atoms have to be built somewhere. The standalone path
  // builds them inline in the launch scaffolding, which a task does not have,
  // so emit a __host__ builder alongside the body. It takes the task's input
  // base pointers, reconstructs the same atoms, and uploads each to global
  // memory; the megakernel calls it once at task-registration time and stores
  // the pointers in the TaskDesc. Everything here is a host-side mirror of what
  // the body derives at compile time.
  if (config.emit_device_body && !tmaParamsList.empty()) {
    CodeKeeper host;
    host.e("");
    host.e("// Host-side TMA atom construction for $.", func_name);
    host.e("static void $_build_tma(void **tma_out, void *const *input_ptrs) {",
           func_name);
    host.e("auto cluster_shape = make_shape(Int<$>{}, Int<$>{}, Int<$>{});",
           g.cluster_dim.x, g.cluster_dim.y, g.cluster_dim.z);
    host << mma_setup;
    for (auto const &p : tmaParamsList) {
      // input_id indexes the graph's KN inputs, which for a task body are
      // exactly the op's operands in order -- the same correspondence
      // register_generated_task relies on for input_strides.
      host.e("$ *dtensor$ = ($*)input_ptrs[$];",
             get_datatype_str(op->input_tensors.at(p.operand_id).data_type),
             p.guid,
             get_datatype_str(op->input_tensors.at(p.operand_id).data_type),
             p.operand_id);
    }
    generate_tma_code_blackwell(host, tmaParamsList, op, config);
    for (size_t i = 0; i < tmaParamsList.size(); i++) {
      size_t const guid = tmaParamsList.at(i).guid;
      host.e("cudaMalloc(&tma_out[$], sizeof(tma_$));", i, guid);
      host.e("cudaMemcpy(tma_out[$], &tma_$, sizeof(tma_$), "
             "cudaMemcpyHostToDevice);",
             i, guid, guid);
    }
    host.e("}");
    code << host;
  }

  // mem_plan.smem_size += tmaParamsList.size() * config.pipeline_stages * 16;
  return CustomOPTranspileResult{CUDA_T_SUCCESS,
                                 func_name,
                                 mem_plan.smem_size,
                                 profiler_buf_size,
                                 code.to_string(),
                                 tmaParamsList};
}
} // namespace transpiler
} // namespace mirage