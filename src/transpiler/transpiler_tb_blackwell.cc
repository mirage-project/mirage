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
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/reduction.h"
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
}

static void emit_async_proxy_publish(mirage::transpiler::CodeKeeper &code) {
  code.e("cutlass::arch::fence_view_async_shared();");
  code.e("tb::wg_sync<CONSUMER_NUM_THREADS>(8);");
  code.e("tb::tcgen05_fence_after_thread_sync();");
}

static std::unordered_set<sguid_t> build_wide_matmul_operand_set(
    tb::Graph const &g,
    std::unordered_map<sguid_t, STensorMeta> const &stensor_metas) {
  std::unordered_set<sguid_t> wide;
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
    bool narrow_ok = mt.is_xor_swizzled && mt.swizzled_dim >= 0 &&
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
      wide.insert(st.guid);
    }
  }
  return wide;
}

struct ChainedMatmulInfo {
  std::unordered_set<sguid_t> chained;
  std::unordered_map<sguid_t, sguid_t> chained_consumer;
  std::unordered_map<sguid_t, sguid_t> generic_antidep;
};

static ChainedMatmulInfo analyze_chained_matmuls(TBSched const &sched) {
  ChainedMatmulInfo info;
  for (TBSchedNode const &prod : sched.loop_nodes) {
    if (prod.type != tb_sched_node_t::OPERATOR ||
        prod.ops.front().first->op_type != type::TB_MATMUL_OP) {
      continue;
    }
    sguid_t const produced = prod.ops.back().first->output_tensors.at(0).guid;
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
            info.chained.insert(produced);
            if (cop->op_type == type::TB_MATMUL_OP) {
              info.chained_consumer[produced] =
                  cons.ops.back().first->output_tensors.at(0).guid;
            }
          }
        }
      }
    }
  }
  for (TBSchedNode const &prod : sched.loop_nodes) {
    if (prod.type != tb_sched_node_t::OPERATOR ||
        prod.ops.front().first->op_type == type::TB_MATMUL_OP) {
      continue; // matmul producers use the chained path
    }
    sguid_t const produced = prod.ops.back().first->output_tensors.at(0).guid;
    for (TBSchedNode const &cons : sched.loop_nodes) {
      if (cons.type != tb_sched_node_t::OPERATOR ||
          cons.ops.front().first->op_type != type::TB_MATMUL_OP) {
        continue;
      }
      for (auto const &in : cons.ops.front().first->input_tensors) {
        if (in.guid == produced) {
          info.generic_antidep[produced] =
              cons.ops.back().first->output_tensors.at(0).guid;
        }
      }
    }
  }
  return info;
}

static bool matmul_swaps_ab(int m) {
  return m != 64 && m != 128;
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

  int tma_barrier_size = 16 * config.pipeline_stages;

  // Get the schedule
  TBSched sched = get_threadblock_schedule(g);

  get_threadblock_swizzle_plan_blackwell(g, sched);

  // Get the memory allocation plan
  TBMemoryPlan mem_plan = get_threadblock_memory_plan(g, sched, true, true);

  std::unordered_set<sguid_t> wide_matmul_operands =
      build_wide_matmul_operand_set(g, stensor_metas);

  std::vector<TMAParams> tmaParamsList;

  // Generate code prologue
  CodeKeeper code;
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

  int const mma_atom_thr_size = 1;
  code.e("bool elect_one_cta = (cta_rank % $) == 0;", mma_atom_thr_size);

  code.e("// STensors");
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
      code.e("uint64_t *$ = (uint64_t*)(buf + $);", mbarrier_ptr_name, addr);
      need_mbarrier = true;
    } else if (guid > mem_plan.mbarrier_buf_guid_offset) {
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

  code.e("*((uint128_t*)buf) = 0ul;");
  code.e("");

  if (g.forloop_range == 1) {
    for (auto const &[guid, meta] : stensor_metas) {
      if (meta.is_pipelined_input) {
        return CustomOPTranspileResult{
            CUDA_T_FL1_PIPELINED_DEADLOCK, func_name, 0, 0, "", {}};
      }
    }
  }

  ChainedMatmulInfo const chain_info = analyze_chained_matmuls(sched);
  auto const &chained = chain_info.chained;
  auto const &chained_consumer = chain_info.chained_consumer;
  auto const &generic_antidep = chain_info.generic_antidep;

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
    generate_Tmem_mbarrier_init_code(
        res,
        num_matmuls > 0 ? g.forloop_range * num_matmuls * g.cluster_dim.x *
                              g.cluster_dim.y * g.cluster_dim.z
                        : -1 /* skip barrier init */,
        tmem_base_ptr_name,
        mbarrier_ptr_name);
    std::unordered_set<sguid_t> barriers_to_init(chained.begin(),
                                                 chained.end());
    for (auto const &[prod, cons] : chained_consumer) {
      barriers_to_init.insert(cons);
    }
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
    }
  }

  // Initialize all reduction max
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
      bool const swap_ab = matmul_swaps_ab(m);
      int const mma_m = swap_ab ? n : m;
      int const mma_n_exact = swap_ab ? m : n;
      bool const pad_n = config.pad_mma_n && swap_ab &&
                         config.target_cc == GPU_CC::B200 &&
                         meta0.innermost_dim == input0.num_dims - 1 &&
                         meta2.innermost_dim == output.num_dims - 1;
      int const mma_n = padded_mma_n(mma_n_exact, pad_n);
      if (config.target_cc == GPU_CC::B200 &&
          ((mma_m != 64 && mma_m != 128) || mma_n % 8 != 0 || mma_n > 256 ||
           mma_n == 0)) {
        return CustomOPTranspileResult{
            CUDA_T_CONFIG_ERROR, func_name, 0, 0, "", {}};
      }

      if (config.target_cc == GPU_CC::B200) {
        int const mma_k_atom =
            32 / static_cast<int>(type::get_datatype_size(
                     swap_ab ? input1.data_type : input0.data_type));
        if (mma_k_atom == 0 || k % mma_k_atom != 0) {
          return CustomOPTranspileResult{
              CUDA_T_CONFIG_ERROR, func_name, 0, 0, "", {}};
        }
      }

      if (config.target_cc == GPU_CC::B200) {

        e_mma("auto tiled_mma_$ = "
              "cutlass::gemm::collective::detail::sm100_make_1sm_trivial_"
              "tiled_mma<$, $, $, Shape<Int<$>, Int<$>>, "
              "decltype(cluster_shape), UMMA::Major::$, UMMA::Major::$>();",
              output.guid,
              get_datatype_str(swap_ab ? input1.data_type : input0.data_type),
              get_datatype_str(swap_ab ? input0.data_type : input1.data_type),
              "float",
              mma_m,
              mma_n,
              swap_ab ? "MN" : "K",
              swap_ab ? "K" : "MN");

        {
          auto row_pitch_bytes = [](tb::STensor const &t,
                                    STensorMeta const &mt) {
            return (size_t)mt.strides[mt.swizzled_dim] *
                   type::get_datatype_size(t.data_type);
          };
          auto operand_ok = [&](tb::STensor const &t, STensorMeta const &mt) {
            if (mt.is_pipelined_input) {
              return true;
            }
            if (wide_matmul_operands.count(t.guid)) {
              return true;
            }
            return mt.is_xor_swizzled && row_pitch_bytes(t, mt) <= 128;
          };
          bool unsupported =
              !operand_ok(input0, meta0) || !operand_ok(input1, meta1);
          if (unsupported) {
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
             (swap_ab ? meta1 : meta0).is_pipelined_input,
             (swap_ab ? meta0 : meta1).is_pipelined_input,
             config.pipeline_stages,
             output.guid, // decltype(tiled_mma_$)
             output.guid, // decltype(mma_tiler_$)
             swap_ab ? "true" : "false",
             config.emit_device_body ? "true" : "false");
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
        return CustomOPTranspileResult{
            CUDA_T_LAYOUT_ERROR, func_name, 0, 0, "", {}};
      }
      code.e("");
    }
  }
  code.e("__syncthreads();");

  // Get matmul stensor_guid2stensor
  std::map<sguid_t, tb::STensor> SGuid2STensor;
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
    }
  }

  // Define G2SCopy for all input STensors
  code.e("// G->S copy atoms");
  std::unordered_set<tb::TBInputOp const *>
      pipelined_input_ops; // A list of input ops that are software pipelined
                           // (asynchronously G->S copied)

  std::map<int64_t, tb::TBInputOp const *> pipeline_inputs;

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
        tb::STensor const &mm_out =
            fusion_chain.at(mm).back()->output_tensors.at(0);
        int const mm_nd = mm_out.num_dims;
        int const mm_m = mm_out.dim[mm_nd - 2];
        bool const mm_swap_ab = matmul_swaps_ab(mm_m);
        bool const role_a = (is_in0 != mm_swap_ab);
        int const nd = stensor.num_dims;
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

      if (!use_chunked_copy) {
        int d_innermost_dim = dtensor_meta.innermost_dim;
        assert(!use_async_copy);
        if (wide_matmul_operands.count(stensor.guid)) {
          emit_wide_operand_atom();
        } else {
          string dtensor_tile_layout = get_dtensor_tile_layout(
              dtensor, dtensor_meta, stensor, stensor_meta, d_innermost_dim);
          code.e("using DTensor$TileLayout = $;",
                 dtensor.guid,
                 dtensor_tile_layout);
          // Non-chunked, synchronous copy
          code.e("using STensor$InputAtom = tb::InputNonChunkedSyncCopy<$, "
                 "$, DTensor$TileLayout, NUM_THREADS>;",
                 stensor.guid,
                 get_datatype_str(stensor.data_type),
                 mov_last_get_stensor_layout(
                     stensor, stensor_meta, d_innermost_dim),
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
          (void)gmem_layout;

          // imap;
          int forloop_dim = cur_op->forloop_dim;
          bool m_input = stensor_meta.m_input;
          string smem_layout = mov_last_get_stensor_layout(
              stensor, stensor_meta, real_innermost_dim, !m_input);

          int lead = 0;
          while (lead < dtensor.num_dims - 2 && stensor.dim[lead] == 1) {
            lead++;
          }
          auto [dims, strides] = get_layout_detail::get_cute_layout_array(
              vector<int>(dtensor.dim + lead, dtensor.dim + dtensor.num_dims),
              vector<size_t>(dtensor_meta.strides + lead,
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
              stensor_meta.num_phy_elems *
                  type::get_datatype_size(stensor.data_type),
              pipeline_num_consumers);

          int const atom_matmul_m =
              stensor_meta.m_input
                  ? stensor.dim[0]
                  : SGuid2STensor[stensor_meta.m_matrix_guid].dim[0];
          bool const atom_swap_ab = matmul_swaps_ab(atom_matmul_m);
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
              "NOT_MULTICAST",
              stensor_meta.m_input
                  ? TiledMMA(get_datatype_str(stensor.data_type),
                             get_datatype_str(
                                 SGuid2STensor[stensor_meta.n_matrix_guid]
                                     .data_type),
                             // ElementAccumulator: tcgen05 accumulates in fp32
                             "float",
                             stensor.dim[0],
                             SGuid2STensor[stensor_meta.n_matrix_guid].dim[1],
                             stensor.dim[1],
                             stensor_meta.c_matrix_guid,
                             matmul_swaps_ab(stensor.dim[0]))
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
                             matmul_swaps_ab(
                                 SGuid2STensor[stensor_meta.m_matrix_guid]
                                     .dim[0])))));

          for (size_t k = 0; k < op->input_tensors.size(); k++) {
            if (op->input_tensors[k].guid == dtensor.guid) {
              tmaParamsList.back().operand_id = k;
              break;
            }
          }

          if (config.emit_device_body) {
            std::vector<TMAParams> just_pushed{tmaParamsList.back()};
            generate_tma_code_blackwell(
                code, just_pushed, op, config, /*types_only=*/true);
            code.e("TMA_$ const &tma_$ = *reinterpret_cast<TMA_$ const *>("
                   "tma_ptr_$);",
                   dtensor.guid,
                   dtensor.guid,
                   dtensor.guid,
                   dtensor.guid);
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
              stensor_meta.m_input != atom_swap_ab,
              g.forloop_range,
              stensor_meta.c_matrix_guid, // decltype(tiled_mma_$)
              stensor_meta.c_matrix_guid, // decltype(mma_tiler_$)
              atom_swap_ab ? "true" : "false",
              config.emit_device_body ? "true" : "false",
              (!stensor_meta.m_input &&
               cur_op->forloop_dim == dtensor.num_dims - 1)
                  ? "true"
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

    if (!config.emit_device_body) {
      code.e_front(tmplt);
    }
    code.inc_indent();
    // code.inc_indent();
  }

  // add mem_size based on tma copies
  mem_plan.smem_size += tmaParamsList.size() * config.pipeline_stages * 16;

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
      emit_async_proxy_publish(code);
    } else {
      auto [op, first_op_meta] = sched_node.ops.front();
      auto [output_op, output_op_meta] = sched_node.ops.back();
      assert(output_op == fusion_chain.at(op).back());
      std::string op_type_str;
      to_json(op_type_str, op->op_type);
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
      bool mma_needs_single_warp = true;

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
          int const mm_m = output.dim[output.num_dims - 2];
          bool const swap_ab = matmul_swaps_ab(mm_m);
          sguid_t const a_guid = swap_ab ? input1.guid : input0.guid;
          sguid_t const b_guid = swap_ab ? input0.guid : input1.guid;

          // always pipeline for MMA
          if (need_advance_pipeline) {

            auto stage_of = [&](sguid_t guid) -> string {
              if (std::find(pipe_ids.begin(), pipe_ids.end(), (int64_t)guid) !=
                  pipe_ids.end()) {
                return fmt("read_idx_$", guid);
              }
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
                   chained.count(output_guid) ? "0" : "for_idx",
                   output_guid,
                   stage_of(a_guid),
                   stage_of(b_guid));
            if (chained.count(output_guid)) {
              code.e("cutlass::arch::umma_arrive($_$);",
                     mbarrier_ptr_name,
                     output_guid);
              code.e("}"); // close the elect_one_cta && elect_one_warp block
              if (g.forloop_range > 1 && chained_consumer.count(output_guid)) {
                code.e("if (for_idx > 0) {");
                code.e("cute::wait_barrier(*$_$, (for_idx - 1) & 1);",
                       mbarrier_ptr_name,
                       chained_consumer.at(output_guid));
                code.e("}");
              }
              code.e("cute::wait_barrier(*$_$, for_idx & 1);",
                     mbarrier_ptr_name,
                     output_guid);
              code.e("Matmul$Kernel::write_tC_to_sC(stensor$_ptr, "
                     "matmul_$_accum, thread_idx);",
                     output_guid,
                     output_guid,
                     output_guid);
              emit_async_proxy_publish(code);
              code.e("if (elect_one_cta && elect_one_warp) {");
            } else {
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

            code.e("Matmul$Kernel::run(matmul_$_accum, stensor$_ptr, "
                   "stensor$_ptr, $, tiled_mma_$, 0);",
                   output_guid,
                   output_guid,
                   a_guid,
                   b_guid,
                   chained.count(output_guid) ? "0" : "for_idx",
                   output_guid);
            if (chained.count(output_guid)) {
              code.e("cutlass::arch::umma_arrive($_$);",
                     mbarrier_ptr_name,
                     output_guid);
              code.e("}"); // close the elect_one_cta && elect_one_warp block
              if (g.forloop_range > 1 && chained_consumer.count(output_guid)) {
                code.e("if (for_idx > 0) {");
                code.e("cute::wait_barrier(*$_$, (for_idx - 1) & 1);",
                       mbarrier_ptr_name,
                       chained_consumer.at(output_guid));
                code.e("}");
              }
              code.e("cute::wait_barrier(*$_$, for_idx & 1);",
                     mbarrier_ptr_name,
                     output_guid);
              code.e("Matmul$Kernel::write_tC_to_sC(stensor$_ptr, "
                     "matmul_$_accum, thread_idx);",
                     output_guid,
                     output_guid,
                     output_guid);
              emit_async_proxy_publish(code);
              code.e("if (elect_one_cta && elect_one_warp) {");
            } else {
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
  if (need_mbarrier && has_tmem_base_ptr) {
    code.e("__syncthreads();");
    code.e("if (elect_one_warp) { ");
    if (!config.emit_device_body) {
      code.e("tmem_allocator.release_allocation_lock(); ");
    }
    code.e("tmem_allocator.free(*tmem_base_ptr, "
           "TmemAllocator::Sm100TmemCapacityColumns); ");
    code.e("}");
  }

  code.e("}"); // kernel

  if (config.emit_device_body && !tmaParamsList.empty()) {
    CodeKeeper host;
    host.e("");
    host.e("// Host-side TMA atom construction for $.", func_name);
    host.e("static void $_build_tma(void **tma_out, void *const *input_ptrs) {",
           func_name);
    host.e("auto cluster_shape = make_shape(Int<$>{}, Int<$>{}, Int<$>{});",
           g.cluster_dim.x,
           g.cluster_dim.y,
           g.cluster_dim.z);
    host << mma_setup;
    for (auto const &p : tmaParamsList) {
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
             i,
             guid,
             guid);
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