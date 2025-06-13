#include "mirage/transpiler/transpiler.h"
#include <unordered_map>
#include <unordered_set>

namespace mirage {
namespace transpiler {

// Get the swizzle plan for a threadblock
// change the layout to a tile of following layouts

static constexpr int MIN_LAYOUT_SiZE = 8;

// Layout_K_SW128_Atom
// Layout_K_SW64_Atom
// Layout_K_SW32_Atom
// Layout_K_INTER_Atom
// K == 64, m == 8
// Sw<3,4,3> o smem_ptr[16b](unset) o (_8,_64):(_64,_1)
// Sw<2,4,3> o smem_ptr[16b](unset) o (_8,_32):(_32,_1)
// Sw<1,4,3> o smem_ptr[16b](unset) o (_8,_16):(_16,_1)
// Sw<0,4,3> o smem_ptr[16b](unset) o (_8,_8):(_8,_1)

// MN major
// Layout_MN_SW128_Atom
// Layout_MN_SW64_Atom
// Layout_MN_SW32_Atom
// Layout_MN_INTER_Atom
// Sw<3,4,3> o smem_ptr[16b](unset) o (_64,_8):(_1,_64)
// Sw<2,4,3> o smem_ptr[16b](unset) o (_32,_8):(_1,_32)
// Sw<1,4,3> o smem_ptr[16b](unset) o (_16,_8):(_1,_16)
// Sw<0,4,3> o smem_ptr[16b](unset) o (_8,_8):(_1,_8)

// Layout atom to tile shape:
// product along the first dimension, then the second
void get_layout_atom_swizzle(tb::STensor const &stensor, STensorMeta &meta) {

  assert(meta.strides[meta.innermost_dim] == 1);
  if (stensor.dim[meta.innermost_dim] % 64 == 0) {
    meta.xor_swizzle_b = 3;
  } else if (stensor.dim[meta.innermost_dim] % 32 == 0) {
    meta.xor_swizzle_b = 2;
  } else if (stensor.dim[meta.innermost_dim] % 16 == 0) {
    meta.xor_swizzle_b = 1;
  } else if (stensor.dim[meta.innermost_dim] % 8 == 0) {
    meta.xor_swizzle_b = 0;
  }

  meta.xor_swizzle_m = 4;
  meta.xor_swizzle_s = 3;
}

void Transpiler::get_threadblock_swizzle_plan_hopper(tb::Graph const &tb_graph,
                                                     TBSched const &sched) {
  // Get a list of all STensors that is not fused
  std::vector<tb::STensor> all_stensors;
  {
    std::unordered_set<sguid_t> seen_guids;
    for (TBSchedNode const &node :
         Combine(Combine(sched.pre_loop_nodes, sched.loop_nodes),
                 sched.post_loop_nodes)) {
      if (node.type != tb_sched_node_t::OPERATOR) {
        continue;
      }
      for (tb::STensor const &stensor :
           Combine(node.ops.front().first->input_tensors,
                   node.ops.back().first->output_tensors)) {
        if (!seen_guids.count(stensor.guid)) {
          seen_guids.insert(stensor.guid);
          all_stensors.push_back(stensor);
        }
      }
    }
  }
  // Resolve `innermost_min_chunk_size` for all STensors
  std::unordered_map<sguid_t, int> innermost_min_chunk_size;
  {
    for (tb::STensor const &stensor : all_stensors) {
      innermost_min_chunk_size[stensor.guid] = 1;
    }
    auto update_innermost_min_chunk_size = [&](sguid_t sguid, int requirement) {
      int &chunk_size = innermost_min_chunk_size[sguid];
      chunk_size = std::max(chunk_size, requirement);
    };
    for (TBSchedNode const &node :
         Combine(Combine(sched.pre_loop_nodes, sched.loop_nodes),
                 sched.post_loop_nodes)) {
      if (node.type != tb_sched_node_t::OPERATOR) {
        continue;
      }
      auto const &[op, op_meta] = node.ops.front();
      auto const &[last_op, last_op_meta] = node.ops.back();
      if (op->op_type == type::TB_INPUT_OP) {
        if (op_meta.is_chunked_input) {
          // For chunked input ops, every num_16B_elems (8 when T is half)
          // elements must be contiguous
          tb::STensor const &output_stensor = last_op->output_tensors.at(0);
          int num_16B_elems = get_num_elems_in_16B(output_stensor.data_type);
          update_innermost_min_chunk_size(output_stensor.guid, num_16B_elems);
        }
      } else if (op->op_type == type::TB_OUTPUT_OP) {
        if (op_meta.is_chunked_output) {
          // For chunked output ops, every num_16B_elems (8 when T is half)
          // elements must be contiguous
          tb::STensor const &input_stensor = op->input_tensors.at(0);
          int num_16B_elems = get_num_elems_in_16B(input_stensor.data_type);
          update_innermost_min_chunk_size(input_stensor.guid, num_16B_elems);
        }
      } else if (op->op_type == type::TB_MATMUL_OP) {
        // For matmul ops when type is half, every 8 elements must be contiguous
        // (Need to rethink this when dtype is not half)
        tb::STensor const &input_stensor0 = op->input_tensors.at(0);
        tb::STensor const &input_stensor1 = op->input_tensors.at(1);
        tb::STensor const &output_stensor = last_op->output_tensors.at(0);
        update_innermost_min_chunk_size(input_stensor0.guid, 8);
        update_innermost_min_chunk_size(input_stensor1.guid, 8);
        update_innermost_min_chunk_size(output_stensor.guid, 8);
      }
    }
  }
  // Decide the way to swizzle each STensor
  for (tb::STensor const &stensor : all_stensors) {
    STensorMeta &meta = stensor_metas.at(stensor.guid);
    int swizzled_dim = meta.swizzled_dim;
    if (swizzled_dim == -1) {
      // No swizzling
      continue;
    }
    if (stensor.dim[swizzled_dim] == 1 ||
        stensor.dim[meta.innermost_dim] == 1) {
      // No need to swizzle
      continue;
    }

    auto is_power_of_2 = [](int x) { return (x & (x - 1)) == 0; };
    auto log2 = [](int x) {
      int result = 0;
      while (x >>= 1) {
        ++result;
      }
      return result;
    };

    size_t dtype_size = type::get_datatype_size(stensor.data_type);
    int num_dims = stensor.num_dims;

    int chunk_size_num_elems = innermost_min_chunk_size.at(stensor.guid);
    if (!is_power_of_2(chunk_size_num_elems)) {
      // Round `chunk_size_num_elems` to the next power of 2
      chunk_size_num_elems = 1 << (log2(chunk_size_num_elems) + 1);
    }
    // The chunk should be at least one bank
    while (chunk_size_num_elems * dtype_size < 4) {
      chunk_size_num_elems *= 2;
    }
    int chunk_size_bytes = chunk_size_num_elems * dtype_size;
    assert(chunk_size_bytes <=
           16); // The chunk size should be less than 16B (8 halfs)
                // The following condition should always hold since:
    // - We pad the real innermost dim to multiple of 16B
    // - chunk_size_bytes cannot be larger than 16B
    assert(meta.strides[swizzled_dim] % chunk_size_num_elems == 0);
    int num_chunks_in_inner_dim =
        meta.strides[swizzled_dim] / chunk_size_num_elems;
    int num_chunks_in_128B =
        128 / chunk_size_bytes; // 128B is the size of all banks

    meta.is_xor_swizzled = true;
    get_layout_atom_swizzle(stensor, meta);
  }
}

} // namespace transpiler
} // namespace mirage
