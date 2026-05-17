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

#pragma once

#include "mirage/config.h"
#include "mirage/cpu/cmem_tensor.h"
#include "mirage/layout.h"
#include "mirage/type.h"
#include "mirage/utils/json_utils.h"
#include <atomic>
#include <cstddef>
#include <functional>

namespace mirage {
namespace kernel {

class KNOperator;

struct alignas(16) DTensor {
  DTensor(void);
  inline bool operator==(DTensor const &b) const {
    if (data_type != b.data_type) {
      return false;
    }
    if (layout != b.layout) {
      return false;
    }
    if (num_dims != b.num_dims) {
      return false;
    }
    for (int i = 0; i < num_dims; i++) {
      if (dim[i] != b.dim[i]) {
        return false;
      }
      if (stride[i] != b.stride[i]) {
        return false;
      }
    }
    if (owner_op != b.owner_op) {
      return false;
    }
    if (owner_ts_idx != b.owner_ts_idx) {
      return false;
    }
    if (base_guid != b.base_guid) {
      return false;
    }
    if (view_offset != b.view_offset) {
      return false;
    }
    assert(data_offset == b.data_offset);
    return true;
  }
  inline bool operator!=(DTensor const &b) const {
    if (data_type != b.data_type) {
      return true;
    }
    if (layout != b.layout) {
      return true;
    }
    if (num_dims != b.num_dims) {
      return true;
    }
    for (int i = 0; i < num_dims; i++) {
      if (dim[i] != b.dim[i]) {
        return true;
      }
      if (stride[i] != b.stride[i]) {
        return true;
      }
    }
    if (owner_op != b.owner_op) {
      return true;
    }
    if (owner_ts_idx != b.owner_ts_idx) {
      return true;
    }
    if (base_guid != b.base_guid) {
      return true;
    }
    if (view_offset != b.view_offset) {
      return true;
    }
    assert(data_offset == b.data_offset);
    return false;
  }

  inline size_t num_elements() const {
    size_t num = 1;
    for (int i = 0; i < num_dims; i++) {
      num *= dim[i];
    }
    return num;
  }

  inline size_t data_size() const {
    using namespace mirage::type;
    size_t data_type_size = get_datatype_size(data_type);
    return num_elements() * data_type_size;
  }

  inline size_t fingerprint_size() const {
    using namespace mirage::type;
    size_t data_type_size = sizeof(FPType);
    return num_elements() * data_type_size;
  }

  // Virtual-tensor (view) helpers.
  //
  // A storage tensor has base_guid == 0 (the default), meaning it owns its
  // underlying memory. A virtual tensor (view) has base_guid set to the GUID
  // of its root storage tensor and view_offset set to its byte offset within
  // that storage. Multi-level views are flattened to root at construction.
  inline bool is_virtual() const {
    return base_guid != 0;
  }
  // Returns the GUID identifying the underlying storage for both storage
  // tensors and views. Dependency analysis keys on this.
  inline type::GuidType resolve_base_guid() const {
    return is_virtual() ? base_guid : guid;
  }
  // Byte size of this tensor's logical window. For views in Phase 1
  // (row-major contiguous), this equals data_size().
  inline size_t bytes_size() const {
    return data_size();
  }

  static DTensor const EMPTY_TENSOR;

  // hash related functions
  size_t get_owner_independent_hash() const;

  bool has_same_fingerprint(mirage::cpu::CTensor const &ref) const;
  mirage::cpu::CTensor copy_fingerprint_to_ctensor() const;

public:
  mirage::type::DataType data_type;
  mirage::layout::DmemLayout layout;
  int num_dims;
  int dim[mirage::config::MAX_TENSOR_DIMS];
  // Per-element strides along each dim, in tensor elements (NOT bytes). For
  // storage tensors created via new_input these come from the user's stride
  // tuple. For views they are computed at view construction time from the
  // parent's stride pattern (same-rank case: inherit; reshape case: row-major
  // from new dims). Codegen reads this directly for per-task offset
  // arithmetic, so a view works without walking back to its parent op.
  int64_t stride[mirage::config::MAX_TENSOR_DIMS];
  type::GuidType guid;
  // GUID of the underlying storage tensor for views; 0 if this tensor owns
  // its own memory. Used by dependency analysis to match producer/consumer
  // edges on the root buffer.
  type::GuidType base_guid;
  // Byte offset of this view within its root storage's allocation. 0 for
  // storage tensors. Set at view construction time so window-overlap analysis
  // works before the memory planner has assigned absolute data_offset.
  int64_t view_offset;
  //  DTensor fields
  KNOperator *owner_op;
  int owner_ts_idx;
  // pointer to data
  // void *data_ptr;
  // offset in device memory
  int64_t data_offset;
  // pointer to fingerprint
  // mirage::type::FPType *fp_ptr;
  // offset in device memory in bytes
  int64_t fp_offset;

  static std::atomic<int64_t> next_guid;
};

// Custom serialization. We can't use NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE
// because cache files serialized before stride / base_guid / view_offset
// existed don't carry those fields, and the strict variant of the macro
// throws on missing keys. Instead:
//   - to_json writes all fields including the new ones;
//   - from_json reads each field with `.value(...)` so missing keys fall
//     back to the default-constructed DTensor's value (all-zero), then
//     calls fixup_legacy_strides to recompute row-major strides when the
//     loaded payload lacked the stride field.
void to_json(nlohmann::json &j, DTensor const &t);
void from_json(nlohmann::json const &j, DTensor &t);

// Recompute row-major strides on a freshly-deserialized DTensor when it
// came from an old JSON cache that didn't carry the stride field (so all
// stride[i] are zero). For tensors that DID carry strides this is a no-op.
// Safe to call after every DTensor JSON parse.
void fixup_legacy_strides(DTensor &t);

} // namespace kernel
} // namespace mirage

namespace std {
template <>
struct hash<mirage::kernel::DTensor> {
  size_t operator()(mirage::kernel::DTensor const &) const;
};
}; // namespace std
