/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "mirage/kernel/graph.h"
#include "mirage/type.h"
#include <cassert>
#include <numeric>
#include <stdexcept>

namespace mirage {
namespace kernel {

namespace {

// Compute the row-major contiguous byte stride at dimension `dim`, i.e. the
// number of bytes you advance to move by one element along that axis.
// Equals product(dim[i] for i > dim) * dtype_size.
inline int64_t row_major_byte_stride(DTensor const &t, int dim) {
  size_t dtype_size = mirage::type::get_datatype_size(t.data_type);
  int64_t stride = static_cast<int64_t>(dtype_size);
  for (int i = t.num_dims - 1; i > dim; i--) {
    stride *= t.dim[i];
  }
  return stride;
}

// A slice inherits its parent's strides; a contiguous reshape recomputes
// strides even when the rank stays the same, e.g. (1, 2048) -> (16, 128).
//
// Writes are made into v.stride[0..v.num_dims). Out-of-rank slots are
// zero-initialized by the DTensor constructor and left alone here.
void set_view_strides(DTensor &v, DTensor const &parent, bool reshape) {
  if (!reshape) {
    for (int i = 0; i < v.num_dims; i++) {
      v.stride[i] = parent.stride[i];
    }
  } else {
    for (int i = v.num_dims - 1; i >= 0; i--) {
      if (i == v.num_dims - 1) {
        v.stride[i] = 1;
      } else {
        v.stride[i] = v.stride[i + 1] * v.dim[i + 1];
      }
    }
  }
}

// Construct a virtual DTensor sharing memory with `parent`. Multi-level views
// flatten to the root: if parent is itself virtual, the new view inherits
// parent.base_guid and accumulates view_offset.
DTensor make_view(DTensor const &parent,
                  std::vector<int> const &new_dims,
                  int64_t additional_byte_offset,
                  bool reshape = false) {
  DTensor v;
  v.data_type = parent.data_type;
  v.layout = parent.layout;
  v.num_dims = static_cast<int>(new_dims.size());
  for (int i = 0; i < v.num_dims; i++) {
    v.dim[i] = new_dims[i];
  }
  set_view_strides(v, parent, reshape);
  // Flatten multi-level views to the root storage. Non-virtual parents
  // (base_guid==0) become this view's base_guid; virtual parents pass their
  // base_guid through.
  v.base_guid = parent.is_virtual() ? parent.base_guid : parent.guid;
  v.view_offset = parent.view_offset + additional_byte_offset;
  v.guid = DTensor::next_guid++;
  // Views are not the output of any KNOperator.
  v.owner_op = nullptr;
  v.owner_ts_idx = -1;
  // data_offset (absolute) will be resolved by codegen as base.data_offset +
  // view_offset.
  v.data_offset = -1;
  v.fp_offset = -1;
  return v;
}

} // namespace

DTensor Graph::view(DTensor const &input, std::vector<int> const &new_shape) {
  if (new_shape.empty() ||
      (int)new_shape.size() > mirage::config::MAX_TENSOR_DIMS) {
    throw std::runtime_error(
        "Graph::view: invalid number of dimensions in new_shape");
  }
  size_t old_elems = input.num_elements();
  size_t new_elems = 1;
  for (int d : new_shape) {
    if (d <= 0) {
      throw std::runtime_error(
          "Graph::view: every dimension in new_shape must be positive");
    }
    new_elems *= static_cast<size_t>(d);
  }
  if (old_elems != new_elems) {
    throw std::runtime_error(
        "Graph::view: total element count must match the input");
  }
  bool same_shape = static_cast<int>(new_shape.size()) == input.num_dims;
  for (int i = 0; same_shape && i < input.num_dims; ++i)
    same_shape = new_shape[i] == input.dim[i];
  // An unchanged shape can retain a strided window. Other reshapes currently
  // require contiguous row-major storage; silently flattening holes is unsafe.
  if (!same_shape) {
    int64_t expected_stride = 1;
    for (int i = input.num_dims - 1; i >= 0; --i) {
      if (input.dim[i] != 1 && input.stride[i] != expected_stride)
        throw std::runtime_error("Graph::view: reshape requires contiguous input");
      expected_stride *= input.dim[i];
    }
  }
  return make_view(input, new_shape, /*additional_byte_offset=*/0,
                   /*reshape=*/!same_shape);
}

DTensor *Graph::view(DTensor const *input, std::vector<int> const &new_shape) {
  DTensor v = view(*input, new_shape);
  view_storage.emplace_back(std::make_unique<DTensor>(v));
  return view_storage.back().get();
}

DTensor Graph::narrow(DTensor const &input, int dim, int start, int length) {
  if (dim < 0 || dim >= input.num_dims) {
    throw std::runtime_error("Graph::narrow: dim out of range");
  }
  if (start < 0 || length <= 0 || start + length > input.dim[dim]) {
    throw std::runtime_error(
        "Graph::narrow: [start, start+length) out of range");
  }
  std::vector<int> new_dims(input.dim, input.dim + input.num_dims);
  new_dims[dim] = length;
  int64_t offset =
      static_cast<int64_t>(start) * row_major_byte_stride(input, dim);
  return make_view(input, new_dims, offset);
}

DTensor *Graph::narrow(DTensor const *input, int dim, int start, int length) {
  DTensor v = narrow(*input, dim, start, length);
  view_storage.emplace_back(std::make_unique<DTensor>(v));
  return view_storage.back().get();
}

std::vector<DTensor>
    Graph::split(DTensor const &input, std::vector<int> const &sizes, int dim) {
  if (dim < 0 || dim >= input.num_dims) {
    throw std::runtime_error("Graph::split: dim out of range");
  }
  int total = 0;
  for (int s : sizes) {
    if (s <= 0) {
      throw std::runtime_error(
          "Graph::split: each split size must be positive");
    }
    total += s;
  }
  if (total != input.dim[dim]) {
    throw std::runtime_error(
        "Graph::split: sum of split sizes does not match input dim");
  }
  std::vector<DTensor> outputs;
  outputs.reserve(sizes.size());
  int64_t stride_bytes = row_major_byte_stride(input, dim);
  int cursor = 0;
  for (int s : sizes) {
    std::vector<int> new_dims(input.dim, input.dim + input.num_dims);
    new_dims[dim] = s;
    int64_t offset = static_cast<int64_t>(cursor) * stride_bytes;
    outputs.push_back(make_view(input, new_dims, offset));
    cursor += s;
  }
  return outputs;
}

std::vector<DTensor>
    Graph::split(DTensor const &input, int chunk_size, int dim) {
  if (dim < 0 || dim >= input.num_dims) {
    throw std::runtime_error("Graph::split: dim out of range");
  }
  if (chunk_size <= 0 || input.dim[dim] % chunk_size != 0) {
    throw std::runtime_error(
        "Graph::split: input.dim[dim] must be divisible by chunk_size");
  }
  int slice_len = input.dim[dim] / chunk_size;
  std::vector<int> sizes(chunk_size, slice_len);
  return split(input, sizes, dim);
}

int Graph::split(DTensor const *input,
                 std::vector<int> const &sizes,
                 int dim,
                 DTensor **outputs) {
  std::vector<DTensor> views = split(*input, sizes, dim);
  for (size_t i = 0; i < views.size(); i++) {
    view_storage.emplace_back(std::make_unique<DTensor>(views[i]));
    outputs[i] = view_storage.back().get();
  }
  return static_cast<int>(views.size());
}

} // namespace kernel
} // namespace mirage
