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

#include "mirage/type.h"
#include <cassert>
#include <cute/layout.hpp>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace cute;

namespace mirage {
namespace transpiler {

// We need to specialize the to_string function for char* type, so we have
// my_to_string function here
template <typename T>
inline static std::string my_to_string(T const &value) {
  return std::to_string(value);
}

template <>
[[maybe_unused]] std::string my_to_string(char const *const &value) {
  return std::string(value);
}

template <>
[[maybe_unused]] std::string my_to_string(void *const &value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

template <>
[[maybe_unused]] std::string my_to_string(std::string const &value) {
  return value;
}

template <>
[[maybe_unused]] std::string my_to_string(char const &value) {
  return std::string(1, value);
}

template <>
[[maybe_unused]] std::string my_to_string(bool const &value) {
  return value ? "true" : "false";
}

// A vector {a, b, c, d} will be converted to "a, b, c, d"
template <typename T>
[[maybe_unused]] std::string my_to_string(std::vector<T> const &vec) {
  std::string result;
  for (size_t i = 0; i < vec.size(); i++) {
    result += my_to_string(vec[i]);
    if (i != vec.size() - 1) {
      result += ", ";
    }
  }
  return result;
}

// A simple `map` function that applies a function to each element in a vector
template <typename InT, typename OutT>
std::vector<OutT> map(std::vector<InT> const &vec,
                      std::function<OutT(InT)> const &f) {
  std::vector<OutT> result(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    result[i] = f(vec[i]);
  }
  return result;
}

template <typename T>
std::vector<std::string> map_to_cute_int(std::vector<T> const &vec) {
  return map<T, std::string>(
      vec, [](T const &x) { return "Int<" + my_to_string(x) + ">"; });
}

// A function that takes a template string and a list of elements
// It replaces each marker in the template string with the corresponding element
// and returns the resulting string. We can use this to generate code elegantly
template <typename... Args>
inline static std::string fmt(std::string const &fmt_str, Args... args) {
  std::string result = fmt_str;
  int num_args = sizeof...(args);
  int num_markers = std::count(result.begin(), result.end(), '$');
  if (num_args != num_markers) {
    std::cerr << "Error encountered during transpiling. ";
    std::cerr << "The number of arguments does not match the number of markers "
                 "in the template string.";
    std::cerr << "fmt_str: " << fmt_str << std::endl;
    std::cerr << "args: ";
    ((std::cerr << my_to_string(args) << " "), ...);
    std::cerr << std::endl;
    std::cerr << "num_args: " << num_args << std::endl;
    std::cerr << "num_markers: " << num_markers << std::endl;
    assert(num_args == num_markers);
  }
  (result.replace(result.find("$"), 1, my_to_string(args)), ...);
  return result;
}

// A helper class for keeping all generated code, and provide some utility
// functions for emitting code
class CodeKeeper {
private:
  static constexpr int NUM_INDENT_SPACES = 2;
  int cur_indent_level = 0;
  std::vector<std::string> lines;

public:
  // Emit a new line
  // Here we support "smart indenting". If the last character is "{", we
  // increase the indent level. If it is "}", we decrease the indent level.
  template <typename... Args>
  void e_front(std::string const &fmt_str, Args... args) {
    std::string line = fmt(fmt_str, args...);
    lines.emplace(lines.begin(), line);
  }

  template <typename... Args>
  void e(std::string const &fmt_str, Args... args) {
    std::string line = fmt(fmt_str, args...);
    char last_char = line.empty() ? EOF : line.back();
    if (last_char == '}') {
      cur_indent_level -= 1;
      if (cur_indent_level < 0) {
        printf("Warning: `cur_indent_level` goes below 0 when transpiling\n");
        cur_indent_level = 0;
      }
    }
    line = std::string(cur_indent_level * NUM_INDENT_SPACES, ' ') + line;
    lines.push_back(line);

    if (last_char == '{') {
      cur_indent_level += 1;
    }
  }

  // Manually increase the indent level by 1
  void inc_indent() {
    cur_indent_level += 1;
  }

  // Manually decrease the indent level by 1
  void dec_indent() {
    cur_indent_level -= 1;
    if (cur_indent_level < 0) {
      printf("Warning: `cur_indent_level` goes below 0 when transpiling\n");
      cur_indent_level = 0;
    }
  }

  // Merge two CodeKeeper objects
  friend void operator<<(CodeKeeper &target, CodeKeeper const &source) {
    for (auto const &line : source.lines) {
      std::string new_line =
          std::string(target.cur_indent_level * NUM_INDENT_SPACES, ' ') + line;
      target.lines.push_back(new_line);
    }
  }

  // Return the generated code as a string
  std::string to_string() const {
    std::string result;
    for (auto const &line : lines) {
      result += line + "\n";
    }
    return result;
  }
};

template <typename T>
inline static T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
inline static T round_to_multiple(T value, T multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

// A mapping from GPU to its compute capability
namespace GPU_CC {
static constexpr int P100 = 60;
static constexpr int V100 = 70;
static constexpr int T4 = 75;
static constexpr int A100 = 80;
static constexpr int H100 = 90;
static constexpr int B200 = 100;
} // namespace GPU_CC

// A handy iterator class for combining two iterators
// You can use like this:
// for (auto elem : Combine(vector1, vector2))
// The combine iterator is a const iterator, which means that, you can do this:
//
// for (const int& x : Combine(v1, v2)) ...
//
// but not this:
//
// for (int& x : Combine(v1, v2)) ...
// (NOTE to do this in the future we may have something like
// `MutCombineIterator`)
template <typename T, class Iter1, class Iter2>
class CombineIterator {
  Iter1 range1_start, range1_end, iter1;
  Iter2 range2_start, range2_end, iter2;

public:
  CombineIterator(Iter1 const &range1_start,
                  Iter1 const &range1_end,
                  Iter1 const &iter1,
                  Iter2 const &range2_start,
                  Iter2 const &range2_end,
                  Iter2 const &iter2)
      : range1_start(range1_start), range1_end(range1_end), iter1(iter1),
        range2_start(range2_start), range2_end(range2_end), iter2(iter2) {}

  bool operator!=(CombineIterator const &other) const {
    return iter1 != other.iter1 || iter2 != other.iter2;
  }

  void operator++() {
    if (iter1 != range1_end) {
      ++iter1;
    } else {
      ++iter2;
    }
  }

  T const &operator*() const {
    if (iter1 != range1_end) {
      return *iter1;
    } else {
      return *iter2;
    }
  }
};

template <typename T1, typename T2>
class Combine {
  using T =
      std::common_type_t<typename T1::value_type, typename T2::value_type>;

  using Iter1 = typename T1::const_iterator;
  using Iter2 = typename T2::const_iterator;

  Iter1 begin1, end1;
  Iter2 begin2, end2;

public:
  using value_type = T;
  using const_iterator = CombineIterator<T, Iter1, Iter2>;

  Combine(T1 const &v1, T2 const &v2)
      : begin1(v1.begin()), end1(v1.end()), begin2(v2.begin()), end2(v2.end()) {
  }

  CombineIterator<T, Iter1, Iter2> begin() const {
    return CombineIterator<T, Iter1, Iter2>(
        begin1, end1, begin1, begin2, end2, begin2);
  }

  CombineIterator<T, Iter1, Iter2> end() const {
    return CombineIterator<T, Iter1, Iter2>(
        begin1, end1, end1, begin2, end2, end2);
  }
};

// Get the number of elements in 16 Bytes for a given datatype
inline static size_t get_num_elems_in_16B(type::DataType datatype) {
  size_t elem_size = type::get_datatype_size(datatype);
  return std::max(16 / elem_size, 1ul);
}

// Generalized function to compute a new layout after partitioning and expanding
// to grid dimensions
inline auto generate_partitioned_and_expanded_layout(
    // Layout<ShapeType, StrideType> original_layout, // Original layout (Shape
    // + Stride)
    dim3 grid_dim, // Grid dimensions (x, y, z)
    std::vector<int> original_shape,
    std::vector<size_t> original_stride,
    std::vector<int>
        partition_logic, // Partitioning logic: gridDim[i] maps to layout_dim[j]
    int forloop_range,
    int forloop_dim) {

  int num_dims = original_shape.size();

  std::vector<int> new_shape_values(num_dims);
  std::vector<size_t> new_stride_values(num_dims);

  int gridX_stride = 1, gridY_stride = 1, gridZ_stride = 1, forloop_stride = 1;
  int gridX_shape = 1, gridY_shape = 1, gridZ_shape = 1, forloop_shape = 1;

  auto exists = [&partition_logic](int val) {
    return std::find(partition_logic.begin(), partition_logic.end(), val) !=
           partition_logic.end();
  };
  // Loop over each dimension in the original layout
  for (int dim = 0; dim < num_dims; ++dim) {
    if (dim != forloop_dim && (!exists(dim))) {
      new_shape_values[dim] = original_shape.at(dim);
      new_stride_values[dim] = original_stride.at(dim);
    }
    int partition_grid_dim = partition_logic[dim];

    if (partition_grid_dim >= 0 && forloop_dim == partition_grid_dim) {
      // partition by forloop and grid
      int partition_factor = (dim == 0)   ? grid_dim.x
                             : (dim == 1) ? grid_dim.y
                                          : grid_dim.z;
      partition_factor *= forloop_range;

      new_shape_values[partition_grid_dim] =
          original_shape.at(partition_grid_dim) / partition_factor;
      new_stride_values[partition_grid_dim] =
          original_stride.at(partition_grid_dim);

      forloop_stride = original_stride.at(partition_grid_dim) *
                       new_shape_values[partition_grid_dim];
      forloop_shape = forloop_range;

      if (dim == 0) {
        gridX_stride = forloop_stride * forloop_shape;
        gridX_shape = grid_dim.x;
      } else if (dim == 1) {
        gridY_stride = forloop_stride * forloop_shape;
        gridY_shape = grid_dim.y;
      } else if (dim == 2) {
        gridZ_stride = forloop_stride * forloop_shape;
        gridZ_shape = grid_dim.z;
      }

      //       new_shape_values[partition_grid_dim],
      //  partition_grid_dim);

    } else if (partition_grid_dim >= 0) {
      int partition_factor = (dim == 0)   ? grid_dim.x
                             : (dim == 1) ? grid_dim.y
                                          : grid_dim.z;
      // Adjust shape by dividing it by the partition factor
      new_shape_values[partition_grid_dim] =
          original_shape.at(partition_grid_dim) / partition_factor;
      // Stride remains the same because partitioning does not
      // affect layout's memory strides
      new_stride_values[partition_grid_dim] =
          original_stride.at(partition_grid_dim);

      if (dim == 0) {
        gridX_stride = original_stride.at(partition_grid_dim) *
                       new_shape_values[partition_grid_dim];
        gridX_shape = grid_dim.x;
      } else if (dim == 1) {
        gridY_stride = original_stride.at(partition_grid_dim) *
                       new_shape_values[partition_grid_dim];
        gridY_shape = grid_dim.y;
      } else if (dim == 2) {
        gridZ_stride = original_stride.at(partition_grid_dim) *
                       new_shape_values[partition_grid_dim];
        gridZ_shape = grid_dim.z;
      }

    } else if (forloop_dim == dim) {
      // forloop dim does not equal to partition dim
      // assert(partition_grid_dim < 0);
      // partition only by forloop dim
      if (forloop_shape > 1) {
        continue;
      }
      new_shape_values[dim] = original_shape.at(dim) / forloop_range;
      new_stride_values[dim] = original_stride.at(dim);
      forloop_stride = original_stride.at(dim) * new_shape_values[dim];
      forloop_shape = forloop_range;

    } else {
    }
  }

  // Add extra dimensions for gridDim.x, gridDim.y, and gridDim.z
  new_shape_values.push_back(gridX_shape); // Add gridDim.x as a new dimension
  new_shape_values.push_back(gridY_shape); // Add gridDim.y as a new dimension
  new_shape_values.push_back(gridZ_shape); // Add gridDim.z as a new dimension
  new_shape_values.push_back(forloop_shape);

  // Append the new strides
  new_stride_values.push_back(gridX_stride);
  new_stride_values.push_back(gridY_stride);
  new_stride_values.push_back(gridZ_stride);
  new_stride_values.push_back(forloop_stride);

  assert(new_shape_values.size() == new_stride_values.size());

  return fmt("Layout<Shape<$>, Stride<$>>",
             map_to_cute_int(new_shape_values),
             map_to_cute_int(new_stride_values));
}

} // namespace transpiler
} // namespace mirage
