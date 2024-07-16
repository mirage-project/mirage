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

#include <iostream>
#include <functional>
#include <string>
#include <vector>

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
  return value;
}

template <>
[[maybe_unused]] std::string my_to_string(std::string const &value) {
  return value;
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
template<typename InT, typename OutT>
std::vector<OutT> map(std::vector<InT> const &vec, std::function<OutT(InT)> const &f) {
  std::vector<OutT> result(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    result[i] = f(vec[i]);
  }
  return result;
}

template<typename T>
std::vector<std::string> map_to_cute_int(std::vector<T> const &vec) {
  return map<T, std::string>(vec, [](T const &x) { return "Int<" + my_to_string(x) + ">"; });
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
    std::cerr << "Error encountered during transpiling.";
    std::cerr << "The number of arguments does not match the number of markers "
                 "in the template string.";
    std::cerr << "fmt_str: " << fmt_str << std::endl;
    std::cerr << "args: ";
    ((std::cerr << my_to_string(args) << " "), ...);
    std::cerr << std::endl;
    std::cerr << "num_args: " << num_args << std::endl;
    std::cerr << "num_markers: " << num_markers << std::endl;
    exit(1);
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
  // Here we support "smart indenting". If the last character is "{", we increase
  // the indent level. If it is "}", we decrease the indent level.
  template <typename... Args>
  void e(std::string const &fmt_str, Args... args) {
    std::string line = fmt(fmt_str, args...);
    char last_char = line.empty() ? EOF : line.back();
    if (last_char == '}') {
      cur_indent_level -= 1;
    }
    line = std::string(cur_indent_level * NUM_INDENT_SPACES, ' ') + line;
    lines.push_back(line);
    if (last_char == '{') {
      cur_indent_level += 1;
    }
  }

  // Merge two CodeKeeper objects
  friend void operator<<(CodeKeeper &target, CodeKeeper const &source) {
    for (auto const &line : source.lines) {
      std::string new_line = std::string(target.cur_indent_level * NUM_INDENT_SPACES, ' ') + line;
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

template<typename T>
inline T round_to_multiple(T value, T multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

// A mapping from GPU to its compute capability
namespace GPU_CC {
  static constexpr int P100 = 60;
  static constexpr int V100 = 70;
  static constexpr int T4 = 75;
  static constexpr int A100 = 80;
  static constexpr int H100 = 90;
}

}
}

