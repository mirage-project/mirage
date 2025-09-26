#pragma once

#include "containers.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void to_json(json &j, int3 const &i);
void from_json(json const &j, int3 &i);
void to_json(json &j, dim3 const &i);
void from_json(json const &j, dim3 &i);

template <typename T>
T load_json(char const *file_path) {
  std::ifstream ifs(file_path);
  json j;
  ifs >> j;
  return j.get<T>();
}