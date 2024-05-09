#pragma once

#include "containers.h"
#include <nlohmann/json.hpp>
#include <vector_types.h>

using json = nlohmann::json;

void to_json(json &j, int3 const &i);
void from_json(json const &j, int3 &i);
void to_json(json &j, dim3 const &i);
void from_json(json const &j, dim3 &i);