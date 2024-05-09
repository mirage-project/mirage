#include "mirage/utils/json_utils.h"

void to_json(json &j, int3 const &i) {
  j["x"] = i.x;
  j["y"] = i.y;
  j["z"] = i.z;
}

void from_json(json const &j, int3 &i) {
  j.at("x").get_to(i.x);
  j.at("y").get_to(i.y);
  j.at("z").get_to(i.z);
}

void to_json(json &j, dim3 const &i) {
  j["x"] = i.x;
  j["y"] = i.y;
  j["z"] = i.z;
}

void from_json(json const &j, dim3 &i) {
  j.at("x").get_to(i.x);
  j.at("y").get_to(i.y);
  j.at("z").get_to(i.z);
}
