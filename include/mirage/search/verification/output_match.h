#pragma once

#include <cstddef>
#include <vector>

namespace mirage {
namespace search {

class OutputMatch {
public:
  OutputMatch(int num_outputs);
  bool next();
  int operator[](size_t) const;
  bool is_valid() const;
  size_t size() const;

  static OutputMatch invalid_match();

private:
  OutputMatch(std::vector<int> match, bool _valid);

  std::vector<int> match;
  bool _valid;
};

} // namespace search
} // namespace mirage
