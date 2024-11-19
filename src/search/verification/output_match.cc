#include "mirage/search/verification/output_match.h"

#include <algorithm>

namespace mirage {
namespace search {

OutputMatch::OutputMatch(int num_outputs) : _valid(true) {
  for (int i = 0; i < num_outputs; ++i) {
    match.push_back(i);
  }
}

OutputMatch::OutputMatch(std::vector<int> match, bool _valid)
    : match(match), _valid(_valid) {}

OutputMatch OutputMatch::invalid_match() {
  return OutputMatch({}, false);
}

bool OutputMatch::next() {
  return std::next_permutation(match.begin(), match.end());
}

int OutputMatch::operator[](size_t i) const {
  return match[i];
}

size_t OutputMatch::size() const {
  return match.size();
}

bool OutputMatch::is_valid() const {
  return _valid;
}

} // namespace search
} // namespace mirage
