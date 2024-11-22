#include "mirage/search/verification/probabilistic_verifier.h"
#include "mirage/search/op_utils.h"

namespace mirage {
namespace search {

std::mutex ProbabilisticVerifier::fp_mutex;

ProbabilisticVerifier::ProbabilisticVerifier(kernel::Graph const &input_graph) {
  for (auto const &op : input_graph.operators) {
    op->fingerprint();
  }

  for (kernel::KNOperator *op : input_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      input_graph_fingerprints.push_back(
          op->input_tensors[0].copy_fingerprint_to_ctensor());
    }
  }
}

OutputMatch ProbabilisticVerifier::verify(kernel::Graph const &graph) {
  std::lock_guard<std::mutex> lock(fp_mutex);

  std::vector<kernel::DTensor> fingerprints;

  for (auto const &op : graph.operators) {
    op->fingerprint();
  }

  for (auto const &op : graph.operators) {
    for (auto const &tensor : op->output_tensors) {
      if (get_num_consumers(graph, tensor) == 0) {
        fingerprints.push_back(tensor);
      }
    }
  }

  assert(fingerprints.size() == input_graph_fingerprints.size());

  auto verify_with_match = [&](OutputMatch const &match) {
    for (size_t i = 0; i < match.size(); i++) {
      if (!fingerprints[match[i]].has_same_fingerprint(
              input_graph_fingerprints[i])) {
        return false;
      }
    }
    return true;
  };

  OutputMatch match(fingerprints.size());
  do {
    if (verify_with_match(match)) {
      assert(match.is_valid());
      return match;
    }
  } while (match.next());
  return OutputMatch::invalid_match();
}

} // namespace search
} // namespace mirage