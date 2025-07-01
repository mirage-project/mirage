#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"
#include "mirage/nki_transpiler/transpile.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace mirage;

int main(int argc, char **argv) {
  bool generate_programs_from_checkpoint;
  if (argc > 1) {
    std::string arg = argv[1];
    if (!strcmp(arg.c_str(), "--from-checkpoint")) {
      generate_programs_from_checkpoint = true;
    } else {
      std::cerr << "Usage: " << argv[0] << " [--from-checkpoint]" << std::endl;
      return 1;
    }
  } else {
    generate_programs_from_checkpoint = false;
  }
  int M = 128, K = 512, N = 512;
  std::string suffix = std::to_string(M) + "_" + std::to_string(K) + "_" + std::to_string(N);
  std::string checkpoint_filename = "nki_rms_norm_checkpoint_" + suffix + ".json";
  if (generate_programs_from_checkpoint) {
    std::ifstream generated_graphs_file(checkpoint_filename.data(), std::ifstream::binary);
    if (!generated_graphs_file) {
      std::cerr << "Error: Could not open file " << checkpoint_filename << std::endl;
      return 1;
    }
    std::filesystem::create_directories("nki_programs" + suffix);
    json j;
    generated_graphs_file >> j;
    int idx = 0;
    for (json const &graph_json : j) {
      kernel::Graph graph;
      from_json(graph_json, graph);
      {
        nki_transpiler::NKITranspilerConfig config{
          /*target_cc=*/10,
        };
        nki_transpiler::NKITranspiler transpiler(&graph, config);
        nki_transpiler::NKITranspileResult transpile_result = transpiler.generate_code();
        if (transpile_result.error_state.errors.empty()) {
          // save transpiled code to nki_programs/nki_program_<idx>.py
          std::string filename = "nki_programs" + suffix + "/nki_program_" + std::to_string(idx) + ".py";
          std::ofstream ofs(filename);
          if (!ofs) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return 1;
          }
          ofs << transpile_result.code;
          ofs.close();
          std::cout << "Transpiled NKI program saved to " << filename << std::endl;
        } else {
          std::cerr << "Error in transpiling graph " << idx << ": ";
          for (const auto &error : transpile_result.error_state.errors) {
            std::cerr << error << " ";
          }
          std::cerr << std::endl;
        }
        idx++;
      }
    }
    return 0;
  }
  kernel::Graph ref_graph;
  {
    kernel::DTensor X = ref_graph.new_input(
        {M, K}, {(size_t)K, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor W = ref_graph.new_input(
        {K, N}, {(size_t)N, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor D = ref_graph.rms_norm(X, {X.dim[1]});
    kernel::DTensor O = ref_graph.matmul(D, W);
    ref_graph.mark_output(O);
  }

  search::GeneratorConfig config =
      search::GeneratorConfig::get_default_config();
  config.verifier_type = search::VerifierType::FORMAL_VERIFIER;
  search::KernelGraphGenerator gen(ref_graph, config, checkpoint_filename.data());
  gen.generate_kernel_graphs();

  return 0;
}
