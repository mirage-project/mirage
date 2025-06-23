#include "mirage/search/config.h"

namespace mirage {
namespace search {

GeneratorConfig GeneratorConfig::get_default_config() {
  return {
      9 /* max_num_threadblock_graph_op */,
      5 /* max_num_kernel_graph_op */,
      1 /* max_num_threadblock_graphs */,
      3 /* max_num_threadblock_graph_inputs */,
      2 /* max_num_threadblock_graph_outputs */,
      16 /* search_thread */,
      VerifierType::PROBABILISTIC_VERIFIER,
      {
          type::KN_MATMUL_OP,
          type::KN_EXP_OP,
          type::KN_SQUARE_OP,
          type::KN_SQRT_OP,
          type::KN_SILU_OP,
          type::KN_GELU_OP,
          type::KN_RELU_OP,
          type::KN_CLAMP_OP,
          type::KN_ADD_OP,
          type::KN_MUL_OP,
          type::KN_DIV_OP,
          type::KN_POW_OP,
          // type::KN_REDUCTION_2_OP,
          type::KN_CUSTOMIZED_OP,
      } /* knop_to_explore */,
      {
          type::TB_MATMUL_OP,
          type::TB_EXP_OP,
          type::TB_SQUARE_OP,
          type::TB_SQRT_OP,
          type::TB_SILU_OP,
          type::TB_GELU_OP,
          type::TB_RELU_OP,
          type::TB_CLAMP_OP,
          type::TB_ADD_OP,
          type::TB_MUL_OP,
          type::TB_DIV_OP,
          type::TB_POW_OP,
          type::TB_RMS_NORM_OP,
          type::TB_FORLOOP_ACCUM_NO_RED_OP,
          type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP,
          // type::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP,
          // type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP,
          type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP,
      } /* tbop_to_explore */,
      {} /* imap_to_explore*/,
      {} /* imap_comb_to_explore */,
      {} /* omap_to_explore */,
      {} /* grid_dim_to_explore*/,
      {} /* block_dim_to_explore */,
      {} /* fmap_to_explore */,
      {
          4,
          16,
          64,
      } /* frange_to_explore */,
      64 /* reduction_dimx */,
      false /* enable_attention_specific_optimization */,
      false /* enable_concat_matmul_transformation */,
      false /* randomized_branches */,
  };
}

void GeneratorConfig::enable_attention_specific_optimization() {
  _enable_attention_specific_optimization = true;
  max_num_threadblock_graphs = 2;
  tbop_to_explore.push_back(type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP);
  deduplicate(tbop_to_explore);
}

void GeneratorConfig::enable_concat_matmul_transformation() {
  _enable_concat_matmul_transformation = true;
}

void GeneratorConfig::show() const {
  printf("========== Search Configuration ==========\n");
  printf("  max num threadblock graph op: %zu\n", max_num_threadblock_graph_op);
  printf("  max num kernel_graph op: %zu\n", max_num_kernel_graph_op);
  printf("  max num threadblock graphs: %zu\n", max_num_threadblock_graphs);
  printf("  max num threadblock graph inputs: %zu\n",
         max_num_threadblock_graph_inputs);
  printf("  max num threadblock graph outputs: %zu\n",
         max_num_threadblock_graph_outputs);
  printf("  search_thread: %zu\n", search_thread);
  printf("  imaps to explore:\n");
  for (auto const &imap : imap_to_explore) {
    printf("    (%d, %d, %d)\n", imap.x, imap.y, imap.z);
  }
  printf("  imap combs to explore:\n");
  for (auto const &imap_comb : imap_comb_to_explore) {
    for (auto const &imap : imap_comb) {
      printf("    (%d, %d, %d), ", imap.x, imap.y, imap.z);
    }
    printf("\n");
  }
  printf("  omaps to explore:\n");
  for (auto const &omap : omap_to_explore) {
    printf("    (%d, %d, %d)\n", omap.x, omap.y, omap.z);
  }
  printf("  grid dims to explore:\n");
  for (auto const &griddim : grid_dim_to_explore) {
    printf("    (%d, %d, %d)\n", griddim.x, griddim.y, griddim.z);
  }
  printf("  block dims to explore:\n");
  for (auto const &blockdim : block_dim_to_explore) {
    printf("    (%d, %d, %d)\n", blockdim.x, blockdim.y, blockdim.z);
  }
  printf("  fmaps to explore:");
  for (auto const &fmap : fmap_to_explore) {
    printf("%d ", fmap);
  }
  printf("\n");
  printf("  franges to explore:");
  for (auto const &frange : frange_to_explore) {
    printf("%d ", frange);
  }
  printf("\n");
}

bool TBGraphConfig::operator==(TBGraphConfig const &other) const {
  return grid_dim == other.grid_dim && block_dim == other.block_dim &&
         imaps == other.imaps && fmaps == other.fmaps && frange == other.frange;
}

void TBGraphConfig::show() const {
  printf("========== Threadblock Graph Configuration ==========\n");
  printf("  grid dim: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
  printf("  block dim: (%d, %d, %d)\n", block_dim.x, block_dim.y, block_dim.z);
  printf("  imaps:\n");
  for (auto const &imap : imaps) {
    printf("    (%d, %d, %d)\n", imap.x, imap.y, imap.z);
  }
  printf("  fmaps:");
  for (auto const &fmap : fmaps) {
    printf("%d ", fmap);
  }
  printf("\n");
  printf("  frange: %d\n", frange);
}

} // namespace search
} // namespace mirage

namespace std {

size_t hash<mirage::search::TBGraphConfig>::operator()(
    mirage::search::TBGraphConfig const &config) const {
  size_t hash = 0;
  hash_combine(hash, config.grid_dim);
  hash_combine(hash, config.block_dim);
  hash_combine(hash, config.imaps);
  hash_combine(hash, config.fmaps);
  hash_combine(hash, config.frange);
  return hash;
}

} // namespace std