#include "mirage/search/config.h"

namespace mirage {
namespace search {

GeneratorConfig GeneratorConfig::get_default_config() {
  return {
      {
          type::KN_MATMUL_OP,
          type::KN_EXP_OP,
          type::KN_SILU_OP,
          type::KN_ADD_OP,
          type::KN_MUL_OP,
          type::KN_DIV_OP,
          type::KN_CUSTOMIZED_OP,
      } /* knop_to_explore */,
      {
          type::TB_MATMUL_OP,
          type::TB_EXP_OP,
          type::TB_SILU_OP,
          type::TB_ADD_OP,
          type::TB_MUL_OP,
          type::TB_DIV_OP,
          type::TB_FORLOOP_ACCUM_NO_RED_OP,
          type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP,
          type::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP,
          type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP,
      } /* tbop_to_explore */,
      {} /* imap_to_explore*/,
      {} /* imap_comb_to_explore */,
      {} /* omap_to_explore */,
      {} /* grid_dim_to_explore*/,
      {} /* block_dim_to_explore */,
      {} /* fmap_to_explore */,
      {4, 16, 64} /* frange_to_explore */,
      64 /* reduction_dimx */,
      false /* enable_attention_specific_optimization */,
      false /* enable_concat_matmul_transformation */,
  };
}

GeneratorConfig GeneratorConfig::get_attention_default_config() {
  return {{
              type::KN_MATMUL_OP,
              type::KN_REDUCTION_0_OP,
              type::KN_REDUCTION_1_OP,
              type::KN_REDUCTION_2_OP,
              type::KN_EXP_OP,
              type::KN_DIV_OP,
              type::KN_CUSTOMIZED_OP,
          },
          {
              type::TB_MATMUL_OP,
              type::TB_EXP_OP,
              type::TB_DIV_OP,
              type::TB_FORLOOP_ACCUM_NO_RED_OP,
              type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP,
              type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP,
          },
          {
              {0, -1, 1},
              {0, 2, -1},
              {0, 1, -1},
              {0, -1, -1},
              {0, 2, -1},
              {0, 1, -1},
          } /* imap_to_explore */,
          {{{0, -1, 1}, {0, 2, -1}, {0, 1, -1}},
           {{0, -1, -1}, {0, 2, -1}, {0, 1, -1}},
           {{0, -1, -1}, {0, -1, -1}},
           {{0, 1, -1}, {0, 1, -1}}} /* imap_comb_to_explore*/,
          {
              {0, 1, -1},
              {0, 2, 1},
              {0, 2, -1},
              {0, -1, -1},
              {-1, 2, 1},
              {-1, 1, -1},
              {-1, 2, -1},
              {-1, -1, -1},
          } /* omap_to_explore */,
          {{16, 1, 1}, {16, 2, 1}} /* grid_dim_to_explore*/,
          {{128, 1, 1}} /* block_dim_to_explore */,
          {-1, 1, 2} /* fmap_to_explore */,
          {4, 8} /* frange_to_explore */,
          {
              layout::SmemRowMajor,
              layout::SmemColumnMajor,
              layout::SmemRowMajorTensorOpMultiplicand_Crosswise16,
              layout::SmemRowMajorTensorOpMultiplicand_Crosswise32,
              layout::SmemRowMajorTensorOpMultiplicand_Crosswise64,
              layout::SmemColumnMajorTensorOpMultiplicand_Crosswise16,
              layout::SmemColumnMajorTensorOpMultiplicand_Crosswise32,
              layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64,
          } /* smem_layout_to_explore*/,
          64 /* reduction_dimx */};
}

GeneratorConfig GeneratorConfig::get_mlp_default_config() {
  return {{type::KN_MATMUL_OP,
           type::KN_EXP_OP,
           type::KN_DIV_OP,
           type::KN_ADD_OP,
           type::KN_MUL_OP,
           type::KN_SILU_OP,
           type::KN_CUSTOMIZED_OP},
          {
              type::TB_MATMUL_OP,
              type::TB_EXP_OP,
              type::TB_DIV_OP,
              type::TB_SILU_OP,
              type::TB_MUL_OP,
              type::TB_ADD_OP,
              type::TB_FORLOOP_ACCUM_NO_RED_OP,
          },
          {{0, -1, -1}, {1, -1, -1}, {-1, -1, -1}} /* imap_to_explore */,
          {} /* imap_comb_to_explore*/,
          {{1, -1, -1}} /* omap_to_explore */,
          {{16, 1, 1}, {16, 2, 1}, {64, 1, 1}} /* grid_dim_to_explore*/,
          {{128, 1, 1}} /* block_dim_to_explore */,
          {-1, 0, 1} /* fmap_to_explore */,
          {4, 16, 64} /* frange_to_explore */,
          {
              layout::SmemRowMajor,
              layout::SmemColumnMajor,
              layout::SmemRowMajorTensorOpMultiplicand_Crosswise16,
              layout::SmemRowMajorTensorOpMultiplicand_Crosswise32,
              layout::SmemRowMajorTensorOpMultiplicand_Crosswise64,
              layout::SmemColumnMajorTensorOpMultiplicand_Crosswise16,
              layout::SmemColumnMajorTensorOpMultiplicand_Crosswise32,
              layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64,
          } /* smem_layout_to_explore*/,
          64 /* reduction_dimx */};
}

GeneratorConfig GeneratorConfig::get_lora_default_config() {
  return {{
              type::KN_MATMUL_OP,
              type::KN_REDUCTION_1_OP,
              type::KN_CUSTOMIZED_OP,
          },
          {
              type::TB_MATMUL_OP,
              type::TB_REDUCTION_1_OP,
              type::TB_REDUCTION_1_TO_DIMX_OP,
              type::TBOperatorType::TB_CONCAT_THEN_MATMUL_OP,
          },
          {{1, -1, -1}, {-1, -1, -1}} /* imap_to_explore */,
          {} /* imap_comb_to_explore*/,
          {{1, -1, -1}} /* omap_to_explore */,
          {{128, 1, 1}} /* grid_dim_to_explore*/,
          {{128, 1, 1}} /* block_dim_to_explore */,
          {-1, 0, 1} /* fmap_to_explore */,
          {2, 4} /* frange_to_explore */,
          {
              layout::SmemRowMajor,
              layout::SmemColumnMajor,
              layout::SmemRowMajorTensorOpMultiplicand_Crosswise16,
              layout::SmemRowMajorTensorOpMultiplicand_Crosswise32,
              layout::SmemRowMajorTensorOpMultiplicand_Crosswise64,
              layout::SmemColumnMajorTensorOpMultiplicand_Crosswise16,
              layout::SmemColumnMajorTensorOpMultiplicand_Crosswise32,
              layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64,
          } /* smem_layout_to_explore*/,
          64 /* reduction_dimx */};
}

void GeneratorConfig::show() const {
  printf("========== Search Configuration ==========\n");
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

} // namespace search
} // namespace mirage
