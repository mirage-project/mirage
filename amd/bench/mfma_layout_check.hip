// Day-one check of the gfx950 MFMA operand layouts moe_experts and gemm_tile depend on (amd/notes/cdna4_isa_facts.md 2.1:
// the ISA's FP8 figure and its general formula disagree). One MFMA on known random inputs, packed per hypothesis, compared with a
// CPU reference; the hypothesis whose result matches is the layout.
//
// v_mfma_scale_f32_16x16x128_f8f6f4, D[16 x 16] = sum_k A[i][k] sa[i][k/32] * B[k][j] sb[j][k/32]:
//   the fp8 operand (e4m3, 8 VGPRs per lane), lane l -> row (A) / column (B) l % 16, group g = l / 16, its 32 bytes hold:
//     fp8-fig  K 16g..16g+15 in bytes 0..15, K 64+16g..64+16g+15 in bytes 16..31   (the ISA's FP8 figure)
//     fp8-lin  K 32g..32g+31 in bytes 0..31                                           (the ISA's general formula)
//   the fp4 operand (e2m1, 4 VGPRs per lane), lane l -> l % 16, group g: K 32g..32g+31, two values per byte:
//     fp4-lo   the lower K in the low nibble
//     fp4-hi   the lower K in the high nibble
//   both arrangements: A fp8 x B fp4, and A fp4 x B fp8 (moe_experts: weights A = 16 output rows, tokens B)
//   scales: lane l -> row (A) / column (B) l % 16, K block l / 16, in byte 0 of the scale VGPR
//   D (4 VGPRs per lane):
//     d-rows  lane l -> column l % 16, register r -> row 4 (l / 16) + r              (the ISA's C/D rule)
//     d-cols  lane l -> row l % 16, register r -> column 4 (l / 16) + r
// And the BF16 MFMA gemm_tile would use, v_mfma_f32_16x16x32_bf16: lane l row (A) / column (B) l % 16, group g:
//     k-fig   K 8g..8g+7 in the 4 VGPRs (2 per VGPR, lower K in the low half)          (the ISA's BF16 figure)
//     k-split K 4g..4g+3 in VGPRs 0-1, K 16+4g..16+4g+3 in VGPRs 2-3
// Output: one line per combination with its largest error relative to the largest |D|; the matching ones are near 0.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 mfma_layout_check.hip -o mfma_layout_check
#include <cmath>
#include <cstring>
#include <random>
#include <string>

#include "common.h"

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));

// in: per lane 8 dwords of A, 8 dwords of B, 1 dword of A scales, 1 of B scales; out: per lane 4 floats of D.
// FA, FB: the formats (0 fp8 e4m3, 4 fp4 e2m1): fp8 x fp4 here; fp4 x fp8 = moe_experts (weights A, tokens B)
template <int FA, int FB>
__global__ void scaled_kernel(const int *in, float *out) {
  int l = threadIdx.x;
  v8i a, b;
  for (int i = 0; i < 8; ++i) {
    a[i] = in[l * 18 + i];
    b[i] = in[l * 18 + 8 + i];
  }
  int sa = in[l * 18 + 16], sb = in[l * 18 + 17];
  v4f c = {0, 0, 0, 0};
  c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, FA, FB, 0, sa, 0, sb);
  for (int r = 0; r < 4; ++r) out[l * 4 + r] = c[r];
}

__global__ void bf16_kernel(const int *in, float *out) {
  int l = threadIdx.x;
  int a4[4], b4[4];
  for (int i = 0; i < 4; ++i) {
    a4[i] = in[l * 8 + i];
    b4[i] = in[l * 8 + 4 + i];
  }
  v8bf a = __builtin_bit_cast(v8bf, *(v4f *)a4), b = __builtin_bit_cast(v8bf, *(v4f *)b4);
  v4f c = {0, 0, 0, 0};
  c = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
  for (int r = 0; r < 4; ++r) out[l * 4 + r] = c[r];
}

static double e4m3(uint8_t v) {
  int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
  double x = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
  return s ? -x : x;
}
static double e2m1(uint8_t v) {
  static const double t[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
  return (v & 8) ? -t[v & 7] : t[v & 7];
}
static float bf16f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

int main() {
  CHECK(hipSetDevice(0));
  print_device(0);
  std::mt19937 rng(7);
  // ---- scaled: fp8 x fp4 (A fp8) and fp4 x fp8 (A fp4, moe_experts' arrangement) ----
  int *din;
  float *dout;
  CHECK(hipMalloc(&din, 64 * 18 * 4));
  CHECK(hipMalloc(&dout, 64 * 4 * 4));
  std::string json;
  for (int a_is_fp8 = 1; a_is_fp8 >= 0; --a_is_fp8) {
    // X[16][128] the fp8 operand (rows / columns by lane), Y[16][128] the fp4 one; SX, SY their scales [16][4]
    uint8_t X[16][128], Y[16][128], SX[16][4], SY[16][4];
    for (auto &r : X)
      for (auto &v : r) {
        do v = rng() & 0xFF;
        while ((v & 0x7F) == 0x7F);  // no NaN
      }
    for (auto &r : Y)
      for (auto &v : r) v = rng() & 0xF;
    for (auto &r : SX)
      for (auto &v : r) v = 124 + rng() % 7;
    for (auto &r : SY)
      for (auto &v : r) v = 124 + rng() % 7;
    // D[i][j] = sum_k A[i][k] B[k][j]: A rows = the A operand's lanes, B columns = the B operand's lanes
    double ref[16][16], refmax = 0;
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < 16; ++j) {
        double s_ = 0;
        for (int k = 0; k < 128; ++k) {
          int xi = a_is_fp8 ? i : j, yi = a_is_fp8 ? j : i;
          s_ += e4m3(X[xi][k]) * std::ldexp(1.0, SX[xi][k / 32] - 127) * e2m1(Y[yi][k]) * std::ldexp(1.0, SY[yi][k / 32] - 127);
        }
        ref[i][j] = s_;
        refmax = std::max(refmax, std::fabs(s_));
      }
    for (int xh = 0; xh < 2; ++xh)
      for (int yh = 0; yh < 2; ++yh) {
        std::vector<int> in(64 * 18, 0);
        for (int l = 0; l < 64; ++l) {
          int row = l % 16, g = l / 16;
          uint8_t *xb = (uint8_t *)&in[l * 18 + (a_is_fp8 ? 0 : 8)], *yb = (uint8_t *)&in[l * 18 + (a_is_fp8 ? 8 : 0)];
          for (int q = 0; q < 32; ++q) {
            int k = xh == 0 ? (q < 16 ? 16 * g + q : 64 + 16 * g + (q - 16)) : 32 * g + q;
            xb[q] = X[row][k];
          }
          for (int q = 0; q < 16; ++q) {
            uint8_t lo = Y[row][32 * g + 2 * q], hi = Y[row][32 * g + 2 * q + 1];
            yb[q] = yh == 0 ? (lo | hi << 4) : (hi | lo << 4);
          }
          in[l * 18 + 16] = a_is_fp8 ? SX[row][g] : SY[row][g];
          in[l * 18 + 17] = a_is_fp8 ? SY[row][g] : SX[row][g];
        }
        CHECK(hipMemcpy(din, in.data(), in.size() * 4, hipMemcpyHostToDevice));
        if (a_is_fp8) {
          hipLaunchKernelGGL((scaled_kernel<0, 4>), dim3(1), dim3(64), 0, 0, din, dout);
        } else {
          hipLaunchKernelGGL((scaled_kernel<4, 0>), dim3(1), dim3(64), 0, 0, din, dout);
        }
        CHECK(hipDeviceSynchronize());
        std::vector<float> d(256);
        CHECK(hipMemcpy(d.data(), dout, 256 * 4, hipMemcpyDeviceToHost));
        for (int dh = 0; dh < 2; ++dh) {
          double err = 0;
          for (int l = 0; l < 64; ++l)
            for (int r = 0; r < 4; ++r) {
              int i = dh == 0 ? 4 * (l / 16) + r : l % 16, j = dh == 0 ? l % 16 : 4 * (l / 16) + r;
              err = std::max(err, std::fabs(d[l * 4 + r] - ref[i][j]) / refmax);
            }
          const char *mf = a_is_fp8 ? "fp8 x fp4" : "fp4 x fp8", *xn = xh ? "fp8-lin" : "fp8-fig", *yn = yh ? "fp4-hi" : "fp4-lo",
                     *dn = dh ? "d-cols" : "d-rows";
          // MI355X (10-10): the matching layout 4e-5 of the largest value (the MFMA adds in fp32), every other one 0.89 or more
          printf("scaled 16x16x128 %s  %-7s %-6s %-6s relative error %.2e%s\n", mf, xn, yn, dn, err, err < 1e-3 ? "   MATCH" : "");
          char row[200];
          snprintf(row, sizeof(row), "%s{\"mfma\": \"scale_16x16x128 %s\", \"fp8\": \"%s\", \"fp4\": \"%s\", \"d\": \"%s\", \"err\": %.3e}",
                   json.empty() ? "" : ", ", mf, xn, yn, dn, err);
          json += row;
        }
      }
  }
  // ---- bf16 ----
  uint16_t Ab[16][32], Bb[32][16];
  for (auto &r : Ab)
    for (auto &v : r) v = 0x3c00 + (rng() % 0x0400) - 0x0200 + ((rng() & 1) << 15);  // |x| in [0.5, 2)
  for (auto &r : Bb)
    for (auto &v : r) v = 0x3c00 + (rng() % 0x0400) - 0x0200 + ((rng() & 1) << 15);
  double refb[16][16], refbmax = 0;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      double s = 0;
      for (int k = 0; k < 32; ++k) s += (double)bf16f(Ab[i][k]) * bf16f(Bb[k][j]);
      refb[i][j] = s;
      refbmax = std::max(refbmax, std::fabs(s));
    }
  for (int kh = 0; kh < 2; ++kh) {
    std::vector<int> in(64 * 8, 0);
    for (int l = 0; l < 64; ++l) {
      int row = l % 16, g = l / 16;
      uint16_t *ab = (uint16_t *)&in[l * 8], *bb = (uint16_t *)&in[l * 8 + 4];
      for (int q = 0; q < 8; ++q) {
        int k = kh == 0 ? 8 * g + q : (q < 4 ? 4 * g + q : 16 + 4 * g + (q - 4));
        ab[q] = Ab[row][k];
        bb[q] = Bb[k][row];
      }
    }
    CHECK(hipMemcpy(din, in.data(), in.size() * 4, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(bf16_kernel, dim3(1), dim3(64), 0, 0, din, dout);
    CHECK(hipDeviceSynchronize());
    std::vector<float> d(256);
    CHECK(hipMemcpy(d.data(), dout, 256 * 4, hipMemcpyDeviceToHost));
    for (int dh = 0; dh < 2; ++dh) {
      double err = 0;
      for (int l = 0; l < 64; ++l)
        for (int r = 0; r < 4; ++r) {
          int i = dh == 0 ? 4 * (l / 16) + r : l % 16, j = dh == 0 ? l % 16 : 4 * (l / 16) + r;
          err = std::max(err, std::fabs(d[l * 4 + r] - refb[i][j]) / refbmax);
        }
      const char *kn = kh ? "k-split" : "k-fig", *dn = dh ? "d-cols" : "d-rows";
      printf("bf16 16x16x32               %-7s %-7s relative error %.2e%s\n", kn, dn, err, err < 1e-5 ? "   MATCH" : "");
      char row[160];
      snprintf(row, sizeof(row), ", {\"mfma\": \"bf16_16x16x32\", \"k\": \"%s\", \"d\": \"%s\", \"err\": %.3e}", kn, dn, err);
      json += row;
    }
  }
  printf("{\"bench\": \"mfma_layout_check\", \"rows\": [%s]}\n", json.c_str());
  return 0;
}
