// gfx950 v_cvt_pk_fp8_f32 vs round-to-nearest-even OCP e4m3: exact values, midpoints, random values
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <vector>
__global__ void cvt(const float *x, uint8_t *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float c = __builtin_amdgcn_fmed3f(x[i], -448.0f, 448.0f);
    y[i] = (uint8_t)(__builtin_amdgcn_cvt_pk_fp8_f32(c, 0.0f, 0, false) & 0xFF);
  }
}
static double e4m3(uint8_t v) {
  int s = v >> 7, ex = (v >> 3) & 15, m = v & 7;
  double x = ex ? std::ldexp(1.0 + m / 8.0, ex - 7) : std::ldexp(m / 8.0, -6);
  return s ? -x : x;
}
static double fnuz(uint8_t v) {  // e4m3fnuz: bias 8, 0x80 = NaN
  int s = v >> 7, ex = (v >> 3) & 15, m = v & 7;
  double x = ex ? std::ldexp(1.0 + m / 8.0, ex - 8) : std::ldexp(m / 8.0, -7);
  return s ? -x : x;
}
static uint8_t rne(double x) {
  uint8_t best = 0; double bd = 1e300;
  for (int v = 0; v < 256; v++) {
    if ((v & 0x7F) == 0x7F) continue;
    double d = std::fabs(e4m3(v) - x);
    if (d < bd || (d == bd && (v & 1) == 0 && (best & 1))) bd = d, best = v;
  }
  if (x == 0) best = std::signbit(x) ? 0x80 : 0;
  return best;
}
int main() {
  std::vector<float> x;
  std::vector<int> kind;  // 0 exact, 1 midpoint, 2 random
  for (int v = 0; v < 256; v++) if ((v & 0x7F) != 0x7F) { x.push_back((float)e4m3(v)); kind.push_back(0); }
  for (int v = 0; v < 126; v++) {  // positive neighbours v, v+1 (codes 0..126)
    double m = (e4m3(v) + e4m3(v + 1)) / 2; x.push_back((float)m); kind.push_back(1); x.push_back((float)-m); kind.push_back(1);
  }
  srand(1);
  for (int i = 0; i < 100000; i++) { x.push_back((float)((rand() / (double)RAND_MAX * 2 - 1) * 448 * std::ldexp(1.0, -(rand() % 12)))); kind.push_back(2); }
  int n = x.size();
  float *dx; uint8_t *dy;
  (void)hipMalloc(&dx, n * 4); (void)hipMalloc(&dy, n);
  (void)hipMemcpy(dx, x.data(), n * 4, hipMemcpyHostToDevice);
  cvt<<<(n + 255) / 256, 256>>>(dx, dy, n);
  std::vector<uint8_t> y(n);
  (void)hipMemcpy(y.data(), dy, n, hipMemcpyDeviceToHost);
  const char *names[3] = {"exact", "midpoint", "random"};
  for (int k = 0; k < 3; k++) {
    int tot = 0, bad = 0, shown = 0, fnuz_ok = 0; std::map<int, int> diff;
    for (int i = 0; i < n; i++) if (kind[i] == k) {
      tot++;
      uint8_t r = rne(x[i]);
      fnuz_ok += std::fabs(fnuz(y[i]) - x[i]) <= std::fabs(x[i]) * 0.07 + 1e-9;
      if (y[i] != r) {
        bad++; diff[(int)(y[i] & 0x7F) - (int)(r & 0x7F)]++;
        if (shown++ < 8) printf("  %s x=%.8g gpu 0x%02x (%.8g) rne 0x%02x (%.8g)\n", names[k], x[i], y[i], e4m3(y[i]), r, e4m3(r));
      }
    }
    printf("%s: %d of %d differ from RNE e4m3; gpu byte read as fnuz within 7%%: %d; code differences:", names[k], bad, tot, fnuz_ok);
    for (auto &kv : diff) printf(" %+d:%d", kv.first, kv.second);
    printf("\n");
  }
}
