#pragma once

namespace mirage {

template <typename T>
inline T round_up_to_multiple(T value, T multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

template <typename T>
inline T mod_power(T base, T exponent, T modulus) {
  T result = 1;
  base %= modulus;
  while (exponent > 0) {
    if (exponent % 2 == 1) {
      result = (result * base) % modulus;
    }
    exponent = exponent / 2;
    base = (base * base) % modulus;
  }
  return result;
}

template <typename T>
inline T mod_inverse(T a, T modulus) {
  return mod_power(a, modulus - 2, modulus);
}

} // namespace mirage
