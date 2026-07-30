#include "mirage/kernel/device_memory_manager.h"
#include "mirage/utils/math_utils.h"

namespace mirage {
namespace kernel {

using namespace mirage::type;
using namespace mirage::config;

DeviceMemoryManager *DeviceMemoryManager::singleton = nullptr;

#ifdef MIRAGE_FINGERPRINT_USE_CPU
DeviceMemoryManager::DeviceMemoryManager() {
  num_devices = 1; // Default to 1 device for non-CUDA environments
  auto initizalize_exp_lookup_table =
      [](FPType *table, int size, int base, int modulus) {
        table[0] = 1;
        for (int i = 1; i < size; ++i) {
          table[i] = (table[i - 1] * base) % modulus;
        }
      };

  auto initizalize_div_lookup_table = [](FPType *table, int size, int modulus) {
    table[0] = 1;
    for (int i = 1; i < size; ++i) {
      table[i] = mod_inverse(i, modulus);
    }
  };

  auto initialize_sqrt_lookup_table = [](FPType *table, int size, int modulus) {
    assert(modulus % 4 == 3 &&
           "Modulus must be of the form 4k + 3 for square roots to exist");
    for (int i = 0; i < size; ++i) {
      table[i] = mod_power(i, (modulus + 1) / 4, modulus);
    }
  };

  exp_lookup_table = new FPType[FP_Q];
  div_p_lookup_table = new FPType[FP_P];
  div_q_lookup_table = new FPType[FP_Q];
  sqrt_p_lookup_table = new FPType[FP_P];
  sqrt_q_lookup_table = new FPType[FP_Q];

  initizalize_exp_lookup_table(exp_lookup_table, FP_Q, FP_EXP_BASE, FP_P);
  initizalize_div_lookup_table(div_p_lookup_table, FP_P, FP_P);
  initizalize_div_lookup_table(div_q_lookup_table, FP_Q, FP_Q);
  initialize_sqrt_lookup_table(sqrt_p_lookup_table, FP_P, FP_P);
  initialize_sqrt_lookup_table(sqrt_q_lookup_table, FP_Q, FP_Q);

  for (int i = 0; i < num_devices; ++i) {
    fp_base_ptr[i] = new char[MAX_DMEM_FP_SIZE];
  }
  stensor_fp_base_ptr =
      new char[MAX_SMEM_FP_SIZE * MAX_NUM_THREADBLOCKS_PER_KERNEL];
}

DeviceMemoryManager::~DeviceMemoryManager() {
  delete[] exp_lookup_table;
  delete[] div_p_lookup_table;
  delete[] div_q_lookup_table;
  delete[] sqrt_p_lookup_table;
  delete[] sqrt_q_lookup_table;

  for (int i = 0; i < num_devices; ++i) {
    delete[] fp_base_ptr[i];
  }
  delete[] stensor_fp_base_ptr;
}

DeviceMemoryManager *DeviceMemoryManager::get_instance() {
  if (singleton == nullptr) {
    singleton = new DeviceMemoryManager();
  }
  return singleton;
}
#endif

} // namespace kernel
} // namespace mirage