// common.h - Some common definitions for the transpiler
#pragma once

#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"
#include "mirage/kernel/device_tensor.h"
#include "mirage/threadblock/smem_tensor.h"

namespace mirage {
namespace transpiler {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;
using dguid_t = decltype(kn::DTensor::guid);
using sguid_t = decltype(tb::STensor::guid);

}
}
