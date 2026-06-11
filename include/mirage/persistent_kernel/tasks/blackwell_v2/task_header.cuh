/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

// blackwell_v2 task implementations, bundled so the v2 megakernel's
// codegen-emitted role dispatch calls resolve. linear and rmsnorm have their
// own namespaces (kernel::linear_v2 / kernel::rmsnorm_v2); the rest share
// kernel::v2 — all distinct from the v1 (kernel::) versions.
#include "mirage/persistent_kernel/runtime_header.h"
#include "linear_sm100_v2.cuh"        // kernel::linear_v2 (Channel-based)
#include "rmsnorm_v2.cuh"             // kernel::rmsnorm_v2
#include "rotary_embedding_v2.cuh"    // kernel::v2
#include "norm_sm100.cuh"             // kernel::v2
#include "attention_sm100.cuh"        // kernel::v2
#include "argmax_sm100.cuh"           // kernel::v2
#include "silu_mul_v2.cuh"            // kernel::v2
#include "embedding_v2.cuh"           // kernel::v2
