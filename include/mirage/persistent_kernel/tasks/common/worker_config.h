/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

constexpr int NUM_THREADS = 128;
constexpr int NUM_THREADS_PER_WARP = 32;
constexpr int NUM_WARPS = 4;
constexpr int WARPGROUP_WARPS = 4;

constexpr float inf = 5e4;
// TODO: only setting this for Hopper can have compilation issues on blackwell
// and presumably ampere
#if defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_GRACE_BLACKWELL)
constexpr int WORKER_NUM_THREADS = 256;   // Grace Hopper setting
constexpr int CONSUMER_NUM_THREADS = 128; // Grace Hopper setting
#endif
