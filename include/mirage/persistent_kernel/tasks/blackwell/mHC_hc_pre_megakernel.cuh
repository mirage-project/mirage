/* Copyright 2026 CMU
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

// mHC hc_pre v3: persistent megakernel fusing rmsnorm + linear + tail
// (K2+K3+K4) into a single launch. Designed for the 148-CTA / 256-thread
// MPK target.
//
// Stage flow (each stage is a grid-stride loop within all 148 CTAs):
//   Stage 1: rmsnorm        (fp32 x -> bf16 x_norm in gmem scratch)
//   Stage 2: linear         (bf16 x_norm @ W^T -> bf16 mixes_pad in gmem)
//   Stage 3: tail v2        (bf16 mixes_pad -> f_pre/h_post/comb)
//
// Cross-stage synchronization uses cooperative_groups::this_grid().sync().
// Requires the kernel to be launched with cudaLaunchCooperativeKernel.
