.. _linear-kernel:

``linear_kernel``
=================

Introduction
------------

The ``linear_kernel`` is a highly optimized device-level CUDA kernel template for performing **linear transformations** of the form:

.. math::

   Y = XW + R

where ``X`` is the input batch, ``W`` is the weight matrix, and ``R`` is an optional residual term. It leverages shared memory tiling, asynchronous copy pipelining, warp-level MMA (matrix-multiply-accumulate) primitives, and flexible memory layouts to achieve high throughput on NVIDIA Tensor Cores.

This kernel is central to implementing fully connected (linear) layers in large-scale machine learning workloads.

----

Template Parameters
-------------------

* ``T``: Data type of input/output elements (e.g., ``bfloat16``, ``half``, ``float``).
* ``BATCH_SIZE``: Number of rows in the input batch.
* ``OUTPUT_SIZE``: Number of output features (columns of the weight matrix).
* ``REDUCTION_SIZE``: Inner dimension size (columns of input, rows of weight).
* ``O_STRIDE`` *(default = ``OUTPUT_SIZE``)*: Stride between consecutive output rows.
* ``K_PIPE_MAX`` *(default = 3)*: Depth of the asynchronous pipeline for double-/triple-buffered shared memory loads.

.. 
   ## Function Signature (commented out in original)

   .. code-block:: cpp

      __device__ __forceinline__
      void linear_kernel(void const *input_ptr,
                         void const *weight_ptr,
                         void const *residual_ptr,
                         void *output_ptr,
                         bool residual = true);

----

Architecture
------------

The kernel is structured in **five main phases**:

1. **Shared Memory Partitioning**  
   Shared memory is divided into regions for zero-fill buffers, pipelined input/weight staging, residuals, intermediate accumulators, and final output. Each buffer is laid out using specialized row/column-major templates (``smem_row``, ``smem_col``) with optional swizzling to reduce bank conflicts.

2. **Asynchronous Copy and Double Buffering**  
   Input and weight tiles are prefetched into shared memory using ``cp.async`` instructions, with a triple-buffered scheme (``K_PIPE_MAX = 3``) to overlap global memory latency with computation.

3. **Matrix Multiplication on Tensor Cores**  
   The kernel issues warp-level MMA instructions (``mma_m16n16k16_bf16bf16bf32``) using fragments loaded via ``ldsm``.

   * Accumulators are stored in registers (``s_frag``) as float32 to preserve precision.
   * A zero buffer handles out-of-bound fragments when dimensions are not divisible by the MMA tile size.

4. **Intermediate Storage and Reduction**  
   Partial results are written back into shared memory.

5. **Residual Addition and Write-back**  
   The final result is optionally fused with the residual tensor before being written to global memory (``output_dmem``). If ``residual = false``, the addition is skipped.

----

Algorithms
----------

1. **Pipelined Copy**
^^^^^^^^^^^^^^^^^^^^^

* Each iteration copies one input tile and one weight tile into shared memory while the previous tile is being consumed by MMA.
* ``cp_async_fence`` and ``cp_async_wait<K>`` synchronize the pipeline.
* Shared memory buffers are rotated in a circular manner across ``K_PIPE_MAX``.

2. **Warp-level MMA Mapping**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Warps are arranged in a 2D grid (``warp_row``, ``warp_col``) across the output tile.
* Each warp computes a sub-tile of size ``16 × 16``.
* The mapping ensures coalesced global memory loads and avoids bank conflicts in shared memory.

3. **Register-based Accumulation**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Each warp maintains FP32 accumulators in registers.
* This reduces shared memory traffic, improves latency, and maximizes Tensor Core utilization.
* After completing the loop, fragments are written back to shared memory.

4. **Residual Fusion**
^^^^^^^^^^^^^^^^^^^^^^

* Residuals are preloaded into shared memory and added elementwise to the accumulator output before final write-back.
* This eliminates the need for an extra global memory read in a separate kernel.

----

Memory Layout
-------------

The kernel relies on flexible memory layout abstractions (``dmem_row_const``, ``dmem_col_const``, ``smem_row``, ``smem_col``) to adapt to both row-major and column-major representations.

* Input (``A``) is stored row-major.
* Weight (``B``) is stored column-major.
* Intermediate accumulators and output (``C``) are stored row-major.
* A **swizzled shared memory layout** avoids bank conflicts for ``ldmatrix``.

----

Problems and Solutions
----------------------

1. **Non-divisible MMA Sizes**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``BATCH_SIZE`` or ``OUTPUT_SIZE`` is not divisible by the ``16×16`` MMA tile size, threads may attempt to load invalid addresses.

* **Solution**: Threads falling out of bounds load from a shared **zero buffer**, ensuring correctness without branching.

2. **Synchronization Tradeoff**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pipeline minimizes global memory latency but requires periodic synchronizations (``__syncthreads``).

* A balance is struck between synchronization overhead and shared memory peak usage.

.. 
   ## Usage Example (commented out in original)

   .. code-block:: cpp

      using T = kernel::bfloat16;
      constexpr int B = 64;   // batch size
      constexpr int O = 128;  // output features
      constexpr int R = 512;  // reduction dimension

      __global__ void launch_linear(void const* input,
                                    void const* weight,
                                    void const* residual,
                                    void* output) {
          kernel::linear_kernel<T, B, O, R>(input, weight, residual, output, true);
      }

   This will compute:

   .. math::

      Y_{64 \times 128} = X_{64 \times 512} \cdot W_{512 \times 128} + R_{64 \times 128}

----

How shared memory buffers rotate across pipeline stages
-------------------------------------------------------

1) Triple-buffer rotation across pipeline stages (``K_PIPE_MAX = 3``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mermaid::

   gantt
       title Triple-buffer cp.async pipeline (idealized)
       dateFormat  HH:mm:ss
       axisFormat  %S
       todayMarker off

       %% Iteration k prefetches k+2, computes k; k+1 is waiting/ready
       section Iter k
       cp.async A[k+2],B[k+2] -> buf#2      :active, 00:00:00, 3s
       cp.async_fence & wait<2>             : 00:00:03, 2s
       set smem ptrs to buf[idx_use=k%3]    : 00:00:05, 1s
       __syncthreads()                      : 00:00:06, 1s
       MMA on A[k],B[k] (buf#idx_use)       : 00:00:07, 4s
       writeback / partial reduce (opt)     : 00:00:11, 2s

       section Iter k+1
       cp.async A[k+3],B[k+3] -> buf#(k+3)%3 : 00:00:13, 3s
       cp.async_fence & wait<2>              : 00:00:16, 2s
       set smem ptrs to buf[(k+1)%3]         : 00:00:18, 1s
       __syncthreads()                       : 00:00:19, 1s
       MMA on A[k+1],B[k+1]                  : 00:00:20, 4s
       writeback / partial reduce (opt)      : 00:00:24, 2s

.. .. mermaid::

..    flowchart LR
..      %% Three-buffer rotation across iterations (K_PIPE_MAX = 3)

..      subgraph Iteration_k
..        U0["idx_use = k%3<br/>compute A[k], B[k]"]
..        R0["idx_ready = (k+1)%3<br/>holds A[k+1], B[k+1]"]
..        F0["idx_fill = (k+2)%3<br/>cp.async A[k+2], B[k+2]"]
..      end

..      subgraph Iteration_k_plus_1
..        U1["idx_use = (k+1)%3<br/>compute A[k+1], B[k+1]"]
..        R1["idx_ready = (k+2)%3<br/>holds A[k+2], B[k+2]"]
..        F1["idx_fill = (k+3)%3<br/>cp.async A[k+3], B[k+3]"]
..      end

..      U0 -->|rotate| F1
..      R0 -->|rotate| U1
..      F0 -->|rotate| R1

.. .. mermaid::

..    sequenceDiagram
..        autonumber
..        participant L as Warp lanes
..        participant S as Shared Mem (buf[idx_use])
..        participant G as Global Mem

..        Note over L,G: Iter k
..        L->>G: cp.async A[k+2], B[k+2] → buf[idx_fill=(k+2)%3]
..        L->>G: cp.async_fence()
..        L->>G: cp.async_wait<2>()
..        L->>S: set SMEM ptrs to buf[idx_use=k%3]
..        L->>L: __syncthreads()
..        L->>S: ldsm(A[k]), load_row_8x8b(B[k])
..        L->>L: MMA m16n16k16 accumulate (C += A*B)
..        L->>S: writeback partials (and/or reduce)
..        Note over L,G: rotate: idx = (idx + 1) % 3

.. note::

   * The “zero buffer” is used whenever a thread’s ``ldmatrix`` address would go OOB (tail tiles).
   * ``cp_async_wait<2>()`` matches the “distance” between issue and consume when ``K_PIPE_MAX=3``.
   * ``__syncthreads()`` is placed after pointer rotation to make the new window visible to all threads.

2) Warp ↔ output-tile mapping (SM80 16×16×16 MMA, 4 warps total)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``OUTPUT_ATOM_SIZE ≤ 128``, the kernel uses **4 warps** arranged as:

.. code-block:: text

   NUM_WARPS_N = min(OUTPUT_ATOM_SIZE/16, 4)
   NUM_WARPS_K = 4 / NUM_WARPS_N

Warps form a (``warp_row`` × ``warp_col``) grid over the C-tile (``BATCH_SIZE × OUTPUT_ATOM_SIZE``).

**Example:** ``OUTPUT_ATOM_SIZE = 128`` → ``NUM_WARPS_N = 4``, ``NUM_WARPS_K = 1`` (grid: 1 row × 4 cols)

.. code-block:: text

   C tile (BATCH_SIZE x 128), decomposed into 16x16 subtiles:

            n=0..7   n=8..15  n=16..23  n=24..31  n=32..39  n=40..47  n=48..55  n=56..63 ...
   m=0..15   [W0]     [W1]      [W2]      [W3]     [W0]      [W1]      [W2]      [W3]   ...
   m=16..31  [W0]     [W1]      [W2]      [W3]     [W0]      [W1]      [W2]      [W3]   ...
      ...     ...       ...       ...       ...      ...       ...       ...       ...

**Warp IDs and indices**

.. code-block:: text

   warp_idx = 0..3
   warp_row = warp_idx >> log2(NUM_WARPS_N)
   warp_col = warp_idx &  (NUM_WARPS_N - 1)

**Lane decomposition inside a warp (per MMA 16×16×16)**

.. code-block:: text

   lane = 0..31
   m_row  = (lane & 0xF)                       // 0..15
   n_col  = ((lane >> 4) << 3) + (lane & 0x7)  // packs 8 elems for ldmatrix

Each warp iterates ``NUM_ITERS_K`` times across K, accumulating FP32 partial sums in registers ``s_frag[m][n][8]``, then writes out two bf16 values per step into the per-warp intermediate smem, followed by optional row-wise reduction if ``NUM_WARPS_K > 1``.

----

Quantization
------------

Introduction
^^^^^^^^^^^^

``linear_kernel_fp8_weight`` computes

.. math::

   Y = X \, \mathrm{dequant}(W_{\text{fp8}}, S) + R

where the weights are stored in **FP8** with **groupwise FP16 scales**. The kernel preserves the original pipeline (triple-buffered ``cp.async``, Tensor Core MMA, shared-memory epilogue) while replacing the B-operand path with **FP8 load + register-space dequant + per-group scaling**.

Compared to the FP16/BF16 baseline (``linear_kernel``), this design cuts GMEM bandwidth for weights by ~2×, often offsetting FP8→BF16 convert cost—especially at large ``REDUCTION_SIZE``.

----

Differences from baseline
^^^^^^^^^^^^^^^^^^^^^^^^^

1) FP8 weight storage + groupwise scaling
"""""""""""""""""""""""""""""""""""""""""

* **Quantization scheme**

  * Weight tensor :math:`W\in\mathbb{R}^{K\times N}` is stored as **FP8 codes** plus **FP16 scale** per **(WB\_K × WB\_N)** block.
  * Defaults: ``WB_K = 128``, ``WB_N = 128``.
  * Scale grid size:

    .. math::

       \text{K_BLOCKS}=\left\lceil \frac{K}{\text{WB_K}} \right\rceil,\quad 
       \text{N_BLOCKS}=\left\lceil \frac{N}{\text{WB_N}} \right\rceil.

* **Dequantization** (done in registers right before MMA):

  .. math::

     W_{\text{bf16}} = S_{(k\_blk,n\_blk)} \cdot \mathrm{decode\_fp8}(W_{\text{fp8}}).

  We load eight FP8 codes, convert to four BF16 pairs, and **scale each BF16** by ``current_scale`` (specified by the current ``(k_blk,n_blk)``).

2) FP8-aware copy path and layouts
""""""""""""""""""""""""""""""""""

* **FP8 chunking** uses ``CHUNK_SIZE_W = 16 / sizeof(FP8) = 16`` bytes per transaction (still 16B aligned), yielding:

  * ``NUM_CHUNKS_B_FP8 = (TILE_SIZE * OUTPUT_ATOM_SIZE) / CHUNK_SIZE_W``
  * ``CHUNKS_PER_COL_B_FP8 = TILE_SIZE / CHUNK_SIZE_W``

* **Shared-memory layout** for weights uses a slightly different swizzle tuple to match FP8 access:

  * Baseline: ``smem_col<T, 3,3,3, ...>``
  * FP8 path: ``smem_col<FP8, 3,4,3, ...>``

**Reason for the change in swizzle mode:**

Shared-memory layout for weights (FP16/BF16 vs FP8)
"""""""""""""""""""""""""""""""""""""""""""""""""""

When staging weights into shared memory, we use the ``smem_col`` layout wrapper:

.. code-block:: cpp

   template <typename T, int B, int M, int S, size_t ROW, size_t COL, size_t STRIDE>
   struct smem_col { ... };

The three integer template parameters ``B, M, S`` determine how logical indices map into physical offsets inside a swizzled banked layout:

* ``M`` controls the size of the **innermost contiguous run** of elements (how many elements a 128-bit transaction covers).
* ``S`` controls the XOR swizzle width (columns XOR’d with the row index to reduce bank conflicts).
* ``B`` controls upper-level blocking of the tensor in shared memory.

Baseline FP16 / BF16 path
"""""""""""""""""""""""""

For 16-bit weights, each element is 2 B. To cover a 16-byte memory transaction, we need ``16/2 = 8`` contiguous elements.

Thus we set:

.. code-block:: cpp

   using WeightSmem = smem_col<T, 3,3,3, TILE_SIZE, OUTPUT_ATOM_SIZE, TILE_SIZE>;
                           //       ^M=3 → 2^3 = 8 contiguous elems = 16 B

FP8 path
""""""""

For 8-bit weights, each element is 1 B. To cover a 16-byte transaction, we need ``16/1 = 16`` contiguous elements.

Thus we must increase ``M`` from 3 → 4:

.. code-block:: cpp

   using WeightSmem = smem_col<FP8, 3,4,3, TILE_SIZE, OUTPUT_ATOM_SIZE, TILE_SIZE>;
                           //        ^M=4 → 2^4 = 16 contiguous elems = 16 B

If we mistakenly left ``M=3``, each contiguous run would only cover 8 bytes. Since the kernel processes 16 elements per round (to match the 128-bit load granularity), the second half of each 16-byte vector would pick up the wrong neighbors after the XOR swizzle, leading to **corrupted fragments**.

By setting ``M=4``, every 16 FP8 codes are guaranteed to be laid out consecutively in shared memory, aligning exactly with 128-bit loads.

3) Register-space FP8 → BF16 convert + scaling (B-operand)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Inside the MMA loop, per warp:

.. code-block:: cpp

   FP8 q_weight[8];
   load_row_8x8b(weight_smem(base_row + dr, base_col + dcp),
                 *reinterpret_cast<uint64_t*>(q_weight));

   // 4×(fp8x2 -> bf16x2), then multiply each bf16 by current_scale
   b_frag[0] = fp8x2_to_bf16(reinterpret_cast<uint16_t*>(q_weight)[0]);
   (reinterpret_cast<T*>(b_frag))[0] *= current_scale;
   (reinterpret_cast<T*>(b_frag))[1] *= current_scale;
   ...
   b_frag[3] = fp8x2_to_bf16(reinterpret_cast<uint16_t*>(q_weight)[3]);
   (reinterpret_cast<T*>(b_frag))[6] *= current_scale;
   (reinterpret_cast<T*>(b_frag))[7] *= current_scale;

**Technique**: load 8 FP8 as a packed ``uint64``, perform **pairwise** decode to BF16, and **scale in place** in the ``b_frag`` register view. This keeps the convert + scale fully in registers, avoiding smem traffic.

4) Per-tile scale selection
"""""""""""""""""""""""""""

* Scales are staged once into smem (``ScaleSmem``) from ``ScaleDmem``:

  .. code-block:: cpp

     // thread-block loads 16B stripes of the scale grid
     if (threadIdx.x < thread_count) {
       load_smem(weight_scale_smem(row, col), weight_scale_dmem(row, col));
     }
     // (Recommended) __syncthreads();  // ensure scale tiles visible

* For MMA iteration ``for_idx`` and output atom index ``output_atom_idx``, we compute:

  .. code-block:: cpp

     int row_scale = (for_idx * TILE_SIZE) / WB_K;
     int col_scale = (output_atom_idx * OUTPUT_ATOM_SIZE) / WB_N;
     current_scale = weight_scale_smem.at(row_scale, col_scale);

This maps the **(k,n) tile** being multiplied to its **(K-block, N-block)** scale.

5) rest unchanged
"""""""""""""""""

* **Triple-buffered cp.async** pipeline with ``K_PIPE_MAX = 3`` (identical structure).
* **A-operand** copies and ``ldsm`` are unchanged.
* **Accumulator** stays FP32 in registers (``s_frag``), identical write-back, optional inter-warp row reduction, and fused residual add.

----

Detailed design
^^^^^^^^^^^^^^^

Memory plan (smem)
""""""""""""""""""

.. mermaid::

   flowchart LR
     %% Shared memory layout (contiguous, left→right = low→high address)
     classDef seg fill:#e8f0fe,stroke:#203864,stroke-width:1px,color:#111,rx:6,ry:6;

     Z["zero (8 * T)"]:::seg
     A["A-buffers (K_PIPE_MAX * BATCH_SIZE * TILE_SIZE * T)"]:::seg
     B["B-buffers (K_PIPE_MAX * TILE_SIZE * OUTPUT_ATOM_SIZE * FP8)"]:::seg
     R["Residual R (BATCH_SIZE * OUTPUT_ATOM_SIZE * T)"]:::seg
     M["MMA_intermediate (NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE * T)"]:::seg
     C["Output C (BATCH_SIZE * OUTPUT_ATOM_SIZE * T)"]:::seg
     S["Scale grid (K_BLOCKS * N_BLOCKS * T)"]:::seg

     Z --> A --> B --> R --> M --> C --> S

Copy + pipeline (B path)
""""""""""""""""""""""""

* **GMEM→SMEM** uses ``load_smem<FP8>(...)`` with 16-byte vectors (same 128-bit lanes).
* Pipeline fences & waits mirror the FP16 version (``cp_async_fence(); cp_async_wait<K_PIPE_MAX-1>();``).
* **Pointer rotation** unmodified (circular modulo ``K_PIPE_MAX``).

Simplified FP8→BF16 Conversion
""""""""""""""""""""""""""""""

.. mermaid::

   flowchart TB
     %% Two FP8 (E4M3) packed in 16 bits → expand to two BF16 in 32 bits

     subgraph FP8x2_input["FP8x2 (uint16_t)"]
       direction LR
       f1s["s1"]:::s --> f1e["e1 e1 e1 e1"]:::e --> f1m["m1 m1 m1"]:::m
       f0s["s0"]:::s --> f0e["e0 e0 e0 e0"]:::e --> f0m["m0 m0 m0"]:::m
     end

     subgraph Expand_and_Map["Bit remap (insert 8-bit gaps)"]
       direction LR
       g1s["s1"]:::s --> g1e["e1×4"]:::e --> g1m["m1×3"]:::m --> gap1["(8 zeros)"]:::z
       g0s["s0"]:::s --> g0e["e0×4"]:::e --> g0m["m0×3"]:::m --> gap0["(8 zeros)"]:::z
     end

     subgraph BF16x2_output["BF16x2 (uint32_t)"]
       direction LR
       b1s["S1"]:::s --> b1e["E1 (8 bits)"]:::E --> b1m["M1 (7 bits)"]:::M
       b0s["S0"]:::s --> b0e["E0 (8 bits)"]:::E --> b0m["M0 (7 bits)"]:::M
     end

     f1s --> g1s --> b1s
     f1e --> g1e --> b1e
     f1m --> g1m --> b1m
     f0s --> g0s --> b0s
     f0e --> g0e --> b0e
     f0m --> g0m --> b0m

     note1["rebias exponent: + (BF16_bias - FP8_bias) = +120\nshift exponent <<7, mantissa <<4; sign <<8"]:::note
     g1e -.-> note1
     g0e -.-> note1

     classDef s fill:#fde2e1,stroke:#b23,rx:4,ry:4,color:#111;
     classDef e fill:#e8f4ff,stroke:#246,rx:4,ry:4,color:#111;
     classDef m fill:#e9f7ef,stroke:#273,rx:4,ry:4,color:#111;
     classDef z fill:#f6f6f6,stroke:#999,rx:4,ry:4,color:#444,stroke-dasharray:3 3;
     classDef E fill:#cfe2ff,stroke:#246,rx:4,ry:4;
     classDef M fill:#cdeccd,stroke:#273,rx:4,ry:4;
     classDef note fill:#fff7cc,stroke:#b99,rx:4,ry:4,color:#333;

Overview
""""""""

In the FP8 quantized linear kernel, the function ``fp8x2_to_bf16`` performs **lightweight decoding** of two FP8 values into two BF16 values packed in a 32-bit register. Unlike a full IEEE-compliant conversion, this function **ignores NaN, infinity, subnormal, and overflow handling**, focusing purely on the fast reconstruction of sign, exponent, and mantissa within normal range.

Code
""""

.. code-block:: cpp

   __device__ __forceinline__ uint32_t fp8x2_to_bf16_v2(uint16_t v) {
     uint32_t vv =
         (v & 0xFF00u) << 8 | (v & 0x00FFu); // insert 8-bit gap for 2nd FP8
     uint32_t m = vv & 0x00070007u;          // extract mantissa (3 bits)
     uint32_t e = (vv >> 3) & 0x000F000Fu;   // extract exponent (4 bits)
     e = ((e + 0x00780078) << 7) | (m << 4); // rebias exponent for BF16
     vv = (vv & 0x00800080u) << 8;           // move sign bit into BF16 position
     return vv | e;                           // merge sign/exponent/mantissa
   }

Explanation
"""""""""""

Each FP8 value follows the **E4M3** format (1-bit sign, 4-bit exponent, 3-bit mantissa):

.. code-block:: text

   [ s | eeee | mmm ]

The function packs two such FP8 codes (16 bits total) into a single 32-bit register containing two BF16 values:

.. code-block:: text

   FP8x2:  [ s1 eeee mmm | s0 eeee mmm ]
   BF16x2: [ s1 (rebased exp) mantissa | s0 (rebased exp) mantissa ]

Key operations:

* **Bit remapping**: ``(v & 0xFF00) << 8 | (v & 0x00FF)`` spaces the two 8-bit FP8s by inserting 8 bits between them.
* **Mantissa extraction**: ``m = vv & 0x00070007u``
* **Exponent extraction and rebiasing**: ``e = ((e + 0x0078'0078) << 7)``
* **Sign alignment**: ``(vv & 0x00800080u) << 8``
* **Merge**: the final ``vv | e`` produces two valid BF16 bit patterns.

Simplification Rationale
""""""""""""""""""""""""

Full IEEE-style handling would require detecting NaN/Inf/subnormals with branching, which is costly and harms warp coherence. In practice, quantized weights are clipped to normal ranges, so the simplified path keeps per-element cost minimal.

Performance Benefit
"""""""""""""""""""

* Without simplification: ~25–30 integer ops with conditionals.
* With simplification: ~6 ops, 0 branches → overhead < 2% of total kernel latency.

Accuracy Implications
"""""""""""""""""""""

* NaN, Inf, and subnormals are not preserved; they map to large finite BF16 values.
* Quantized neural weights rarely use such codes due to clipping.

----

Performance Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Bandwidth**: Weights are half the bytes; K-heavy GEMMs are typically memory-bound on B.  
2. **Locality**: Convert+scale in registers right before MMA maximizes reuse and avoids extra smem cycles.  
3. **Bank behavior**: A tailored ``smem_col<FP8, 3,4,3,...>`` swizzle ensures that every 16 FP8 codes are laid out contiguously in shared memory, matching the 128-bit vectorized copy requirements.

Accuracy & configurability
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Group size** (``WB_K``, ``WB_N``) trades accuracy vs. metadata overhead.
* **Decode** path assumes a consistent FP8 format (e.g., **E4M3**).
* Accumulation is still FP32.

.. 
   ## Edge cases & correctness (omitted in rendered page; kept for reference)

   * Tail tiles: zero-buffer strategy for ``ldsm``.
   * Scale coverage: ensure preload covers the tile grid; barrier before first use.
   * Alignment: keep 16-byte alignment for shared buffers; mind aliasing with ``reinterpret_cast``.

API & template changes
^^^^^^^^^^^^^^^^^^^^^^

* New template parameter: ``typename FP8 = __uint8_t``.
* New GMEM inputs: ``weight_fp8_ptr`` (FP8), ``weight_scale_ptr`` (FP16).
* Other parameters match the baseline.

Situations where FP8 quantization helps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Large K** and **moderate N** where B-bandwidth dominates.
* On SM80 (no native FP8 MMA): BF16 MMA + register dequant is typically best.
* On newer arch with native FP8 MMA: prefer native FP8; pipeline remains similar.
