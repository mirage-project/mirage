# MPK TMA Settings Instruction

## Overview

**Tensor Memory Accelerator (TMA)** is a new component in **Hopper/Blackwell** Architecture GPUs to move tiles between **global memory (GMEM)** and **shared memory (SMEM)** using `cp.async.bulk.tensor.5d` with an **mbarrier** for completion. Unlike `ldmatrix` or `cp.async.cg` in Ampere GPUs where all threads work cooperatively, only one thread issues the `cp.async.bulk.tensor` instructions. In mpk settings, we support the following 2-4 dimensional `TMA` usage.

* `tma_2d<T, …>` — 2-D tiles (row, col)
* `tma_3d<T, …>` — 3-D tiles (depth, row, col)
* `tma_4d<T, …>` — 4-D tiles (outermost, depth, row, col)

Each helper encodes a `CUtensorMap` on the host (shape, strides, swizzle) and exposes device functions to issue async copies and stores per tile.

## Prerequisites

* Compile for **Hopper** (`-arch=sm_90a`) and **define** `MIRAGE_GRACE_HOPPER`.
* mbarrier object lives in `__shared__` and is 8-byte aligned.


## Key Parameters

* `T`: element type (default: bfloat16).
* `B`: **swizzle class**: `1→32B`, `2→64B`, `3→128B`. (Choose based on bank-conflict behavior; 128B is generally effective.)
* `GMEM_*`: logical sizes of the source/destination tensor in **elements**.
* `SMEM_*`: the **tile** you move per operation in **elements** (each dim ≤ 256).
* `GMEM_STRIDE_*`: strides in **elements**. Non-innermost byte strides must be **16B-aligned**.
* `SMEM_REPEAT_ROW_/COL_`: intra-call repetition for **tma.cp_aysnc** or **tma.store_async**. A typical usage is to set `SMEM_REPEAT_COL` to **SMEM_COL/64**. This is because `TMA` hardware can only transfer 64 bfloat16 elements under the **swizzle<3,3,3>** (**swizzle<3,3,3>** means 8 elements as a new unit, each row contain 8 units, 8 rows in total, and thus 64 elements in each row).
* `SMEM_STRIDE`: offsets in **elements** between each  `SMEM_REPEAT_ROW_/COL_` copy.

**Coordinate order to pass at tma_cp_async/tma_store_async call** (matches the `CUtensorMap` encoding):

* 2-D: `{col, row}`
* 3-D: `{col, row, depth}`
* 4-D: `{col, row, depth, outermost}`

## TMA Completion Mechanism
### TL;DR

* **Load (`tma_cp_async`)**: *declare bytes* → *issue copy* → *`wait(mbar[slot], phase)`*.
* **Store (`tma_store_async`)**: *issue store(s)* → *`store_commit_group`* → *`store_async_wait`*.

---
### GMEM → SMEM (`tma_cp_async` with **mbarrier**)

Loads are issued with `cp.async.bulk.tensor.*.tile.mbarrier::complete_tx`.
Because completion is signaled **to an mbarrier**, you must:

1. **Set the expected transaction bytes** on the mbarrier (exactly the number of bytes you will fetch):

```cpp
size_t bytes = tile_elems * sizeof(T); 
set_barrier_transaction_bytes(mbar[slot], bytes);    
```

2. **Launch the TMA copy** (coords ordered as encoded in the CUtensorMap):

```cpp
tma.tma_cp_async(mbar, smem_tile_ptr, coords);
```

3. **Wait on the mbarrier** before consuming data in SMEM:

```cpp
wait(mbar[slot], phase);
```

**Note:** the hardware will “arrive” the mbarrier only after all bytes you declared have been transferred to SMEM. If the byte count is wrong, the wait will not complete correctly and could wait forever.

---

### SMEM → GMEM (`tma_store_async` with **commit/wait group**)

Stores are issued with `cp.async.bulk.tensor.*.global.shared::cta.bulk_group`.
This path does **not** use an mbarrier. Instead, it uses the **bulk-group**:

1. **Launch the async store(s)**:

```cpp
tma.tma_store_async(smem_tile_ptr, coords);
```

2. **Commit the current group** of stores:

```cpp
kernel::tma::store_commit_group();
```

3. **Wait for the committed group(s)** to finish:

```cpp
kernel::tma::store_async_wait<0>(); // cp.async.bulk.wait_group 0  (drains all)
kernel::tma::async_proxy_fence();   // fence.proxy.async.shared::cta (visibility)
```

**Note:** store completion is tracked by “groups” rather than an mbarrier. You explicitly bound a set of store ops with `commit_group`, then block on them with `wait_group<N>` (using `N=0` to drain all outstanding groups). The proxy fence finalizes visibility ordering before you reuse SMEM or depend on GMEM results.

---


## Minimal Usage Example
 **Transfer tile of shape *(SMEM_TILE_ROW=64, SMEM_TILE_COL=128)* from global memory of shape *(GMEM_ROW=4096, GMEM_COL=1024)* to shared memory**
### Host 

```cpp

using TMA_2D = kernel::tma::tma_2d<
  /*T=*/T, 
  /*B=*/3, 
  /*M=*/3, 
  /*S=*/3,
  /*GMEM_ROW=*/4096, 
  /*GMEM_COL=*/1024,
  /*SMEM_ROW=*/64, 
  /*SMEM_COL=*/64,
  /*GMEM_STRIDE_ROW=*/4096, // usually C if global tensor is Row-Major 
  /*GMEM_STRIDE_COL=*/1,
  /*SMEM_REPEAT_ROW=*/1, 
  /*SMEM_REPEAT_COL=*/2, // (128+63)/64
  /*SMEM_STRIDE=*/64*64 // each SMEM_REPEAT_COL copy is 64*64 elements continuously in raw memory
  >;

TMA_2D tma(gmem_ptr);          // encodes CUtensorMap on host and pass into kernels
```

### Device (issue async copy → compute → async store)

```cpp
extern __shared__ char smem[];
__shared__ kernel::tma::Barrier mbar;

// 1) Tell the barrier how many bytes to expect for this transaction.
size_t bytes = TileR * TileC * sizeof(T);
set_barrier_transaction_bytes(mbar[slot_residual], bytes);

// 2) GMEM -> SMEM (coords are {col, row} in elements):
int coords[2] = { ITER_COL * TileC, ITER_ROW * TileR };
tma.tma_cp_async(mbar, smem_ptr, coords);

// 3) Wait for producing completion before consuming:
wait(mbar[slot], phase); // wait for `slot` to flip on `phase`, threads wait until **internal phase != phase**

// ... computation on 'tile' ...

// 4) SMEM -> GMEM store back (same coords):
tma.tma_store_async(tile, coords); // issue tma_store_async
kernel::tma::store_commit_group(); // commit point
kernel::tma::store_async_wait<N>(); // wait store_async, at most N on the fly 
kernel::tma::async_proxy_fence(); // make sync proxy's changes visable to async proxies 
```

## TMA Requirements
> TMA has a lot of requirements, there are many assertions in each TMA_Nd.cuh files, including
* **GMEM base address** is **16B aligned**.
* All **byte strides (except the first element size)** are **multiples of 16B** and `< 2^40`.
* Every SMEM box dimension is in **\[1, 256]**; SMEM box strides are in **\[1, 8]**.



