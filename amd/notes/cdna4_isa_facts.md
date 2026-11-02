# CDNA4 (gfx950, MI355X) hardware facts for the static megakernel port

Collected 2026-10-09. No GPU was used. Labels used below:
- **[ISA p.N]** = AMD "CDNA4 Instruction Set Architecture Reference Guide", 5-August-2025, printed page N
  (https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf).
- **[LLVM]** = https://llvm.org/docs/AMDGPUUsage.html, section "Memory Model GFX942" and table
  "AMDHSA Memory Model Code Sequences GFX942" (fetched 2026-10-09).
- **[asm]** = what ROCm 7.2.4 hipcc (`rocm/dev-ubuntu-24.04:7.2.4`, LLVM 22) emits for `--offload-arch=gfx950 -O3`;
  test sources are in the session scratchpad (`hip/mm.hip`, `hip/mf.hip`, `hip/lds.hip`).
- **[drv]** = Linux `drivers/gpu/drm/amd/amdgpu/gmc_v9_0.c` and `amdgpu_drv.c`, torvalds/linux master, fetched 2026-10-09.
- **[ROCm-wl]** = https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/workload.html
- **[ROCm-blog]** = https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html
- **inferred** = my reading, not stated in a source. **not found** = searched, not stated.

Terms: *XCD/XCC* = one compute chiplet (MI355X has 8, each with its own L2). *Agent* = one GPU as
seen by HIP (in SPX mode all 8 XCDs form one agent, so one agent has 8 L2s). *MTYPE* = per-page
memory type in the GPU page table that sets how L2 caches the page. *MALL / Infinity Cache* =
the 256 MB memory-side cache [ROCm-wl]; the ISA calls it "last-level cache".

---

## 1. Memory model

### 1.1 sc0 / sc1 / nt bits on vector memory (global, buffer, flat) [ISA p.84-85, Tables 49-50]
`SC[1:0]` is a scope: 0 = wave, 1 = group, 2 = device, 3 = system (sc1 is the high bit).
`NT` = non-temporal (1 = no reuse expected).

Loads (CU cache = L1 / L2 / last-level cache):

| scope | sc1 sc0 | nt=0 | nt=1 |
|---|---|---|---|
| wave | 0 0 | hit LRU / hit LRU / hit LRU | miss-evict / hit stream / hit evict |
| group | 0 1 | hit LRU / hit LRU / hit evict | miss-evict / hit stream / hit evict |
| device | 1 0 | miss-evict / (1 L2: hit LRU; >1 L2: **coherent cache bypass**) / hit LRU | miss-evict / (1 L2: hit stream; >1 L2: coherent cache bypass) / hit evict |
| system | 1 1 | miss-evict / coherent cache bypass / hit LRU | miss-evict / coherent cache bypass / hit evict |

Stores: same rows, except the CU cache column is "Miss LRU" for wave/group (stores write through L1).
In TG_SPLIT mode, group scope loads become Miss LRU in L1.

Atomics: `sc0` = return the pre-op value (1) or not (0). `sc1` = 0 device-scope atomic, 1 system-scope
atomic. `nt` = last-level-cache allocate (0) / no-allocate (1). [ISA p.85]
Atomics always bypass L1 [LLVM].

MI355X in SPX mode has 8 L2s per agent, so the ">1 L2" column applies: device-scope (sc1=1) loads
and stores bypass L2 ("coherent cache bypass") and are served from MALL/memory. **inferred**: an
agent-scope flag poll (`global_load ... sc1`) never hits in L2; its latency is MALL/HBM latency.

Scalar loads have only GLC: GLC=1 misses L1 and L2 and fetches from the data fabric [ISA p.84].
LLVM does not use scalar loads for memory that can change during a dispatch [LLVM].

### 1.2 buffer_wbl2 and buffer_inv [ISA p.86 Tables 51-52; p.505 opcode text]
`BUFFER_WBL2` (opcode 40, "Write back L2 cache. Returns ACK to shader"):

| sc1 sc0 | L2 behavior |
|---|---|
| 0 x | NOP |
| 1 0 | 1 L2: NOP; >1 L2: write back dirty data |
| 1 1 | write back dirty data |

`BUFFER_INV` (opcode 41):

| sc1 sc0 | CU (L1) | L2 |
|---|---|---|
| 0 0 | NOP | NOP |
| 0 1 | invalidate only in TG-split mode, else NOP | NOP |
| 1 0 | invalidate | 1 L2: NOP; >1 L2: invalidate non-coherently cached lines |
| 1 1 | invalidate | invalidate non-coherently cached lines |

Neither instruction takes an address: both act on the whole cache of the issuing CU / its XCD's L2
[ISA p.86]. Which lines count: LLVM GFX942 text says `buffer_wbl2` "will writeback dirty L2 cache
lines", `buffer_inv sc1` "will invalidate non-local L2 cache lines" (MTYPE NC); "MTYPE RW and CC
memory will never be stale due to the memory probes" [LLVM]. LLVM GFX90A text (older chip, same
mechanism) adds: wbl2 writes back MTYPE RW and NC lines; MTYPE CC writes through to DRAM and
MTYPE UC bypasses L2, so neither leaves dirty lines [LLVM, GFX90A section].
Cost: **not found** (no cycle numbers in either source). **inferred**: wbl2 cost grows with the
number of dirty lines in that XCD's 4 MB L2, so every agent-scope release in a kernel that has
just written large outputs pays for writing them back; vmcnt wait follows it.

### 1.3 Code sequences, gfx942/gfx950 [LLVM table; confirmed on gfx950 by asm]
LLVM's doc has no separate GFX950 memory-model table; the hipcc output for gfx950 matches the
GFX942 table exactly (**inferred**: gfx950 uses the GFX942 model). Non-TG-split mode, global
address space:

| operation | workgroup | agent | system |
|---|---|---|---|
| load acquire | `global_load sc0` (no wait/inv) | `global_load sc1; s_waitcnt vmcnt(0); buffer_inv sc1` | `global_load sc0 sc1; s_waitcnt vmcnt(0); buffer_inv sc0 sc1` |
| store release | `s_waitcnt vmcnt(0)`(if needed); `global_store sc0` | `buffer_wbl2 sc1; s_waitcnt vmcnt(0) lgkmcnt(0); global_store sc1` | `buffer_wbl2 sc0 sc1; s_waitcnt ...; global_store sc0 sc1` |
| atomicrmw acq_rel | atomic only | `buffer_wbl2 sc1; s_waitcnt vmcnt(0); global_atomic_* [sc0 if returning]; s_waitcnt vmcnt(0); buffer_inv sc1` | `buffer_wbl2 sc0 sc1; s_waitcnt vmcnt(0); global_atomic_* sc1 [sc0]; s_waitcnt vmcnt(0); buffer_inv sc0 sc1` |
| atomicrmw relaxed | `global_atomic` | `global_atomic` (no sc1) | `global_atomic sc1` |
| fence release | `s_waitcnt` only | `buffer_wbl2 sc1; s_waitcnt vmcnt(0)` | `buffer_wbl2 sc0 sc1; s_waitcnt vmcnt(0)` |
| fence acquire | `s_waitcnt` only | `s_waitcnt vmcnt(0); buffer_inv sc1` | `s_waitcnt vmcnt(0); buffer_inv sc0 sc1` |
| fence seq_cst (agent) | | `buffer_wbl2 sc1; s_waitcnt vmcnt(0); buffer_inv sc1` | |
| volatile load | `flat/global_load sc0 sc1; s_waitcnt vmcnt(0)` | | |
| nontemporal load | `global_load ... nt` | | |

Notes:
- `s_waitcnt vmcnt` for stores decrements "when the data has been written to the L2 cache" [ISA p.19].
- The atomic optimizer folded my per-lane `atomicAdd` into one lane per wave (`v_mbcnt`, `s_bcnt1`) [asm].
- "The presence of AV metadata can suppress cache operations" in these sequences [LLVM].
- Monotonic (relaxed) load/store at agent scope still carry `sc1` (so they bypass L2 on >1-L2 agents) [LLVM].

### 1.4 MTYPE and coarse vs fine-grained memory
MTYPEs: RW (read-write, cacheable, for memory local to the L2), NC (non-coherent, cacheable),
CC (cache-coherent), UC (uncached, bypasses L2) [LLVM].
What the gfx950 driver picks (IP 9.5.0 branch, dGPU) [drv `gmc_v9_0_get_coherence_flags`]:
- own VRAM, ordinary allocation (hipMalloc, coarse-grained): `MTYPE_RW` by default
  (module param `mtype_local`: 0=RW, 1=NC, 2=CC; "default: ASIC dependent").
- own VRAM with `AMDGPU_GEM_CREATE_EXT_COHERENT`: CC. With `UNCACHED`: UC.
- peer GPU VRAM (over xGMI) and system memory, not uncached: NC (on gfx950; gfx942 used UC for
  non-VRAM). Ext-coherent remote memory: UC.
- KFD maps `KFD_IOC_ALLOC_MEM_FLAGS_COHERENT / EXT_COHERENT / UNCACHED` to these GEM flags
  [drv `amdgpu_amdkfd_gpuvm.c`]. Which HIP flag sets which KFD flag: **not verified here**.

Coherence rules [LLVM GFX942]:
- "An L2 cache can be kept coherent with other L2 caches by using the MTYPE RW for memory local to
  the L2, and MTYPE NC with the PTE C-bit set for memory not local to the L2."
- "Any local memory cache lines will be automatically invalidated by writes from CUs associated with
  other L2 caches, or writes from the CPU, due to the cache probe caused by the PTE C-bit."
- Writes, same agent, different L2: `buffer_wbl2` (sc1) needed. Reads: `buffer_inv sc1`.
- Different agents: `buffer_wbl2 sc0 sc1` / `buffer_inv sc0 sc1`.
- PCIe to CPU memory: MTYPE UC.

Is L2 coherent across XCDs? **Not automatically** for normal stores: the producer must write back
(wbl2 sc1) and agent-scope loads/stores bypass L2 [ISA p.85-86, LLVM]. For RW lines the probe
invalidates other L2s' copies once the write reaches memory; NC lines need the reader's buffer_inv
[LLVM]. CC: write-through to DRAM (GFX90A text) [LLVM]; UC: no L2 at all.
The exact per-MTYPE behavior of the 8-L2 case is described only by the LLVM text quoted above;
the ISA itself does not mention MTYPE (**not found** in ISA).

### 1.5 Where agent-scope atomics execute
**not found** as an explicit statement. Facts: atomics bypass L1 [LLVM]; `sc1=0` means
"device scope atomic" [ISA p.85]; the `nt` bit on atomics selects "last level cache allocate /
no allocate" [ISA p.85]. **inferred**: since device scope must be coherent across the 8 L2s and
the atomic carries a last-level-cache allocation hint, device-scope atomics are performed past the
XCD L2 (at the MALL/memory side), so their latency is fabric latency, not L2 latency. Confirm with a
microbenchmark before relying on it.

---

## 2. MFMA on gfx950

### 2.1 Scaled FP8/FP6/FP4 MFMA [ISA p.43, 50-51, 55-57]
`V_MFMA_SCALE_F32_16X16X128_F8F6F4` and `V_MFMA_SCALE_F32_32X32X64_F8F6F4`. 4-dword encoding: a
VOP3P-like "load scale" word pair (bits [31:16] = 0xCC35) followed by the MFMA VOP3P [ISA p.56].

Operand registers per lane (A and B independently) [ISA p.50, asm]:

| format | cbsz (A) / blgp (B) | VGPRs for A or B |
|---|---|---|
| FP8 E4M3 | 0 | 8 |
| BF8 E5M2 | 1 | 8 |
| FP6 E2M3 | 2 | 6 |
| BF6 E3M2 | 3 | 6 |
| FP4 E2M1 | 4 | 4 |

C/D: 4 VGPRs (16x16, f32) or 16 VGPRs (32x32, f32). For the F8F6F4 ops CBSZ/BLGP lose their
broadcast meaning (behave as BLGP=0) [ISA p.54-55]. Operands A/B must be VGPR or AGPR, even-aligned;
C may be a constant [ISA p.50]. asm example: `v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[2:9], v[10:13], v[14:17], v1, v19 op_sel_hi:[0,0,0] blgp:4` (fp8 A, fp4 B).

Scales [ISA p.51, 55-57]:
- Format E8M0 (8-bit exponent, bias 127, 0xFF = NaN). One scale per 32 K-elements (block size 32).
- SRC0 of the load-scale word = A scales, SRC1 = B scales; each is a VGPR or an inline constant.
- Byte select: A uses `{OP_SEL_HI[0], OP_SEL[0]}`, B uses `{OP_SEL_HI[1], OP_SEL[1]}`:
  00 = bits [7:0], 01 = [15:8], 10 = [23:16], 11 = [31:24]. Confirmed: opselA=1, opselB=2 assembles
  as `op_sel:[1,0,0] op_sel_hi:[0,1,0]` [asm]. **inferred**: one VGPR can carry the scales of 4
  consecutive MFMAs.
- Lane layout of the scale byte: 16x16x128: lane l holds the scale of row (A) / column (B) `l % 16`,
  K-block `l / 16` (K = 32*(l/16) .. +31). 32x32x64: lane l holds row/column `l % 32`, K-block `l / 32`.
- `ABID[0]` must be 1 for the SCALE ops; ABID[0]=0 forces all scales to 1.0 [ISA p.51]. (The
  assembler output does not print abid; encoding not checked.)
- Math: `d_exp = sum(a_exp + b_exp) + c_exp + scale_a + scale_b` [ISA p.51].
- Register `SH_MEM_CONFIG` bit 8 must be 1 for correct FP8/BF8 results (driver-set) [ISA p.60].

A/B per-lane layout (A row = M index, B column = N index; "group g" = lanes 16g..16g+15) [ISA p.51-52 figures]:
- **16x16x128**: lane l -> row `l % 16`, group g = l/16.
  - FP4: v0-v3 hold k = 32g .. 32g+31 (8 values per VGPR, item 0 in bits 3:0).
  - FP6: v0-v2 hold k = 32g .. 32g+15, v3-v5 hold k = 32g+16 .. 32g+31 (6-bit values packed across a 96-bit group).
  - FP8: v0-v3 hold k = 16g .. 16g+15, v4-v7 hold k = 64+16g .. 64+16g+15.
- **32x32x64**: lane l -> row `l % 32`, half h = l/32.
  - FP4: v0-v3 hold k = 32h .. 32h+31.
  - FP6: v0-v2 hold k = 32h .. 32h+15, v3-v5 hold k = 32h+16 .. 32h+31.
  - FP8: v0-v3 hold k = 16h .. 16h+15, v4-v7 hold k = 32+16h .. 32+16h+15.
- Note: the general formula on p.48-49 (`K_L = K/(64/(M*B))`, item `k % K_L` of lane `i + M*(k/K_L)`)
  matches the FP4/FP6 figures but **not** the FP8 figure (which splits each lane's 32 K-values into two
  16-value halves 64 apart). Take the per-format figure; confirm with the AMD matrix instruction
  calculator or a unit test.

C/D layout (general rule, H = 4) [ISA p.49-50]:
- 16x16 f32 (4 VGPRs): lane l -> column `l % 16`; register r -> row `4*(l/16) + r`.
- 32x32 f32 (16 VGPRs): lane l -> column `l % 32`; register r -> row `8*(r/4) + 4*(l/32) + (r % 4)`.

Cycles [ISA p.43 Table 28, p.50]; the ISA defines "PASS = 4 clock cycles" [ISA p.66]:

| instruction | both operands FP6/FP4 | either operand FP8/BF8 |
|---|---|---|
| (scale_)f32_16x16x128_f8f6f4 | 16 cycles (4 passes) | 32 cycles (8 passes) |
| (scale_)f32_32x32x64_f8f6f4 | 32 cycles (8 passes) | 64 cycles (16 passes) |

So **A = FP8 with B = FP4 runs at the FP8 rate, half the FP4xFP4 rate**. The SCALE variants have the
same cycle counts as the unscaled ones. LLVM agrees: the hazard NOPs it inserts after the MFMA
are the same for fp8xfp4 and fp8xfp8 (`s_nop 11` for 16x16, `s_nop 15; s_nop 3` for 32x32) and smaller
for fp4xfp4 (`s_nop 7` / `s_nop 11`) [asm]. **inferred** peak per CU (4 SIMDs, one MFMA pipe each):
16x16x128 = 65,536 FLOP per instruction -> FP4 16,384 FLOP/clk/CU, FP8 8,192 FLOP/clk/CU.
Issue rate when several waves share a SIMD: **not found** (the ISA gives only per-instruction cycles).

### 2.2 BF16 MFMA shapes on gfx950 [ISA p.43 Table 28]
| shape | blocks | cycles |
|---|---|---|
| v_mfma_f32_32x32x4_2b_bf16 | 2 | 64 |
| v_mfma_f32_16x16x4_4b_bf16 | 4 | 32 |
| v_mfma_f32_4x4x4_16b_bf16 | 16 | 8 |
| v_mfma_f32_32x32x8_bf16 | 1 | 32 |
| v_mfma_f32_16x16x16_bf16 | 1 | 16 |
| **v_mfma_f32_16x16x32_bf16** (new) | 1 | 16 |
| **v_mfma_f32_32x32x16_bf16** (new) | 1 | 32 |

Same shapes/cycles for F16. Sparse: `v_smfmac_f32_16x16x32_bf16` 16, `32x32x16_bf16` 32 [ISA p.60].
New BF16 A/B layouts [ISA p.52-53]: 16x16x32: lane l row `l%16`, group g: v0..v3 hold k = 8g..8g+7
(2 per VGPR). 32x32x16: lane l row `l%32`, half h: v0..v3 hold k = 8h..8h+7. asm: A and B are 4 VGPRs each.
Other: FP8 16x16x32 (16 cycles), 32x32x16 (32); I8 16x16x64 (16), 32x32x32 (32); F64 16x16x4 (64).

### 2.3 Hazards worth knowing [ISA p.66-69 Table 38]
- VALU writes VGPR -> MFMA reads it: 2 wait states.
- Back-to-back MFMA with srcC = previous vDst, same opcode/size: 0-2 waits (accumulation forwarding).
- MFMA result read by VALU / memory / as srcA/B: needs the full pass count (4..18 extra waits depending on passes); the compiler inserts `s_nop`.

- Hand-off between two CUs of one XCD (amd/bench/l2_handoff.hip, MI355X 10-15, measured): after a plain store by one CU,
  a load with sc0 (workgroup scope) or `buffer_inv sc0` + load on the other CU never sees it (its L1 keeps the stale
  line); `buffer_inv sc1` + load sees it in 560 ns, a load with nt in 480 ns, a load with sc1 (device scope) in 480 ns
  (medians). So a device-scope poll is as fast as reading through the shared L2; one CU's flag hop to another is
  ~0.4-0.5 us idle or with HBM busy (amd/bench/poll_depth.hip; 2 or 4 polls in flight do not shorten it) [measured].

---

## 3. LDS [ISA p.6, 13, 83-84, 95-99; LLVM; asm]
- 160 KB per CU; 64 banks x 640 entries x 4 bytes; one work-group may use all 160 KB [ISA p.6, p.95].
- Allocation granule 1280 bytes, 1280-byte aligned [ISA p.13]; LLVM: GFX950 granule `lds-size/(320*4)` [LLVM kernel descriptor].
- 32 integer atomic units; bank conflicts serialize; an op takes 2 to 64 cycles [ISA p.6, p.96].
- Direct global->LDS loads (no VGPR pass):
  - MUBUF `buffer_load_{ubyte,sbyte,ushort,sshort,dword,dwordx3,dwordx4,format_x}` with lds=1 [ISA p.83].
  - `global_load_lds_{ubyte,sbyte,ushort,sshort,dword,dwordx3,dwordx4}` (dwordx3 = opcode 126, dwordx4 = 125; x3/x4 new in CDNA4) and `scratch_load_lds_*` [ISA p.91, p.546-547].
  - LDS address = M0[17:0] (byte offset) + inst offset + TIDinWave * size; for 3/4-dword sizes TIDinWave*16 (x3 writes 3 dwords and skips the 4th) [ISA p.83-84]. So one wave writes a contiguous 64*16 = 1 KB block per dwordx4 instruction (**inferred** from the formula).
  - asm: `__builtin_amdgcn_global_load_lds(..., 16, ...)` -> `s_mov_b32 m0, ...; global_load_lds_dwordx4 v1, s[0:1]`.
  - Completion is tracked by vmcnt (**inferred**; not stated for the LDS variant).
- Transposing LDS reads for MFMA operands: `ds_read_b64_tr_b16`, `ds_read_b64_tr_b8`, `ds_read_b64_tr_b4`, `ds_read_b96_tr_b6`; EXEC must be all ones; two instructions load a full operand [ISA p.98-99].

---

## 4. Clocks and IDs
- `s_memrealtime`: 64-bit "real time counter", "constant 100MHz clock (not affected by power modes or core clock frequency changes)" [ISA p.73]. HIP `wall_clock64()` compiles to `s_memrealtime` [asm].
- `s_memtime`: "64-bit clock counter" [ISA p.72]; rate **not found** in ISA. HIP `clock64()` and `__builtin_readcyclecounter()` compile to `s_memtime` [asm]. **inferred**: shader clock (up to 2,400 MHz on MI355X [ROCm-wl]); varies with DVFS and per XCD.
  Both are SMEM ops: result needs `s_waitcnt lgkmcnt(0)`; counts like an `s_load_dwordx2` [ISA p.19].
- `s_getreg_b32` IDs [ISA p.16, p.29 Table 19]: `HW_REG_HW_ID` (id 4, "Read only. Debug only"):
  WAVE_ID [3:0], SIMD_ID [5:4], PIPE_ID [7:6], CU_ID [11:8], SH_ID [12] (always 0), SE_ID [15:13],
  TG_ID [19:16], VM_ID [23:20], QUEUE_ID [26:24], STATE_ID [29:27] (unused), ME_ID [31:30].
  `HW_REG_XCC_ID` (id 20): XCC_ID [3:0]. asm: `s_getreg_b32 s2, hwreg(HW_REG_XCC_ID)`.
  ISA warning: "It is not safe to rely on these values as they may change over the lifetime of a wave" [ISA p.16].
- XCD layout: 8 XCDs, 36 CUs per XCD of which 32 active, 256 CUs total; 4 MB L2 per XCD; 256 MB Infinity Cache; "clock speed variation of 3% to 10% among different XCDs" (stated for MI300X) [ROCm-wl].
  **inferred**: CU_ID (4 bits) + SE_ID (3 bits) is enough to name 32 CUs per XCD; the SE/CU split per XCD is not found.
- Work-group to XCD mapping: in SPX mode "workgroups launched to the device are distributed
  round-robin to the XCDs", and "the programmer cannot have explicit control over which XCD a
  workgroup is assigned to" [ROCm-blog]. Exact formula (e.g. `wg_id % 8`) **not found** in AMD docs;
  [ROCm-wl] recommends grid sizes in multiples of 8 and pid remapping, which assumes `wg_id % 8`
  (**inferred**). Deterministic? **not found**. For a persistent kernel, read `HW_REG_XCC_ID` at start
  and record it, instead of assuming the mapping. CPX mode: each XCD is its own device [ROCm-blog].

---

## 5. Wavefront, registers, occupancy
- Wave64 only: the ISA describes 64 work-items per wavefront [ISA p.5]; `-mno-wavefrontsize64` for
  gfx950 still produced `.wavefront_size: 64`, and `-mwavefrontsize32` is rejected by hipcc [asm].
- VGPRs: up to 512 per wave, 256 arch + 256 AGPR at most, split is flexible below 512 [ISA p.12-13];
  allocated in groups of 8 [ISA p.12]; LLVM: GFX90A/GFX942 granule 8 with `vgprs_used = align(arch,4) + acc`, AGPR start (`ACCUM_OFFSET`) granularity 4 [LLVM]. [ROCm-wl] says "512 available VGPRs, which are allocated in blocks of 16" (MI300X example), which conflicts with the ISA's 8; unresolved.
- SGPRs: 16 to 102 per wave, VCC in SGPR 106-107 [ISA p.12].
- Occupancy: max 8 waves per SIMD (compiler reports `; Occupancy: 8` for a trivial gfx950 kernel) [asm]; 4 SIMDs per CU [ROCm-wl]. VGPR limit = floor(512 / granule-rounded VGPRs) waves per SIMD; LDS limit = floor(163840 / LDS bytes per WG) WGs per CU [ROCm-wl]. Work-group max 16 waves (1024 work-items) [ISA p.19].
- `s_sleep N`: sleeps 64*(N-1) .. 64*N clocks, N = SIMM16[6:0] (max ~8128 clocks); `s_sleep 0` = no sleep [ISA p.146]. `s_wakeup` wakes the other waves of the same work-group early [ISA p.18].
- `s_setprio N`: N in 0..3 (3 highest); full priority = {SPIPrio[1:0], UserPrio[1:0], WaveAge[3:0]} [ISA p.146].
- `s_nop N`: 0..15 extra wait states [ISA p.145].

---

## 6. Cross-device (xGMI) and system scope
- ISA: only the system scope bits: sc1=sc0=1 on loads/stores is "system" (L1 miss, L2 coherent-cache bypass, MALL hit) and sc1=1 on atomics is "system scope atomic" [ISA p.85]. Nothing else about xGMI in the ISA (**not found**).
- LLVM: system-scope release = `buffer_wbl2 sc0 sc1`, acquire = `buffer_inv sc0 sc1`; "Ensures that following loads will not see stale MTYPE NC global data. MTYPE RW and CC memory will never be stale due to the memory probes"; "XGMI accesses from the CPU to local memory may be cached on the CPU. Subsequent access from the GPU will automatically invalidate or writeback the CPU cache due to the L2 probe filter" [LLVM].
- Driver: peer-GPU VRAM is mapped MTYPE NC on gfx950 (UC if allocated ext-coherent or uncached) [drv]. **inferred**: a peer's flag/data mapped NC can sit stale in the local L2 until a system-scope acquire (`buffer_inv sc0 sc1`) or an sc0+sc1 load; remote atomics need sc1=1 (system scope) to be coherent with the peer.
- Remaining cross-device atomic placement (which die executes a remote atomic) and its cost: **not found**.

## Compiler / hardware lessons (MI355X, 2026-10-13)
- Stores and loads share vmcnt on gfx950: a task that ends with system-scope pushes to 8 GPUs delays the NEXT task on the
  same CU by ~6 us (its first load wait also waits for the pushes' acks). Plan the consumers of a push on other CUs.
- `__shfl_xor` compiles to ds_bpermute (an LDS round trip per step). Reductions inside 8 / 16 lanes: DPP quad_perm (xor 1,
  xor 2) + row_half_mirror + row_mirror (the same adds as the xor butterfly); 64 lanes: those + readlane of lanes 0, 16, 32,
  48 (core.cuh wave_red64). KDA -4 us, MLA -8 us with the data-as-flag residual and the tile-major push.
- The compiler's waitcnt pass drains the pipeline at control-flow joins: a guarded load in a pipelined loop (`if (tok < T)`
  around the MFMA's B fragment, a `cr >= 0 ?` branch around a prefetch load, a value converted right after its load) made
  it emit vmcnt(0)/vmcnt(1..6) inside the loop. Make the loads unconditional (valid dummy address, select on use) and
  consume prefetched values only where the wait is due anyway; check the loop in --save-temps asm with `asm volatile("; MARK")`.
- Copying a weight slice through registers in batches of 4 is ~8 serial round trips per 128 KB; global_load_lds (async
  DMA into LDS, M0 base + 16 B per lane, the source address per lane free -> any XOR swizzle) issues all at once.
- LDS b128 reads: rows 128 B apart in the same column hit 2 bank groups per 8 lanes (4-way); XOR the 16-B chunk index
  with (row >> 3) ^ row (mla_kt_chunk).
- One-line atomic start barrier over 256 CUs ~4 us; in-kernel resets ~2-3 us; with server flags the launch costs ~11.5 us
  of e2e over span (dispatch ~3.6 of it: launch_gap bench).
