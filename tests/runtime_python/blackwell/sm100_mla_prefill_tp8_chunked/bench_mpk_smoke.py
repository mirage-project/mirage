"""Measure run_test_mode overhead at the chunked-attention smoke shape."""
import os, sys, math, time
import torch
import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

D_QK_NOPE = 128
D_QK_ROPE = 64
D_V = 128

H, B, q_len, kv_len = 16, 1, 256, 1024
q_start = kv_len - q_len
device = "cuda"
dt = torch.bfloat16
torch.manual_seed(0)
q_nope = torch.randn(B*q_len, H, D_QK_NOPE, dtype=dt, device=device) * 0.2
q_pe = torch.randn(B*q_len, H, D_QK_ROPE, dtype=dt, device=device) * 0.2
kv_combined = torch.randn(B*kv_len, H, D_QK_NOPE+D_V, dtype=dt, device=device) * 0.2
k_nope = kv_combined[..., :D_QK_NOPE]
v = kv_combined[..., D_QK_NOPE:]
k_rope = torch.randn(B*kv_len, 1, D_QK_ROPE, dtype=dt, device=device) * 0.2
o = torch.zeros(B*q_len, H, D_V, dtype=dt, device=device)

nw, nsch = mirage.get_configurations_from_gpu(0)
params = PersistentKernel.get_default_init_parameters()
params.update(test_mode=True, num_workers=nw, num_local_schedulers=nsch,
              max_num_batched_tokens=q_len, max_num_batched_requests=B,
              max_seq_length=kv_len)
pk = PersistentKernel(**params)
qn_dt = pk.attach_input(q_nope, name="q_nope")
qp_dt = pk.attach_input(q_pe, name="q_pe")
kn_dt = pk.attach_input(k_nope, name="k_nope")
kr_dt = pk.attach_input(k_rope, name="k_rope")
v_dt = pk.attach_input(v, name="v")
o_dt = pk.attach_input(o, name="o")
pk.mla_prefill_tp8_chunked_layer(
    q_nope=qn_dt, q_pe=qp_dt, k_nope=kn_dt, k_rope=kr_dt, v=v_dt, output=o_dt,
    mla_params=(H, q_len, kv_len, q_start),
    grid_dim=(H, (q_len + 63) // 64, B), block_dim=(128, 1, 1))
folder = os.path.dirname(os.path.abspath(__file__))
pk.compile(output_dir=folder)
for _ in range(20): pk.run_test_mode()
torch.cuda.synchronize()
s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(50): pk.run_test_mode()
e.record(); torch.cuda.synchronize()
print(f"chunked attn MPK: {s.elapsed_time(e)/50*1000:.1f} us")
