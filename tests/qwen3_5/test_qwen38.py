"""CPU tests for 27B geometry and grouped GDN prefill semantics."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/mirage/mpk/models"))
from qwen3_5.config import Qwen35TextConfig
from qwen3_5.weight_loader import expected_text_shapes
from qwen3_5.prefill import _gdn, _attention
from qwen3_5.reference import causal_conv_step, gated_delta_step, delta_gates, gated_rms_norm
from qwen3_5.attention_reference import full_attention_forward, text_rope


def test_official_27b_checkpoint_geometry():
    c = Qwen35TextConfig.from_json(Path(__file__).parent / "fixtures/qwen3_8_27b_config.json")
    shapes = expected_text_shapes(c)
    assert (c.hidden_size, c.intermediate_size, c.num_hidden_layers) == (5120, 17408, 64)
    assert (len(c.gdn_layers), len(c.attention_layers)) == (48, 16)
    assert shapes["layers.0.linear_attn.in_proj_qkv.weight"] == (10240, 5120)
    assert shapes["layers.0.linear_attn.in_proj_z.weight"] == (6144, 5120)
    assert shapes["layers.3.self_attn.q_proj.weight"] == (12288, 5120)
    assert shapes["layers.3.self_attn.k_proj.weight"] == (1024, 5120)
    assert shapes["lm_head.weight"] == (248320, 5120)
    assert not c.tie_word_embeddings


def test_loader_rejects_non_27b_geometry(tmp_path):
    from qwen3_5.weight_loader import Qwen35WeightLoader
    import pytest
    source = json.loads((Path(__file__).parent / "fixtures/qwen3_8_27b_config.json").read_text())
    source["text_config"]["hidden_size"] = 1024
    (tmp_path / "config.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="27B dense only"):
        Qwen35WeightLoader(tmp_path)


def test_grouped_prefill_agrees_with_independent_recurrent_updates():
    torch.manual_seed(38)
    # Keep the 3:1 head mapping and native 128-wide states; reduce projection width.
    kh, vh, d, hidden = 2, 6, 128, 16
    c = SimpleNamespace(linear_num_key_heads=kh, linear_num_value_heads=vh,
                        linear_key_head_dim=d, conv_dim=(2*kh+vh)*d, rms_norm_eps=1e-6)
    def rand(*shape): return (torch.randn(shape)*.1).bfloat16()
    t = dict(qkv_weight=rand(c.conv_dim,hidden), z_weight=rand(vh*d,hidden),
             ab_weight=rand(64,hidden), conv_weight=rand(c.conv_dim,4),
             parameters=torch.randn(2,vh)*.1, gated_norm_weight=torch.ones(d,dtype=torch.bfloat16),
             out_weight=rand(hidden,vh*d), state=torch.zeros(vh,d,d),
             conv_state=torch.zeros(c.conv_dim,3,dtype=torch.bfloat16))
    x = rand(1,7,hidden)
    state=t["state"].unsqueeze(0).clone(); history=t["conv_state"].unsqueeze(0).clone()
    expected=[]
    for at in range(x.shape[1]):
        h=x[:,at]
        mixed,history=causal_conv_step(F.linear(h,t["qkv_weight"]),history,t["conv_weight"].unsqueeze(1))
        q,k,v=mixed.split((kh*d,kh*d,vh*d),-1)
        gates=F.linear(h,t["ab_weight"])
        decay,beta=delta_gates(gates[:,:vh],gates[:,vh:2*vh],*t["parameters"])
        raw,state=gated_delta_step(q.reshape(1,kh,d),k.reshape(1,kh,d),v.reshape(1,vh,d),decay,beta,state)
        z=F.linear(h,t["z_weight"]).reshape(vh,d)
        gated=gated_rms_norm(raw.reshape(vh,d),t["gated_norm_weight"],z).reshape(1,vh*d)
        expected.append(F.linear(gated,t["out_weight"]))
    actual,cache=_gdn(x,t,c,0)
    torch.testing.assert_close(actual,torch.stack(expected,1),rtol=.03,atol=2e-5)
    # The chunked triangular solve and tokenwise recurrence differ slightly
    # in FP32 order, including at state entries whose reference is near zero.
    torch.testing.assert_close(cache["state"],state[0],rtol=.01,atol=2e-5)
    torch.testing.assert_close(cache["conv_state"],history[0],rtol=0,atol=0)
    first,new=_gdn(x[:,:3],t,c,0)
    t.update(new)
    rest,last=_gdn(x[:,3:],t,c,3)
    torch.testing.assert_close(torch.cat((first,rest),1),actual,rtol=.03,atol=2e-5)
    torch.testing.assert_close(last["state"],cache["state"],rtol=.01,atol=2e-5)


def test_27b_gqa_prefill_and_cached_continuation():
    torch.manual_seed(3827)
    c=Qwen35TextConfig.from_json(Path(__file__).parent / "fixtures/qwen3_8_27b_config.json")
    # Keep all 24 query / 4 KV heads but reduce hidden width for CPU execution.
    from dataclasses import replace
    c=replace(c,hidden_size=16)
    def rand(*shape): return (torch.randn(shape)*.1).bfloat16()
    weights={"q_proj.weight":rand(12288,16),"k_proj.weight":rand(1024,16),
             "v_proj.weight":rand(1024,16),"o_proj.weight":rand(16,6144),
             "q_norm.weight":rand(256),"k_norm.weight":rand(256)}
    t={"qg_weight":weights["q_proj.weight"],"k_weight":weights["k_proj.weight"],
       "v_weight":weights["v_proj.weight"],"out_weight":weights["o_proj.weight"],
       "q_norm":weights["q_norm.weight"].float(),"k_norm":weights["k_norm.weight"].float()}
    cos,sin=text_rope(c,torch.arange(10))
    mixer=SimpleNamespace(tensors=t,attn_implementation="eager",rope_table=torch.stack((cos[0],sin[0]),1))
    x=rand(1,7,16)
    expected=full_attention_forward(x,weights,c)
    actual,cache=_attention(x,mixer,c,0)
    torch.testing.assert_close(actual,expected["output"],rtol=0,atol=0)
    first,old=_attention(x[:,:3],mixer,c,0)
    t.update(old)
    rest,last=_attention(x[:,3:],mixer,c,3)
    torch.testing.assert_close(torch.cat((first,rest),1),actual,rtol=0,atol=0)
    for name in cache: torch.testing.assert_close(last[name],cache[name],rtol=0,atol=0)


def test_external_shards_require_explicit_trusted_storage(tmp_path):
    from qwen3_5.weight_loader import Qwen35WeightLoader
    import pytest
    model, storage = tmp_path / "model", tmp_path / "storage"
    model.mkdir(); storage.mkdir()
    (model / "config.json").write_bytes((Path(__file__).parent / "fixtures/qwen3_8_27b_config.json").read_bytes())
    (storage / "shard.safetensors").write_bytes(b"metadata-only path test")
    (model / "shard.safetensors").symlink_to(storage / "shard.safetensors")
    index = model / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map":{"test":"shard.safetensors"}}))
    with pytest.raises(ValueError, match="Invalid shard path"):
        Qwen35WeightLoader(model)._files()
    _, files = Qwen35WeightLoader(model, shard_roots=[storage])._files()
    assert files["shard.safetensors"] == storage / "shard.safetensors"
    index.write_text(json.dumps({"weight_map":{"test":"../storage/shard.safetensors"}}))
    with pytest.raises(ValueError, match="Invalid shard path"):
        Qwen35WeightLoader(model, shard_roots=[storage])._files()
