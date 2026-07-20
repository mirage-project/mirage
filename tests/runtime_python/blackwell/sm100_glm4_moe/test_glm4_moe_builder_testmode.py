"""End-to-end builder test for GLM-4.6 MPK support.

Builds a 4-layer GLM-4.6 (full-size dims: hidden 5120, 96/8 heads, 3 dense +
1 MoE layer with 160 routed + 1 shared experts) with randomly initialized
weights, runs one decode step through the full MPK pipeline in test mode,
and compares the lm-head logits and argmax token against the HuggingFace
Glm4MoeForCausalLM reference running the exact same weights.

This exercises every layer the Glm4MoeBuilder wires: embed, rmsnorm,
qkv linear + bias (via fused residual), paged attention with qk-norm
(eps 1e-5) + partial RoPE (64/128, theta 1e6), o_proj residual, dense
silu MLP, MoE gate linear + glm_moe_router + w13/silu/w2/mul_sum_add with
the folded shared expert, final norm, lm head, argmax.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

NUM_LAYERS = 4  # 3 dense + 1 MoE
PROMPT_TOKEN = 4242


def build_hf_model():
    from transformers.models.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM

    config = Glm4MoeConfig(
        num_hidden_layers=NUM_LAYERS,
        hidden_size=5120,
        intermediate_size=12288,
        num_attention_heads=96,
        num_key_value_heads=8,
        head_dim=128,
        partial_rotary_factor=0.5,
        rope_theta=1e6,
        use_qk_norm=True,
        attention_bias=True,
        rms_norm_eps=1e-5,
        first_k_dense_replace=3,
        n_routed_experts=160,
        num_experts_per_tok=8,
        n_shared_experts=1,
        moe_intermediate_size=1536,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        vocab_size=151552,
        tie_word_embeddings=False,
        max_position_embeddings=4096,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = Glm4MoeForCausalLM(config).to(torch.bfloat16).eval()
    # make qkv biases non-trivial (HF inits them to zero)
    with torch.no_grad():
        for layer in model.model.layers:
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                         layer.self_attn.v_proj):
                proj.bias.normal_(0.0, 0.02)
    return model


def main():
    device = "cuda"
    model = build_hf_model()

    # HF reference: single-token prompt, next-token logits
    input_ids = torch.tensor([[PROMPT_TOKEN]], device=device)
    with torch.no_grad():
        ref_logits = model(input_ids=input_ids).logits[0, -1].float()
    ref_token = int(ref_logits.argmax().item())

    state_dict = dict(model.state_dict())

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["max_num_batched_tokens"] = 1
    params["max_num_batched_requests"] = 1
    params["page_size"] = 64  # must be a multiple of the 64-key KV tile
    params["max_num_pages"] = 1
    params["max_seq_length"] = 64
    tokens = torch.zeros((1, 64), dtype=torch.int64, device=device)
    tokens[0, 0] = PROMPT_TOKEN
    params["meta_tensors"] = {
        "tokens": tokens,
        "prompt_lengths": torch.tensor([1], dtype=torch.int32, device=device),
    }
    params["eos_token_id"] = 151329
    pk = PersistentKernel(**params)

    from mirage.mpk.models.glm4_moe import Glm4MoeBuilder
    builder = Glm4MoeBuilder(pk)
    # pre-seed argmax_in with an externally attached tensor so the test can
    # read the lm-head logits back after the run (new_tensor buffers are
    # internal allocations)
    argmax_in_t = torch.zeros(1, 151552, dtype=torch.bfloat16, device=device)
    builder._bufs["argmax_in"] = pk.attach_input(argmax_in_t, name="argmax_in")
    builder.build_from_dict(state_dict, with_lm_head=True)

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    mpk_logits = argmax_in_t[0].float()
    mpk_token = int(pk.meta_tensors["output_tokens"][0].item())

    vocab = ref_logits.shape[0]
    mpk_v = mpk_logits[:vocab]
    max_diff = (mpk_v - ref_logits).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        mpk_v, ref_logits, dim=0).item()
    print(f"logits max diff: {max_diff:.4f}, cosine: {cos:.6f}")
    print(f"argmax token: mpk={mpk_token}, hf={ref_token}")

    pk.finalize()
    if mpk_token != ref_token or cos < 0.99:
        print("FAILED")
        sys.exit(1)
    print("PASSED: GLM-4.6 builder matches HF reference end-to-end")


if __name__ == "__main__":
    main()
