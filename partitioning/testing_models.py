"""Testing models for partitioning and benchmarking.

This module contains neural network architectures designed for testing
the Mirage partitioning system and performance benchmarking. All models
use float16 precision.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TestMLP(nn.Module):
    """5-layer MLP with tanh and sigmoid activations.
    
    A simple multilayer perceptron for testing purposes with the following
    architecture: 1024 -> 8192 -> 16384 -> 16384 -> 8192 -> 1024.
    Uses tanh activation after layers 1 and 4, and sigmoid at the output.
    All weights are float16 and initialized with Xavier uniform.
    """
    
    def __init__(self):
        super().__init__()
        # Define weights manually instead of nn.Linear layers
        self.w1 = nn.Parameter(torch.empty(1024, 8192, dtype=torch.float16))
        self.w2 = nn.Parameter(torch.empty(8192, 16384, dtype=torch.float16))
        self.w3 = nn.Parameter(torch.empty(16384, 16384, dtype=torch.float16))
        self.w4 = nn.Parameter(torch.empty(16384, 8192, dtype=torch.float16))
        self.w5 = nn.Parameter(torch.empty(8192, 1024, dtype=torch.float16))
        
        # Initialize weights similar to Linear
        for w in [self.w1, self.w2, self.w3, self.w4, self.w5]:
            nn.init.xavier_uniform_(w)

    def forward(self, x):
        """Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, 1024)
            
        Returns:
            Output tensor of shape (batch_size, 1024) with sigmoid activation
        """
        # Using torch.matmul instead of nn.Linear
        x = torch.tanh(torch.matmul(x, self.w1))
        x = torch.matmul(x, self.w2)
        x = torch.matmul(x, self.w3)
        x = torch.tanh(torch.matmul(x, self.w4))
        x = torch.sigmoid(torch.matmul(x, self.w5))
        return x

class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention and feedforward.
    
    Implements a standard transformer block with:
    - RMSNorm pre-normalization
    - Multi-head self-attention
    - Feedforward network with SiLU activation
    - Residual connections
    
    All parameters use float16 precision.
    """
    
    def __init__(self, d_model, n_heads, ff_mult=4):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimensionality (must be divisible by n_heads)
            n_heads: Number of attention heads
            ff_mult: Feedforward hidden dimension multiplier (default: 4)
        """

        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.wq   = nn.Linear(d_model, d_model, bias=False, dtype=torch.float16)
        self.wk   = nn.Linear(d_model, d_model, bias=False, dtype=torch.float16)
        self.wv   = nn.Linear(d_model, d_model, bias=False, dtype=torch.float16)
        self.proj = nn.Linear(d_model, d_model, bias=False, dtype=torch.float16)

        self.rms1 = nn.RMSNorm(d_model, eps=1e-6, dtype=torch.float16)
        self.rms2 = nn.RMSNorm(d_model, eps=1e-6, dtype=torch.float16)

        hidden = ff_mult * d_model
        self.fc1 = nn.Linear(d_model, hidden, bias=False, dtype=torch.float16)
        self.fc2 = nn.Linear(hidden, d_model, bias=False, dtype=torch.float16)

    def forward(self, x):
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # --- Attention ---
        h = self.rms1(x)
        q = self.wq(h)  # (B, T, D)
        k = self.wk(h)  # (B, T, D)
        v = self.wv(h)  # (B, T, D)

        B, T, _ = q.shape
        H, Hd = self.n_heads, self.head_dim

        # reshape to (B, H, T, Hd) explicitly
        q = q.view(B, T, H, Hd).transpose(1, 2)
        k = k.view(B, T, H, Hd).transpose(1, 2)
        v = v.view(B, T, H, Hd).transpose(1, 2)

        # attn logits: (B, H, T, T)
        attn = torch.matmul(q, k.transpose(-2, -1))

        # explicit scaling: create (1,1,1,1) tensor and expand to attn.shape
        scale = (1.0 / math.sqrt(Hd))
        scale = torch.tensor(scale, dtype=attn.dtype, device=attn.device).view(1, 1, 1, 1)
        attn = attn * scale.expand_as(attn)

        attn = attn.softmax(dim=-1)

        # attention output: (B, H, T, Hd)
        a = torch.matmul(attn, v)

        # back to (B, T, D)
        a = a.transpose(1, 2).contiguous().view(B, T, self.d_model)
        x = x + self.proj(a)
        # --- Feedforward ---
        h = self.rms2(x)
        h = self.fc2(F.silu(self.fc1(h)))
        return x + h


class TestTransformer(nn.Module):
    """Large-scale transformer model for inference benchmarking.
    
    A complete transformer architecture designed for testing partitioning
    strategies on large models. Features:
    - Token and positional embeddings
    - Stacked transformer blocks with RMSNorm
    - Language modeling head
    - Default configuration: 8192 hidden dim, 8 heads, 16K vocab
    
    All parameters use float16 precision for efficient inference.
    """
    
    def __init__(self, vocab_size=16384, max_seq_len=1024,
                 d_model=8192, n_heads=8, n_layers=1, ff_mult=4):
        """Initialize transformer model.
        
        Args:
            vocab_size: Vocabulary size (default: 16384)
            max_seq_len: Maximum sequence length (default: 1024)
            d_model: Model dimensionality (default: 8192)
            n_heads: Number of attention heads per block (default: 8)
            n_layers: Number of transformer blocks (default: 1)
            ff_mult: Feedforward hidden dimension multiplier (default: 4)
        """

        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model, dtype=torch.float16)
        self.pos_emb = nn.Embedding(max_seq_len, d_model, dtype=torch.float16)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)]
        )
        self.rms_f = nn.RMSNorm(d_model, eps=1e-6, dtype=torch.float16)
        self.head = nn.Linear(d_model, vocab_size, bias=False, dtype=torch.float16)

    def forward(self, input_ids):
        """Forward pass through the transformer.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len).
                      Automatically converted to long dtype if needed.
                      
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        B, T = input_ids.shape
        device = input_ids.device

        # dummy positions = 1 for all tokens, shape (T,)
        pos = torch.ones((1, T), dtype=torch.long, device=device)
        pos_e = self.pos_emb(pos)                 # (1, T, D)
        pos_e = pos_e.expand(B, T, -1).contiguous()  # (B, T, D) explicit

        tok_e = self.tok_emb(input_ids)           # (B, T, D)
        x = tok_e + pos_e                         # same shape, no implicit broadcast

        for blk in self.blocks:
            x = blk(x)

        x = self.rms_f(x)
        return self.head(x)
