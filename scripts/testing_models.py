
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 8192, dtype=torch.float16)
        self.fc2 = nn.Linear(8192, 16384, dtype=torch.float16)
        self.fc3 = nn.Linear(16384, 16384, dtype=torch.float16)
        self.fc4 = nn.Linear(16384, 8192, dtype=torch.float16)
        self.fc5 = nn.Linear(8192, 1024, dtype=torch.float16)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.rms1 = nn.RMSNorm(d_model, eps=1e-6)
        self.rms2 = nn.RMSNorm(d_model, eps=1e-6)

        hidden = ff_mult * d_model
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        # --- Attention ---
        h = self.rms1(x)
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h)

        B, T, _ = q.shape
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        a = torch.matmul(attn, v)
        a = a.transpose(1, 2).reshape(B, T, self.d_model)
        x = x + self.proj(a)

        # --- Feedforward ---
        h = self.rms2(x)
        h = self.fc2(F.silu(self.fc1(h)))
        return x + h

class TestTransformer(nn.Module):
    """Simplified large transformer for inference benchmarking (RMSNorm)."""
    def __init__(self, vocab_size=16384, max_seq_len=2048,
                 d_model=4096, n_heads=32, n_layers=6, ff_mult=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)]
        )
        self.rms_f = nn.RMSNorm(d_model, eps=1e-6)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        _, T = input_ids.shape

        # dummy positions, eliminates Range and Unsqueeze ops
        pos = torch.ones((1, T), dtype=torch.long)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.rms_f(x)
        return self.head(x)
