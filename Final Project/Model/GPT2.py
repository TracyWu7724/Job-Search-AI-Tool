# -*- coding = utf-8 -*-
# @Time : 4/8/25 12:07
# @Author : Tracy
# @File : GPT2.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
    return mask  # shape: (1, 1, seq_len, seq_len)

class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = attn @ v
        concat = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.out(concat)

        return output, attn


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.2):
        super().__init__()
        self.attn = MaskedSelfAttention(n_embd, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = self.ln1(x)
        q = k = v = x
        attn_output, _ = self.attn(q, k, v, mask=mask)
        x = x + attn_output
        mlp_output = self.mlp(self.ln2(x))
        x = x + mlp_output
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layers, n_heads, n_embds):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embds)
        self.position_embedding_table = nn.Embedding(block_size, n_embds)
        self.blocks = nn.Sequential(*[DecoderBlock(n_embds, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embds)
        self.lm_head = nn.Linear(n_embds, vocab_size, bias=False)
        self.block_size = block_size

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x, mask=None):
        B, T = x.size()
        tokens = self.token_embedding_table(x)
        positions_id = torch.arange(0, x.size(1), device=x.device)
        positions = self.position_embedding_table(positions_id)
        x = tokens + positions

        if mask is not None:
            mask = generate_causal_mask(T, device=x.device)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, idx, max_len, mask=None):
        for _ in range(max_len):
            # only keep last context
            idx_cond = idx[:, -self.block_size:]

            # get the prediction
            logits = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  #(batch_size, seq_len, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # randomly sample from the multinominal distribution
            idx_next = torch.multinomial(probs, 1)

            # add the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, seq_len + 1)

        return idx  # shape (B, max_len + 1)

