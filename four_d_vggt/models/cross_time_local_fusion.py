"""Parameter-matched alternating frame / sliding-window temporal attention."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def window_indices(sequence_length, window_size, device=None):
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1 or window_size % 2 != 1:
        raise ValueError("window_size must be a positive odd integer")
    if sequence_length < 1:
        raise ValueError("The sequence must contain at least one frame")
    offsets = torch.arange(window_size, device=device) - window_size // 2
    indices = torch.arange(sequence_length, device=device)[:, None] + offsets
    valid = (indices >= 0) & (indices < sequence_length)
    return indices.clamp(0, sequence_length - 1), valid


def temporal_visibility(sequence_length, window_size, device=None):
    """Small frame-level golden-test helper, never a full token attention mask."""
    window_indices(sequence_length, window_size, device)
    t = torch.arange(sequence_length, device=device)
    return (t[:, None] - t[None, :]).abs() <= window_size // 2


class WindowAttention(nn.Module):
    """Own an already-copied attention's weights, adding signed time bias only.

    Q comes from one frame; K/V gather clipped neighbouring frames. Special tokens
    obey the same window. No global S*P by S*P attention matrix is constructed.
    """

    def __init__(self, copied_attention, window_size):
        super().__init__()
        window_indices(1, window_size)
        for name in ("qkv", "q_norm", "k_norm", "attn_drop", "proj", "proj_drop", "rope"):
            setattr(self, name, getattr(copied_attention, name))
        for name in ("num_heads", "head_dim", "scale", "fused_attn"):
            setattr(self, name, getattr(copied_attention, name))
        self.window_size = window_size
        self.relative_time_bias = nn.Parameter(self.qkv.weight.new_zeros(self.num_heads, window_size))

    def forward(self, x, pos=None):
        b, s, p, c = x.shape
        qkv = self.qkv(x).reshape(b * s, p, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)
        indices, valid = window_indices(s, self.window_size, x.device)

        def gather_windows(tensor):
            tensor = tensor.reshape(b, s, self.num_heads, p, self.head_dim)[:, indices]
            return tensor.permute(0, 1, 3, 2, 4, 5).reshape(b * s, self.num_heads, self.window_size * p, self.head_dim)

        k, v = gather_windows(k), gather_windows(v)
        # Broadcast over Q tokens instead of allocating a P x (window*P) mask.
        bias = self.relative_time_bias.to(q.dtype)[None, :, :].expand(s, -1, -1)
        bias = bias.masked_fill(~valid[:, None, :], float("-inf"))
        bias = bias.repeat_interleave(p, dim=-1)[None, :, :, None, :]
        bias = bias.expand(b, -1, -1, -1, -1).reshape(b * s, self.num_heads, 1, self.window_size * p)
        if self.fused_attn:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                 dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            weights = ((q * self.scale) @ k.transpose(-2, -1) + bias).softmax(dim=-1)
            out = self.attn_drop(weights) @ v
        out = out.transpose(1, 2).reshape(b, s, p, c)
        return self.proj_drop(self.proj(out))


class WindowTemporalBlock(nn.Module):
    def __init__(self, copied_block, window_size):
        super().__init__()
        if copied_block.sample_drop_ratio != 0:
            raise ValueError("Minimal temporal branch requires the original zero stochastic-depth setting")
        for name in ("norm1", "norm2", "ls1", "ls2", "mlp"):
            setattr(self, name, getattr(copied_block, name))
        self.attn = WindowAttention(copied_block.attn, window_size)

    def forward(self, x, pos=None):
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos))
        return x + self.ls2(self.mlp(self.norm2(x)))


class CrossTimeLocalFusion(nn.Module):
    def __init__(self, spatial_branch, window_size=5):
        super().__init__()
        window_indices(1, window_size)
        if list(spatial_branch.aa_order) != ["frame", "global"]:
            raise ValueError("Minimal temporal branch requires frame/global alternation")
        self.window_size = window_size
        self.depth = spatial_branch.depth
        self.aa_block_size = spatial_branch.aa_block_size
        self.cached_layer_indices = set(spatial_branch.cached_layer_indices)
        self.frame_blocks = deepcopy(spatial_branch.frame_blocks)
        self.temporal_blocks = nn.ModuleList([
            WindowTemporalBlock(deepcopy(block), window_size) for block in spatial_branch.global_blocks
        ])

    def initialize_from_spatial(self, spatial_branch):
        """Copy after upstream loading, never share Parameter objects/storage."""
        self.frame_blocks.load_state_dict(spatial_branch.frame_blocks.state_dict(), strict=True)
        for temporal, spatial in zip(self.temporal_blocks, spatial_branch.global_blocks):
            weights = dict(spatial.state_dict())
            weights["attn.relative_time_bias"] = torch.zeros_like(temporal.attn.relative_time_bias)
            temporal.load_state_dict(weights, strict=True)

    def forward(self, encoded):
        tokens, pos, b, s, patch_start_idx = encoded
        _, p, c = tokens.shape
        outputs = []
        for start in range(0, self.depth, self.aa_block_size):
            frames, temporal = [], []
            for index in range(start, start + self.aa_block_size):
                tokens = tokens.reshape(b * s, p, c)
                block = self.frame_blocks[index]
                tokens = checkpoint(block, tokens, pos, use_reentrant=False) if self.training else block(tokens, pos)
                frames.append(tokens.reshape(b, s, p, c))
            for index in range(start, start + self.aa_block_size):
                tokens = tokens.reshape(b, s, p, c)
                block = self.temporal_blocks[index]
                tokens = checkpoint(block, tokens, pos, use_reentrant=False) if self.training else block(tokens, pos)
                temporal.append(tokens)
            for offset, (frame, time) in enumerate(zip(frames, temporal)):
                outputs.append(torch.cat((frame, time), dim=-1) if start + offset in self.cached_layer_indices else None)
        return outputs, patch_start_idx


def branch_parameter_report(spatial, temporal):
    def count(module):
        return sum(p.numel() for p in module.parameters())
    frame, global_ = count(spatial.frame_blocks), count(spatial.global_blocks)
    time_frame, window = count(temporal.frame_blocks), count(temporal.temporal_blocks)
    ratio = (time_frame + window) / (frame + global_)
    if not 0.99 <= ratio <= 1.01:
        raise ValueError(f"Attention parameter ratio outside [0.99,1.01]: {ratio}")
    return {"spatial_frame": frame, "spatial_global": global_, "P_S": frame + global_,
            "temporal_frame": time_frame, "temporal_window": window, "P_T": time_frame + window, "ratio": ratio}
