import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from primesim.utils.gradient import gradient_checkpoint_forward

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention,
    )

    # One compiled flex_attention serves every (shape, BlockMask) pair the causal
    # model produces: the teacher-forcing and intra-view masks in training, plus one
    # pair per chunk in autoregressive sampling. That is more than dynamo's default
    # 8 variants, and running out silently falls back to an eager kernel that
    # materialises the full score matrix -- so raise the limit and fail loudly.
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = 32
    else:
        torch._dynamo.config.cache_size_limit = 32
    if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
        torch._dynamo.config.fail_on_recompile_limit_hit = True
    compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
except (ImportError, AttributeError):  # PyTorch < 2.5: no causal model
    BlockMask = create_block_mask = flex_attention = compiled_flex_attention = None

# The world model sees NUM_VIEWS camera views, stacked along H; self-attention spans all
# views every CROSS_VIEW_INTERVAL-th block and stays within each view otherwise.
NUM_VIEWS = 3
CROSS_VIEW_INTERVAL = 3
# The causal model (stages 1-2) attends chunk-causally over CAUSAL_CHUNK_SIZE latent
# frames: a frame sees every frame of its own chunk and of earlier chunks.
CAUSAL_CHUNK_SIZE = 3
# Parameters PrimeSimModel adds to the base Wan2.2 DiT.
WORLD_MODEL_PREFIXES = ("action_mlp1.", "frame_action_adapter.", "view_embedding.")

# ── Shape Debug ─────────────────────────────────────────────────────────────
# Set to True to print key tensor shapes throughout the WAN pipeline.
# Can be toggled at runtime:
#   import primesim.modules.primesim_dit as _dit
#   _dit.WAN_DEBUG_SHAPES = True
# WAN_DEBUG_SHAPES = True
WAN_DEBUG_SHAPES = False


def _dbg(*args):
    """Print only when WAN_DEBUG_SHAPES is True."""
    if WAN_DEBUG_SHAPES:
        print(*args)


# ────────────────────────────────────────────────────────────────────────────

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode=False,
):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def framewise_action_cross_attn_mask(
    batch_size: int,
    f: int,
    h: int,
    w: int,
    text_len: int,
    T_latent: int,
    device: torch.device,
    dtype: torch.dtype,
    frame_offset: int = 0,
    action_start_frame: int = 0,
    action_target_offset: int = 1,
) -> torch.Tensor:
    """Cross-attention mask binding each latent frame to its own action token.

    The f frames are global frames ``frame_offset..``. Action token 0 targets
    ``action_start_frame + action_target_offset``. Opening windows use offset one
    (latent 0 is observed); stage-4 continuation windows use offset zero.

    Every video token attends to all text tokens. With the opening offset, latent
    frame t attends to action token t-1 and conditioning frame 0 attends to none;
    continuation blocks bind their local action tokens directly to new latents.

    Returns float mask [B, 1, f*h*w, text_len + T_latent]:
      0      = attend
      -inf   = blocked
    """
    hw = h * w
    q_len = f * hw
    k_len = text_len + T_latent

    # video token i belongs to frame i // hw
    q_frames = torch.arange(q_len, device=device) // hw + frame_offset  # [q_len]

    # All tokens attend to text; build causal mask only for action tokens
    mask = torch.ones(q_len, k_len, dtype=torch.bool, device=device)
    if T_latent > 0:
        if action_target_offset not in (0, 1):
            raise ValueError("action_target_offset must be 0 or 1.")
        action_indices = torch.arange(T_latent, device=device)
        action_targets = action_indices + action_start_frame + action_target_offset
        causal = q_frames.unsqueeze(1) == action_targets.unsqueeze(0)
        mask[:, text_len:] = causal

    float_mask = torch.zeros(q_len, k_len, dtype=dtype, device=device)
    float_mask.masked_fill_(~mask, float("-inf"))

    return float_mask[None, None].expand(
        batch_size, 1, q_len, k_len
    )  # [B, 1, q_len, k_len]


def _flex_block_mask(mask_fn, length, device):
    """BlockMask over `length` tokens padded to FlexAttention's 128-token blocks;
    padding rows attend only to themselves so they stay finite."""
    if create_block_mask is None:
        raise RuntimeError("The causal model needs FlexAttention (PyTorch >= 2.5).")
    padded = math.ceil(length / 128) * 128

    def masked(batch_idx, head_idx, query_idx, key_idx):
        return mask_fn(query_idx, key_idx) | (query_idx == key_idx)

    return create_block_mask(
        masked,
        B=None,
        H=None,
        Q_LEN=padded,
        KV_LEN=padded,
        device=device,
        _compile=False,
    )


def chunk_causal_block_mask(f: int, h: int, w: int, device: torch.device):
    """Self-attention over f latent frames of h*w tokens, chunk-causal: a token sees
    every token up to the end of its own CAUSAL_CHUNK_SIZE-frame chunk."""
    if f % CAUSAL_CHUNK_SIZE != 0:
        raise ValueError(
            f"{f} latent frames are not whole {CAUSAL_CHUNK_SIZE}-frame chunks."
        )
    tokens_per_chunk = CAUSAL_CHUNK_SIZE * h * w
    total = f * h * w

    def mask_fn(query_idx, key_idx):
        chunk_end = (query_idx // tokens_per_chunk + 1) * tokens_per_chunk
        return (query_idx < total) & (key_idx < total) & (key_idx < chunk_end)

    return _flex_block_mask(mask_fn, total, device)


def teacher_forcing_block_mask(f: int, h: int, w: int, device: torch.device):
    """Self-attention over [clean frames | noisy frames], f each (stage 1 training).

    A clean token sees clean tokens of its own and earlier chunks; a noisy token sees
    clean tokens of earlier chunks and noisy tokens of its own chunk -- the context a
    chunk has when it is generated autoregressively.
    """
    if f % CAUSAL_CHUNK_SIZE != 0:
        raise ValueError(
            f"{f} latent frames are not whole {CAUSAL_CHUNK_SIZE}-frame chunks."
        )
    half = f * h * w
    tokens_per_chunk = CAUSAL_CHUNK_SIZE * h * w

    def mask_fn(query_idx, key_idx):
        query_clean, key_clean = query_idx < half, key_idx < half
        query_noisy = (query_idx >= half) & (query_idx < 2 * half)
        key_noisy = (key_idx >= half) & (key_idx < 2 * half)
        query_chunk = (query_idx % half) // tokens_per_chunk
        key_chunk = (key_idx % half) // tokens_per_chunk
        clean_visible = query_clean & key_clean & (query_chunk >= key_chunk)
        noisy_visible = query_noisy & (
            (key_clean & (query_chunk > key_chunk))
            | (key_noisy & (query_chunk == key_chunk))
        )
        return clean_visible | noisy_visible

    return _flex_block_mask(mask_fn, 2 * half, device)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, attn_mask=None):
        if BlockMask is not None and isinstance(attn_mask, BlockMask):
            # Causal self-attention: pad to the mask's 128-token blocks, then crop.
            q, k, v = (
                rearrange(t, "b s (n d) -> b n s d", n=self.num_heads)
                for t in (q, k, v)
            )
            length, padded = q.shape[2], int(attn_mask.shape[-1])
            q, k, v = (F.pad(t, (0, 0, 0, padded - length)) for t in (q, k, v))
            flex = (
                compiled_flex_attention if q.device.type == "cuda" else flex_attention
            )
            x = flex(q, k, v, block_mask=attn_mask)[:, :, :length]
            return rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        if attn_mask is not None:
            # The action cross-attention mask needs SDPA; flash attention takes none.
            q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            return rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, attn_mask=None, kv_cache=None, start_token=0):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        if kv_cache is not None:
            # Causal sampling: this block's keys go into the cache and its queries
            # attend to every cached token (earlier blocks, and this block).
            k, v = kv_cache.write(k, v, start_token)
        x = self.attn(q, k, v, attn_mask=attn_mask)
        return self.o(x)


class KVCache:
    """One self-attention layer's keys and values during causal sampling.

    Holds at most `size` tokens -- the last local_attn_size latent frames -- in
    sampling order; writing past the end rolls the oldest tokens out. Rewriting the
    newest block (its denoising steps, then the final clean pass) overwrites it.
    The cache holds detached values: with gradients on (the stage 3 rollout), only
    the current block's keys and values carry them.
    """

    def __init__(self, batch_size, size, dim, dtype, device, sink_size=0):
        self.k = torch.zeros(batch_size, size, dim, dtype=dtype, device=device)
        self.v = torch.zeros(batch_size, size, dim, dtype=dtype, device=device)
        self.sink_size = int(sink_size)
        self.global_end = 0  # tokens of the video written so far
        self.local_end = 0  # tokens held in the cache

    def write(self, k, v, start_token):
        end = start_token + k.shape[1]
        if end < self.global_end:
            raise ValueError(
                f"KV cache cannot move backwards: block ends at {end}, cache at "
                f"{self.global_end}."
            )
        new = end - self.global_end
        local_end = self.local_end + new
        if local_end > self.k.shape[1]:  # roll the oldest tokens out
            evicted = local_end - self.k.shape[1]
            kept = self.local_end - evicted - self.sink_size
            if kept < 0:
                raise ValueError(
                    "KV cache is too small for its sink and current causal chunk."
                )
            if kept:
                source = slice(
                    self.sink_size + evicted,
                    self.sink_size + evicted + kept,
                )
                target = slice(self.sink_size, self.sink_size + kept)
                self.k[:, target] = self.k[:, source].clone()
                self.v[:, target] = self.v[:, source].clone()
            local_end -= evicted
        start = local_end - k.shape[1]
        self.k[:, start:local_end] = k.detach().to(self.k.dtype)
        self.v[:, start:local_end] = v.detach().to(self.v.dtype)
        self.global_end, self.local_end = end, local_end
        if torch.is_grad_enabled():
            return (
                torch.cat([self.k[:, :start], k.to(self.k.dtype)], dim=1),
                torch.cat([self.v[:, :start], v.to(self.v.dtype)], dim=1),
            )
        return self.k[:, :local_end], self.v[:, :local_end]


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor, attn_mask=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)
        x = self.attn(q, k, v, attn_mask=attn_mask)
        return self.o(x)


class FrameActionAdapter(nn.Module):
    """Zero-init per-frame residual action adapter.

    The adapter binds each latent-frame token to the action chunk that produced
    that latent frame.  Frame 0 is the visual prefix/conditioning frame and gets
    a zero action embedding; frame k>0 gets action chunk k-1.  The final
    projection is zero-initialized, so enabling this module preserves pretrained
    Wan behavior until it is trained.
    """

    def __init__(self, chunk_input_size: int, model_dim: int):
        super().__init__()
        self.action_mlp = nn.Sequential(
            nn.Linear(chunk_input_size, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.norm = nn.LayerNorm(model_dim)
        self.fuse = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.gate = nn.Parameter(torch.randn(()))
        self._init_weights()

    def forward(
        self,
        x: torch.Tensor,
        action_chunks: torch.Tensor,
        f: int,
        h: int,
        w: int,
        action_target_offset: int = 0,
    ) -> torch.Tensor:
        # x: [B, f*h*w, D]. ``action_target_offset`` reserves a clean prefix in
        # the first opening block; continuation blocks have no such prefix.
        if action_chunks is None:
            return x
        if action_target_offset not in (0, 1):
            raise ValueError("action_target_offset must be 0 or 1.")
        num_chunks = action_chunks.shape[1]
        if action_target_offset + num_chunks > f:
            raise ValueError(
                "FrameActionAdapter action/latent mismatch: "
                f"offset={action_target_offset}, chunks={num_chunks}, frames={f}."
            )
        action_emb = self.action_mlp(action_chunks.to(device=x.device, dtype=x.dtype))
        zero = torch.zeros(
            action_emb.shape[0],
            1,
            action_emb.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        parts = []
        if action_target_offset:
            parts.append(zero)
        parts.append(action_emb)
        suffix = f - action_target_offset - num_chunks
        if suffix:
            parts.append(zero.expand(-1, suffix, -1))
        per_frame = torch.cat(parts, dim=1)
        per_token = (
            per_frame[:, :, None, None, :]
            .expand(-1, -1, h, w, -1)
            .reshape(x.shape[0], f * h * w, -1)
        )
        delta = self.fuse(torch.cat([self.norm(x), per_token], dim=-1))
        return x + self.gate.to(dtype=x.dtype) * delta

    def _init_weights(self):
        for module in self.action_mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)

        # First linear of fuse: Kaiming He init
        nn.init.kaiming_uniform_(self.fuse[0].weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fuse[0].weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.fuse[0].bias, -bound, bound)
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)


class GateModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        _debug_block_id: int = -1,
        cross_view_attn=False,
        f=0,
        h=0,
        w=0,
        cross_attn_mask=None,
        self_attn_mask=None,
        intra_view_attn_mask=None,
        teacher_forcing=False,
        kv_cache=None,
        start_frame=0,
    ):
        """x is [B, f*h*w, D], or [B, 2*f*h*w, D] with teacher_forcing (clean frames
        then noisy frames). The causal model passes BlockMasks for full (cross-view)
        and intra-view self-attention."""
        if WAN_DEBUG_SHAPES and _debug_block_id == 0:
            _dbg(
                f"  [DiTBlock-0]  x={tuple(x.shape)} t_mod={tuple(t_mod.shape)}  "
                f"freqs={tuple(freqs.shape)}"
            )
            _dbg(
                f"  context={tuple(context.shape)}" if context is not None else None
            )  # [B, text_seq, 4096]")
            _dbg(
                f"  [DiTBlock-0]  cross_view_attn={cross_view_attn}  f={f} h={h} w={w}"
            )
        # t_mod is per token, [B, f*h*w, 6, dim]. msa: self-attention, mlp: feed-forward
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=2)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shift_msa.squeeze(2),
            scale_msa.squeeze(2),
            gate_msa.squeeze(2),
            shift_mlp.squeeze(2),
            scale_mlp.squeeze(2),
            gate_mlp.squeeze(2),
        )
        # Self-Attention (with intra/cross-view support)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        if cross_view_attn:
            # Full attention across all views
            attn_out = self.self_attn(
                input_x,
                freqs,
                attn_mask=self_attn_mask,
                kv_cache=kv_cache,
                start_token=start_frame * h * w,
            )
        else:
            # Intra-view: each view attends only to itself. With teacher forcing the
            # sequence holds two streams (s = 2), [clean | noisy], each f frames.
            B_x, _, D = input_x.shape
            h_pv = h // NUM_VIEWS
            s = 2 if teacher_forcing else 1
            # [B, s*f*h*w, D] → [B, s, f, V, h_pv, w, D] → [B*V, s*f*h_pv*w, D]
            input_x = input_x.view(B_x, s, f, NUM_VIEWS, h_pv, w, D)
            input_x = input_x.permute(0, 3, 1, 2, 4, 5, 6).reshape(
                B_x * NUM_VIEWS, s * f * h_pv * w, D
            )
            # RoPE freqs: take one view slice (all identical due to H-reset)
            freqs_iv = freqs.view(s, f, NUM_VIEWS, h_pv, w, 1, -1)[:, :, 0].reshape(
                s * f * h_pv * w, 1, -1
            )
            attn_out = self.self_attn(
                input_x,
                freqs_iv,
                attn_mask=intra_view_attn_mask,
                kv_cache=kv_cache,
                start_token=start_frame * h_pv * w,
            )
            # [B*V, s*f*h_pv*w, D] → [B, V, s, f, h_pv, w, D]
            #                      → [B, s, f, V, h_pv, w, D] → [B, s*f*h*w, D]
            attn_out = attn_out.view(B_x, NUM_VIEWS, s, f, h_pv, w, D)
            attn_out = attn_out.permute(0, 2, 3, 1, 4, 5, 6).reshape(
                B_x, s * f * h * w, D
            )
        x = self.gate(x, gate_msa, attn_out)
        # Cross-Attention (always full sequence — text attends to all views)
        x = x + self.cross_attn(self.norm3(x), context, attn_mask=cross_attn_mask)
        # FFN
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        # t_mod is per token, [B, f*h*w, dim]
        shift, scale = (
            self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device)
            + t_mod.unsqueeze(2)
        ).chunk(2, dim=2)
        return self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))


class PrimeSimModel(torch.nn.Module):
    """Wan2.2-TI2V-5B DiT as an action-conditioned multiview world model.

    The Wan2.2 DiT plus the world-model parts (action conditioning and the view
    embedding). forward() takes multiview latents whose frame 0 is the clean first
    frame (TI2V) and predicts the flow-matching velocity for the rest.
    """

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        action_dim: int,
        causal: bool = False,
    ):
        """`causal` selects chunk-causal self-attention (stages 1-2) instead of full
        bidirectional attention (fine-tuning); it adds no parameters."""
        super().__init__()
        self.causal = causal
        self._block_masks = {}  # causal BlockMasks by shape, built on first use
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        # World-model parts (parameters named WORLD_MODEL_PREFIXES). Actions condition
        # the video two ways: every 4 action steps (one latent frame's worth -- the
        # Wan VAE compresses time 4x) become one cross-attention context token
        # through action_mlp1, appended after the text tokens, with the framewise
        # mask letting latent frame t see only chunk t-1; and FrameActionAdapter adds
        # the chunks residually to the frames' tokens. The NUM_VIEWS views are
        # stacked along H; each view restarts RoPE's H coordinate and gets an
        # additive view embedding, and self-attention spans all views every
        # CROSS_VIEW_INTERVAL-th block and stays within each view otherwise.
        self.action_dim = action_dim
        # [B, T, action_dim] -> [B, T/4, action_dim*4] -> model dim
        self.action_mlp1 = nn.Sequential(
            nn.Linear(action_dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.frame_action_adapter = FrameActionAdapter(
            chunk_input_size=4 * action_dim,
            model_dim=dim,
        )
        self.view_embedding = nn.Embedding(NUM_VIEWS, dim)

    def unfreeze_world_model(self):
        """Make the world-model parts (WORLD_MODEL_PREFIXES) trainable."""
        for name, param in self.named_parameters():
            if name.startswith(WORLD_MODEL_PREFIXES):
                param.requires_grad_(True)

    def embed_timestep(self, timestep_tokens: torch.Tensor) -> torch.Tensor:
        """[B, N] per-token timesteps -> [B, N, dim] time embeddings."""
        batch_size, num_tokens = timestep_tokens.shape
        t = sinusoidal_embedding_1d(self.freq_dim, timestep_tokens.reshape(-1))
        t = t.to(
            dtype=next(self.time_embedding.parameters()).dtype,
            device=timestep_tokens.device,
        )
        return self.time_embedding(t.reshape(batch_size, num_tokens, self.freq_dim))

    def new_kv_cache(
        self,
        latents: torch.Tensor,
        local_attn_size: int = 20,
        sink_size: int = 0,
    ) -> list:
        """Empty per-layer KV caches for causal sampling of latents [B, V, C, T, H, W],
        holding the last local_attn_size latent frames. Cross-view layers cache all
        views' tokens; intra-view layers one view each."""
        batch_size, _, _, num_frames, h, w = latents.shape
        h_pv, w_p = h // self.patch_size[1], w // self.patch_size[2]
        cache_frames = num_frames if local_attn_size == -1 else local_attn_size
        if cache_frames < num_frames:
            raise ValueError(
                f"KV cache ({cache_frames} frames) is smaller than this segment "
                f"({num_frames} frames)."
            )
        if not 0 <= sink_size < cache_frames:
            raise ValueError(
                "sink_size must be non-negative and smaller than the cache."
            )
        caches = []
        for block_id in range(len(self.blocks)):
            if block_id % CROSS_VIEW_INTERVAL == 0:
                batch, tokens_per_frame = batch_size, NUM_VIEWS * h_pv * w_p
            else:
                batch, tokens_per_frame = batch_size * NUM_VIEWS, h_pv * w_p
            size = cache_frames * tokens_per_frame
            caches.append(
                KVCache(
                    batch,
                    size,
                    self.dim,
                    latents.dtype,
                    latents.device,
                    sink_size=sink_size * tokens_per_frame,
                )
            )
        return caches

    def causal_block_mask(self, teacher_forcing, f, h, w, device):
        """The causal self-attention BlockMask for this shape, built once."""
        key = (teacher_forcing, f, h, w, str(device))
        if key not in self._block_masks:
            build = (
                teacher_forcing_block_mask
                if teacher_forcing
                else (chunk_causal_block_mask)
            )
            self._block_masks[key] = build(f, h, w, device)
        return self._block_masks[key]

    def patchify(self, x: torch.Tensor):
        return self.patch_embedding(x)  # [B, C, T, H, W] -> [B, dim, T, H/2, W/2]

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        clean_latents: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        start_frame: int = 0,
        action_start_frame: int = 0,
        action_target_offset: int = 1,
        use_gradient_checkpointing: bool = False,
    ):
        """latents [B, V, C, T, H, W], frame 0 the clean first frame -> velocity, same
        shape.

        timestep [B] is the denoising time of frames 1..T-1 (frame 0 is embedded at
        t=0), or [B, T] one time per frame (the causal model). context is the T5
        embedding [B, L, text_dim]; action [B, 4*(T-1), action_dim], or [B, 4*T, ...]
        for a prefix in autoregressive sampling. clean_latents, the clean video
        [B, V, C, T, H, W], turns on teacher forcing (causal model, stage 1): each
        noisy chunk then sees the clean frames of earlier chunks.

        kv_cache (from new_kv_cache) is causal sampling one block at a time: latents
        are the block starting at latent frame start_frame and earlier blocks' keys
        and values come from the cache. ``action_start_frame`` places a local action
        trajectory on the global latent clock; ``action_target_offset`` is one for
        an observed opening prefix and zero for a continuation segment.
        """
        teacher_forcing = clean_latents is not None
        if (teacher_forcing or kv_cache is not None) and not self.causal:
            raise ValueError("clean_latents and kv_cache need causal=True.")
        if teacher_forcing and kv_cache is not None:
            raise ValueError("teacher forcing (clean_latents) is for training only.")
        # Timestep: one embedding per token, frame 0 as a clean prefix
        batch_size, latent_frames = latents.shape[0], latents.shape[3]
        spatial_tokens = latents.shape[1] * latents.shape[4] * latents.shape[5] // 4
        timestep = timestep.to(dtype=latents.dtype, device=latents.device)
        if timestep.ndim == 2:
            frame_timesteps = timestep
        else:
            frame_timesteps = torch.cat(
                [
                    torch.zeros_like(timestep[:, None]),
                    timestep[:, None].expand(batch_size, latent_frames - 1),
                ],
                dim=1,
            )
        timestep_tokens = (
            frame_timesteps[:, :, None]
            .expand(batch_size, latent_frames, spatial_tokens)
            .reshape(batch_size, -1)
        )
        t = self.embed_timestep(timestep_tokens)
        t_mod = self.time_projection(t).unflatten(2, (6, self.dim))
        if teacher_forcing:  # the clean stream is embedded at t=0
            t_clean = self.embed_timestep(torch.zeros_like(timestep_tokens))
            t_mod_clean = self.time_projection(t_clean).unflatten(2, (6, self.dim))
            t_mod = torch.cat([t_mod_clean, t_mod], dim=1)

        _dbg("\n=== WAN SHAPE DEBUG [PrimeSimModel.forward] ===")
        _dbg(
            f"  [IN] latents:        {tuple(latents.shape)}  # [B, V, C, T_latent, "
            f"H_vae, W_vae]"
        )
        _dbg(f"  [IN] timestep:       {tuple(timestep.shape)}")
        _dbg(f"  [IN] context (T5):   {tuple(context.shape)}  # [B, text_seq, 4096]")
        _dbg(
            f"  [TIME] t_mod:        {tuple(t_mod.shape)}  # [B, tokens, 6, {self.dim}]"
        )

        context = self.text_embedding(context)
        x = latents

        action_start_frame = int(action_start_frame)
        action_target_offset = int(action_target_offset)
        if action_target_offset not in (0, 1):
            raise ValueError("action_target_offset must be 0 or 1.")

        # Action -> cross-attention context tokens, one per target latent frame.
        action_context_len = 0
        action_frame_chunks = None
        if action is not None:
            action = action.unsqueeze(0) if action.dim() == 2 else action
            action = action.to(dtype=context.dtype, device=context.device)
            B_act, T_act, A = action.shape
            if T_act % 4 != 0 or A != self.action_dim:
                raise ValueError(
                    f"action is [B, T, A] = {tuple(action.shape)}; "
                    f"T must be a multiple of 4 "
                    f"(4 action steps per latent frame) and A must be "
                    f"action_dim={self.action_dim}."
                )
            mlp_param = next(self.action_mlp1.parameters())
            action_frame_chunks = action.to(
                dtype=mlp_param.dtype, device=mlp_param.device
            ).reshape(B_act, T_act // 4, A * 4)
            action_emb_ctx = self.action_mlp1(action_frame_chunks)  # [B, T/4, dim]
            context = context.to(
                device=action_emb_ctx.device, dtype=action_emb_ctx.dtype
            )
            action_context_len = action_emb_ctx.shape[1]
            context = torch.cat([context, action_emb_ctx], dim=1).to(dtype=x.dtype)
            _dbg(
                f"  [ACTION-XATTN] context after action tokens: {tuple(context.shape)}"
            )

        # Patchify, views stacked along H
        b, v = x.shape[:2]
        if v != NUM_VIEWS:
            raise ValueError(
                f"latents have {v} views; the world model takes {NUM_VIEWS}."
            )
        x = rearrange(x, "b v c t h w -> (b v) c t h w")
        x = self.patchify(x)
        x = rearrange(x, "(b v) c t h w -> b c t (v h) w", b=b)
        f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        _dbg(f"  [PATCHIFY] f={f}, h={h}, w={w}  → seq_len={f * h * w}")
        if teacher_forcing:
            clean = clean_latents.to(device=x.device, dtype=latents.dtype)
            clean = self.patchify(rearrange(clean, "b v c t h w -> (b v) c t h w"))
            clean = rearrange(clean, "(b v) c f h w -> b (f v h w) c", b=b)
            if clean.shape != x.shape:
                raise ValueError(
                    f"clean_latents give {tuple(clean.shape)} tokens, latents "
                    f"{tuple(x.shape)}."
                )

        cross_attn_mask = None
        if action_frame_chunks is not None:
            cross_attn_mask = framewise_action_cross_attn_mask(
                x.shape[0],
                f,
                h,
                w,
                context.shape[1] - action_context_len,
                action_context_len,
                device=x.device,
                dtype=x.dtype,
                frame_offset=start_frame,
                action_start_frame=action_start_frame,
                action_target_offset=action_target_offset,
            )
            first_target = action_start_frame + action_target_offset
            block_prefix = min(f, max(first_target - start_frame, 0))
            first_action = max(start_frame - first_target, 0)
            num_actions = min(
                f - block_prefix,
                action_frame_chunks.shape[1] - first_action,
            )
            if num_actions > 0:
                frame_chunks = action_frame_chunks[
                    :, first_action : first_action + num_actions
                ]
                x = self.frame_action_adapter(
                    x,
                    frame_chunks.to(device=x.device),
                    f=f,
                    h=h,
                    w=w,
                    action_target_offset=block_prefix,
                )
                if teacher_forcing:
                    clean = self.frame_action_adapter(
                        clean,
                        frame_chunks.to(device=x.device),
                        f=f,
                        h=h,
                        w=w,
                        action_target_offset=block_prefix,
                    )
            if teacher_forcing:
                cross_attn_mask = torch.cat([cross_attn_mask, cross_attn_mask], dim=2)
        if teacher_forcing:
            # The whole clean stream goes before the noisy one.
            x = torch.cat([clean, x], dim=1)

        # Multi-view: reset RoPE H-coordinate per view
        h_per_view = h // NUM_VIEWS
        h_freqs = self.freqs[1][:h_per_view].repeat(NUM_VIEWS, 1)

        freqs = (
            torch.cat(
                [
                    self.freqs[0][start_frame : start_frame + f]
                    .view(f, 1, 1, -1)
                    .expand(f, h, w, -1),
                    h_freqs.view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        streams = 2 if teacher_forcing else 1
        freqs = freqs.repeat(streams, 1, 1)
        self_attn_mask = intra_view_attn_mask = None
        if self.causal and kv_cache is None:
            self_attn_mask = self.causal_block_mask(teacher_forcing, f, h, w, x.device)
            intra_view_attn_mask = self.causal_block_mask(
                teacher_forcing, f, h_per_view, w, x.device
            )

        # Multi-view: additive view embedding
        view_ids = torch.arange(NUM_VIEWS, device=x.device).repeat_interleave(
            h_per_view * w
        )
        view_ids = view_ids.unsqueeze(0).expand(f, -1).reshape(f * h * w)
        view_ids = view_ids.repeat(streams).to(self.view_embedding.weight.device)
        x = x + self.view_embedding(view_ids).unsqueeze(0).to(
            device=x.device, dtype=x.dtype
        )

        context = context.to(device=x.device, dtype=x.dtype)
        _dbg(
            f"  [BLOCKS] entering {len(self.blocks)} DiTBlocks  x={tuple(x.shape)}  "
            f"context={tuple(context.shape)}"
        )

        for block_id, block in enumerate(self.blocks):
            cross_view = block_id % CROSS_VIEW_INTERVAL == 0
            if WAN_DEBUG_SHAPES and not use_gradient_checkpointing:
                x = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                    _debug_block_id=block_id,
                    cross_attn_mask=cross_attn_mask,
                    cross_view_attn=cross_view,
                    f=f,
                    h=h,
                    w=w,
                    self_attn_mask=self_attn_mask,
                    intra_view_attn_mask=intra_view_attn_mask,
                    teacher_forcing=teacher_forcing,
                    kv_cache=None if kv_cache is None else kv_cache[block_id],
                    start_frame=start_frame,
                )
            else:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    cross_view_attn=cross_view,
                    f=f,
                    h=h,
                    w=w,
                    cross_attn_mask=cross_attn_mask,
                    self_attn_mask=self_attn_mask,
                    intra_view_attn_mask=intra_view_attn_mask,
                    teacher_forcing=teacher_forcing,
                    kv_cache=None if kv_cache is None else kv_cache[block_id],
                    start_frame=start_frame,
                )

        if teacher_forcing:
            x = x[:, f * h * w :]  # drop the clean stream
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        x = rearrange(x, "b c t (v h) w -> b v c t h w", v=v)
        _dbg(
            f"  [OUT] x (unpatchify): {tuple(x.shape)}  # [B, V, out_C, T_latent, "
            f"H_vae, W_vae]"
        )
        _dbg("=== END WAN SHAPE DEBUG ===")
        return x
