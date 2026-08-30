"""CPU reference for CS265 Llama-3-8B-Instruct operators.

Toy sizes by default so it runs anywhere. Set CS265_FULL_SHAPES=1 to use
real 8B dimensions (slow, memory-heavy). Numerics follow the Spring 2026
project handout: RMSNorm, GQA, rotate_half RoPE with base 500000, SwiGLU.
"""

from __future__ import annotations

import os

import numpy as np

EPS = 1e-5
ROPE_BASE = 500_000.0


def rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = EPS) -> np.ndarray:
    # x: (s, d), gamma: (d,)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)) * gamma


def silu(z: np.ndarray) -> np.ndarray:
    return z * (1.0 / (1.0 + np.exp(-z)))


def swiglu(x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray) -> np.ndarray:
    # weights stored (out, in) as in Hugging Face
    gate = x @ w_gate.T
    up = x @ w_up.T
    hidden = silu(gate) * up
    return hidden @ w_down.T


def rope_angles(s: int, hd: int, base: float = ROPE_BASE) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(hd // 2, dtype=np.float64)
    theta = 1.0 / (base ** (2.0 * i / hd))
    p = np.arange(s, dtype=np.float64)[:, None]
    ang = p * theta[None, :]
    return np.cos(ang), np.sin(ang)


def apply_rope_rotate_half(q: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """q: (heads, s, hd). Pair first half with second half (Llama 3 / HF rotate_half)."""
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    # cos/sin: (s, hd/2)
    cos_b = cos[None, :, :]
    sin_b = sin[None, :, :]
    out1 = q1 * cos_b - q2 * sin_b
    out2 = q1 * sin_b + q2 * cos_b
    return np.concatenate([out1, out2], axis=-1)


def softmax_stable(scores: np.ndarray) -> np.ndarray:
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def gqa_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = True,
) -> np.ndarray:
    """q (h,s,hd), k/v (hk,s,hd) → (h,s,hd)."""
    h, s, hd = q.shape
    hk = k.shape[0]
    group = h // hk
    scale = 1.0 / np.sqrt(hd)
    outs = []
    mask = None
    if causal:
        mask = np.triu(np.ones((s, s), dtype=bool), k=1)
    for i in range(h):
        g = i // group
        scores = (q[i] @ k[g].T) * scale
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        alpha = softmax_stable(scores)
        outs.append(alpha @ v[g])
    return np.stack(outs, axis=0)


def decoder_block(
    x: np.ndarray,
    weights: dict[str, np.ndarray],
    h: int,
    hk: int,
    hd: int,
) -> np.ndarray:
    s, d = x.shape
    xn = rmsnorm(x, weights["input_layernorm"])
    q = (xn @ weights["q_proj"].T).reshape(s, h, hd).transpose(1, 0, 2)
    k = (xn @ weights["k_proj"].T).reshape(s, hk, hd).transpose(1, 0, 2)
    v = (xn @ weights["v_proj"].T).reshape(s, hk, hd).transpose(1, 0, 2)
    cos, sin = rope_angles(s, hd)
    q = apply_rope_rotate_half(q, cos, sin)
    k = apply_rope_rotate_half(k, cos, sin)
    o = gqa_attention(q, k, v)
    o_flat = o.transpose(1, 0, 2).reshape(s, h * hd)
    attn_out = o_flat @ weights["o_proj"].T
    x = x + attn_out
    xn = rmsnorm(x, weights["post_attention_layernorm"])
    x = x + swiglu(xn, weights["gate_proj"], weights["up_proj"], weights["down_proj"])
    return x


def random_weights(s: int, d: int, h: int, hk: int, hd: int, dff: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    def w(out: int, inn: int) -> np.ndarray:
        return rng.standard_normal((out, inn)).astype(np.float64) * 0.02

    return {
        "input_layernorm": rng.standard_normal(d) * 0.1 + 1.0,
        "post_attention_layernorm": rng.standard_normal(d) * 0.1 + 1.0,
        "q_proj": w(h * hd, d),
        "k_proj": w(hk * hd, d),
        "v_proj": w(hk * hd, d),
        "o_proj": w(d, h * hd),
        "gate_proj": w(dff, d),
        "up_proj": w(dff, d),
        "down_proj": w(d, dff),
    }


def shapes() -> dict[str, int]:
    if os.environ.get("CS265_FULL_SHAPES") == "1":
        return {"s": 4, "d": 4096, "h": 32, "hk": 8, "hd": 128, "dff": 14336}
    return {"s": 8, "d": 64, "h": 4, "hk": 2, "hd": 16, "dff": 128}


def self_check() -> None:
    rng = np.random.default_rng(0)
    sh = shapes()
    s, d, h, hk, hd, dff = sh["s"], sh["d"], sh["h"], sh["hk"], sh["hd"], sh["dff"]
    assert h * hd == d
    x = rng.standard_normal((s, d))
    w = random_weights(s, d, h, hk, hd, dff, rng)
    y = decoder_block(x, w, h, hk, hd)
    assert y.shape == x.shape
    assert np.isfinite(y).all()

    # RMSNorm: mean of squares of normalized (pre-gamma) should be ~1
    g = np.ones(d)
    n = rmsnorm(x, g)
    rms = np.sqrt(np.mean(n * n, axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-5)

    # Causal: first position output of attention should ignore future K
    q = rng.standard_normal((h, s, hd))
    k = rng.standard_normal((hk, s, hd))
    v = rng.standard_normal((hk, s, hd))
    o = gqa_attention(q, k, v, causal=True)
    v0 = v.copy()
    v0[:, 1:, :] = 999.0
    o2 = gqa_attention(q, k, v0, causal=True)
    assert np.allclose(o[:, 0, :], o2[:, 0, :], atol=1e-6)

    print(f"ok shapes={sh} residual_std={y.std():.4f}")


if __name__ == "__main__":
    self_check()
