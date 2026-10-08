"""TF_DSV41_DRAFT_GROUP (dspark.py): the drafter's attention with wo_a as one grouped launch and wq_a beside wkv on
Par streams equals the serial slices bit for bit: the block output and the window keys it stores, for one stream's
rows (attention, N 1..16) and several streams' (attention_multi, M x N up to 32), eagerly and replayed from a CUDA
graph; under TF_EXL3_ROT_FUSE the grouped launch stays off (the slices)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3 import linear
from tensorfold.families.deepseek_v41.cuda import dspark as DS
from tensorfold.families.deepseek_v41.cuda import kernels as K
from tensorfold.families.deepseek_v41.cuda import serial
from tensorfold.families.deepseek_v41.cuda.weights import AttnW

D, QA, Dh, H, G, NO = 4096, 1024, 512, 32, 4, 1024      # a rank's draft block (hidden, q rank, heads, wo_a groups)
SLOTS = 4


def _layer(seed: int, k: int, n: int, bits: int = 4) -> linear.Exl3Linear:
    rng = np.random.default_rng(seed)
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(k // 16, n // 16, fmt.tile_words(bits)))
                               .astype(np.int16))
    suh = torch.from_numpy((rng.standard_normal(k) * 0.05).astype(np.float16))
    svh = torch.from_numpy((rng.standard_normal(n) * 0.05).astype(np.float16))
    return linear.Exl3Linear.from_tensors(trellis, suh, svh, "mul1")


@pytest.fixture(scope="module")
def ds():
    g = torch.Generator(device="cuda").manual_seed(3)
    a = AttnW(_layer(1, D, QA), _layer(2, D, Dh), (1 + 0.1 * torch.randn(QA, generator=g, device="cuda")).bfloat16(),
              (1 + 0.1 * torch.randn(Dh, generator=g, device="cuda")).bfloat16(), _layer(3, QA, H * Dh),
              [_layer(10 + i, (H // G) * Dh, NO) for i in range(G)], _layer(20, G * NO, D),
              torch.randn(H, generator=g, device="cuda"), 0)
    a.wo_a_grouped = linear.GroupedLinear(a.wo_a)          # as weights.py: the slices read the grouped copy
    freqs = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
    c = SimpleNamespace(head_dim=Dh, sliding_window=128, rms_norm_eps=1e-6)
    eng = SimpleNamespace(tables_rope={0: K.rope_tables(freqs, 1 << 14)}, comm=SimpleNamespace(partials=lambda t: t),
                          par=serial.Par())
    d = DS.DSpark.__new__(DS.DSpark)
    d.c, d.eng, d.dev, d.ring, d.swa_q = c, eng, torch.device("cuda"), serial.DRING, serial.SWA_Q
    d.swa_big = [(torch.randn((SLOTS * serial.DRING, Dh), generator=g, device="cuda")).bfloat16()]
    d.g_base = torch.full((1,), serial.DRING, dtype=torch.long, device="cuda")
    return d, SimpleNamespace(attn=a)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32 if t.element_size() == 4 else torch.int16)


def _one(d, block, x, P, N):
    pos = P + torch.arange(N, device="cuda")
    return d.attention(0, block, x, pos, P)


def _multi(d, block, x, P, base, N):
    M = P.shape[0]
    pos = (P[:, None] + torch.arange(N, device="cuda")[None, :]).reshape(-1)
    rbase = base[:, None].expand(M, N).reshape(-1)
    d.N = N
    return d.attention_multi(0, block, x, pos, P, base, rbase, M)


def _ab(monkeypatch, d, fn, graph=False):
    """fn() with GROUP off, then on, from the same window keys: (outputs, keys) of each."""

    keep = d.swa_big[0].clone()
    got = []
    for on in (False, True):
        monkeypatch.setattr(DS, "GROUP", on)
        d.swa_big[0].copy_(keep)
        if graph and on:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    fn()
            torch.cuda.current_stream().wait_stream(side)
            d.swa_big[0].copy_(keep)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                y = fn()
            d.swa_big[0].copy_(keep)
            gr.replay()
        else:
            y = fn()
        torch.cuda.synchronize()
        got.append((y.clone(), d.swa_big[0].clone()))
    d.swa_big[0].copy_(keep)
    return got


class _Count:
    def __init__(self, f):
        self.f, self.n = f, 0

    def __call__(self, *args, **kw):
        self.n += 1
        return self.f(*args, **kw)


@pytest.mark.parametrize("graph", [False, True])
def test_attention_grouped_equals_slices(monkeypatch, ds, graph):
    d, block = ds
    grouped, par = _Count(block.attn.wo_a_grouped), _Count(d.eng.par)
    monkeypatch.setattr(block.attn, "wo_a_grouped", grouped)
    monkeypatch.setattr(d.eng, "par", par)
    gen = torch.Generator(device="cuda").manual_seed(11)
    for N in ([1, 2, 3, 6, 8, 16] if not graph else [3, 6]):
        x = torch.randn((N, D), generator=gen, device="cuda").bfloat16()
        P = torch.full((1,), 300 + 7 * N, dtype=torch.long, device="cuda")
        (y0, k0), (y1, k1) = _ab(monkeypatch, d, lambda x=x, P=P, N=N: _one(d, block, x, P, N), graph)
        assert y0.dtype == torch.float32 and torch.equal(_bits(y1), _bits(y0)), N
        assert torch.equal(_bits(k1), _bits(k0)), N
    assert grouped.n > 0 and par.n == grouped.n                # the on arm took both


@pytest.mark.parametrize("graph", [False, True])
def test_attention_multi_grouped_equals_slices(monkeypatch, ds, graph):
    d, block = ds
    gen = torch.Generator(device="cuda").manual_seed(12)
    for M, N in ([(1, 6), (2, 3), (3, 6), (4, 8)] if not graph else [(2, 6), (4, 8)]):
        x = torch.randn((M * N, D), generator=gen, device="cuda").bfloat16()
        P = 200 + torch.randint(0, 900, (M,), generator=gen, device="cuda")
        base = torch.arange(M, device="cuda") * serial.DRING
        (y0, k0), (y1, k1) = _ab(monkeypatch, d, lambda x=x, P=P, base=base, N=N: _multi(d, block, x, P, base, N), graph)
        assert torch.equal(_bits(y1), _bits(y0)), (M, N)
        assert torch.equal(_bits(k1), _bits(k0)), (M, N)


def test_rot_fuse_keeps_slices(monkeypatch, ds):
    """TF_EXL3_ROT_FUSE: the grouped call is not taken (its multi-row linear_rot is untested here); the slices run."""

    a = ds[1].attn
    o = torch.randn((3, H, Dh), device="cuda").bfloat16()
    want = DS.DSpark._wo_a(a, o, 3, H, Dh)
    monkeypatch.setattr(linear, "ROT_FUSE", True)
    monkeypatch.setattr(DS, "GROUP", True)
    monkeypatch.setattr(a, "wo_a_grouped", object())        # not callable: taking it raises
    assert torch.equal(_bits(DS.DSpark._wo_a(a, o, 3, H, Dh)), _bits(want))
