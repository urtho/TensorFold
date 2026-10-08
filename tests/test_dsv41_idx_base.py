"""TF_DSV41_IDX_BASE: a decode graph piece's index tensors (``SerialEngine._round_bases``) and the shifted top-k
(``_shifted``) equal the expressions every layer computes otherwise, for the sources the piece reads only; the memo
lives exactly as long as one ``layers`` call. CPU, plus a captured-graph check on a side stream (GPU)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import serial as S
from tensorfold.families.deepseek_v41.cuda.serial import DRING, SerialEngine

CFG = Config.from_dict(json.loads((Path(__file__).parent / "fixtures" / "deepseek_v41" / "config.json").read_text()))


def engine(R: int, seed: int, device: str = "cpu", slots: int = 16) -> SerialEngine:
    """A SerialEngine shell with the fixture's layers (ratio, compressor, indexer as weights.py assigns them)."""

    g = torch.Generator().manual_seed(seed)
    c = CFG
    eng = SerialEngine.__new__(SerialEngine)
    eng.c = c
    eng.w = SimpleNamespace(layers=[
        SimpleNamespace(index=i, attn=SimpleNamespace(
            ratio=c.compress_ratios[i], compressor=object() if i in c.kv_source_layer_ids else None,
            indexer=object() if i in c.index_source_layer_ids else None))
        for i in range(c.num_hidden_layers)])
    # aligned extents of up to ~10M tokens (ratio-2 sources: odd bases never occur, but // 2 must floor anyway)
    eng.ebase = (torch.randint(0, 10_000_000, (slots,), generator=g) * torch.randint(1, 3, (slots,), generator=g)
                 ).to(device)
    eng._sid = torch.randint(0, slots, (R,), generator=g).to(device)
    eng.topk = {}
    return eng


def positions(R: int, seed: int, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    near = torch.tensor([0, 1, 2, DRING - 1, DRING, DRING + 1, 2 * DRING - 1, 65535, 65536, 614399])
    pos = torch.where(torch.rand((R,), generator=g) < 0.5, near[torch.randint(0, len(near), (R,), generator=g)],
                      torch.randint(0, 700_000, (R,), generator=g))
    return pos.to(device)


def kv_src(L: int) -> int:
    return max(s for s in CFG.kv_source_layer_ids if s <= L)


def idx_src(L: int) -> int:
    return max(s for s in CFG.index_source_layer_ids if s <= L)


def pieces():
    c = CFG
    return [(0, c.engram_layer_ids[0]), (c.engram_layer_ids[0], c.engram_layer_ids[1]),
            (c.engram_layer_ids[1], c.num_hidden_layers)]


def test_fixture_shape():
    """The layout these tests assume: 38 compressed layers, kv sources 2/8/14/20, index sources up to 36."""

    c = CFG
    assert sum(1 for r in c.compress_ratios if r > 0) == 38
    assert tuple(c.kv_source_layer_ids) == (2, 8, 14, 20)
    assert len({(idx_src(L), kv_src(L)) for L in range(c.num_hidden_layers) if c.compress_ratios[L] > 0}) == 8


@pytest.mark.parametrize("R", [1, 2, 5, 6, 16, 17, 32])
@pytest.mark.parametrize("piece", range(3))
def test_bases_equal_the_per_layer_expressions(R, piece):
    first, last = pieces()[piece]
    eng = engine(R, R * 7 + piece)
    pos = positions(R, R + 100 * piece)
    b = eng._round_bases(pos, first, last)
    sid = eng._sid
    for k, want in [("wbase", sid * DRING), ("wslot", sid * DRING + pos % DRING)]:
        assert b[k].dtype == want.dtype and torch.equal(b[k], want), k
    lays = range(first, last)
    ratio2 = any(CFG.compress_ratios[L] == 2 and L in CFG.kv_source_layer_ids for L in lays)
    assert ("rprev" in b) == ratio2
    if ratio2:                                                  # compress: roff + (ends - 1).clamp(min=0) % rs
        want = sid * DRING + (pos - 1).clamp(min=0) % DRING
        assert b["rprev"].dtype == want.dtype and torch.equal(b["rprev"], want)
    used = {kv_src(L) for L in lays if CFG.compress_ratios[L] > 0}
    assert {k[1] for k in b if isinstance(k, tuple) and k[0] == "e"} == used   # only the sources read
    for s in used:
        old = eng.ebase[eng._sid] // CFG.layer_ratios[s]        # _ebase as before
        assert b[("e", s)].dtype == old.dtype and torch.equal(b[("e", s)], old)
        assert b[("ei", s)].dtype == old.int()[:, None].dtype and torch.equal(b[("ei", s)], old.int()[:, None])


def test_pieces_build_only_their_sources():
    eng = engine(3, 1)
    pos = positions(3, 1)
    got = [sorted(k[1] for k in eng._round_bases(pos, f, t) if isinstance(k, tuple) and k[0] == "e")
           for f, t in pieces()]
    assert got == [[], [2, 8], [14, 20]]


def test_ebase_reads_the_memo_else_computes():
    eng = engine(4, 2)
    pos = positions(4, 2)
    old = {s: eng._ebase(s) for s in CFG.kv_source_layer_ids}           # no memo: the expression
    eng._ebc = eng._round_bases(pos, 14, CFG.num_hidden_layers)
    for s in (14, 20):
        assert eng._ebase(s) is eng._ebc[("e", s)] and torch.equal(eng._ebase(s), old[s])
    for s in (2, 8):                                                    # not in this piece: computed as before
        assert torch.equal(eng._ebase(s), old[s])


def shift_old(eng, idx, src):
    return torch.where(idx >= 0, idx + eng._ebase(src).int()[:, None], idx)


@pytest.mark.parametrize("R", [1, 6, 32])
def test_shifted_once_a_pair_and_equal(R):
    g = torch.Generator().manual_seed(R)
    eng = engine(R, R)
    pos = positions(R, R)
    for first, last in pieces()[1:]:
        eng._ebc = eng._round_bases(pos, first, last)
        eng._idxc = {}
        built = 0
        for L in range(first, last):
            if CFG.compress_ratios[L] == 0:
                continue
            if L in CFG.index_source_layer_ids:                         # select(), then attention's invalidation
                t = torch.randint(-1, 4_000_000, (R, CFG.index_topk), generator=g, dtype=torch.int32)
                t[:, -7:] = -1
                eng.topk[L] = t
                for k in [k for k in eng._idxc if k[0] == L]:
                    del eng._idxc[k]
            n = len(eng._idxc)
            got = eng._shifted(idx_src(L), kv_src(L))
            built += len(eng._idxc) - n
            want = shift_old(eng, eng.topk[idx_src(L)], kv_src(L))
            assert got.dtype == want.dtype == torch.int32 and torch.equal(got, want), L
        pairs = {(idx_src(L), kv_src(L)) for L in range(first, last) if CFG.compress_ratios[L] > 0}
        assert built == len(pairs) and set(eng._idxc) == pairs
    eng._ebc = eng._idxc = None


def test_shifted_follows_a_new_topk():
    """A top-k replaced without the invalidation (identity check) is shifted again, never served stale."""

    eng = engine(2, 5)
    eng._ebc = eng._round_bases(positions(2, 5), 14, CFG.num_hidden_layers)
    eng._idxc = {}
    eng.topk[14] = torch.zeros((2, 8), dtype=torch.int32)
    a = eng._shifted(14, 14)
    assert eng._shifted(14, 14) is a                                    # cached
    eng.topk[14] = torch.full((2, 8), 3, dtype=torch.int32)
    b = eng._shifted(14, 14)
    assert b is not a and torch.equal(b, shift_old(eng, eng.topk[14], 14))


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("static", [False, True])
def test_layers_memo_lifetime(monkeypatch, on, static):
    monkeypatch.setattr(S, "IDX_BASE", on)
    eng = engine(2, 9)
    pos = positions(2, 9)
    seen = {}

    def body(carry, pos_, rows, first, last, static_, attn_last):
        seen["ebc"], seen["idxc"] = eng._ebc, eng._idxc
        return carry

    monkeypatch.setattr(eng, "_layers", body)
    assert eng.layers(("c",), pos, {}, 14, CFG.num_hidden_layers, static) == ("c",)
    if on and static:
        assert seen["ebc"] is not None and seen["idxc"] == {}
    else:
        assert seen["ebc"] is None and seen["idxc"] is None             # the old per-layer code path
    assert eng._ebc is None and eng._idxc is None

    def boom(*a):
        raise RuntimeError("x")

    monkeypatch.setattr(eng, "_layers", boom)
    with pytest.raises(RuntimeError):
        eng.layers(("c",), pos, {}, 14, CFG.num_hidden_layers, static)
    assert eng._ebc is None and eng._idxc is None                       # cleared on the way out too


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_memo_read_on_a_side_stream_in_a_graph():
    """The race the eager build fixes: a Par-like fork whose two branches both read wslot / rprev (kv_branch,
    comp_branch of part_b's first layer). Built on main before the fork, read on two side streams inside a captured
    graph, replayed with new slots and positions: the rows written equal the per-layer expressions'."""

    R = 6
    eng = engine(R, 11, device="cuda")
    pos = positions(R, 11, device="cuda")
    io_sid, io_pos = eng._sid.clone(), pos.clone()
    eng._sid = io_sid
    ring_a = torch.zeros((16 * DRING,), dtype=torch.long, device="cuda")
    ring_b = torch.zeros((16 * DRING,), dtype=torch.long, device="cuda")
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    vals = torch.arange(1, R + 1, device="cuda")

    def step():
        b = eng._round_bases(io_pos, 14, CFG.num_hidden_layers)
        main = torch.cuda.current_stream()
        for st in (s1, s2):
            st.wait_stream(main)
        with torch.cuda.stream(s1):
            ring_a.index_copy_(0, b["wslot"], vals)
        with torch.cuda.stream(s2):
            ring_b.index_copy_(0, b["rprev"], vals * 10)
            ring_b.index_copy_(0, b["wslot"], vals * 100)
        for st in (s1, s2):
            main.wait_stream(st)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        step()
    g = torch.Generator().manual_seed(3)
    for _ in range(200):
        io_sid.copy_(torch.randperm(16, generator=g)[:R])                # distinct slots: no write collisions
        io_pos.copy_(positions(R, int(torch.randint(0, 1 << 30, (1,), generator=g))))
        ring_a.zero_()
        ring_b.zero_()
        gr.replay()
        want_a = torch.zeros_like(ring_a).index_copy_(0, io_sid * DRING + io_pos % DRING, vals)
        want_b = torch.zeros_like(ring_b)
        want_b.index_copy_(0, io_sid * DRING + (io_pos - 1).clamp(min=0) % DRING, vals * 10)
        want_b.index_copy_(0, io_sid * DRING + io_pos % DRING, vals * 100)
        torch.cuda.synchronize()
        assert torch.equal(ring_a, want_a) and torch.equal(ring_b, want_b)
