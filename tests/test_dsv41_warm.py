"""The boot warm-ups (TF_DSV41_WARM_SERVING / _RARE / _TRACE): the full battery's prompts are the old ones draw for
draw, the trim battery's chunks reach the same Triton integer classes, the switches stay out of the calibration and
NVMe keys, ``late_kernels``' trace records first loads only. GPU: ``SerialEngine.warm_rare`` on a shell engine
reaches its two rare shapes, leaves every cache byte as it was, and covers a later call of the same shapes."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda import late_kernels as LK
from tensorfold.families.deepseek_v41.cuda import engine as EN
from tensorfold.families.deepseek_v41.cuda import kvdisk as D


def old_battery():
    """engine._warm_serving's waves before the switch (a5dd0ca), as (prompt, tokens, sampled)."""

    rng = random.Random(0)

    def ids(n: int) -> list[int]:
        return [rng.randrange(1000, 100000) for _ in range(n)]

    doc = ids(3000)
    waves = [[ids(n)] for n in (1, 7, 16, 100, 2048, 2048 + 33, 2048 + 48)]
    waves += [[doc], [doc + ids(40)], [doc + ids(2100)], [ids(500), ids(2600), ids(5000)]]
    waves = [[(p, 12, False) for p in wave] for wave in waves]
    waves.append([(ids(300), 12, True)])
    return waves


def test_full_battery_is_the_old_one():
    assert [specs for _, specs in EN.serving_waves("full")] == old_battery()


def test_modes(monkeypatch):
    for v, mode in [("", "full"), ("1", "full"), ("full", "full"), ("0", "0"), ("trim", "trim"), ("audit", "audit"),
                    ("lean", "full")]:
        monkeypatch.setenv("TF_DSV41_WARM_SERVING", v)
        assert EN.warm_serving_mode() == mode
    monkeypatch.delenv("TF_DSV41_WARM_SERVING")
    monkeypatch.delenv("TF_DSV41_WARM_RARE", raising=False)
    monkeypatch.delenv("TF_DSV41_WARM_TRACE", raising=False)
    assert EN.warm_serving_mode() == "full" and not EN.warm_rare_on() and not EN.warm_trace_on()
    monkeypatch.setenv("TF_DSV41_WARM_SERVING", "trim")
    assert EN.warm_rare_on() and not EN.warm_trace_on()
    monkeypatch.setenv("TF_DSV41_WARM_RARE", "0")
    assert not EN.warm_rare_on()
    monkeypatch.setenv("TF_DSV41_WARM_SERVING", "audit")
    assert EN.warm_trace_on()
    monkeypatch.setenv("TF_DSV41_WARM_SERVING", "")
    monkeypatch.setenv("TF_DSV41_WARM_RARE", "1")
    monkeypatch.setenv("TF_DSV41_WARM_TRACE", "1")
    assert EN.warm_rare_on() and EN.warm_trace_on()


# -- the chunks a battery prefills (multi._step + serial.prefill, a stream at a time) ----------------------------------
MAX_ROWS, ROWS, KEEP_MIN, MARGIN = 2048, 32, 256, 256 - 130     # serial / multi constants (DRING - WINDOW_ROWS)


def steps(n: int, cached: int, rows: int) -> list[tuple[int, int]]:
    """(p0, R) of every forward a prompt of n tokens takes from ``cached``, ``rows`` a fill step."""

    out, pos = [], cached
    while pos < n:
        stop = min(n, pos + rows)
        points = sorted({p for p in [(n - 1) // MAX_ROWS * MAX_ROWS] if p >= KEEP_MIN and cached < p < n - MARGIN})
        split = next((p for p in points if pos + ROWS < p < stop), None)
        stop = split if split is not None else stop
        base, length = pos, stop - pos
        if 0 < length < ROWS + 1 and pos > 0:
            base = pos - (ROWS + 1 - length)
        starts = list(range(base, stop, MAX_ROWS))
        if len(starts) > 1 and stop - starts[-1] < ROWS + 1:
            starts[-1] = stop - ROWS - 1
        out += [(s, e - s) for s, e in zip(starts, starts[1:] + [stop])]
        pos = stop
    return out


def cls(v: int) -> str:
    return "one" if v == 1 else "16" if v % 16 == 0 else "other"


def classes(chunks: list[tuple[int, int]]) -> set:
    eager = [(p0, R) for p0, R in chunks if R > ROWS]
    return ({("R", p0 > 0, cls(R)) for p0, R in eager}
            | {("entries", r, cls((p0 + R) // r)) for p0, R in eager for r in (1, 2)})


def test_steps_model():
    assert steps(2081, 0, 8 * MAX_ROWS) == [(0, 2048), (2048, 33)]
    assert steps(5100, 3000, 8 * MAX_ROWS) == [(3000, 1096), (4096, 1004)]
    assert steps(2610, 2600, MAX_ROWS) == [(2577, 33)]                         # a short tail backs up


def test_trim_reaches_the_full_batterys_classes():
    full = [steps(n, 0, 8 * MAX_ROWS) for n in (1, 7, 16, 100, 2048, 2081, 2096, 3000)]
    full += [steps(3040, 3000, 8 * MAX_ROWS), steps(5100, 3000, 8 * MAX_ROWS), steps(300, 0, 8 * MAX_ROWS)]
    for n in (500, 2600, 5000):                                                 # filled alone or beside a decode
        full += [steps(n, 0, 8 * MAX_ROWS), steps(n, 0, MAX_ROWS)]
    waves = dict(EN.serving_waves("trim"))
    doc = waves["doc 2600"][0][0]
    assert len(doc) == 2600 and waves["100 sampled | doc + 48"][1][0][:2600] == doc
    assert waves["100 sampled | doc + 48"][0][2] and len(waves["7"][0][0]) <= ROWS
    trim = [steps(2600, 0, 8 * MAX_ROWS), steps(100, 0, 8 * MAX_ROWS), steps(2648, 2600, MAX_ROWS),
            steps(2648, 2600, 8 * MAX_ROWS)]
    assert trim[0] == [(0, 2048), (2048, 552)] and trim[2] == trim[3] == [(2600, 48)]
    want = set().union(*(classes(c) for c in full))
    got = set().union(*(classes(c) for c in trim))
    assert want <= got, want - got
    assert sum(R for c in trim[:3] for _, R in c) + 7 < 3000                 # prompt rows prefilled (full: ~19.9K)


# -- keys -------------------------------------------------------------------------------------------------------------
def test_warm_switches_stay_out_of_the_nvme_key(monkeypatch):
    for k in [k for k in list(__import__("os").environ) if k.startswith("TF_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("TF_DSV41_WARM_SERVING", "")
    base = D.knobs_from_env()
    assert base == {"TF_DSV41_WARM_SERVING": ""}                # (as the compose file sets it today)
    for v in ("trim", "full", "audit"):
        monkeypatch.setenv("TF_DSV41_WARM_SERVING", v)
        monkeypatch.setenv("TF_DSV41_WARM_RARE", "1")
        monkeypatch.setenv("TF_DSV41_WARM_TRACE", "1")
        assert D.knobs_from_env() == base
    for v in ("1", "0"):                                         # old values: their old key
        monkeypatch.setenv("TF_DSV41_WARM_SERVING", v)
        assert D.knobs_from_env()["TF_DSV41_WARM_SERVING"] == v


def test_warm_switches_stay_out_of_the_calibration_key(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tensorfold.families.deepseek_v41.cuda.multi import MultiDecoder

    for k in [k for k in list(__import__("os").environ) if k.startswith("TF_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("TF_REVISION", "abc")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "GB10")
    m = MultiDecoder.__new__(MultiDecoder)
    m.e = SimpleNamespace(cap=65536, slots=4, drafter=None)
    m.drafts, m.rank = 5, 0
    monkeypatch.setenv("TF_DSV41_WARM_SERVING", "")
    base = m._calib_path([65536])
    for v in ("trim", "full", "audit"):
        monkeypatch.setenv("TF_DSV41_WARM_SERVING", v)
        monkeypatch.setenv("TF_DSV41_WARM_RARE", "0")
        monkeypatch.setenv("TF_DSV41_WARM_TRACE", "1")
        assert m._calib_path([65536]) == base
    monkeypatch.setenv("TF_DSV41_WARM_SERVING", "0")
    assert m._calib_path([65536]) != base                       # (an old value keys as before)
    monkeypatch.setenv("TF_DSV41_WARM_SERVING", "")
    monkeypatch.setenv("TF_DSV41_HC_SIDE", "0")
    assert m._calib_path([65536]) != base


# -- late_kernels.trace -----------------------------------------------------------------------------------------------
def test_trace_records_first_loads_only(monkeypatch):
    chain = []
    monkeypatch.setattr(LK, "_chain", lambda: SimpleNamespace(add=chain.append))
    monkeypatch.setattr(LK, "_tracing", [])
    monkeypatch.setattr(LK, "_seen", set())
    monkeypatch.setattr(LK, "_new", [])
    monkeypatch.setattr(LK, "_armed", [])
    monkeypatch.setattr(LK, "late", {})
    assert LK.take() == []
    assert LK.trace() and LK.trace() and chain == [LK._trace_hook]       # installed once
    LK._trace_hook(None, None, "_k1", None, "0123456789abcdef")
    LK._trace_hook(None, None, "_k2", None, "ffffffffffffffff")
    LK._trace_hook(None, None, "_k1", None, "0123456789abcdef")
    assert LK.take() == ["_k1:0123456789ab", "_k2:ffffffffffff"]
    LK._trace_hook(None, None, "_k2", None, "ffffffffffffffff")
    LK._trace_hook(None, None, "_k2", None, "eeeeeeeeeeeeeeee")         # another specialization
    assert LK.take() == ["_k2:eeeeeeeeeeee"] and LK.take() == []
    LK._hook(None, None, "_k3", None, "1")                                 # not armed: no late load
    assert LK.late == {}


def test_trace_without_the_hook(monkeypatch):
    monkeypatch.setattr(LK, "_chain", lambda: None)
    monkeypatch.setattr(LK, "_tracing", [])
    assert not LK.trace() and not LK.arm()


# -- warm_rare (GPU) --------------------------------------------------------------------------------------------------
def _shell(limit: int):
    import torch

    from tensorfold.families.deepseek_v41.config import Config
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda.serial import RING, Caches, SerialEngine
    from tensorfold.families.deepseek_v41.reference import inv_freq

    c = Config.from_dict(json.loads((Path(__file__).parent / "fixtures" / "deepseek_v41" / "config.json").read_text()))
    dev, BF, H = torch.device("cuda"), torch.bfloat16, 32
    g = torch.Generator(device="cuda").manual_seed(0)

    class Lin:
        def __init__(self, i: int, o: int) -> None:
            self.n, self.w = o, (torch.randn((o, i), generator=g, device=dev) * i ** -0.5).to(BF)

        def __call__(self, x):
            return x @ self.w.T

    wq_a, ix = Lin(c.hidden_size, c.q_lora_rank), SimpleNamespace(
        wq_b=Lin(c.q_lora_rank, c.index_n_heads * c.index_head_dim),
        weights_proj=torch.randn((c.index_n_heads, c.hidden_size), generator=g, device=dev).half())
    lays = [SimpleNamespace(index=i, attn=SimpleNamespace(
        ratio=c.layer_ratios[i], indexer=ix if i in c.index_source_layer_ids else None, wq_a=wq_a,
        q_norm=torch.ones(c.q_lora_rank, dtype=BF, device=dev), wq_b=SimpleNamespace(n=H * c.head_dim),
        sink=torch.zeros(H, dtype=torch.float32, device=dev))) for i in range(c.num_hidden_layers)]
    e = SerialEngine.__new__(SerialEngine)
    e.c, e.dev, e.limit, e.w = c, dev, limit, SimpleNamespace(layers=lays)
    e.tables_rope = {r: K.rope_tables(inv_freq(c, r, dev), limit) for r in set(c.layer_ratios)}
    e.attnbuf = K.AttnBuffers(K.FULL_ROWS, H, c.head_dim, c.index_topk + c.sliding_window, device=dev)
    e.candidates = torch.full((7, 3), 5, device=dev)

    def rows(n: int, dim: int, **kw):
        t = K.Fp4Rows(n, dim, device=dev, **kw)
        t.q.copy_(torch.randint(0, 256, t.q.shape, generator=g, device=dev, dtype=torch.uint8))
        t.s.copy_(torch.randint(100, 140, t.s.shape, generator=g, device=dev, dtype=torch.uint8))
        return t

    srcs = c.kv_source_layer_ids
    e.state = Caches(limit, [torch.randn((RING, c.head_dim), generator=g, device=dev).to(BF) for _ in lays],
                     {s: rows(limit // c.layer_ratios[s] + 1, c.head_dim) for s in srcs}, {},
                     {s: rows(limit // c.layer_ratios[s] + 1, c.index_head_dim, group=32, scale="ue8m0") for s in srcs})
    return e


def _bytes(e) -> list:
    st = e.state
    return [t.clone() for t in [*st.swa, *(x for r in st.comp.values() for x in (r.q, r.s)),
                                *(x for r in st.ik.values() for x in (r.q, r.s))]]


@pytest.mark.parametrize("limit", [34816])
def test_warm_rare_reaches_its_shapes_and_changes_nothing(limit, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA only")
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda.weights import Linear

    monkeypatch.setattr(K, "FULL_DEQ_MIB", 16)                  # past the shared decode at 16K entries (ratio 1 and 2)
    e = _shell(limit)
    ns, deq = [], []
    dequant_rows, deq_entries = K.Fp4Rows.dequant_rows, K.deq_entries

    def rec_rows(self, out, off, n):
        ns.append((self.group, n))
        return dequant_rows(self, out, off, n)

    def rec_deq(comp, n_comp):
        out = deq_entries(comp, n_comp)
        deq.append(out is None)
        return out

    monkeypatch.setattr(K.Fp4Rows, "dequant_rows", rec_rows)
    monkeypatch.setattr(K, "deq_entries", rec_deq)
    before, cand = _bytes(e), e.candidates
    LK.trace()
    LK.take()
    assert e.warm_rare() == []
    warmed = LK.take()
    assert (32, 1) in ns                                        # an indexer key segment of one key
    assert deq and all(deq)                                     # prompt attention over packed entries (FMT 2)
    assert all(torch.equal(a, b) for a, b in zip(before, _bytes(e)))
    assert e.candidates is cand and not Linear.prompt_mode
    # the same calls on other values load nothing new (the specializations are the warmed ones)
    c, st, R = e.c, e.state, 2048
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn((R, c.hidden_size), generator=g, device="cuda").to(torch.bfloat16)
    Linear.prompt_mode = True
    try:
        topk = {}
        for lw in e.w.layers:
            a = lw.attn
            if a.indexer is not None:
                pos = torch.arange((K.SELECT_SEG + 1) * a.ratio - R, (K.SELECT_SEG + 1) * a.ratio, device="cuda")
                qr = K.rmsnorm(a.wq_a(x), a.q_norm, c.rms_norm_eps)
                topk[lw.index] = (pos, e.select(lw, qr, x, pos, static=False))
        for L in (2, 20, 39):
            a = e.w.layers[L].attn
            src = max(s for s in c.kv_source_layer_ids if s <= L)
            pos, idx = topk[max(s for s in c.index_source_layer_ids if s <= L)]
            cos, sin = e.tables_rope[a.ratio]
            q = torch.randn((R, 32, c.head_dim), generator=g, device="cuda").to(torch.bfloat16)
            K.mqa(q, st.comp[src], idx, st.swa[L], pos, a.sink, c.sliding_window, e.attnbuf, c.head_dim ** -0.5,
                  cos, sin)
        torch.cuda.synchronize()
    finally:
        Linear.prompt_mode = False
    assert LK.take() == [], warmed
