"""TF_DSV41_COSTS=depth, host side: the switch's values, the allocator pricing one-stream rounds by the one-stream curve
(and nothing else: off, several streams or another key width plan as before), calibration's layout (rank 0's depth on
both ranks, the curve timed through ``step_multi`` in slot 0 after a prompt and before every slot's reset, the
gathered lengths and the cache key). Draft counts only, never a token; on the GPUs: ``--decoder-test`` and
``--jaybench --jb-serial`` with TF_DSV41_COSTS=depth (notes/dsv41/DEV.md)."""

import pytest

pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v41.cuda import multi as M
from tests.test_dsv41_kept import FakeEngine
from tests.test_dsv41_multi_copy import decoder, live, old_allocate

CALIB = [24.0, 29.0, 34.0, 38.0, 43.0, 49.0] + [49.0 + 6.0 * r for r in range(1, M.ROWS - 5)]
DEPTH = [23.6, 28.0, 32.1, 35.8, 38.0, 40.7]          # decode-bench 2K, 2026-10-08


def test_switch_values():
    assert M.costs_depth(None) == M.costs_depth("") == M.costs_depth("calib") == M.costs_depth("0") == 0
    assert M.costs_depth("depth") == 2048 and M.costs_depth("depth:32768") == 32768
    for bad in ("deep", "depth:", "depth:-1", "depth:0", "one"):
        with pytest.raises(ValueError):
            M.costs_depth(bad)


def priced(d, one=None, width=1 << 20):
    d.curves = [(width, list(CALIB))]
    d.costs = list(CALIB)
    d.draft_curve = [3.6 + 0.4 * i for i in range(16)]
    d.draft_ms = 3.6
    d.overhead = 2.0
    if one is not None:
        d.one_rows, d.one_depth = len(one), 2048
        d.one = (width, one + [one[-1] + CALIB[r] - CALIB[len(one) - 1] for r in range(len(one), M.ROWS)])
    return d


def stream(d, prompt_len=500, acc=0.8, slot_prompt=None):
    s = live(d, slot_prompt or list(range(100, 100 + prompt_len)), [7])
    s.acc = [acc] * len(s.acc)
    return s


def test_off_plans_as_before():
    d = priced(decoder())
    s = stream(d)
    assert d.one is None
    assert d._allocate([s]) == old_allocate(d, [s]) == [4]


def test_one_stream_priced_by_its_curve():
    """acceptance 0.8 a position: the random-row curve stops at 4 drafts, the measured one-stream windows pay for 5."""

    d = priced(decoder(), DEPTH)
    s = stream(d)
    assert d._allocate([s]) == [5]
    assert d.costs == d.one[1]


def test_several_streams_or_another_width_ignore_it():
    d = priced(decoder(), DEPTH)
    a, b = stream(d), stream(d)
    ref = priced(decoder())
    ra, rb = stream(ref), stream(ref)
    assert d._allocate([a, b]) == old_allocate(ref, [ra, rb])
    assert d.costs == CALIB
    d = priced(decoder(), DEPTH)
    d.curves = [(1 << 10, [c - 1 for c in CALIB]), (1 << 20, list(CALIB))]
    d.one = (1 << 10, d.one[1])                        # timed at the narrow width; the stream needs the wide one
    s = stream(d, prompt_len=2000)
    assert d._allocate([s]) == [4] and d.costs == CALIB


class CalEngine(FakeEngine):
    """FakeEngine plus what calibration touches: graphs by rows, a drafter, step_multi recording its rows."""

    def __init__(self) -> None:
        super().__init__(3, 1 << 16, 1 << 14, 1)
        self.widths, self.cap = [1 << 13], 1 << 14
        self.graphs = {r: {} for r in range(1, M.ROWS + 1)}
        self.calls, self.resets, self.prefills = [], [], []
        self.drafter = type("D", (), {"multi_graphs": {1: 0, 2: 0}, "propose_multi": lambda s, items: None,
                                      "propose": lambda s, a, p: None})()
        for i in range(self.slots):
            self.extents[i] = (0, 1 << 14)

    def graph_for(self, rows, need):
        return {"tok": torch.zeros(rows, dtype=torch.long), "pos": torch.zeros(rows, dtype=torch.long),
                "sid": torch.zeros(rows, dtype=torch.long)}

    def _replay_free(self, g):
        pass

    def prefill(self, tokens, final=None):
        self.prefills.append((self.slot, len(self.state.ids), len(tokens), final))
        self.state.ids.extend(tokens)

    def reset(self):
        self.resets.append(self.slot)
        super().reset()

    def step_multi(self, rows):
        call = []
        for slot, token in rows:
            call.append((slot, len(self.views[slot].ids)))
            self.views[slot].ids.append(token)
        self.calls.append(call)
        return None, [0] * len(rows)


def calibrated(monkeypatch, depths, drafts=5, rank=0):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(M, "COSTS_DEPTH", depths[rank])
    monkeypatch.delenv("TF_REVISION", raising=False)
    e = CalEngine()
    d = M.MultiDecoder(e, lambda v: v, rank=rank, drafts=drafts)
    sent = []

    def gather(values):
        sent.append(list(values))
        if len(values) == 1 and values[0] in depths:      # the depth exchange: each rank's own
            return [[depths[0]], [depths[1]]]
        return [list(values), list(values)]

    d.calibrate(gather)
    return d, e, sent


def test_calibration_off_takes_the_old_layout(monkeypatch):
    d, e, sent = calibrated(monkeypatch, (0, 0))
    assert d.one is None and not e.calls and not e.prefills
    assert len(sent[-1]) == M.ROWS * 2 + 2              # two widths' curves, two drafting passes
    assert len(d.draft_curve) == 2


def test_calibration_times_one_stream_before_the_resets(monkeypatch):
    d, e, sent = calibrated(monkeypatch, (2048, 0), rank=1)   # rank 0's depth wins on both
    assert d.one_depth == 2048 and d.one_rows == 6
    assert e.prefills == [(0, 0, 2048, 2048)]
    assert len(e.calls) == 6 * 6
    assert all(c == [(0, 2048 + j) for j in range(len(c))] for c in e.calls)   # slot 0, consecutive at the depth
    assert [len(c) for c in e.calls[::6]] == [1, 2, 3, 4, 5, 6]
    assert len(sent[-1]) == M.ROWS * 2 + 2 + 6
    assert e.resets[-e.slots - 1:] == [0, 0, 1, 2]      # the prompt's slot reset first, then every slot after
    assert all(not v.ids for v in e.views)
    w, curve = d.one
    assert w == 1 << 13 and len(curve) == M.ROWS        # 2048 + 6 rows fit the narrow width's graphs


def test_curves_split_and_extrapolate():
    d = decoder()
    widths = [1 << 13, 1 << 14]
    d.one_rows, d.one_depth = 3, 2048
    rand = [10_000 * (r + 1) for r in range(M.ROWS)]
    raw = rand + [x + 1 for x in rand] + [3600, 4000] + [9000, 15000, 20000]
    d.e.drafter = type("D", (), {"multi_graphs": {1: 0, 2: 0}})()
    d._curves([raw, raw], widths)
    assert d.draft_curve == [3.6, 4.0]
    assert d.one[0] == 1 << 13
    assert d.one[1][:3] == [9.0, 15.0, 20.0]
    assert d.one[1][3:] == [20.0 + 10.0 * j for j in range(1, M.ROWS - 2)]


def test_cache_key(monkeypatch):
    d = decoder()
    d.e.widths, d.e.cap = [], 1 << 14
    monkeypatch.setenv("TF_REVISION", "abc")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "fake")
    a = d._calib_path([1 << 14])
    monkeypatch.setenv("TF_DSV41_COSTS", "")             # (compose passes it empty): the calib arm keeps its key
    assert d._calib_path([1 << 14]) == a
    d.one_rows, d.one_depth = 6, 2048
    b = d._calib_path([1 << 14])
    assert b != a
    d.one_depth = 32768
    assert d._calib_path([1 << 14]) not in (a, b)


def test_natural_ids_without_a_tokenizer():
    ids = M.natural_ids(None, 3000)
    assert len(ids) == 3000 and ids[:5] == M.natural_ids("/nonexistent", 5)


def test_natural_ids_from_the_source(tmp_path):
    tk = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers

    vocab = {"[UNK]": 0, "def": 1, "self": 2, "the": 3, "return": 4}
    tok = tk.Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    ids = M.natural_ids(path, 2100)
    assert len(ids) == 2100 and {1, 2, 3, 4} <= set(ids)
    assert ids == M.natural_ids(path, 2100)              # the same text every call (and on both ranks)
