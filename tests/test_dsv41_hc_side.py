"""TF_DSV41_HC_SIDE in serial.layers: the HC side stream is used only from ``layers`` (never the drafter's
``eng.hc``), and joined before every reader of post / comb / pre and where ``layers`` ends (CPU: recorded calls)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v41.cuda import serial as S
from tensorfold.families.deepseek_v41.cuda.serial import SerialEngine

SIDE = object()                                             # the HC stream (a token: nothing runs)


class _Main:
    def __init__(self, log):
        self.log = log

    def wait_stream(self, st):
        assert st is SIDE
        self.log.append("join")


def engine(monkeypatch, log, side_on=True, n_layers=4, engram=(1,)):
    eng = SerialEngine.__new__(SerialEngine)
    eng.c = SimpleNamespace(rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20, dspark_target_layer_ids=())
    eng.w = SimpleNamespace(layers=[SimpleNamespace(index=i, engram=object() if i in engram else None,
                                                    hc_attn=SimpleNamespace(fn=0, base=0, scale=0, norm=0),
                                                    hc_ffn=SimpleNamespace(fn=0, base=0, scale=0, norm=0))
                                    for i in range(n_layers)])
    eng.hcbuf, eng.drafter, eng.debug = None, None, None
    eng._hc_stream = SIDE if side_on else None
    eng._join = lambda final=False: None
    eng.engram = lambda layer, X, rows: log.append("engram") or X
    eng.attention = lambda layer, x, pos, static: log.append("attn") or x
    eng.moe = lambda layer, x, R: log.append("moe") or x

    def pre(X, *a, side=None):
        log.append("pre-side" if side is not None else "pre")
        R = X.shape[0]
        return torch.zeros(R, 4), torch.zeros(R, 4, 4), torch.zeros(R, 8), torch.zeros(R, 4)

    def post(b, X, post_w, comb):
        log.append("post")
        return X

    monkeypatch.setattr(S.hcf, "pre", pre)
    monkeypatch.setattr(S.hcf, "post", post)
    monkeypatch.setattr(S.torch.cuda, "current_stream", lambda: _Main(log))
    return eng


def check(log):
    """Every read of a side pre's outputs (post, the next pre's collapse) comes after a join; the log ends joined."""

    live = False
    for e in log:
        if e in ("post", "pre", "pre-side") and live:
            raise AssertionError(f"{e} before the join: {log}")
        if e == "pre-side":
            live = True
        elif e == "join":
            live = False
    assert not live, log


@pytest.mark.parametrize("R", [1, 6, 32])
@pytest.mark.parametrize("first", [0, 1])
def test_layers_joins_before_every_reader(monkeypatch, R, first):
    log = []
    eng = engine(monkeypatch, log)
    X = torch.zeros(R, 4, 8)
    carry = (X, torch.zeros(R, 4), None if first == 0 else X[:, 0], None, None)
    eng.layers(carry, torch.zeros(R, dtype=torch.long), {1: None}, first, 4, static=True)
    assert log.count("pre-side") == 2 * (4 - first) and "pre" not in log
    check(log)
    assert eng._hc_live is False


def test_attn_last_break_ends_joined(monkeypatch):
    log = []
    eng = engine(monkeypatch, log)
    X = torch.zeros(2, 4, 8)
    eng.layers((X, torch.zeros(2, 4), X[:, 0], None, None), torch.zeros(2, dtype=torch.long), {1: None}, 0, 3,
               static=True, attn_last=True)
    check(log)
    assert log[-1] == "join" and eng._hc_live is False


def test_prompt_rows_and_switch_off_stay_on_the_main_stream(monkeypatch):
    log = []
    eng = engine(monkeypatch, log)
    R = S.PROMPT_ROWS + 1
    monkeypatch.setattr(S, "FUSE_HC", False)                 # (the prompt's fused post_pre is not this test)
    X = torch.zeros(R, 4, 8)
    eng.layers((X, torch.zeros(R, 4), None, None, None), torch.zeros(R, dtype=torch.long), {1: None}, 0, 2, False)
    assert "pre-side" not in log and "join" not in log
    log.clear()
    eng = engine(monkeypatch, log, side_on=False)
    X = torch.zeros(2, 4, 8)
    eng.layers((X, torch.zeros(2, 4), None, None, None), torch.zeros(2, dtype=torch.long), {1: None}, 0, 2, True)
    assert "pre-side" not in log and "join" not in log


def test_drafter_and_bench_calls_stay_on_the_main_stream(monkeypatch):
    """dspark.py and the sublayer bench call eng.hc(w, X, pre) without ``side``: never forked, nothing to join."""

    log = []
    eng = engine(monkeypatch, log)
    w = eng.w.layers[0].hc_attn
    eng.hc(w, torch.zeros(5, 4, 8), torch.zeros(5, 4))
    assert log == ["pre"] and eng._hc_live is False
    eng.hc(w, torch.zeros(5, 4, 8), torch.zeros(5, 4), side=True)
    assert log[-1] == "pre-side" and eng._hc_live is True
    eng._hc_join()
    assert log[-1] == "join" and eng._hc_live is False
