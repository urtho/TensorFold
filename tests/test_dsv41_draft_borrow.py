"""TF_DSV41_DRAFT_BORROW: a round that kept all its k drafts pulls the undrafted positions' estimates toward the
k-th's, so a stream planned at k drafts can learn its way to k + 1 (off: those positions stay at the prior)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")
import tensorfold.families.deepseek_v41.cuda.multi as M


def run(monkeypatch, borrow, rounds=20, k=4, m=4):
    monkeypatch.setattr(M, "DRAFT_BORROW", borrow)
    d = M.MultiDecoder.__new__(M.MultiDecoder)
    d.prior = [0.8] * 5
    s = SimpleNamespace(acc=[0.8] * 5)
    for _ in range(rounds):
        d._learn(s, k, m)
    return s.acc


def test_off_leaves_undrafted_positions(monkeypatch):
    acc = run(monkeypatch, False)
    assert acc[4] == 0.8 and acc[3] > 0.95


def test_on_lifts_undrafted_positions_after_full_rounds(monkeypatch):
    acc = run(monkeypatch, True)
    assert acc[3] > 0.95 and acc[4] > 0.9


def test_on_ignores_rounds_with_a_rejection(monkeypatch):
    acc = run(monkeypatch, True, m=2)                      # positions 0, 1 kept, 2 rejected: 3, 4 unobserved
    assert acc[3] == 0.8 and acc[4] == 0.8


def test_on_and_off_plan_the_first_round_alike(monkeypatch):
    assert run(monkeypatch, True, rounds=0) == run(monkeypatch, False, rounds=0)
