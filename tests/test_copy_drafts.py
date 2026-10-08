"""Copy drafts propose what followed the latest earlier occurrence of the context's last tokens (copy_drafts.py is
adapted from MiaAI-Lab's GLM-5.3-Flash TensorFold recipe, Apache-2.0; see THIRD_PARTY_NOTICES.md)."""

import pytest

from tensorfold.cuda.copy_drafts import CopyDrafts, CopySettings

S = CopySettings(match=3, most=4)


def test_no_match_no_drafts():
    assert CopyDrafts([1, 2, 3, 4, 5, 6], S).propose() == []


def test_latest_occurrence_with_room():
    ctx = [9, 1, 2, 3, 10, 11, 12, 13, 7, 1, 2, 3, 20, 21, 22, 23, 8, 1, 2, 3]
    assert CopyDrafts(ctx, S).propose() == [20, 21, 22, 23]
    assert CopyDrafts(ctx, S).propose(room=2) == [20, 21]


def test_short_tail_takes_the_longest_continuation():
    ctx = [1, 2, 3, 10, 11, 12, 13, 14, 5, 1, 2, 3, 30, 1, 2, 3]
    assert CopyDrafts(ctx, S).propose() == [30, 1, 2, 3]               # both have 4 after them: the latest wins
    ctx = [1, 2, 3, 10, 11, 1, 2, 3]
    assert CopyDrafts(ctx, S).propose() == [10, 11, 1, 2]


def test_extend_and_truncate():
    c = CopyDrafts([5, 1, 2, 3, 6, 7, 8, 9], S)
    c.extend([1, 2])
    assert c.propose() == []
    c.extend([3])
    assert c.propose() == [6, 7, 8, 9]
    c.truncate(9)
    assert len(c) == 9 and c.propose() == []


def test_settings_from_env():
    assert CopySettings.from_env(5, env={"TF_COPY_DRAFTS": "0"}) is None
    assert CopySettings.from_env(5, env={}).most == 5
    s = CopySettings.from_env(5, env={"TF_COPY_MAX": "15"})
    assert s.match == 8 and s.most == 5
    with pytest.raises(ValueError):
        CopySettings.from_env(5, env={"TF_COPY_DRAFTS": "1", "TF_COPY_MATCH": "1"})


def test_shorter_proposal_is_exactly_that_long():
    """propose(k) after propose(cap) gave at least k tokens returns exactly k (a concurrent round sizes a copy with
    the one and verifies the other); fuzzed over small alphabets, where matches are many."""

    import random

    rng = random.Random(0)
    for _ in range(2000):
        s = CopySettings(match=rng.randrange(2, 5), most=rng.randrange(1, 16))
        c = CopyDrafts([rng.randrange(3) for _ in range(rng.randrange(1, 80))], s)
        cap = rng.randrange(1, 20)
        first = c.propose(cap)
        for k in range(1, len(first) + 1):
            assert len(c.propose(k)) == k


def test_window_tail_index_proposes_the_same(monkeypatch):
    """An index of the prompt's last WINDOW tokens and the reply proposes what the whole context's index does (the
    search reads only the last WINDOW tokens)."""

    import random

    import tensorfold.cuda.copy_drafts as CD

    monkeypatch.setattr(CD, "WINDOW", 64)
    rng = random.Random(1)
    for _ in range(300):
        prompt = [rng.randrange(4) for _ in range(rng.randrange(1, 200))]
        whole = CopyDrafts(prompt, S)
        tail = CopyDrafts(prompt[-64:], S)
        for _ in range(rng.randrange(1, 30)):
            new = [rng.randrange(4) for _ in range(rng.randrange(1, 4))]
            whole.extend(new)
            tail.extend(new)
            for k in (1, 2, 4):
                assert whole.propose(k) == tail.propose(k)
