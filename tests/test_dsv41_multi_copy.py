"""Copy drafts in DeepSeek-V4.1's concurrent rounds (TF_MULTI_COPY), host side: the allocator's copy choice (and the
DSpark allocation unchanged without a match), the plan's k + COPY_FLAG decoded once, the kept-share back-off, both
ranks' copy indexes equal, and two MultiDecoders over a fake engine staying in step through copy rounds with the
replies of the copy-off decoder. On the GPUs: ``tools/dsv41_serial_run.py --decoder-test`` runs both arms against
greedy serial decoding (copy rounds are host-side: no kernel or graph changes)."""

import random

import pytest

pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from tensorfold.cuda.streams import Stream
from tensorfold.families.deepseek_v41.cuda import multi as M
from tests.test_dsv41_kept import FakeEngine, pair, run, same

MASK = M.COPY_FLAG - 1


@pytest.fixture(autouse=True)
def host_sampling(monkeypatch):
    monkeypatch.setattr(M, "sample_rows", lambda logits, positions, sampling: [0] * len(positions))


@pytest.fixture
def copy_on(monkeypatch):
    monkeypatch.setattr(M, "MULTI_COPY", True)
    monkeypatch.delenv("TF_COPY_MAX", raising=False)
    monkeypatch.delenv("TF_COPY_DRAFTS", raising=False)


def curve(d, drafts=5):
    """A synthetic cost curve (25 + 3.5 ms a row, 3.6 ms a drafting pass) on rank 0's decoder."""

    costs = [25.0 + 3.5 * r for r in range(1, M.ROWS + 1)]
    d.curves = [(1 << 20, costs)]
    d.costs = costs
    d.draft_curve = [3.6 + 0.4 * i for i in range(16)] if drafts else []
    d.draft_ms = d.draft_curve[0] if drafts else 0.0


def decoder(slots=4, drafts=5):
    e = FakeEngine(slots, 1 << 16, 1 << 14, 1)
    if drafts:
        e.drafter = object()                       # (only _verify and calibrate call it)
    d = M.MultiDecoder(e, lambda v: v, rank=0, drafts=drafts)
    curve(d, drafts)
    return d


def live(d, prompt, out, count=500, constraint=None):
    """A decoding stream: its slot's rows hold prompt + out[:-1], out[-1] pending."""

    s = Stream(list(prompt), count)
    s.constraint = constraint
    d._queue(s)
    d.filling.remove(s)
    d.e.extents[s.slot] = (0, 1 << 14)
    d.e.views[s.slot].ids = list(prompt) + list(out[:-1])
    s.out = list(out)
    s.sid = s.slot
    d.streams[s.sid] = s
    return s


def old_allocate(d, live_):
    """``_allocate`` before copy drafts (bbc5ee8), the reference."""

    ks = [0] * len(live_)
    caps = []
    for s in live_:
        room = min(d.e.limit, d.e.extents[s.slot][1]) - len(d.e.views[s.slot].ids) - 1
        ok = s.draft and s.constraint is None and d.drafts
        caps.append(max(0, min(d.drafts, room, s.count - len(s.out) - 1)) if ok else 0)
    if not any(caps) or d.costs is None:
        return ks
    need = max(len(d.e.views[s.slot].ids) for s in live_) + M.ROWS
    d.costs = next((c for w, c in d.curves if need <= w), d.curves[-1][1])
    rows, drafting = len(live_), 0
    tokens = float(len(live_))
    rate = tokens / (d.costs[rows - 1] + d.overhead)
    while rows < M.ROWS:
        best = None
        for i, s in enumerate(live_):
            base = d._expected(s.acc, ks[i])
            for j in range(1, min(caps[i] - ks[i], M.ROWS - rows) + 1):
                gain = d._expected(s.acc, ks[i] + j) - base
                cost = d.costs[rows + j - 1] + d.overhead + d._draft_cost(drafting + (ks[i] == 0))
                r = (tokens + gain) / cost
                if r > rate and (best is None or r > best[0]):
                    best = (r, i, j, gain)
        if best is None:
            break
        rate, i, j, gain = best
        drafting += ks[i] == 0
        ks[i] += j
        tokens += gain
        rows += j
    return ks


def rand(rng, n):
    return [rng.randrange(10, 50000) for _ in range(n)]


def edit_stream(rng, n=300):
    """A prompt holding a 'file' and a reply that has started quoting it (the copy proposal: its next tokens)."""

    body = rand(rng, n)
    prompt = rand(rng, 20) + body + rand(rng, 10)
    return prompt, [0, *body[:40]]


# -- the allocator --------------------------------------------------------------------------------------------------
def test_copy_chosen_for_a_quoting_stream(copy_on):
    d = decoder()
    rng = random.Random(0)
    prompt, out = edit_stream(rng)
    s = live(d, prompt, out)
    (k,) = d._allocate([s])
    assert k >= M.COPY_FLAG and 1 <= k & MASK <= d.copy.most == 15
    assert d._copies(s).propose(k & MASK) == prompt[20 + 40:20 + 40 + (k & MASK)]
    assert (k & MASK) + 1 <= M.ROWS


def test_no_match_allocates_as_before(copy_on):
    """Without a copy proposal (or with copies off) the DSpark draft counts equal the old allocator's."""

    rng = random.Random(1)
    for trial in range(40):
        n = rng.randrange(1, 9)
        on, off = decoder(slots=n), decoder(slots=n)
        off.copy = None
        streams = {}
        for d in (on, off):
            r = random.Random(trial)
            streams[id(d)] = [live(d, rand(r, r.randrange(20, 400)), rand(r, r.randrange(1, 30)),
                                   count=r.randrange(2, 600)) for _ in range(n)]
            for s in streams[id(d)]:
                s.acc = [r.uniform(0.2, 0.95) for _ in s.acc]
        ref = old_allocate(off, streams[id(off)])
        assert off._allocate(streams[id(off)]) == ref
        assert on._allocate(streams[id(on)]) == ref


def test_grammar_and_serial_streams_never_copy(copy_on):
    d = decoder()
    rng = random.Random(2)
    a = live(d, *edit_stream(rng), constraint=object())
    b = live(d, *edit_stream(rng))
    b.draft = False
    assert all(k < M.COPY_FLAG for k in d._allocate([a, b]))
    assert d._proposals([a, b], [1000, 1000]) == [[], []]


def test_sixteen_copying_streams_fit_the_rows(copy_on):
    d = decoder(slots=16)
    rng = random.Random(3)
    streams = [live(d, *edit_stream(rng)) for _ in range(16)]
    ks = d._allocate(streams)
    assert sum(k & MASK for k in ks) + len(ks) <= M.ROWS
    assert any(k >= M.COPY_FLAG for k in ks)


def test_copy_without_a_drafter(copy_on):
    d = decoder(drafts=0)
    prompt, out = edit_stream(random.Random(4))
    (k,) = d._allocate([live(d, prompt, out)])
    assert k >= M.COPY_FLAG


def test_plan_round_trip(copy_on):
    d = decoder()
    rng = random.Random(5)
    streams = [live(d, *edit_stream(rng)), live(d, rand(rng, 200), [0, 1, 2])]
    plan = d._plan(streams)
    decoded = [(sid, k & MASK, k >= M.COPY_FLAG) for sid, k in plan]
    assert decoded[0][2] and decoded[0][1] >= 1
    assert all(k < M.COPY_FLAG for _, k, copied in decoded if not copied)


def test_backoff_after_misses(copy_on):
    d = decoder()
    s = live(d, *edit_stream(random.Random(6)))
    d._learn_copy(s, 10, 0)                         # 1.0 -> 0.5: still copying
    assert s.cwait == 0 and s.copy_rounds == 1
    d._learn_copy(s, 10, 0)                         # 0.25: wait 2 rounds
    assert s.cwait == 2 and s.cfrac == M.MULTI_COPY_MIN
    assert d._proposals([s], [1000]) == [[]]
    d._learn_copy(s, 10, 1)                         # a retry keeping less than the share: wait 4
    assert s.cwait == 4 and s.cmiss == 2
    d._learn_copy(s, 10, 10)
    assert s.cmiss == 0 and s.copy_accepted == 11
    s.cwait = 0
    assert d._proposals([s], [1000])[0]


def test_copy_cap_from_env(copy_on, monkeypatch):
    assert decoder().copy.most == 15
    monkeypatch.setenv("TF_COPY_MAX", "40")
    assert decoder().copy.most == M.ROWS - 1
    monkeypatch.setenv("TF_COPY_MAX", "7")
    assert decoder().copy.most == 7
    monkeypatch.setenv("TF_COPY_DRAFTS", "0")
    assert decoder().copy is None
    monkeypatch.setattr(M, "MULTI_COPY", False)
    monkeypatch.delenv("TF_COPY_DRAFTS")
    assert decoder().copy is None


def test_copy_switch_kept_out_of_the_calibration_key(copy_on, monkeypatch):
    d = decoder()
    d.e.widths, d.e.cap = [], 1 << 14
    monkeypatch.setenv("TF_REVISION", "abc")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "fake")
    a = d._calib_path([1 << 14])
    monkeypatch.setenv("TF_MULTI_COPY", "1")
    monkeypatch.setenv("TF_MULTI_COPY_MIN", "0.5")
    assert d._calib_path([1 << 14]) == a


# -- both ranks' copy indexes ---------------------------------------------------------------------------------------
def test_rank_indexes_agree(copy_on, monkeypatch):
    """Rank 0's stream grows out by take(), rank 1's by out.extend(): equal proposals every round, also after a
    replay (continued(): a fresh stream) and with the prompt longer than the search window."""

    monkeypatch.setattr(M, "COPY_WINDOW", 256)
    import tensorfold.cuda.copy_drafts as CD

    monkeypatch.setattr(CD, "WINDOW", 256)
    d0, d1 = decoder(), decoder()
    rng = random.Random(7)
    body = rand(rng, 120)
    prompt = rand(rng, 400) + body + rand(rng, 5) + body[:60]
    reply = body[60:] + rand(rng, 30) + body[:50]
    a, b = Stream(list(prompt), 400), Stream(list(prompt), 400)
    for d, s in ((d0, a), (d1, b)):
        d._queue(s)
        d._copies(s)
    full = CD.CopyDrafts(prompt, d0.copy)               # the reference: the whole context
    i = 0
    while i < len(reply):
        n = rng.randrange(1, 6)
        new = reply[i:i + n]
        i += n
        a.take(new)
        b.out.extend(new)
        full.extend(new)
        for k in (1, 3, 15):
            assert d0._copies(a).propose(k) == d1._copies(b).propose(k) == full.propose(k)
        assert len(d0._copies(a)) <= 256 + len(a.out) + 1
    c = a.continued()
    d0._queue(c)
    c.out = list(a.out[:10])
    assert d0._copies(c).propose(15) == CD.CopyDrafts(prompt + a.out[:10], d0.copy).propose(15)


# -- two ranks in step through copy rounds --------------------------------------------------------------------------
class ModelEngine(FakeEngine):
    """FakeEngine with rounds: a row's greedy token is the next token of its stream's ``truth`` (prompt + reply, by
    the prompt's first 8 tokens; set by the test), whatever the other rows (row-invariant); past it, 5."""

    def step_multi(self, rows):
        greedy = []
        for slot, token in rows:
            ids = self.views[slot].ids
            base, _ = self.extents[slot]
            self.arena[base + len(ids)] = token
            ids.append(token)
            t = self.truth[tuple(ids[:8])]
            greedy.append(t[len(ids)] if len(ids) < len(t) else 5)
        return torch.zeros((len(rows), 8)), greedy


def workload(seed):
    """Prompts and their true replies: two quoting a file with edits every 37 tokens, one that never repeats."""

    rng = random.Random(seed)
    out = []
    for kind in ("edit", "edit", "plain"):
        body = rand(rng, 300)
        prompt = rand(rng, 30) + body + rand(rng, 10)
        if kind == "edit":
            reply = [0] + [3 if j % 37 == 36 else t for j, t in enumerate(body)]
        else:
            reply = [0] + rand(rng, 200)
        out.append((prompt, prompt + reply, len(reply)))
    return out


def serve(d0, d1, jobs, check=False):
    """The jobs decoded together on both ranks; rank 1's streams and rows compared with rank 0's after every round."""

    for d in (d0, d1):
        d.e.truth = {tuple(t[:8]): t for _, t, _ in jobs}
        d.check = check
    curve(d0, drafts=0)
    streams = []
    for prompt, _, n in jobs:
        s = Stream(list(prompt), n, stop_eos=False)
        run(d0, s)
        streams.append(s)
    while any(not s.done for s in streams):
        d0.round()
        d1.sync()
        for s in streams:
            r = d1.streams[s.sid]
            assert r.out == s.out
            assert d1.e.views[r.slot].ids == d0.e.views[s.slot].ids
    same(d0, d1)
    stats = [M.stream_stats(s) for s in streams]
    d0.finish(streams)
    same(d0, d1)
    return [s.out for s in streams], stats


@pytest.mark.parametrize("check", [False, True])
def test_copy_rounds_stay_in_step(monkeypatch, check):
    jobs = workload(8)
    monkeypatch.setattr(M, "MULTI_COPY", False)
    ref, ref_stats = serve(*pair(slots=3, engine=ModelEngine), jobs)
    assert ref == [t[len(p):] for p, t, _ in jobs]       # (the fake model's replies)
    monkeypatch.setattr(M, "MULTI_COPY", True)
    d0, d1 = pair(slots=3, engine=ModelEngine)
    gathers = []
    if check:                                            # the copy windows' hash goes through gather on both ranks
        for d in (d0, d1):
            g = d.gather
            d.gather = lambda v, g=g: gathers.append(list(v)) or g(v)
    outs, stats = serve(d0, d1, jobs, check=check)
    assert outs == ref                                   # exact: copies only change how many rows a round keeps
    edit, _, plain = stats
    assert edit["copy_rounds"] > 0 and edit["copy_accepted"] > 5 * edit["copy_rounds"]
    assert edit["rounds"] < ref_stats[0]["rounds"] // 2
    assert plain["copy_rounds"] == 0 and plain["rounds"] == ref_stats[2]["rounds"]
    assert bool(gathers) == check


def test_random_traffic_with_copies_stays_in_step(copy_on):
    """Streams arriving and finishing at random, quoting their prompts or not, some ending early: in step."""

    d0, d1 = pair(slots=3, engine=ModelEngine)
    curve(d0, drafts=0)
    rng = random.Random(9)
    truth = {}
    for e in (d0.e, d1.e):
        e.truth = truth
    streams = []
    for step in range(150):
        if len(streams) < 3 and rng.random() < 0.3:
            body = rand(rng, rng.randrange(20, 200))
            prompt = rand(rng, 12) + body + rand(rng, rng.randrange(1, 20))
            reply = [0] + [rng.randrange(10, 50000) if rng.random() < 0.05 else t for t in body] + rand(rng, 20)
            truth[tuple(prompt[:8])] = prompt + reply
            s = Stream(list(prompt), rng.randrange(2, len(reply) + 1), stop_eos=False)
            run(d0, s)
            streams.append(s)
        if streams:
            d0.round()
            d1.sync()
            for s in streams:
                assert d1.streams[s.sid].out == s.out
                assert s.out == truth[tuple(s.prompt[:8])][len(s.prompt):len(s.prompt) + len(s.out)]
            done = [s for s in streams if s.done]
            if done:
                d0.finish(done)
                streams = [s for s in streams if not s.done]
        same(d0, d1)

