"""The loop guard (``call_gate.ThinkLoop``): a reasoning that repeats itself is closed after three dry 1,024-token
windows, a varied one never; the cut goes after the window's last token, through ``generate_gated``."""

import random

from tensorfold.engine.call_gate import ThinkLoop, generate_gated

END, CLOSE = 9, [7, 9, 8]


def feed(gate, tokens, batch=5):
    """Commit ``tokens`` in batches as an engine would: (index of the cut in ``tokens`` or None, its fix)."""

    for at in range(0, len(tokens), batch):
        new = tokens[at:at + batch]
        hit = gate.cut(new)
        if hit is not None:
            i, fix = hit
            for t in [*new[:i], *fix]:
                gate.observe(t)
            return at + i, fix
        for t in new:
            gate.observe(t)
    return None, None


def test_loop_closed_after_three_dry_windows():
    loop = [100 + i for i in range(37)]                     # a 37-token phrase, over and over
    reply = (loop * 400)[:8000]
    gate = ThinkLoop(CLOSE, END)
    at, fix = feed(gate, reply)
    assert fix == CLOSE and at == 4 * 1024                  # window 1 is all new; windows 2-4 dry: cut after the 4th
    assert gate.fired and not gate.open                     # its </think> closed the gate


def test_varied_reasoning_never_cut():
    rng = random.Random(0)
    reply = [rng.randrange(1000, 50000) for _ in range(12000)]
    gate = ThinkLoop(CLOSE, END)
    assert feed(gate, reply) == (None, None) and not gate.fired


def test_answer_after_think_end_is_never_cut():
    loop = [100 + i for i in range(37)]
    reply = [*range(200, 1500), END, *(loop * 400)]         # the reasoning ended; the answer may repeat
    gate = ThinkLoop(CLOSE, END)
    assert feed(gate, reply) == (None, None)


def test_a_new_window_resets_the_dry_count():
    loop = [100 + i for i in range(37)]
    rng = random.Random(1)
    fresh = [rng.randrange(1000, 50000) for _ in range(1024)]
    reply = [*(loop * 60)[:3 * 1024], *fresh, *(loop * 60)[:2 * 1024]]   # dry, dry, new, dry, dry: never 3 in a row
    gate = ThinkLoop(CLOSE, END)
    assert feed(gate, reply) == (None, None)


def test_generate_gated_goes_on_after_the_close():
    loop = [100 + i for i in range(37)]
    script = (loop * 400)[:6000]
    calls = []

    def generate(ids, count, take):
        calls.append(len(ids))
        if len(calls) == 1:                                 # the looping run, until the gate cuts it
            for at in range(0, len(script), 3):
                if take(script[at:at + 3]):
                    return {}
        else:                                               # the run after the close: an answer
            take([42, 43, 1])
        return {}

    out = []
    generate_gated(generate, [1, 2, 3], 10000, [ThinkLoop(CLOSE, END)], lambda new: out.extend(new) or False)
    assert out[:4 * 1024] == script[:4 * 1024] and out[4 * 1024:4 * 1024 + 3] == CLOSE and out[-3:] == [42, 43, 1]
    assert calls == [3, 3 + 4 * 1024 + 3]                   # the second run: the prompt, the reply and the close


def test_numbered_loop_caught_with_digits_as_one_symbol():
    """A cycle of lines with an increasing line number: new 8-grams every line plainly, a loop with numbers alike."""

    digits = frozenset(range(5000, 6000))                    # this test's "number" tokens
    lines = [[100 + 7 * j + i for i in range(6)] for j in range(13)]
    reply = []
    for k in range(1200):
        reply += [5000 + k % 1000, *lines[k % 13]]
    reply = reply[:8000]
    assert feed(ThinkLoop(CLOSE, END), reply) == (None, None)
    at, fix = feed(ThinkLoop(CLOSE, END, same=digits), reply)
    assert fix == CLOSE and at <= 5 * 1024
