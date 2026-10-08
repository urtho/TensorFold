"""The drafter's Markov steps as kernels (``deepseek_v41/cuda/markov.py``) against the torch loop they replace
(``dspark.py``: ``torch.mm(e.half(), head.T, out_dtype=F32)`` and argmax), on random weights at the model's shapes
(GPU): the bias bits, the drafts of a two-rank vocabulary split emulated in one process, ties across the split, and
cached rows equal to computed ones."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from types import SimpleNamespace  # noqa: E402

from tensorfold.families.deepseek_v41.cuda import markov as MK  # noqa: E402

V, RANK = 129280, 256
F32 = torch.float32


@pytest.fixture(scope="module")
def dw():
    g = torch.Generator(device="cuda").manual_seed(0)
    head = (torch.randn((V, RANK), generator=g, device="cuda") * 0.06).half()
    emb = (torch.randn((V, RANK), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    return SimpleNamespace(markov_head=head, markov_embed=emb)


def ref_bias(dw, prev):
    return torch.mm(dw.markov_embed[prev].half(), dw.markov_head.T, out_dtype=F32)


def markov(dw, rank, world, rows=0, tokens=None):
    return MK.Markov(dw, SimpleNamespace(world=world, rank=rank), V, rows=rows, tokens=tokens)


@pytest.mark.parametrize("M", [1, 2, 6])
def test_bias_bits_equal_cublas(dw, M):
    mk = markov(dw, 0, 1)
    n = 64 // M * M
    toks = torch.randint(0, V, (n,), device="cuda")
    bad = 0
    for i in range(0, n, M):
        t = toks[i:i + M]
        st = torch.empty((M, V), dtype=F32, device="cuda")
        mk.bias(t, 1, mk.slot, st, V, M)
        ref = ref_bias(dw, t)
        bad += int((st.view(torch.int32) != ref.view(torch.int32)).sum())
    print(f"M={M}: {bad} of {n * V} bias elements differ from cuBLAS")
    assert bad == 0


def run_split(dw, logits, anchors, N, rows=0, tokens=None):
    """Both ranks' steps in lockstep, the gather done by hand: [M, N] drafts (rank 0's; rank 1's checked equal)."""

    M = anchors.shape[0]
    ranks = [markov(dw, r, 2, rows, tokens) for r in (0, 1)]
    half = V // 2
    lg = [logits[:, r * half:(r + 1) * half].contiguous() for r in (0, 1)]
    outs = []
    for _ in ranks:
        o = torch.empty((M, N + 1), dtype=torch.long, device="cuda")
        o[:, 0] = anchors
        outs.append(o)
    for i in range(N):
        sends = [mk.local_best(lg[r], outs[r], N, i).clone() for r, mk in enumerate(ranks)]
        g = torch.stack(sends)
        for r, mk in enumerate(ranks):
            mk.pick(g, outs[r], i)
    assert torch.equal(outs[0], outs[1])
    return outs[0][:, 1:]


def ref_loop(dw, logits, anchors, N):
    M = anchors.shape[0]
    view = logits.view(M, N, -1)
    prev, out = anchors, []
    for j in range(N):
        prev = (view[:, j] + ref_bias(dw, prev)).argmax(-1)
        out.append(prev)
    return torch.stack(out, dim=1)


@pytest.mark.parametrize("M", [1, 3, 6])
def test_split_drafts_equal_torch_loop(dw, M):
    N = 5
    g = torch.Generator(device="cuda").manual_seed(M)
    logits = torch.randn((M * N, V), generator=g, device="cuda") * 4
    anchors = torch.randint(0, V, (M,), generator=g, device="cuda")
    assert torch.equal(run_split(dw, logits, anchors, N), ref_loop(dw, logits, anchors, N))


def test_ties_across_the_split_go_to_rank_0(dw):
    N, M = 5, 2
    logits = torch.full((M * N, V), -50.0, device="cuda")
    logits[:, 7] = logits[:, V // 2 + 7] = 1e4                 # equal best on both halves: the lower index wins
    anchors = torch.tensor([3, 5], device="cuda")
    got = run_split(dw, logits, anchors, N)
    ref = ref_loop(dw, logits, anchors, N)
    assert torch.equal(got, ref)


def test_cached_rows_change_nothing(dw):
    N, M = 5, 3
    g = torch.Generator(device="cuda").manual_seed(9)
    logits = torch.randn((M * N, V), generator=g, device="cuda") * 4
    anchors = torch.randint(0, V, (M,), generator=g, device="cuda")
    plain = run_split(dw, logits, anchors, N)
    hot = ref_loop(dw, logits, anchors, N).flatten().tolist() + anchors.tolist()   # every token the steps read
    assert torch.equal(run_split(dw, logits, anchors, N, rows=len(hot), tokens=hot), plain)
