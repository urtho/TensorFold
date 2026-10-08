"""The bounded radix-select decode top-k equals the reference selection (kernels.top_entries / candidate_blocks /
mask_to_blocks) on random and edge-case rows (GPU)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from tensorfold.families.deepseek_v41.cuda import kernels as K  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import topk as TK  # noqa: E402


@pytest.fixture(autouse=True, params=[8, 11], ids=["digits8", "digits11"])
def digits(request, monkeypatch):
    """Every test with the 8-bit and the 11 / 11 / 10-bit radix passes (TF_DSV41_TOPK_DIGITS)."""

    monkeypatch.setattr(TK, "TOPK_DIGITS", request.param)
    return request.param


def rows(R, S, ratio, kind, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pos = torch.randint(0, S * ratio, (R,), generator=g, device="cuda")
    pos[0] = S * ratio - 1                                  # a full row
    pos[1] = 0                                              # nothing visible yet (ratio 2) / one entry
    s = torch.randn((R, S), generator=g, device="cuda")
    if kind == "relu":                                      # many exact zeros, ties among them
        s = torch.relu(s) * (torch.rand((R, S), generator=g, device="cuda") < 0.3)
    if kind == "ties":                                      # few distinct non-zero values
        s = torch.randint(-3, 4, (R, S), generator=g, device="cuda").float()
        s[s == -0.0] = -0.0
    if kind == "negzero":
        s = torch.where(torch.rand((R, S), generator=g, device="cuda") < 0.5, torch.full_like(s, -0.0), s)
    vis = torch.arange(S, device="cuda")[None, :] < ((pos + 1) // ratio)[:, None]
    return torch.where(vis, s, torch.full_like(s, float("-inf"))), pos


def ref_dense(scores, topk):
    return K.top_entries(scores, topk)


@pytest.mark.parametrize("S,ratio", [(700, 2), (5000, 1), (32769, 2), (65537, 1), (200000, 1)])
@pytest.mark.parametrize("kind", ["randn", "relu", "negzero"])
def test_dense_equals_reference(S, ratio, kind):
    scores, pos = rows(6, S, ratio, kind, S + ratio)
    got = TK.top_entries(scores, pos, ratio, 512)
    assert torch.equal(got, ref_dense(scores, 512))


def test_ties_lower_index_and_same_set_when_unique():
    scores, pos = rows(5, 3000, 1, "ties", 7)
    got = TK.top_entries(scores, pos, 1, 512)
    for r in range(5):
        v = K.untie(scores[r:r + 1])[0]
        n = int((v != float("-inf")).sum())
        sel = got[r][got[r] >= 0]
        assert len(sel) == min(512, n) and torch.all(sel[1:] > sel[:-1])
        if n > 512:                                         # every chosen >= every unchosen; ties: lower index
            chosen = torch.zeros_like(v, dtype=torch.bool)
            chosen[sel.long()] = True
            lo = v[chosen].min()
            assert torch.all(v[~chosen & (v != float("-inf"))] <= lo)
            eq = (v == lo).nonzero().flatten()
            assert torch.all(chosen[eq[:int(chosen[eq].sum())]])


@pytest.mark.parametrize("S,ratio", [(4000, 1), (65537, 1), (300001, 1)])
def test_candidates_and_masked_equal_reference(S, ratio):
    scores, pos = rows(5, S, ratio, "randn", S)
    block, keep = 8, 2048
    flags = TK.candidate_flags(scores, pos, ratio, block, keep)
    cand = K.candidate_blocks(scores, pos, ratio, block, keep)
    nb = -(-S // block)
    want = torch.zeros((5, nb + 1), dtype=torch.bool, device="cuda")
    want.scatter_(1, torch.where(cand >= 0, cand, nb), True)
    assert torch.equal(flags.bool(), want[:, :nb])
    later, _ = rows(5, S, ratio, "relu", S + 1)
    later = torch.where(scores == float("-inf"), scores, later)
    got = TK.top_entries(later, pos, ratio, 512, flags=flags, block=block)
    assert torch.equal(got, K.top_entries(K.mask_to_blocks(later, cand, block), 512))


@pytest.mark.parametrize("S", [5000, 65537])
def test_adversarial_rows_same_for_both_digit_widths(S):
    """Rows built against the 11-bit digits: every score in one top-11-bit bin (1 + i 2^-23), many equal non-zero
    scores straddling the k-th place, all zeros (untied), and rows with 0 / 1 visible entries."""

    R, k = 6, 512
    i = torch.arange(S, device="cuda", dtype=torch.float32)
    s = torch.empty((R, S), device="cuda")
    s[0] = 1.0 + i * 2.0 ** -23
    s[1] = torch.where(i < 3 * k, 2.0, 1.0)
    s[2] = 0.0
    s[3] = torch.randn((S,), device="cuda")
    s[4] = torch.randn((S,), device="cuda")
    s[5] = (i % 7).float()
    pos = torch.full((R,), S - 1, device="cuda")
    pos[3], pos[4] = 0, 1
    vis = torch.arange(S, device="cuda")[None, :] < (pos + 1)[:, None]
    scores = torch.where(vis, s, torch.full_like(s, float("-inf")))
    got = {}
    for d in (8, 11):
        TK.TOPK_DIGITS = d
        got[d] = TK.top_entries(scores, pos, 1, k)
    assert torch.equal(got[8], got[11])
    assert torch.equal(got[11], ref_dense(scores, k))
