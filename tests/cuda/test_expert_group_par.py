"""TF_EXPERT_GROUP=par: the grouping kernel with its members written one thread a pick gives the serial fill's uids,
count and whole members window: random and duplicate picks, skipped slots (pick E), member lists past maxm (cut)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from tensorfold.cuda.exl3 import experts as ex3


def _group(pick, R, slots, E, maxm, par):
    n = R * slots
    uids = torch.full((min(n, E),), -7, dtype=torch.int32, device="cuda")
    count = torch.full((1,), -7, dtype=torch.int32, device="cuda")
    members = torch.full((min(n, E), maxm), -7, dtype=torch.int32, device="cuda")
    ex3._ext().group(pick, uids, count, members, R, slots, E, par)
    return uids, count, members


@pytest.mark.parametrize("E", [24, 384])
@pytest.mark.parametrize("maxm", [1, 3, 16])
@pytest.mark.parametrize("kind", ["random", "duplicates", "skip"])
def test_par_equals_serial(E, maxm, kind):
    g = torch.Generator(device="cuda").manual_seed(E * 100 + maxm)
    for R in range(1, 17):
        slots = 7
        if kind == "random":
            pick = torch.randint(0, E, (R, slots), generator=g, device="cuda")
        elif kind == "duplicates":
            pick = torch.randint(0, 3, (R, slots), generator=g, device="cuda")
        else:
            pick = torch.randint(0, E, (R, slots), generator=g, device="cuda")
            pick[:, -1] = E                                  # a skipped slot
        pick = pick.int().contiguous()
        a = _group(pick, R, slots, E, maxm, 0)
        b = _group(pick, R, slots, E, maxm, 1)
        torch.cuda.synchronize()
        cnt = int(a[1])
        assert cnt == int(b[1]), (R, kind)
        assert torch.equal(a[0][:cnt], b[0][:cnt]) and torch.equal(a[2][:cnt], b[2][:cnt]), (R, kind, maxm)
