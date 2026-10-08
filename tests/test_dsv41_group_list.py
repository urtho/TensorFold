"""Prompt chunks' expert grouping on the device (``serial.group_device``, ``work_list``) == ``group_members``' tables,
and the work list names every (place, member group) with members exactly once, then only out-of-range places."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.deepseek_v41.cuda.serial import group_device, group_members, work_list  # noqa: E402


def scratch(rows, slots, E):
    maxu = min(rows * slots, E)
    return SimpleNamespace(ids=torch.zeros((maxu,), dtype=torch.int32), count=torch.zeros((1,), dtype=torch.int32),
                           members_buf=torch.full((maxu * rows,), -1, dtype=torch.int32))


@pytest.mark.parametrize("R,E,slots,skew", [(2048, 256, 6, 0.0), (2048, 256, 6, 3.0), (100, 256, 6, 1.0),
                                            (40, 384, 7, 0.0), (2048, 384, 7, 5.0)])
def test_device_tables_equal_group_members(R, E, slots, skew):
    g = torch.Generator().manual_seed(R + E)
    weights = torch.tensor([(1 + i) ** -skew for i in range(E)])
    pick = torch.stack([torch.multinomial(weights, slots, replacement=False, generator=g) for _ in range(R)])
    s0, s1 = scratch(R, slots, E), scratch(R, slots, E)
    ids0, mem0 = group_members(pick.int(), E, s0)
    ids1, mem1, cnt = group_device(pick.int(), E, s1, R)
    nu = int(s0.count)
    assert int(s1.count) == nu and torch.equal(ids1[:nu], ids0)
    busiest = mem0.shape[1]
    assert torch.equal(mem1[:nu, :busiest], mem0) and bool((mem1[:nu, busiest:] == -1).all())
    assert torch.equal(cnt[:nu], (mem0 >= 0).sum(1)) and bool((cnt[nu:] == 0).all())
    for rows in (32, 64):
        w = work_list(cnt, rows, R * slots).tolist()
        want = [(p, j) for p in range(nu) for j in range(-(-int(cnt[p]) // rows))]
        got = [(p, j) for p, j in w if p < nu]
        assert got == want and all(p >= nu for p, _ in w[len(got):])
