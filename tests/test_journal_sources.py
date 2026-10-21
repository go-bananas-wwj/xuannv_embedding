import torch

from xuannv_embedding.export.journal_sources import remove_local_inputs


def test_missing_local_inputs_preserves_public_queries_and_never_mutates_originals():
    public = torch.full((2, 64, 4, 4), 3.0)
    spectral = torch.full((2, 95, 4, 4), 2.0)
    detail = torch.ones_like(public)
    local = torch.full((2, 6, 1024, 2, 2), 7.0)
    mask = torch.ones((2, 6, 1, 2, 2))
    query = torch.full((2, 12, 64, 2, 2), 4.0)
    query_mask = torch.ones((2, 12, 1, 2, 2))
    support = torch.ones((2, 6, 12))
    batch = (
        public,
        spectral,
        detail,
        (local, mask, query, query_mask, support),
        None,
        detail,
        spectral.bool(),
        {},
    )
    result = remove_local_inputs(batch)
    assert result[0] is public and result[1] is spectral
    assert result[2].count_nonzero() == 0
    assert result[3][0].count_nonzero() == result[3][1].count_nonzero() == 0
    assert result[3][2] is query and result[3][3] is query_mask
    assert result[3][4] is support
    assert result[5] is None
    assert local.min() == 7 and mask.min() == 1 and detail.min() == 1
