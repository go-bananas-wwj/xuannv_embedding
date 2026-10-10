import numpy as np
import pytest
import torch
from test_neural_readouts import example

from xuannv_embedding.downstream import neural_readouts as neural
from xuannv_embedding.downstream import neural_trajectory as trajectory
from xuannv_embedding.export.context import sha


@pytest.mark.parametrize("kind", ["mlp", "conv3x3"])
def test_long_training_keeps_the_original_100_step_prefix_and_selects_only_on_calibration(
    tmp_path, kind
):
    torch.set_num_threads(1)
    data = example()
    reference = neural.fit_neural(kind, *data)
    checkpoints = trajectory.fit(kind, *data, checkpoints=[100, 110])
    assert (
        checkpoints[100].metadata["final_weights_sha256"]
        == reference.metadata["final_weights_sha256"]
    )
    np.testing.assert_array_equal(
        checkpoints[100].predict(data[3], data[5]), reference.predict(data[3], data[5])
    )
    best = trajectory.choose(checkpoints)
    assert best in [100, 110]
    assert checkpoints[best].metadata["validation_ap"] == max(
        m.metadata["validation_ap"] for m in checkpoints.values()
    )
    root = tmp_path / "saved"
    trajectory.save(checkpoints[110], root)
    loaded = trajectory.load(root, sha(root / "identity.json"))
    assert loaded.metadata["optimizer_steps"] == 110
    np.testing.assert_array_equal(
        loaded.predict(data[3], data[5]), checkpoints[110].predict(data[3], data[5])
    )


def test_trajectory_rejects_unsorted_or_duplicate_stops():
    for stops in [[100, 100], [300, 100], [0], [True], []]:
        with pytest.raises(ValueError):
            trajectory.fit("mlp", *example(), checkpoints=stops)
