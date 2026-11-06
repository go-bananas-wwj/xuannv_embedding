from pathlib import Path

import numpy as np

from xuannv_embedding.data_process.omnicloudmask_npu import (
    AscendOmniCloudMaskV4Predictor,
)


class _FakeOmRunner:
    def __init__(self, model_paths: list[Path], device_id: int) -> None:
        assert len(model_paths) == 2
        assert device_id == 3
        self.calls: list[np.ndarray] = []

    def infer(self, array: np.ndarray) -> list[np.ndarray]:
        self.calls.append(array)
        first = np.zeros((4, 96, 96), dtype=np.float32)
        second = np.zeros_like(first)
        first[1] = 4.0
        second[1] = 2.0
        return [first, second]

    def close(self) -> None:
        pass


def test_npu_predictor_uses_96_pixel_tiles_and_ensembles_two_models() -> None:
    predictor = AscendOmniCloudMaskV4Predictor(
        model_paths=[Path("model-0.om"), Path("model-1.om")],
        device_id=3,
        runner_factory=_FakeOmRunner,
    )
    image = np.arange(3 * 128 * 128, dtype=np.float32).reshape(3, 128, 128) + 1

    labels, confidence = predictor.predict_batch([image])

    assert labels.shape == (1, 128, 128)
    assert confidence.shape == (1, 128, 128)
    assert np.all(labels == 1)
    assert len(predictor.runner.calls) == 4
    assert all(call.shape == (3, 96, 96) for call in predictor.runner.calls)
    assert all(np.allclose(call.mean(axis=(1, 2)), 0, atol=1e-5) for call in predictor.runner.calls)


def test_npu_predictor_keeps_nodata_clear_but_zero_confidence() -> None:
    predictor = AscendOmniCloudMaskV4Predictor(
        model_paths=[Path("model-0.om"), Path("model-1.om")],
        device_id=3,
        runner_factory=_FakeOmRunner,
    )
    image = np.ones((3, 96, 96), dtype=np.float32)
    image[:, :4, :4] = 0

    labels, confidence = predictor.predict_batch([image])

    assert np.all(labels[0, :4, :4] == 0)
    assert np.all(confidence[0, :4, :4] == 0)


def test_pointer_adapter_preserves_numpy_lifetime_and_filters_only_known_warning():
    import warnings
    from types import SimpleNamespace

    from xuannv_embedding.data_process.omnicloudmask_npu import _host_array_pointer

    def pointer(array):
        warnings.warn(
            "acl.util.numpy_to_ptr will be deprecated. Please use acl.util.bytes_to_ptr instead."
        )
        assert array.shape == (3, 96, 96)
        return 123

    acl = SimpleNamespace(util=SimpleNamespace(numpy_to_ptr=pointer))
    with warnings.catch_warnings(record=True) as captured:
        value = _host_array_pointer(acl, np.ones((3, 96, 96), dtype=np.float32))
    assert value == 123 and not captured
