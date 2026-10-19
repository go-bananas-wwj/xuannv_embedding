import numpy as np
import pytest
import torch

from xuannv_embedding.models.prithvi import (
    PrithviEncoder,
    PrithviEncoderConfig,
    load_encoder_state,
    sincos_positions,
)


def tiny_config(**changes):
    return PrithviEncoderConfig(
        **{
            "image_size": 32,
            "num_frames": 4,
            "in_channels": 6,
            "embed_dim": 32,
            "depth": 2,
            "num_heads": 4,
            **changes,
        }
    )


@pytest.mark.parametrize("frames,height,width", [(4, 32, 32), (12, 32, 64), (1, 16, 48)])
def test_encoder_preserves_actual_temporal_and_spatial_grid(frames, height, width):
    model = PrithviEncoder(tiny_config()).eval()
    with torch.no_grad():
        x = torch.randn(1, 6, frames, height, width)
        tokens = model.forward_tokens(x)
        grid = model(x)
    assert tokens.shape == (1, 1 + frames * (height // 16) * (width // 16), 32)
    assert grid.shape == (1, 32, frames, height // 16, width // 16)
    torch.testing.assert_close(grid.flatten(2).transpose(1, 2), tokens[:, 1:])
    assert torch.isfinite(grid).all()


@pytest.mark.parametrize("shape", [(1, 6, 2, 31, 32), (1, 5, 4, 32, 32), (1, 6, 32, 32)])
def test_encoder_rejects_implicit_crop_or_wrong_band_layout(shape):
    with pytest.raises(ValueError, match="shape|divisible"):
        PrithviEncoder(tiny_config())(torch.zeros(shape))


def test_encoder_rejects_nonfinite_observed_tensor():
    with pytest.raises(ValueError, match="finite"):
        PrithviEncoder(tiny_config())(torch.full((1, 6, 1, 16, 16), float("nan")))


@pytest.mark.parametrize(
    "changes",
    [
        {"embed_dim": 30},
        {"depth": 0},
        {"num_frames": True},
        {"num_heads": 3},
        {"patch_size": (1, 0, 16)},
        {"image_size": 31},
        {"mlp_ratio": 0},
        {"layer_norm_eps": float("nan")},
    ],
)
def test_encoder_config_rejects_unsupported_dimensions(changes):
    with pytest.raises(ValueError):
        tiny_config(**changes)


def test_positions_follow_width_height_time_order_and_zero_cls():
    positions = sincos_positions(32, (2, 2, 3))
    assert positions.shape == (13, 32)
    np.testing.assert_array_equal(positions[0], 0)
    # Width changes first; time changes after all spatial cells.
    np.testing.assert_array_equal(positions[1, 12:], positions[2, 12:])
    assert not np.array_equal(positions[1, :12], positions[2, :12])
    np.testing.assert_array_equal(positions[1, :24], positions[7, :24])
    assert not np.array_equal(positions[1, 24:], positions[7, 24:])


def test_pretrained_encoder_loading_consumes_every_encoder_key():
    source = PrithviEncoder(tiny_config())
    target = PrithviEncoder(tiny_config())
    state = {"encoder." + k: v.clone() for k, v in source.state_dict().items()}
    state["decoder.mask_token"] = torch.zeros(1)
    receipt = load_encoder_state(target, state)
    assert receipt == {"encoder_keys": len(source.state_dict()), "ignored_decoder_keys": 1}
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)


def test_pretrained_encoder_loading_rejects_missing_or_unexpected_keys():
    model = PrithviEncoder(tiny_config())
    state = {"encoder." + k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match="prefix"):
        load_encoder_state(model, {**state, "unknown.weight": torch.ones(1)})
    missing = dict(state)
    missing.pop("encoder.cls_token")
    with pytest.raises(ValueError, match="keys"):
        load_encoder_state(model, missing)
    with pytest.raises(ValueError, match="keys"):
        load_encoder_state(model, {**state, "encoder.unknown": torch.ones(1)})


def test_pretrained_encoder_loading_rejects_bad_shape_or_nonfinite_weights():
    model = PrithviEncoder(tiny_config())
    state = {"encoder." + k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match="shape"):
        load_encoder_state(model, {**state, "encoder.cls_token": torch.zeros(1, 1, 16)})
    with pytest.raises(ValueError, match="finite"):
        load_encoder_state(
            model, {**state, "encoder.cls_token": state["encoder.cls_token"] * float("nan")}
        )


def test_actual_frame_change_keeps_positions_on_encoder_device():
    model = PrithviEncoder(tiny_config())
    positions = model.position_encoding((12, 32, 64))
    assert positions.device == model.pos_embed.device
    assert positions.shape == (1, 12 * 2 * 4 + 1, 32)
    assert model.position_encoding((4, 32, 32)) is model.pos_embed
