from __future__ import annotations

# 模型模块单元测试。
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from xuannv_embedding.models.blocks import (
    EmbeddingUpsampleHead,
    LearnedSpatialResampling,
    MonthlyEmbeddingModule,
    MultiResolutionSTPBlock,
    STPEncoder,
    STPPrecisionOperator,
    STPSpaceOperator,
    STPTimeOperator,
)
from xuannv_embedding.models.bottleneck import VMFBottleneck
from xuannv_embedding.models.decoders import CategoricalDecoder, ContinuousDecoder
from xuannv_embedding.models.highres_fusion import AvailabilityAwareFusion
from xuannv_embedding.models.model import AEFModel, AEFOutput
from xuannv_embedding.models.sensor_encoders import (
    NativeResolutionHighResEncoder,
    SensorEncoder,
    SensorEncoderBank,
    SourceAwareTemporalFusion,
)
from xuannv_embedding.models.time_encoding import TimeEncoding, WindowCode


def test_sensor_encoder_shape() -> None:
    """单个 SensorEncoder 输出形状应与输入空间尺寸一致、通道数为 out_channels。"""
    batch_size = 2
    in_channels = 13
    out_channels = 64
    height, width = 32, 32

    encoder = SensorEncoder(in_channels, out_channels)
    x = torch.randn(batch_size, in_channels, height, width)
    y = encoder(x)

    assert y.shape == (batch_size, out_channels, height, width)


def test_sensor_encoder_bank() -> None:
    """SensorEncoderBank 应能为不同数据源独立编码并输出统一通道数。"""
    sensor_configs = {"s2": 13, "s1": 2}
    out_channels = 64
    batch_size = 2
    height, width = 32, 32

    bank = SensorEncoderBank(sensor_configs, out_channels)

    # S2 输入
    x_s2 = torch.randn(batch_size, sensor_configs["s2"], height, width)
    y_s2 = bank(x_s2, source="s2")
    assert y_s2.shape == (batch_size, out_channels, height, width)

    # S1 输入
    x_s1 = torch.randn(batch_size, sensor_configs["s1"], height, width)
    y_s1 = bank(x_s1, source="s1")
    assert y_s1.shape == (batch_size, out_channels, height, width)


def test_sensor_encoder_bank_unknown_source() -> None:
    """传入未注册的数据源名称时应抛出 KeyError。"""
    bank = SensorEncoderBank({"s2": 13}, out_channels=64)
    x = torch.randn(1, 13, 16, 16)

    with pytest.raises(KeyError, match="未知数据源"):
        bank(x, source="landsat")


def test_source_aware_temporal_fusion_shape_and_masks() -> None:
    """SourceAwareTemporalFusion 应保持 stem 维度，并让无效 source 权重为 0。"""
    batch_size, time_steps, channels, height, width = 2, 3, 8, 4, 4
    fusion = SourceAwareTemporalFusion(["s2", "s1"], stem_dim=channels)
    features = {
        "s2": torch.randn(batch_size, time_steps, channels, height, width),
        "s1": torch.randn(batch_size, time_steps, channels, height, width),
    }
    masks = {
        "s2": torch.ones(batch_size, time_steps),
        "s1": torch.zeros(batch_size, time_steps),
    }

    out = fusion(features, masks, ["s2", "s1"])

    expected = features["s2"] + fusion.source_embeddings[0].view(1, 1, channels, 1, 1)
    assert out.shape == (batch_size, time_steps, channels, height, width)
    assert torch.allclose(out, expected, atol=1e-6)


def test_source_aware_temporal_fusion_all_masked_returns_zero() -> None:
    """当某个时间步所有 source 都不可用时，融合输出应为 0 且无 NaN。"""
    batch_size, time_steps, channels, height, width = 1, 2, 8, 4, 4
    fusion = SourceAwareTemporalFusion(["s2", "s1"], stem_dim=channels)
    features = {
        "s2": torch.randn(batch_size, time_steps, channels, height, width),
        "s1": torch.randn(batch_size, time_steps, channels, height, width),
    }
    masks = {
        "s2": torch.zeros(batch_size, time_steps),
        "s1": torch.zeros(batch_size, time_steps),
    }

    out = fusion(features, masks, ["s2", "s1"])

    assert not torch.isnan(out).any()
    assert torch.allclose(out, torch.zeros_like(out), atol=0.0)


def test_native_resolution_highres_encoder() -> None:
    """NativeResolutionHighResEncoder 应将任意输入对齐到目标分辨率。"""
    batch_size = 2
    in_channels = 4
    out_channels = 16
    target_size = (8, 8)

    encoder = NativeResolutionHighResEncoder(in_channels, out_channels, target_size=target_size)
    x = torch.randn(batch_size, in_channels, 64, 64)
    y = encoder(x)

    assert y.shape == (batch_size, out_channels, *target_size)

    # 每个 stride=2 卷积后都应有 GroupNorm。
    assert isinstance(encoder.norm1, nn.GroupNorm)
    assert isinstance(encoder.norm2, nn.GroupNorm)
    assert isinstance(encoder.norm3, nn.GroupNorm)


def test_stp_encoder_precision_scale_one_keeps_full_resolution() -> None:
    """precision_scale=1 时，STP 精度路径应保持输入空间分辨率。"""
    encoder = STPEncoder(
        input_channels=8,
        space_dim=32,
        time_dim=16,
        precision_dim=16,
        num_blocks=1,
        num_heads=2,
        time_attention_mode="none",
        precision_scale=1,
    )
    x = torch.randn(2, 3, 16, 16, 8)
    timestamps = torch.tensor([[202501, 202502, 202503], [202501, 202502, 202503]])
    mask = torch.ones(2, 3)

    feats, input_size = encoder(x, timestamps, mask)

    assert input_size == (16, 16)
    assert feats.shape == (2, 3, 16, 16, 16)


def test_aef_model_uses_native_resolution_highres_encoder() -> None:
    """AEFModel 应为每个高分辨率源使用独立的 NativeResolutionHighResEncoder。"""
    sensor_channels = {
        "s2": 10,
        "highres_optical_haidian": 4,
        "highres_sar_haidian": 1,
    }
    embed_dim = 16
    model = AEFModel(
        sensor_channels,
        embed_dim,
        target_heads={"s2_recon": ("continuous", 10)},
        num_space_heads=2,
    )

    assert hasattr(model, "highres_encoders")
    assert set(model.highres_encoders.keys()) == {
        "highres_optical_haidian",
        "highres_sar_haidian",
    }
    for encoder in model.highres_encoders.values():
        assert isinstance(encoder, NativeResolutionHighResEncoder)
        assert encoder.out_channels == embed_dim


def test_stp_precision_operator_preserves_channels_last_layout_with_zero_convs() -> None:
    """精度算子应正确处理 channels-last 输入，不能用裸 reshape 打乱布局。"""
    op = STPPrecisionOperator(dim=3)
    with torch.no_grad():
        op.conv1.weight.zero_()
        op.conv1.bias.zero_()
        op.conv2.weight.zero_()
        op.conv2.bias.zero_()

    x = torch.arange(2 * 2 * 4 * 5 * 3, dtype=torch.float32).view(2, 2, 4, 5, 3)
    y = op(x)

    assert torch.equal(y, x)


def test_multi_resolution_block_preserves_layout_when_cross_resampling_zeroed() -> None:
    """STP block 内部转为 channels-first 时不应打乱空间和通道顺序。"""

    class _Identity(nn.Module):
        def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            return x

    class _ZeroResample(nn.Module):
        def __init__(self, out_channels: int) -> None:
            super().__init__()
            self.out_channels = out_channels

        def forward(
            self,
            x: torch.Tensor,
            target_size: tuple[int, int] | None = None,
        ) -> torch.Tensor:
            height, width = target_size if target_size is not None else x.shape[-2:]
            return x.new_zeros((x.shape[0], self.out_channels, height, width))

    block = MultiResolutionSTPBlock(
        space_dim=3,
        time_dim=3,
        precision_dim=3,
        num_heads=1,
        time_attention_mode="none",
        precision_scale=1,
    )
    block.space_op = _Identity()
    block.time_op = _Identity()
    block.precision_op = _Identity()
    block.space_to_time = _ZeroResample(3)
    block.space_to_precision = _ZeroResample(3)
    block.time_to_space = _ZeroResample(3)
    block.time_to_precision = _ZeroResample(3)
    block.precision_to_space = _ZeroResample(3)
    block.precision_to_time = _ZeroResample(3)

    space = torch.arange(1 * 2 * 2 * 2 * 3, dtype=torch.float32).view(1, 2, 2, 2, 3)
    time = torch.arange(1 * 2 * 3 * 3 * 3, dtype=torch.float32).view(1, 2, 3, 3, 3)
    precision = torch.arange(1 * 2 * 4 * 4 * 3, dtype=torch.float32).view(1, 2, 4, 4, 3)
    timestamps = torch.tensor([[202501, 202502]], dtype=torch.long)

    out_space, out_time, out_precision = block(space, time, precision, timestamps)

    assert torch.equal(out_space, space)
    assert torch.equal(out_time, time)
    assert torch.equal(out_precision, precision)


def test_aef_model_uses_gated_temporal_fusion() -> None:
    """gated_sum 模式应启用源感知融合，并保持 STP 输入通道为 stem_dim。"""
    sensor_channels = {"s2": 10, "s1": 2, "landsat": 6}
    embed_dim = 16
    stem_dim = 8
    target_heads = {"s2_recon": ("continuous", 10)}
    model = AEFModel(
        sensor_channels,
        embed_dim,
        target_heads,
        stem_dim=stem_dim,
        stp={
            "space_dim": 64,
            "time_dim": 32,
            "precision_dim": 32,
            "num_blocks": 1,
            "num_heads": 2,
            "temporal_fusion": "gated_sum",
        },
        num_months=2,
    )

    assert isinstance(model.temporal_fusion, SourceAwareTemporalFusion)
    assert model.total_stem_channels == stem_dim
    assert model.stp_encoder.input_projection.in_features == stem_dim

    batch_size, time_steps, height, width = 1, 2, 16, 16
    source_frames = {
        "s2": torch.randn(batch_size, time_steps, 10, height, width),
        "s1": torch.randn(batch_size, time_steps, 2, height, width),
        "landsat": torch.randn(batch_size, time_steps, 6, height, width),
    }
    source_masks = {k: torch.ones(batch_size, time_steps) for k in source_frames}
    source_masks["s1"][:, 1] = 0
    timestamps = _make_yyyymm_timestamps(batch_size, time_steps, start=202501)

    out = model(source_frames, source_masks, timestamps)

    assert out.embedding_map.shape == (batch_size, 2, embed_dim, height, width)
    assert out.reconstructions["s2_recon"].shape == (batch_size, 2, 10, height, width)


def test_vmf_bottleneck() -> None:
    """VMFBottleneck 应保持 (B, C, H, W) 形状并将输出约束到单位球面。"""
    batch_size = 2
    in_dim = 64
    out_dim = 32
    height, width = 8, 8

    bottleneck = VMFBottleneck(in_dim, out_dim, kappa=100.0)
    x = torch.randn(batch_size, in_dim, height, width)
    z = bottleneck(x)

    assert z.shape == (batch_size, out_dim, height, width)

    # 每个空间位置的 L2 范数应接近 1。
    norms = z.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_vmf_bottleneck_log_kappa_is_learnable() -> None:
    """log_kappa 应是 nn.Parameter，默认初始化为 log(10.0)，并能被优化器收集。"""
    bottleneck = VMFBottleneck(16, 8)

    assert hasattr(bottleneck, "log_kappa")
    assert isinstance(bottleneck.log_kappa, nn.Parameter)
    assert bottleneck.log_kappa.shape == ()
    assert torch.allclose(bottleneck.log_kappa, torch.log(torch.tensor(10.0)), atol=1e-6)

    params = {name for name, _ in bottleneck.named_parameters()}
    assert "log_kappa" in params


def test_vmf_bottleneck_kappa_initializes_log_kappa() -> None:
    """显式传入 kappa 时，log_kappa 应初始化为 log(kappa)，且不保留冲突的 self.kappa。"""
    bottleneck = VMFBottleneck(16, 8, kappa=100.0)

    assert torch.allclose(bottleneck.log_kappa, torch.log(torch.tensor(100.0)), atol=1e-6)
    assert not hasattr(bottleneck, "kappa")


def test_vmf_bottleneck_noise_std_inverse_to_exp_log_kappa() -> None:
    """训练时噪声标准差应为 1 / exp(log_kappa)，且随 log_kappa 增大而减小。"""
    in_dim, out_dim, height, width = 16, 8, 8, 8
    x = torch.randn(2, in_dim, height, width)

    bottleneck = VMFBottleneck(in_dim, out_dim)

    # 保存并恢复 RNG 状态，避免影响其它测试。
    rng_state = torch.get_rng_state()
    try:
        # 低 kappa（高温度）：噪声标准差大。
        bottleneck.log_kappa.data.fill_(torch.log(torch.tensor(1.0)).item())
        torch.manual_seed(0)
        z_low = bottleneck(x)

        # 高 kappa（低温度）：噪声标准差小。
        bottleneck.log_kappa.data.fill_(torch.log(torch.tensor(100.0)).item())
        torch.manual_seed(0)
        z_high = bottleneck(x)
    finally:
        torch.set_rng_state(rng_state)

    # 使用同一组投影权重计算无噪声归一化结果。
    z_clean = F.normalize(bottleneck.proj(x), dim=1).detach()
    diff_low = (z_low - z_clean).norm(dim=1).mean().item()
    diff_high = (z_high - z_clean).norm(dim=1).mean().item()
    assert diff_low > diff_high


def test_vmf_bottleneck_eval_disables_noise() -> None:
    """推理模式下不应注入切空间噪声，输出仅由 L2 归一化决定。"""
    in_dim, out_dim, height, width = 16, 8, 8, 8
    bottleneck = VMFBottleneck(in_dim, out_dim)
    x = torch.randn(2, in_dim, height, width)

    bottleneck.eval()
    z1 = bottleneck(x)
    z2 = bottleneck(x)

    assert torch.allclose(z1, z2, atol=0.0)

    expected = F.normalize(bottleneck.proj(x), dim=1)
    assert torch.allclose(z1, expected, atol=1e-6)


def test_vmf_bottleneck_log_kappa_grad_flows() -> None:
    """log_kappa 应能接收梯度，说明噪声幅度是可通过训练调整的。"""
    bottleneck = VMFBottleneck(16, 8)
    x = torch.randn(2, 16, 8, 8)
    z = bottleneck(x)
    loss = z.sum()
    loss.backward()

    assert bottleneck.log_kappa.grad is not None
    assert not torch.isnan(bottleneck.log_kappa.grad)


def test_continuous_decoder() -> None:
    """ContinuousDecoder 应保持空间尺寸并将通道映射到 out_channels。"""
    batch_size = 2
    embed_dim = 64
    out_channels = 13
    height, width = 8, 8

    decoder = ContinuousDecoder(embed_dim, out_channels)
    emb = torch.randn(batch_size, embed_dim, height, width)
    y = decoder(emb)

    assert y.shape == (batch_size, out_channels, height, width)


def test_categorical_decoder() -> None:
    """CategoricalDecoder 应保持空间尺寸并输出 num_classes 个 logits。"""
    batch_size = 2
    embed_dim = 64
    num_classes = 11
    height, width = 8, 8

    decoder = CategoricalDecoder(embed_dim, num_classes)
    emb = torch.randn(batch_size, embed_dim, height, width)
    y = decoder(emb)

    assert y.shape == (batch_size, num_classes, height, width)


def test_availability_aware_fusion() -> None:
    """AvailabilityAwareFusion 应保持形状，并且可用性掩码应影响输出。"""
    batch_size = 2
    dim = 64
    height, width = 8, 8

    fusion = AvailabilityAwareFusion(dim)
    base_feat = torch.randn(batch_size, dim, height, width)
    highres_feat = torch.randn(batch_size, dim, height, width)

    mask_zero = torch.zeros(batch_size, 1, height, width)
    mask_one = torch.ones(batch_size, 1, height, width)

    out_zero = fusion(base_feat, highres_feat, mask_zero)
    out_one = fusion(base_feat, highres_feat, mask_one)

    assert out_zero.shape == (batch_size, dim, height, width)
    assert out_one.shape == (batch_size, dim, height, width)
    assert not torch.allclose(out_zero, out_one)


def test_availability_aware_fusion_masks_highres_features() -> None:
    """高分不可用时，不同 highres_feat 不应改变融合输出。"""
    batch_size = 2
    dim = 16
    height, width = 8, 8

    fusion = AvailabilityAwareFusion(dim)
    base_feat = torch.randn(batch_size, dim, height, width)
    highres_a = torch.randn(batch_size, dim, height, width)
    highres_b = highres_a + 100.0
    mask_zero = torch.zeros(batch_size, 1, height, width)

    out_a = fusion(base_feat, highres_a, mask_zero)
    out_b = fusion(base_feat, highres_b, mask_zero)

    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_availability_aware_fusion_rejects_size_mismatch() -> None:
    """AvailabilityAwareFusion 在输入尺寸不一致时应抛出断言错误。"""
    fusion = AvailabilityAwareFusion(16)
    base_feat = torch.randn(1, 16, 8, 8)
    highres_feat = torch.randn(1, 16, 4, 4)
    mask = torch.ones(1, 1, 8, 8)

    with pytest.raises(AssertionError):
        fusion(base_feat, highres_feat, mask)


def test_time_encoding() -> None:
    """TimeEncoding 应将时间戳编码为 (B, T, embed_dim)。"""
    batch_size, time_steps, embed_dim = 2, 6, 16

    encoding = TimeEncoding(embed_dim, max_periods=64)
    timestamps = torch.arange(time_steps).float().unsqueeze(0).expand(batch_size, -1)
    out = encoding(timestamps)

    assert out.shape == (batch_size, time_steps, embed_dim)


def test_window_code() -> None:
    """WindowCode 应将窗口 ID 映射为 embedding。"""
    num_windows, embed_dim = 12, 16
    batch_size = 2

    window_code = WindowCode(num_windows, embed_dim)
    window_ids = torch.randint(0, num_windows, (batch_size,))
    out = window_code(window_ids)

    assert out.shape == (batch_size, embed_dim)


def _make_yyyymm_timestamps(batch_size: int, time_steps: int, start: int = 202501) -> torch.Tensor:
    """生成 YYYYMM 格式时间戳，超出单个月份后自动进位到下个月。"""
    year = start // 100
    month = start % 100
    values = []
    for _ in range(time_steps):
        values.append(year * 100 + month)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return torch.tensor(values, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)


def test_monthly_embedding_module_shape_and_mask() -> None:
    """MonthlyEmbeddingModule 应输出正确的月度特征与月度掩码。"""
    batch_size, time_steps, height, width, in_channels = 2, 4, 4, 4, 8
    embed_dim = 16
    num_months = 6

    module = MonthlyEmbeddingModule(in_channels, embed_dim, num_months)
    feats = torch.randn(batch_size, time_steps, height, width, in_channels)
    timestamps = _make_yyyymm_timestamps(batch_size, time_steps, start=202501)
    mask = torch.ones(batch_size, time_steps)
    mask[0, 1] = 0.0

    monthly_feats, monthly_mask = module(feats, timestamps, mask)

    assert monthly_feats.shape == (batch_size, num_months, height, width, embed_dim)
    assert monthly_mask.shape == (batch_size, num_months)
    # 202501, 202502, 202503, 202504 四个月份有观测。
    assert monthly_mask[0, 0].item() == 1.0
    assert monthly_mask[0, 1].item() == 0.0
    assert monthly_mask[0, 2].item() == 1.0
    assert monthly_mask[0, 5].item() == 0.0
    assert not torch.isnan(monthly_feats).any()


def test_monthly_embedding_module_custom_ref_month() -> None:
    """自定义参考月应把跨年月份映射到连续月度 bin。"""
    module = MonthlyEmbeddingModule(
        in_channels=8,
        embed_dim=16,
        num_months=2,
        ref_year=2025,
        ref_month=12,
    )
    feats = torch.randn(1, 2, 4, 4, 8)
    timestamps = torch.tensor([[202512, 202601]], dtype=torch.long)
    mask = torch.ones(1, 2)

    monthly_feats, monthly_mask = module(feats, timestamps, mask)

    assert monthly_feats.shape == (1, 2, 4, 4, 16)
    assert monthly_mask.tolist() == [[1.0, 1.0]]
    assert module._yyyymm_to_index(timestamps).tolist() == [[0, 1]]


def test_monthly_embedding_module_missing_token_used() -> None:
    """缺失月份的特征应接近可学习 missing_token，而有观测月份应与之不同。"""
    batch_size, time_steps, height, width, in_channels = 1, 2, 4, 4, 8
    embed_dim = 16
    num_months = 4

    module = MonthlyEmbeddingModule(in_channels, embed_dim, num_months)
    # 固定输入以便复现。
    torch.manual_seed(0)
    feats = torch.randn(batch_size, time_steps, height, width, in_channels)
    timestamps = torch.tensor([[202501, 202503]], dtype=torch.long)
    mask = torch.ones(batch_size, time_steps)

    monthly_feats, monthly_mask = module(feats, timestamps, mask)

    missing_token = module.missing_token
    # 月份 1（202502）与 3（202504）缺失，应使用 missing_token。
    assert torch.allclose(monthly_feats[0, 1], missing_token.squeeze(0), atol=1e-6)
    assert torch.allclose(monthly_feats[0, 3], missing_token.squeeze(0), atol=1e-6)
    # 月份 0 与 2 有观测，不应等于 missing_token。
    assert not torch.allclose(monthly_feats[0, 0], missing_token.squeeze(0), atol=1e-6)
    assert not torch.allclose(monthly_feats[0, 2], missing_token.squeeze(0), atol=1e-6)
    assert monthly_mask[0, 0].item() == 1.0
    assert monthly_mask[0, 1].item() == 0.0


def test_aef_model_raises_when_all_valid_timestamps_out_of_month_range() -> None:
    """所有有效观测越界时，AEFModel 不应静默输出全 missing_token 月度特征。"""
    model = AEFModel(
        sensor_channels={"s2": 3},
        embed_dim=8,
        target_heads={"s2_recon": ("continuous", 3)},
        stem_dim=8,
        stp={
            "space_dim": 16,
            "time_dim": 16,
            "precision_dim": 16,
            "num_blocks": 1,
            "num_heads": 2,
        },
        num_months=2,
        ref_year=2025,
        ref_month=12,
    )
    source_frames = {"s2": torch.randn(1, 2, 3, 16, 16)}
    source_masks = {"s2": torch.ones(1, 2)}
    timestamps = torch.tensor([[202501, 202502]], dtype=torch.long)

    with pytest.raises(ValueError, match="月度范围外"):
        model(source_frames, source_masks, timestamps)


def test_aef_model_forward() -> None:
    """AEFModel 应能前向传播并返回月度正确形状的输出（含高分辨率数据）。"""
    num_months = 4
    sensor_channels = {
        "s2": 10,
        "s1": 2,
        "landsat": 6,
        "highres_optical_haidian": 4,
        "highres_sar_haidian": 1,
    }
    embed_dim = 16
    target_heads = {
        "s2_recon": ("continuous", 10),
        "s1_recon": ("continuous", 2),
        "landsat_recon": ("continuous", 6),
        "worldcover": ("categorical", 9),
    }
    model = AEFModel(
        sensor_channels,
        embed_dim,
        target_heads,
        num_space_heads=2,
        num_months=num_months,
    )

    batch_size, time_steps, height, width = 2, 4, 16, 16
    source_frames = {
        "s2": torch.randn(batch_size, time_steps, 10, height, width),
        "s1": torch.randn(batch_size, time_steps, 2, height, width),
        "landsat": torch.randn(batch_size, time_steps, 6, height, width),
    }
    source_masks = {k: torch.ones(batch_size, time_steps) for k in source_frames}
    timestamps = _make_yyyymm_timestamps(batch_size, time_steps, start=202501)
    highres_frames = {
        "highres_optical_haidian": torch.randn(batch_size, 4, height, width),
        "highres_sar_haidian": torch.randn(batch_size, 1, height, width),
    }
    highres_masks = {
        "highres_optical_haidian": torch.ones(batch_size, 1, height, width),
        "highres_sar_haidian": torch.ones(batch_size, 1, height, width),
    }

    out = model(
        source_frames,
        source_masks,
        timestamps,
        highres_frames,
        highres_masks,
    )

    assert isinstance(out, AEFOutput)
    assert out.embedding.shape == (batch_size, num_months, embed_dim)
    assert out.embedding_map.shape == (batch_size, num_months, embed_dim, height, width)
    assert set(out.reconstructions.keys()) == set(target_heads.keys())
    assert out.reconstructions["s2_recon"].shape == (batch_size, num_months, 10, height, width)
    assert out.reconstructions["s1_recon"].shape == (batch_size, num_months, 2, height, width)
    assert out.reconstructions["landsat_recon"].shape == (batch_size, num_months, 6, height, width)
    assert out.reconstructions["worldcover"].shape == (batch_size, num_months, 9, height, width)

    # embedding_map 应位于单位球面上。
    norms = out.embedding_map.norm(dim=2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    # 场景级 embedding 也应是单位向量。
    emb_norms = out.embedding.norm(dim=-1)
    assert torch.allclose(emb_norms, torch.ones_like(emb_norms), atol=1e-6)


def test_aef_model_without_highres() -> None:
    """AEFModel 在没有高分辨率数据时也应能正常前向传播。"""
    num_months = 3
    sensor_channels = {"s2": 10, "s1": 2, "landsat": 6}
    embed_dim = 16
    target_heads = {"s2_recon": ("continuous", 10)}
    model = AEFModel(
        sensor_channels,
        embed_dim,
        target_heads,
        num_space_heads=2,
        num_months=num_months,
    )

    batch_size, time_steps, height, width = 2, 3, 16, 16
    source_frames = {
        "s2": torch.randn(batch_size, time_steps, 10, height, width),
        "s1": torch.randn(batch_size, time_steps, 2, height, width),
        "landsat": torch.randn(batch_size, time_steps, 6, height, width),
    }
    source_masks = {k: torch.ones(batch_size, time_steps) for k in source_frames}
    timestamps = _make_yyyymm_timestamps(batch_size, time_steps, start=202501)

    out = model(source_frames, source_masks, timestamps)

    assert isinstance(out, AEFOutput)
    assert out.embedding.shape == (batch_size, num_months, embed_dim)
    assert out.embedding_map.shape == (batch_size, num_months, embed_dim, height, width)
    assert "s2_recon" in out.reconstructions
    assert out.reconstructions["s2_recon"].shape == (batch_size, num_months, 10, height, width)


def test_p10c_contract_outputs_64_by_128_by_128() -> None:
    """精简后的生产核心必须保持 P10C 的 64×128×128 dense embedding 合同。"""
    model = AEFModel(
        sensor_channels={"s2": 3},
        embed_dim=64,
        target_heads={},
        stem_dim=8,
        stp={
            "space_dim": 16,
            "time_dim": 16,
            "precision_dim": 16,
            "precision_scale": 1,
            "num_blocks": 1,
            "num_heads": 2,
            "time_attention_mode": "none",
        },
        num_months=1,
        ref_year=2025,
        ref_month=12,
    ).eval()

    with torch.no_grad():
        output = model(
            source_frames={"s2": torch.randn(1, 1, 3, 128, 128)},
            source_masks={"s2": torch.ones(1, 1)},
            timestamps=torch.tensor([[202512]]),
        )

    assert output.embedding_map.shape == (1, 1, 64, 128, 128)
    assert torch.isfinite(output.embedding_map).all()
    assert torch.allclose(
        output.embedding_map.norm(dim=2),
        torch.ones(1, 1, 128, 128),
        atol=1e-6,
    )


def test_aef_model_time_attention_none_keeps_month_state_local() -> None:
    """关闭时间 attention 后，修改后一个月份不应影响前一个月份 embedding。"""
    num_months = 2
    sensor_channels = {"s2": 10}
    embed_dim = 16
    target_heads = {"s2_recon": ("continuous", 10)}
    model = AEFModel(
        sensor_channels,
        embed_dim,
        target_heads,
        stem_dim=16,
        stp={
            "space_dim": 64,
            "time_dim": 32,
            "precision_dim": 32,
            "num_blocks": 2,
            "num_heads": 2,
            "time_attention_mode": "none",
        },
        num_months=num_months,
    )
    model.eval()

    batch_size, time_steps, height, width = 1, 2, 16, 16
    source_frames = {
        "s2": torch.randn(batch_size, time_steps, 10, height, width),
    }
    source_masks = {"s2": torch.ones(batch_size, time_steps)}
    timestamps = _make_yyyymm_timestamps(batch_size, time_steps, start=202501)

    changed_frames = {"s2": source_frames["s2"].clone()}
    changed_frames["s2"][:, 1] = changed_frames["s2"][:, 1] + 10.0

    with torch.no_grad():
        out_a = model(source_frames, source_masks, timestamps)
        out_b = model(changed_frames, source_masks, timestamps)

    assert torch.allclose(out_a.embedding_map[:, 0], out_b.embedding_map[:, 0], atol=1e-5)
    assert not torch.allclose(out_a.embedding_map[:, 1], out_b.embedding_map[:, 1], atol=1e-5)


def test_aef_model_can_disable_highres_fusion_to_embedding() -> None:
    """关闭高分融合后，静态高分输入不应改变导出的主 embedding。"""
    num_months = 2
    sensor_channels = {"s2": 10, "highres_optical_haidian": 3}
    embed_dim = 16
    target_heads = {
        "s2_recon": ("continuous", 10),
        "highres_optical_haidian_recon": ("continuous", 3),
    }
    model = AEFModel(
        sensor_channels,
        embed_dim,
        target_heads,
        stem_dim=16,
        stp={
            "space_dim": 64,
            "time_dim": 32,
            "precision_dim": 32,
            "num_blocks": 1,
            "num_heads": 2,
            "time_attention_mode": "none",
            "highres_fusion_to_embedding": False,
        },
        num_months=num_months,
    )
    model.eval()

    batch_size, time_steps, height, width = 1, 2, 16, 16
    source_frames = {"s2": torch.randn(batch_size, time_steps, 10, height, width)}
    source_masks = {"s2": torch.ones(batch_size, time_steps)}
    timestamps = _make_yyyymm_timestamps(batch_size, time_steps, start=202501)
    highres_masks = {"highres_optical_haidian": torch.ones(batch_size, 1, height * 2, width * 2)}
    highres_a = {"highres_optical_haidian": torch.randn(batch_size, 3, height, width)}
    highres_b = {"highres_optical_haidian": highres_a["highres_optical_haidian"] + 10.0}

    with torch.no_grad():
        out_a = model(source_frames, source_masks, timestamps, highres_a, highres_masks)
        out_b = model(source_frames, source_masks, timestamps, highres_b, highres_masks)

    assert torch.allclose(out_a.embedding_map, out_b.embedding_map, atol=1e-5)
    assert out_a.reconstructions["highres_optical_haidian_recon"].shape == (
        batch_size,
        num_months,
        3,
        height,
        width,
    )


def test_aef_model_highres_fusion_changes_embedding() -> None:
    """开启高分融合后，同一低分输入的不同高分影像应改变 embedding。"""
    num_months = 2
    sensor_channels = {"s2": 10, "highres_optical_haidian": 3}
    model = AEFModel(
        sensor_channels,
        embed_dim=16,
        target_heads={
            "s2_recon": ("continuous", 10),
            "highres_optical_haidian_recon": ("continuous", 3),
        },
        stem_dim=16,
        stp={
            "space_dim": 64,
            "time_dim": 32,
            "precision_dim": 32,
            "precision_scale": 1,
            "num_blocks": 1,
            "num_heads": 2,
            "time_attention_mode": "none",
            "highres_fusion_to_embedding": True,
        },
        num_months=num_months,
    )
    model.eval()

    batch_size, time_steps, height, width = 1, 2, 16, 16
    source_frames = {"s2": torch.randn(batch_size, time_steps, 10, height, width)}
    source_masks = {"s2": torch.ones(batch_size, time_steps)}
    timestamps = _make_yyyymm_timestamps(batch_size, time_steps, start=202501)
    highres_masks = {"highres_optical_haidian": torch.ones(batch_size, 1, height * 2, width * 2)}
    highres_a = {"highres_optical_haidian": torch.randn(batch_size, 3, height * 2, width * 2)}
    highres_b = {"highres_optical_haidian": highres_a["highres_optical_haidian"] + 5.0}

    with torch.no_grad():
        out_a = model(source_frames, source_masks, timestamps, highres_a, highres_masks)
        out_b = model(source_frames, source_masks, timestamps, highres_b, highres_masks)

    assert out_a.embedding_map.shape == (batch_size, num_months, 16, height, width)
    assert not torch.allclose(out_a.embedding_map, out_b.embedding_map, atol=1e-5)


def test_aef_model_missing_months_no_nan() -> None:
    """当部分月份无观测时，AEFModel 输出不应出现 NaN。"""
    num_months = 6
    sensor_channels = {"s2": 10}
    embed_dim = 16
    target_heads = {"s2_recon": ("continuous", 10)}
    model = AEFModel(
        sensor_channels,
        embed_dim,
        target_heads,
        stem_dim=16,
        stp={"space_dim": 64, "time_dim": 32, "precision_dim": 32, "num_blocks": 2, "num_heads": 2},
        num_months=num_months,
    )

    batch_size, time_steps, height, width = 1, 2, 16, 16
    source_frames = {
        "s2": torch.randn(batch_size, time_steps, 10, height, width),
    }
    source_masks = {"s2": torch.ones(batch_size, time_steps)}
    # 只有 202501 与 202503 有数据。
    timestamps = torch.tensor([[202501, 202503]], dtype=torch.long)

    out = model(source_frames, source_masks, timestamps)

    assert not torch.isnan(out.embedding_map).any()
    assert not torch.isnan(out.embedding).any()
    assert not torch.isnan(out.reconstructions["s2_recon"]).any()
    assert out.embedding_map.shape == (batch_size, num_months, embed_dim, height, width)


def test_aef_model_global_timestamps_tensor() -> None:
    """AEFModel 应接受单一全局 timestamps 张量并输出月度结果。"""
    num_months = 3
    sensor_channels = {"s2": 10, "s1": 2}
    embed_dim = 16
    target_heads = {"s2_recon": ("continuous", 10)}
    model = AEFModel(
        sensor_channels,
        embed_dim,
        target_heads,
        stem_dim=16,
        stp={"space_dim": 64, "time_dim": 32, "precision_dim": 32, "num_blocks": 2, "num_heads": 2},
        num_months=num_months,
    )

    batch_size, time_steps, height, width = 1, 3, 16, 16
    source_frames = {
        "s2": torch.randn(batch_size, time_steps, 10, height, width),
        "s1": torch.randn(batch_size, time_steps, 2, height, width),
    }
    source_masks = {k: torch.ones(batch_size, time_steps) for k in source_frames}
    timestamps = _make_yyyymm_timestamps(batch_size, time_steps, start=202501)

    out = model(source_frames, source_masks, timestamps)
    assert out.embedding_map.shape == (batch_size, num_months, embed_dim, height, width)
    assert out.embedding.shape == (batch_size, num_months, embed_dim)


def test_stp_space_operator() -> None:
    """STPSpaceOperator 应保持 (B, T, H, W, C) 形状。"""
    batch_size, time_steps, height, width, dim = 2, 3, 4, 4, 64
    operator = STPSpaceOperator(dim, num_heads=8)
    x = torch.randn(batch_size, time_steps, height, width, dim)
    y = operator(x)
    assert y.shape == (batch_size, time_steps, height, width, dim)


def test_stp_time_operator() -> None:
    """STPTimeOperator 应保持 (B, T, H, W, C) 形状并支持时间掩码。"""
    batch_size, time_steps, height, width, dim = 2, 3, 4, 4, 64
    operator = STPTimeOperator(dim, num_heads=8)
    x = torch.randn(batch_size, time_steps, height, width, dim)
    timestamps = torch.arange(time_steps).float().unsqueeze(0).expand(batch_size, -1)
    mask = torch.ones(batch_size, time_steps)
    mask[0, -1] = 0
    y = operator(x, timestamps, mask=mask)
    assert y.shape == (batch_size, time_steps, height, width, dim)


def test_stp_precision_operator() -> None:
    """STPPrecisionOperator 应保持 (B, T, H, W, C) 形状。"""
    batch_size, time_steps, height, width, dim = 2, 3, 4, 4, 64
    operator = STPPrecisionOperator(dim)
    x = torch.randn(batch_size, time_steps, height, width, dim)
    y = operator(x)
    assert y.shape == (batch_size, time_steps, height, width, dim)


def test_multi_resolution_stp_block() -> None:
    """MultiResolutionSTPBlock 应保持三个路径各自的形状。"""
    batch_size, time_steps = 2, 3
    space_dim, time_dim, precision_dim = 64, 32, 16
    space_h, space_w = 1, 1
    time_h, time_w = 2, 2
    precision_h, precision_w = 8, 8

    block = MultiResolutionSTPBlock(space_dim, time_dim, precision_dim, num_heads=4)
    space_x = torch.randn(batch_size, time_steps, space_h, space_w, space_dim)
    time_x = torch.randn(batch_size, time_steps, time_h, time_w, time_dim)
    precision_x = torch.randn(batch_size, time_steps, precision_h, precision_w, precision_dim)
    timestamps = torch.arange(time_steps).float().unsqueeze(0).expand(batch_size, -1)

    s_out, t_out, p_out = block(space_x, time_x, precision_x, timestamps)
    assert s_out.shape == space_x.shape
    assert t_out.shape == time_x.shape
    assert p_out.shape == precision_x.shape


def test_stp_encoder() -> None:
    """STPEncoder 应输出 1/2L 精度的特征。"""
    batch_size, time_steps, height, width, channels = 2, 3, 16, 16, 32
    encoder = STPEncoder(
        input_channels=channels,
        space_dim=64,
        time_dim=32,
        precision_dim=16,
        num_blocks=2,
        num_heads=4,
    )
    x = torch.randn(batch_size, time_steps, height, width, channels)
    timestamps = torch.arange(time_steps).float().unsqueeze(0).expand(batch_size, -1)
    y, input_size = encoder(x, timestamps)
    assert input_size == (height, width)
    assert y.shape == (batch_size, time_steps, height // 2, width // 2, 16)


def test_embedding_upsample_head() -> None:
    """EmbeddingUpsampleHead 应将 (B, H, W, C) 上采样到 (B, 2H, 2W, C)。"""
    batch_size, height, width, dim = 2, 4, 4, 16
    head = EmbeddingUpsampleHead(dim)
    x = torch.randn(batch_size, height, width, dim)
    y = head(x)
    assert y.shape == (batch_size, height * 2, width * 2, dim)


def test_embedding_upsample_head_temporal_reshape() -> None:
    """EmbeddingUpsampleHead 应支持将 (B*T, H, W, C) 整体处理后恢复形状。"""
    batch_size, time_steps, height, width, dim = 2, 3, 4, 4, 16
    head = EmbeddingUpsampleHead(dim)
    x = torch.randn(batch_size * time_steps, height, width, dim)
    y = head(x)
    assert y.shape == (batch_size * time_steps, height * 2, width * 2, dim)


def test_spatial_resampling_avoids_transposed_conv() -> None:
    """空间重采样不应使用 Ascend 反传不稳定的 ConvTranspose2d。"""
    up = LearnedSpatialResampling(8, 16, 16.0)
    down = LearnedSpatialResampling(16, 8, 1.0 / 16.0)
    x_up = torch.randn(2, 8, 4, 4)
    x_down = torch.randn(2, 16, 64, 64)

    assert not any(isinstance(m, nn.ConvTranspose2d) for m in up.modules())
    assert not any(isinstance(m, nn.ConvTranspose2d) for m in down.modules())
    assert up(x_up, target_size=(64, 64)).shape == (2, 16, 64, 64)
    assert down(x_down, target_size=(4, 4)).shape == (2, 8, 4, 4)


def test_aef_model_odd_size() -> None:
    """AEFModel 在奇数输入尺寸下应输出与输入相同空间分辨率的月度 embedding_map。"""
    num_months = 2
    sensor_channels = {"s2": 10, "s1": 2}
    embed_dim = 16
    target_heads = {"s2_recon": ("continuous", 10)}
    model = AEFModel(
        sensor_channels,
        embed_dim,
        target_heads,
        stem_dim=16,
        stp={"space_dim": 64, "time_dim": 32, "precision_dim": 32, "num_blocks": 2, "num_heads": 2},
        num_months=num_months,
    )
    batch_size, time_steps, height, width = 1, 2, 17, 17
    source_frames = {
        "s2": torch.randn(batch_size, time_steps, 10, height, width),
        "s1": torch.randn(batch_size, time_steps, 2, height, width),
    }
    source_masks = {k: torch.ones(batch_size, time_steps) for k in source_frames}
    timestamps = _make_yyyymm_timestamps(batch_size, time_steps, start=202501)
    out = model(source_frames, source_masks, timestamps)
    assert out.embedding_map.shape == (batch_size, num_months, embed_dim, height, width)
    assert out.reconstructions["s2_recon"].shape == (batch_size, num_months, 10, height, width)


def test_stp_time_operator_all_masked_no_nan() -> None:
    """STPTimeOperator 对全被掩码的样本行应输出 0 而非 NaN。"""
    from xuannv_embedding.models.blocks import STPTimeOperator

    dim = 64
    operator = STPTimeOperator(dim, num_heads=8)
    x = torch.randn(2, 3, 4, 4, dim)
    timestamps = torch.arange(3).float().unsqueeze(0).expand(2, -1)
    mask = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)
    out = operator(x, timestamps, mask=mask)
    assert out.shape == (2, 3, 4, 4, dim)
    assert not torch.isnan(out).any()


def test_stp_encoder_rejects_too_small_input() -> None:
    """STPEncoder 应对小于 SPACE_SCALE 的输入抛出清晰的 ValueError。"""
    encoder = STPEncoder(
        input_channels=8,
        space_dim=64,
        time_dim=32,
        precision_dim=32,
        num_blocks=2,
        num_heads=2,
    )
    x = torch.randn(1, 2, 8, 15, 15)
    ts = torch.tensor([[1.0, 2.0]])
    with pytest.raises(ValueError, match="input height/width >= 16"):
        encoder(x, ts)


def test_aef_model_skips_empty_temporal_source() -> None:
    """AEFModel 应安全跳过时间维度为 0 的 source。"""
    num_months = 2
    sensor_channels = {"s2": 10, "s1": 2}
    embed_dim = 16
    target_heads = {"s2_recon": ("continuous", 10)}
    model = AEFModel(
        sensor_channels,
        embed_dim,
        target_heads,
        stem_dim=16,
        stp={"space_dim": 64, "time_dim": 32, "precision_dim": 32, "num_blocks": 2, "num_heads": 2},
        num_months=num_months,
    )
    batch_size, height, width = 1, 16, 16
    source_frames = {
        "s2": torch.randn(batch_size, 0, 10, height, width),
        "s1": torch.randn(batch_size, 2, 2, height, width),
    }
    source_masks = {"s2": torch.zeros(batch_size, 0), "s1": torch.ones(batch_size, 2)}
    timestamps = _make_yyyymm_timestamps(batch_size, 2, start=202501)
    out = model(source_frames, source_masks, timestamps)
    assert out.embedding_map.shape == (batch_size, num_months, embed_dim, height, width)
