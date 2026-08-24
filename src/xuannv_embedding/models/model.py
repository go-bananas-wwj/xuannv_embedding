# AEF 主模型：多源时序遥感数据 -> 多分辨率 STP 编码器 -> 月度嵌入 ->
# 嵌入上采样 -> 可选高分辨率融合（逐月）-> vMF 瓶颈 -> 解码器。

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from xuannv_embedding.models.blocks import (
    EmbeddingUpsampleHead,
    MonthlyEmbeddingModule,
    STPEncoder,
)
from xuannv_embedding.models.bottleneck import VMFBottleneck
from xuannv_embedding.models.decoders import CategoricalDecoder, ContinuousDecoder
from xuannv_embedding.models.highres_fusion import AvailabilityAwareFusion
from xuannv_embedding.models.sensor_encoders import (
    NativeResolutionHighResEncoder,
    SensorEncoderBank,
    SourceAwareTemporalFusion,
)

logger = logging.getLogger(__name__)


@dataclass
class AEFOutput:
    """AEFModel 前向输出容器。"""

    embedding_map: torch.Tensor  # [B, T_month, D, H, W]，与输入相同空间分辨率
    embedding: torch.Tensor  # [B, T_month, D]
    reconstructions: dict[str, torch.Tensor]  # [B, T_month, C_out, H, W]


class AEFModel(nn.Module):
    """AEF（Angular Embedding Field）月度地理嵌入模型。

    输入支持多源时序遥感数据（如 S2/S1/Landsat），以及可选的稀疏高分辨率
    数据。模型通过 per-sensor stem、多分辨率 Space-Time-Precision 编码器、
    月度嵌入模块、嵌入上采样头、vMF 瓶颈生成单位球面上的逐月像素级与场景级
    嵌入，并解码回各目标模态。输出 ``embedding_map`` 与输入保持相同空间分辨率。
    """

    def __init__(
        self,
        sensor_channels: dict[str, int],
        embed_dim: int,
        target_heads: dict[str, tuple[str, int]],
        stem_dim: int = 32,
        stp: dict[str, Any] | None = None,
        num_space_heads: int = 8,
        num_months: int = 17,
        ref_year: int = 2025,
        ref_month: int = 1,
        gradient_checkpointing: bool = False,
    ) -> None:
        """初始化 AEFModel。

        Args:
            sensor_channels: 数据源到输入通道数的映射，例如
                ``{"s2": 12, "s1": 2, "landsat": 7,
                "highres_optical_haidian": 4, "highres_sar_haidian": 1}``。
                若使用高分辨率融合，需要为每个以 ``"highres"`` 开头的 source
                注册对应的通道数。
            embed_dim: 统一嵌入维度，也是最终 embedding_map 的通道数。
            target_heads: 解码器配置，格式为 ``name -> (kind, channels)``，
                其中 ``kind`` 为 ``"continuous"`` 或 ``"categorical"``。
            stem_dim: 各时序数据源经 stem encoder 后输出的统一通道数。
            stp: STP 编码器配置字典，可包含 ``space_dim``、``time_dim``、
                ``precision_dim``、``num_blocks``、``num_heads``。缺失项使用默认值。
            num_space_heads: ``stp["num_heads"]`` 的默认值。
            num_months: 月度 bin 数量，阶段一默认为 17（2025-01 至 2026-05）。
            ref_year: 月度 bin 起始年份，应与数据集首月一致。
            ref_month: 月度 bin 起始月份，应与数据集首月一致。
            gradient_checkpointing: 是否启用 STP 编码器的梯度检查点。
        """
        super().__init__()
        self.sensor_channels = sensor_channels
        self.embed_dim = embed_dim
        self.target_heads = target_heads
        self.stem_dim = stem_dim
        self.num_months = num_months
        self.ref_year = ref_year
        self.ref_month = ref_month

        stp_cfg = dict(stp) if stp is not None else {}
        stp_defaults = {
            "space_dim": 512,
            "time_dim": 256,
            "precision_dim": 128,
            "precision_scale": 2,
            "num_blocks": 6,
            "num_heads": num_space_heads,
            "temporal_fusion": "concat",
            "time_attention_mode": "full",
            "highres_fusion_to_embedding": True,
        }
        for key, value in stp_defaults.items():
            stp_cfg.setdefault(key, value)
        stp_cfg.setdefault("gradient_checkpointing", gradient_checkpointing)
        self.stp_cfg = stp_cfg

        temporal_channels = {
            name: ch for name, ch in sensor_channels.items() if not name.startswith("highres")
        }
        highres_channels = {
            name: ch for name, ch in sensor_channels.items() if name.startswith("highres")
        }

        self.temporal_source_order = list(temporal_channels.keys())
        self.temporal_stem_bank = SensorEncoderBank(temporal_channels, stem_dim)
        self.temporal_fusion_mode = stp_cfg["temporal_fusion"]
        if self.temporal_fusion_mode not in {"concat", "gated_sum"}:
            raise ValueError(
                "stp.temporal_fusion 仅支持 'concat' 或 'gated_sum'，"
                f"实际为 {self.temporal_fusion_mode!r}"
            )
        self.highres_fusion_to_embedding = bool(stp_cfg["highres_fusion_to_embedding"])
        self.temporal_fusion = (
            SourceAwareTemporalFusion(self.temporal_source_order, stem_dim)
            if self.temporal_fusion_mode == "gated_sum"
            else None
        )
        self.highres_encoders = nn.ModuleDict(
            {
                source: NativeResolutionHighResEncoder(in_ch, embed_dim)
                for source, in_ch in highres_channels.items()
            }
        )

        total_stem_channels = (
            stem_dim
            if self.temporal_fusion_mode == "gated_sum"
            else stem_dim * len(temporal_channels)
        )
        self.total_stem_channels = total_stem_channels
        self.stp_encoder = STPEncoder(
            input_channels=total_stem_channels,
            space_dim=stp_cfg["space_dim"],
            time_dim=stp_cfg["time_dim"],
            precision_dim=stp_cfg["precision_dim"],
            num_blocks=stp_cfg["num_blocks"],
            num_heads=stp_cfg["num_heads"],
            gradient_checkpointing=stp_cfg["gradient_checkpointing"],
            time_attention_mode=stp_cfg["time_attention_mode"],
            precision_scale=stp_cfg["precision_scale"],
        )

        self.monthly_embed = MonthlyEmbeddingModule(
            in_channels=stp_cfg["precision_dim"],
            embed_dim=embed_dim,
            num_months=num_months,
            ref_year=ref_year,
            ref_month=ref_month,
        )
        self.upsample_head = EmbeddingUpsampleHead(embed_dim, embed_dim)
        self.highres_fusion = AvailabilityAwareFusion(embed_dim)
        self.bottleneck = VMFBottleneck(embed_dim, embed_dim)

        # 构建各目标模态的 decoder head。
        self.decoders = nn.ModuleDict()
        for name, (kind, ch) in target_heads.items():
            if kind == "continuous":
                self.decoders[name] = ContinuousDecoder(embed_dim, ch)
            elif kind == "categorical":
                self.decoders[name] = CategoricalDecoder(embed_dim, ch)
            else:
                raise ValueError(f"不支持的 decoder 类型: {kind!r}")

    def forward(
        self,
        source_frames: dict[str, torch.Tensor],
        source_masks: dict[str, torch.Tensor],
        timestamps: torch.Tensor,
        highres_frames: dict[str, torch.Tensor] | None = None,
        highres_masks: dict[str, torch.Tensor] | None = None,
    ) -> AEFOutput:
        """AEFModel 前向传播。

        Args:
            source_frames: 各时序数据源的输入，格式 ``{source: (B, T, C, H, W)}``。
            source_masks: 各时序数据源的时间有效掩码，格式 ``{source: (B, T)}``。
            timestamps: 全局时间戳，形状 ``(B, T)``，用于所有时序 source。推荐格式为
                ``YYYYMM`` 整数，例如 ``202501``。
            highres_frames: 可选的高分辨率单帧输入字典，格式 ``{source: (B, C, H, W)}``。
            highres_masks: 高分辨率可用性掩码字典，格式 ``{source: (B, 1, H, W)}``；
                提供 ``highres_frames`` 时也必须提供，且 key 需与 ``highres_frames`` 一致。

        Returns:
            AEFOutput，包含月度 embedding_map、embedding 与 reconstructions。
        """
        if not source_frames:
            raise ValueError("source_frames 不能为空字典")

        # 1) 编码每个时序数据源到 stem 维度，并屏蔽缺失时间步。
        # 跳过时间维度为 0 或未在 sensor_channels 中注册的 source
        # （未注册 source 通常作为 target-only 标签，例如 worldcover）。
        temporal_sources = [
            s
            for s in source_frames
            if not s.startswith("highres")
            and source_frames[s].shape[1] > 0
            and s in self.sensor_channels
        ]
        if not temporal_sources:
            raise ValueError("source_frames 中至少需要一个有效的时序数据源")

        global_timestamps = timestamps

        # 第一个 source 用于确定 batch/time/space 维度。
        first_source = temporal_sources[0]
        first_x = source_frames[first_source]
        B, T, _, H, W = first_x.shape
        input_size = (H, W)
        temporal_input = torch.zeros(
            B,
            T,
            H,
            W,
            self.total_stem_channels,
            device=first_x.device,
            dtype=first_x.dtype,
        )
        masks: list[torch.Tensor] = []
        encoded_sources: dict[str, torch.Tensor] = {}
        encoded_masks: dict[str, torch.Tensor] = {}

        for source in temporal_sources:
            x = source_frames[source]
            mask = source_masks.get(source)
            if mask is None:
                raise KeyError(f"缺少 source mask: {source!r}")

            x_flat = x.reshape(B * T, x.shape[2], H, W)
            x_enc = self.temporal_stem_bank(x_flat, source)  # (B*T, stem_dim, H, W)
            x_enc = x_enc.view(B, T, self.stem_dim, H, W)
            mask_5d = mask[:, :, None, None, None]
            x_enc = x_enc * mask_5d

            if self.temporal_fusion_mode == "concat":
                source_idx = self.temporal_source_order.index(source)
                start = source_idx * self.stem_dim
                end = start + self.stem_dim
                temporal_input[..., start:end] = x_enc.permute(0, 1, 3, 4, 2)
            else:
                encoded_sources[source] = x_enc
                encoded_masks[source] = mask
            masks.append(mask)

        if self.temporal_fusion_mode == "gated_sum":
            if self.temporal_fusion is None:
                raise RuntimeError("gated_sum 模式缺少 temporal_fusion 模块")
            fused = self.temporal_fusion(encoded_sources, encoded_masks, temporal_sources)
            temporal_input = fused.permute(0, 1, 3, 4, 2)

        # 2) 合并时间有效性掩码。
        combined_mask = torch.stack(masks, dim=0).amax(dim=0)  # (B, T)
        self._validate_month_range(global_timestamps, combined_mask)

        # 3) STP 编码器输出精度路径特征，同时拿回原始输入尺寸。
        feats, _ = self.stp_encoder(
            temporal_input, global_timestamps, mask=combined_mask
        )  # (B, T, H//precision_scale, W//precision_scale, precision_dim)

        # 4) 月度嵌入：按 YYYYMM 分 bin，缺失月份用 missing_token。
        monthly_feats, monthly_mask = self.monthly_embed(
            feats, global_timestamps, combined_mask
        )  # (B, T_month, H', W', embed_dim), (B, T_month)

        # 5) 逐月恢复到原始分辨率。precision_scale=1 时避免无意义的上采样再缩回。
        Bm, M, Hh, Wh, D = monthly_feats.shape
        if (Hh, Wh) == input_size:
            mu_up = monthly_feats
        else:
            monthly_feats_flat = monthly_feats.reshape(Bm * M, Hh, Wh, D)
            mu_up_flat = self.upsample_head(
                monthly_feats_flat, target_size=input_size
            )  # (B*T_month, H, W, embed_dim)
            mu_up = mu_up_flat.view(Bm, M, H, W, D)
        embedding_map = mu_up.permute(0, 1, 4, 2, 3)  # (B, T_month, D, H, W)

        # 6) 可选高分辨率 availability-aware 融合（逐月重复同一高分辨率特征）。
        if highres_frames and self.highres_fusion_to_embedding:
            if highres_masks is None:
                raise ValueError("提供 highres_frames 时必须同时提供 highres_masks")
            base_feat = embedding_map.reshape(Bm * M, D, H, W)
            for source, highres_frame in highres_frames.items():
                if source not in self.sensor_channels:
                    raise KeyError(f"sensor_channels 中未注册 {source!r}，无法融合高分辨率数据")
                highres_mask = highres_masks.get(source)
                if highres_mask is None:
                    raise KeyError(f"缺少 highres_mask: {source!r}")
                highres_feat = self.highres_encoders[source](
                    highres_frame, target_size=input_size
                )  # (B, D, H, W)
                highres_mask = highres_mask.to(device=highres_feat.device, dtype=highres_feat.dtype)
                if highres_mask.shape[-2:] != input_size:
                    highres_mask = F.interpolate(
                        highres_mask,
                        size=input_size,
                        mode="nearest",
                    )
                highres_feat_rep = (
                    highres_feat.unsqueeze(1).expand(-1, M, -1, -1, -1).reshape(Bm * M, D, H, W)
                )
                highres_mask_rep = (
                    highres_mask.unsqueeze(1).expand(-1, M, -1, -1, -1).reshape(Bm * M, 1, H, W)
                )
                base_feat = self.highres_fusion(base_feat, highres_feat_rep, highres_mask_rep)
            embedding_map = base_feat.view(Bm, M, D, H, W)

        # 7) vMF 瓶颈：输出位于单位球面（逐月）。
        emb_map_flat = self.bottleneck(
            embedding_map.reshape(Bm * M, D, H, W)
        )  # (B*T_month, D, H, W)
        emb_map = emb_map_flat.view(Bm, M, D, H, W)

        # 8) 每个场景/月份平均池化得到场景级嵌入并 L2 归一化。
        emb = emb_map.mean(dim=[3, 4])  # (B, T_month, D)
        emb = F.normalize(emb, p=2, dim=-1)

        # 9) 解码器重建目标模态（逐月）。
        emb_map_dec = emb_map.reshape(Bm * M, D, H, W)
        reconstructions: dict[str, torch.Tensor] = {}
        for name, decoder in self.decoders.items():
            out = decoder(emb_map_dec)  # (B*T_month, C_out, H, W)
            _, C_out, _, _ = out.shape
            reconstructions[name] = out.view(Bm, M, C_out, H, W)

        return AEFOutput(embedding_map=emb_map, embedding=emb, reconstructions=reconstructions)

    def _validate_month_range(self, timestamps: torch.Tensor, mask: torch.Tensor) -> None:
        """避免所有有效观测静默落到月度 bin 范围外。"""
        valid_obs = mask.bool()
        if not bool(valid_obs.any().item()):
            return
        month_index = self.monthly_embed._yyyymm_to_index(timestamps)
        in_range = (month_index >= 0) & (month_index < self.num_months)
        if bool((valid_obs & in_range).any().item()):
            return

        observed = timestamps[valid_obs].detach().cpu().unique().tolist()
        observed_preview = ", ".join(str(int(v)) for v in observed[:8])
        if len(observed) > 8:
            observed_preview += ", ..."
        end_month = self.ref_month + self.num_months - 1
        end_year = self.ref_year + (end_month - 1) // 12
        end_month = ((end_month - 1) % 12) + 1
        raise ValueError(
            "所有有效 timestamps 都落在 MonthlyEmbeddingModule 月度范围外: "
            f"observed=[{observed_preview}], "
            f"range={self.ref_year}-{self.ref_month:02d}..{end_year}-{end_month:02d}"
        )
