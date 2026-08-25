from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


class ConfigError(ValueError):
    """配置解析或合同校验失败。"""


class _UniqueKeyLoader(yaml.SafeLoader):
    """拒绝 YAML 重复键，避免后值静默覆盖前值。"""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConfigError(f"YAML 包含重复字段: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{section} 必须是 mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{section} 的字段名必须是字符串")
    return value


def _strict(
    value: Any,
    section: str,
    *,
    allowed: set[str],
    required: set[str] = frozenset(),
) -> dict[str, Any]:
    raw = _mapping(value, section)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"{section} 包含未知字段: {', '.join(unknown)}")
    missing = sorted(required - set(raw))
    if missing:
        raise ConfigError(f"{section} 缺少必填字段: {', '.join(missing)}")
    return raw


def _reject_base(value: Any, section: str = "config") -> None:
    if isinstance(value, dict):
        if "_base_" in value:
            raise ConfigError(f"{section} 禁止使用 _base_")
        for key, child in value.items():
            _reject_base(child, f"{section}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_base(child, f"{section}[{index}]")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field_name} 必须是正整数")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{field_name} 必须是非负整数")
    return value


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} 必须是布尔值")
    return value


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field_name} 必须是非空字符串")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ConfigError(f"{field_name} 必须是列表")
    return [_string(item, field_name) for item in value]


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} 必须是数值")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{field_name} 必须是有限数值")
    return result


def _non_negative_float(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if result < 0:
        raise ConfigError(f"{field_name} 必须非负")
    return result


def _positive_float(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if result <= 0:
        raise ConfigError(f"{field_name} 必须大于 0")
    return result


@dataclass(frozen=True)
class PathsConfig:
    data_root: Path
    output_root: Path
    artifact_root: Path


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int = 42
    output_dir: Path | None = None


@dataclass(frozen=True)
class InputSourceConfig:
    channels: int
    role: Literal["temporal", "highres"]


@dataclass(frozen=True)
class TargetHeadConfig:
    source: str
    loss_type: Literal["continuous", "categorical"]
    channels: int
    weight: float

    @property
    def training_loss_type(self) -> str:
        return "l1" if self.loss_type == "continuous" else "ce"


@dataclass(frozen=True)
class STPConfig:
    space_dim: int = 512
    time_dim: int = 256
    precision_dim: int = 128
    precision_scale: int = 2
    num_blocks: int = 6
    num_heads: int = 8
    temporal_fusion: Literal["concat", "gated_sum"] = "concat"
    time_attention_mode: Literal["full", "none"] = "full"
    highres_fusion_to_embedding: bool = True


@dataclass(frozen=True)
class ModelConfig:
    embed_dim: int
    input_sources: dict[str, InputSourceConfig]
    target_heads: dict[str, TargetHeadConfig]
    stem_dim: int = 32
    num_months: int = 17
    ref_year: int = 2025
    ref_month: int = 1
    stp: STPConfig = field(default_factory=STPConfig)

    @property
    def sensor_channels(self) -> dict[str, int]:
        return {name: source.channels for name, source in self.input_sources.items()}

    @property
    def source_roles(self) -> dict[str, str]:
        return {name: source.role for name, source in self.input_sources.items()}

    @property
    def decoder_specs(self) -> dict[str, tuple[str, int]]:
        return {name: (head.loss_type, head.channels) for name, head in self.target_heads.items()}

    @property
    def loss_specs(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "loss_type": head.training_loss_type,
                "channels": head.channels,
                "weight": head.weight,
            }
            for name, head in self.target_heads.items()
        }


@dataclass(frozen=True)
class InputMaskingConfig:
    enabled: bool = False
    drop_availability_masks: bool = True
    modality_dropout_probs: dict[str, float] = field(default_factory=dict)
    month_dropout_prob: float = 0.0
    max_months_per_sample: int = 1
    spatial_block_prob: float = 0.0
    spatial_block_size: int = 16
    spatial_block_ratio: float = 0.15


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    gradient_accumulation_steps: int
    save_every: int
    amp: bool = True
    gradient_checkpointing: bool = True
    uniformity_weight: float = 0.0
    uniformity_warmup_epochs: int = 0
    uniformity_temperature: float = 2.0
    semantic_probe_weight: float = 0.0
    semantic_probe_warmup_epochs: int = 0
    semantic_probe_tasks: list[str] = field(default_factory=list)
    semantic_probe_task_weights: dict[str, float] = field(default_factory=dict)
    semantic_probe_pos_weight: float = 1.0
    semantic_probe_pos_weights: dict[str, float] = field(default_factory=dict)
    semantic_probe_hidden_dim: int = 64
    semantic_probe_hard_negative_ratio: float = 0.0
    semantic_probe_hard_negative_weight: float = 0.0
    semantic_probe_hard_negative_warmup_epochs: int = 0
    input_masking: InputMaskingConfig = field(default_factory=InputMaskingConfig)


@dataclass(frozen=True)
class RegionDatasetConfig:
    region: str
    manifest_path: Path
    statistics_dir: Path
    patch_grid_path: Path
    source_map: dict[str, str]
    supervised_label_roots: dict[str, Path]
    sampling_weight: float = 1.0


@dataclass(frozen=True)
class DataConfig:
    months: list[str]
    datasets: list[RegionDatasetConfig]
    batch_size: int = 4
    num_workers: int = 8
    patch_size: int = 128

    def dataset_for_region(self, region: str) -> RegionDatasetConfig:
        matches = [dataset for dataset in self.datasets if dataset.region == region]
        if not matches:
            raise ConfigError(f"未配置区域: {region}")
        return matches[0]


@dataclass(frozen=True)
class Config:
    schema_version: str
    paths: PathsConfig
    experiment: ExperimentConfig
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(child) for key, child in value.items()}
            if isinstance(value, list):
                return [convert(child) for child in value]
            return value

        return convert(asdict(self))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        config_path = Path(path)
        try:
            raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        except ConfigError:
            raise
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"无法读取配置 {config_path}: {exc}") from exc

        top = _strict(
            raw,
            "config",
            allowed={"schema_version", "paths", "experiment", "model", "training", "data"},
            required={"schema_version", "paths", "experiment", "model", "training", "data"},
        )
        _reject_base(top)
        schema_version = _string(top["schema_version"], "schema_version")
        if schema_version != "1":
            raise ConfigError(f"不支持的 schema_version: {schema_version!r}")

        paths = _parse_paths(top["paths"])
        experiment = _parse_experiment(top["experiment"])
        model = _parse_model(top["model"])
        training = _parse_training(top["training"], model)
        data = _parse_data(top["data"], model)
        _validate_cross_contracts(model, training, data)
        return cls(schema_version, paths, experiment, model, training, data)


def _parse_paths(value: Any) -> PathsConfig:
    raw = _strict(
        value,
        "paths",
        allowed={"data_root", "output_root", "artifact_root"},
        required={"data_root", "output_root", "artifact_root"},
    )
    return PathsConfig(
        *(
            Path(_string(raw[name], f"paths.{name}"))
            for name in ("data_root", "output_root", "artifact_root")
        )
    )


def _parse_experiment(value: Any) -> ExperimentConfig:
    raw = _strict(
        value,
        "experiment",
        allowed={
            "name",
            "seed",
            "output_dir",
        },
        required={"name"},
    )
    return ExperimentConfig(
        name=_string(raw["name"], "experiment.name"),
        seed=_non_negative_int(raw.get("seed", 42), "experiment.seed"),
        output_dir=(
            Path(_string(raw["output_dir"], "experiment.output_dir"))
            if raw.get("output_dir") is not None
            else None
        ),
    )


def _parse_input_source(name: str, value: Any) -> InputSourceConfig:
    raw = _strict(
        value,
        f"model.input_sources.{name}",
        allowed={"channels", "role"},
        required={"channels", "role"},
    )
    role = _string(raw["role"], f"model.input_sources.{name}.role")
    if role not in {"temporal", "highres"}:
        raise ConfigError(f"model.input_sources.{name}.role 非法: {role!r}")
    return InputSourceConfig(
        channels=_positive_int(raw["channels"], f"model.input_sources.{name}.channels"),
        role=role,  # type: ignore[arg-type]
    )


def _parse_target_head(name: str, value: Any) -> TargetHeadConfig:
    raw = _strict(
        value,
        f"model.target_heads.{name}",
        allowed={"source", "loss_type", "channels", "weight"},
        required={"source", "loss_type", "channels", "weight"},
    )
    loss_type = _string(raw["loss_type"], f"model.target_heads.{name}.loss_type")
    if loss_type not in {"continuous", "categorical"}:
        raise ConfigError(f"model.target_heads.{name}.loss_type 非法: {loss_type!r}")
    weight = _non_negative_float(raw["weight"], f"model.target_heads.{name}.weight")
    return TargetHeadConfig(
        source=_string(raw["source"], f"model.target_heads.{name}.source"),
        loss_type=loss_type,  # type: ignore[arg-type]
        channels=_positive_int(raw["channels"], f"model.target_heads.{name}.channels"),
        weight=weight,
    )


def _parse_stp(value: Any) -> STPConfig:
    raw = _strict(
        value,
        "model.stp",
        allowed={
            "space_dim",
            "time_dim",
            "precision_dim",
            "precision_scale",
            "num_blocks",
            "num_heads",
            "temporal_fusion",
            "time_attention_mode",
            "highres_fusion_to_embedding",
        },
    )
    temporal_fusion = _string(raw.get("temporal_fusion", "concat"), "model.stp.temporal_fusion")
    time_attention_mode = _string(
        raw.get("time_attention_mode", "full"), "model.stp.time_attention_mode"
    )
    if temporal_fusion not in {"concat", "gated_sum"}:
        raise ConfigError(f"model.stp.temporal_fusion 非法: {temporal_fusion!r}")
    if time_attention_mode not in {"full", "none"}:
        raise ConfigError(f"model.stp.time_attention_mode 非法: {time_attention_mode!r}")
    return STPConfig(
        space_dim=_positive_int(raw.get("space_dim", 512), "model.stp.space_dim"),
        time_dim=_positive_int(raw.get("time_dim", 256), "model.stp.time_dim"),
        precision_dim=_positive_int(raw.get("precision_dim", 128), "model.stp.precision_dim"),
        precision_scale=_positive_int(raw.get("precision_scale", 2), "model.stp.precision_scale"),
        num_blocks=_positive_int(raw.get("num_blocks", 6), "model.stp.num_blocks"),
        num_heads=_positive_int(raw.get("num_heads", 8), "model.stp.num_heads"),
        temporal_fusion=temporal_fusion,  # type: ignore[arg-type]
        time_attention_mode=time_attention_mode,  # type: ignore[arg-type]
        highres_fusion_to_embedding=_boolean(
            raw.get("highres_fusion_to_embedding", True),
            "model.stp.highres_fusion_to_embedding",
        ),
    )


def _parse_model(value: Any) -> ModelConfig:
    raw = _strict(
        value,
        "model",
        allowed={
            "embed_dim",
            "stem_dim",
            "num_months",
            "ref_year",
            "ref_month",
            "input_sources",
            "target_heads",
            "stp",
        },
        required={"embed_dim", "num_months", "input_sources", "target_heads"},
    )
    input_raw = _mapping(raw["input_sources"], "model.input_sources")
    target_raw = _mapping(raw["target_heads"], "model.target_heads")
    if any(not name for name in input_raw):
        raise ConfigError("model.input_sources 名称必须是非空字符串")
    if any(not name for name in target_raw):
        raise ConfigError("model.target_heads 名称必须是非空字符串")
    input_sources = {name: _parse_input_source(name, source) for name, source in input_raw.items()}
    target_heads = {name: _parse_target_head(name, head) for name, head in target_raw.items()}
    if not input_sources or not any(source.role == "temporal" for source in input_sources.values()):
        raise ConfigError("model.input_sources 至少需要一个 temporal source")
    return ModelConfig(
        embed_dim=_positive_int(raw["embed_dim"], "model.embed_dim"),
        input_sources=input_sources,
        target_heads=target_heads,
        stem_dim=_positive_int(raw.get("stem_dim", 32), "model.stem_dim"),
        num_months=_positive_int(raw["num_months"], "model.num_months"),
        ref_year=_positive_int(raw.get("ref_year", 2025), "model.ref_year"),
        ref_month=_positive_int(raw.get("ref_month", 1), "model.ref_month"),
        stp=_parse_stp(raw.get("stp", {})),
    )


def _parse_masking(value: Any, model: ModelConfig) -> InputMaskingConfig:
    raw = _strict(
        value,
        "training.input_masking",
        allowed={
            "enabled",
            "drop_availability_masks",
            "modality_dropout_probs",
            "month_dropout_prob",
            "max_months_per_sample",
            "spatial_block_prob",
            "spatial_block_size",
            "spatial_block_ratio",
        },
    )
    dropout_raw = _mapping(
        raw.get("modality_dropout_probs", {}),
        "training.input_masking.modality_dropout_probs",
    )
    dropout = {
        _string(name, "training.input_masking.modality_dropout_probs source"): _finite_float(
            probability,
            f"training.input_masking.modality_dropout_probs.{name}",
        )
        for name, probability in dropout_raw.items()
    }
    unknown = sorted(set(dropout) - set(model.input_sources))
    if unknown:
        raise ConfigError(f"input masking 引用了未知 source: {', '.join(unknown)}")
    ratios = {
        "month_dropout_prob": _finite_float(
            raw.get("month_dropout_prob", 0.0), "training.input_masking.month_dropout_prob"
        ),
        "spatial_block_prob": _finite_float(
            raw.get("spatial_block_prob", 0.0), "training.input_masking.spatial_block_prob"
        ),
        "spatial_block_ratio": _finite_float(
            raw.get("spatial_block_ratio", 0.15), "training.input_masking.spatial_block_ratio"
        ),
        **{f"modality_dropout_probs.{name}": probability for name, probability in dropout.items()},
    }
    for name, ratio in ratios.items():
        if not 0.0 <= ratio <= 1.0:
            raise ConfigError(f"training.input_masking.{name} 必须位于 [0, 1]")
    return InputMaskingConfig(
        enabled=_boolean(raw.get("enabled", False), "training.input_masking.enabled"),
        drop_availability_masks=_boolean(
            raw.get("drop_availability_masks", True),
            "training.input_masking.drop_availability_masks",
        ),
        modality_dropout_probs=dropout,
        month_dropout_prob=ratios["month_dropout_prob"],
        max_months_per_sample=_positive_int(
            raw.get("max_months_per_sample", 1),
            "training.input_masking.max_months_per_sample",
        ),
        spatial_block_prob=ratios["spatial_block_prob"],
        spatial_block_size=_positive_int(
            raw.get("spatial_block_size", 16),
            "training.input_masking.spatial_block_size",
        ),
        spatial_block_ratio=ratios["spatial_block_ratio"],
    )


def _parse_training(value: Any, model: ModelConfig) -> TrainingConfig:
    fields = {
        "epochs",
        "lr",
        "weight_decay",
        "warmup_epochs",
        "gradient_accumulation_steps",
        "save_every",
        "amp",
        "gradient_checkpointing",
        "uniformity_weight",
        "uniformity_warmup_epochs",
        "uniformity_temperature",
        "semantic_probe_weight",
        "semantic_probe_warmup_epochs",
        "semantic_probe_tasks",
        "semantic_probe_task_weights",
        "semantic_probe_pos_weight",
        "semantic_probe_pos_weights",
        "semantic_probe_hidden_dim",
        "semantic_probe_hard_negative_ratio",
        "semantic_probe_hard_negative_weight",
        "semantic_probe_hard_negative_warmup_epochs",
        "input_masking",
    }
    required = {
        "epochs",
        "lr",
        "weight_decay",
        "warmup_epochs",
        "gradient_accumulation_steps",
        "save_every",
    }
    raw = _strict(value, "training", allowed=fields, required=required)
    tasks_raw = raw.get("semantic_probe_tasks", [])
    if not isinstance(tasks_raw, list):
        raise ConfigError("training.semantic_probe_tasks 必须是字符串列表")
    tasks = [
        _string(task, f"training.semantic_probe_tasks[{index}]")
        for index, task in enumerate(tasks_raw)
    ]
    if len(set(tasks)) != len(tasks):
        raise ConfigError("training.semantic_probe_tasks 不得重复")
    task_weights = {
        _string(name, "training.semantic_probe_task_weights task"): _non_negative_float(
            weight, f"training.semantic_probe_task_weights.{name}"
        )
        for name, weight in _mapping(
            raw.get("semantic_probe_task_weights", {}),
            "training.semantic_probe_task_weights",
        ).items()
    }
    pos_weights = {
        _string(name, "training.semantic_probe_pos_weights task"): _positive_float(
            weight, f"training.semantic_probe_pos_weights.{name}"
        )
        for name, weight in _mapping(
            raw.get("semantic_probe_pos_weights", {}),
            "training.semantic_probe_pos_weights",
        ).items()
    }
    return TrainingConfig(
        epochs=_positive_int(raw["epochs"], "training.epochs"),
        lr=_positive_float(raw["lr"], "training.lr"),
        weight_decay=_non_negative_float(raw["weight_decay"], "training.weight_decay"),
        warmup_epochs=_non_negative_int(raw["warmup_epochs"], "training.warmup_epochs"),
        gradient_accumulation_steps=_positive_int(
            raw["gradient_accumulation_steps"],
            "training.gradient_accumulation_steps",
        ),
        save_every=_positive_int(raw["save_every"], "training.save_every"),
        amp=_boolean(raw.get("amp", True), "training.amp"),
        gradient_checkpointing=_boolean(
            raw.get("gradient_checkpointing", True), "training.gradient_checkpointing"
        ),
        uniformity_weight=_non_negative_float(
            raw.get("uniformity_weight", 0.0), "training.uniformity_weight"
        ),
        uniformity_warmup_epochs=_non_negative_int(
            raw.get("uniformity_warmup_epochs", 0),
            "training.uniformity_warmup_epochs",
        ),
        uniformity_temperature=_positive_float(
            raw.get("uniformity_temperature", 2.0), "training.uniformity_temperature"
        ),
        semantic_probe_weight=_non_negative_float(
            raw.get("semantic_probe_weight", 0.0), "training.semantic_probe_weight"
        ),
        semantic_probe_warmup_epochs=_non_negative_int(
            raw.get("semantic_probe_warmup_epochs", 0),
            "training.semantic_probe_warmup_epochs",
        ),
        semantic_probe_tasks=tasks,
        semantic_probe_task_weights=task_weights,
        semantic_probe_pos_weight=_positive_float(
            raw.get("semantic_probe_pos_weight", 1.0), "training.semantic_probe_pos_weight"
        ),
        semantic_probe_pos_weights=pos_weights,
        semantic_probe_hidden_dim=_non_negative_int(
            raw.get("semantic_probe_hidden_dim", 64), "training.semantic_probe_hidden_dim"
        ),
        semantic_probe_hard_negative_ratio=_finite_float(
            raw.get("semantic_probe_hard_negative_ratio", 0.0),
            "training.semantic_probe_hard_negative_ratio",
        ),
        semantic_probe_hard_negative_weight=_non_negative_float(
            raw.get("semantic_probe_hard_negative_weight", 0.0),
            "training.semantic_probe_hard_negative_weight",
        ),
        semantic_probe_hard_negative_warmup_epochs=_non_negative_int(
            raw.get("semantic_probe_hard_negative_warmup_epochs", 0),
            "training.semantic_probe_hard_negative_warmup_epochs",
        ),
        input_masking=_parse_masking(raw.get("input_masking", {}), model),
    )


def _parse_dataset(index: int, value: Any) -> RegionDatasetConfig:
    section = f"data.datasets[{index}]"
    raw = _strict(
        value,
        section,
        allowed={
            "region",
            "manifest_path",
            "statistics_dir",
            "patch_grid_path",
            "source_map",
            "supervised_label_roots",
            "sampling_weight",
        },
        required={
            "region",
            "manifest_path",
            "statistics_dir",
            "patch_grid_path",
            "source_map",
            "supervised_label_roots",
            "sampling_weight",
        },
    )
    source_map = {
        _string(physical, f"{section}.source_map physical source"): _string(
            canonical, f"{section}.source_map.{physical}"
        )
        for physical, canonical in _mapping(raw["source_map"], f"{section}.source_map").items()
    }
    duplicates = sorted(
        canonical
        for canonical in set(source_map.values())
        if list(source_map.values()).count(canonical) > 1
    )
    if duplicates:
        raise ConfigError(f"{section}.source_map 包含重复映射: {', '.join(duplicates)}")
    label_roots = {
        _string(name, f"{section}.supervised_label_roots task"): Path(
            _string(path, f"{section}.supervised_label_roots.{name}")
        )
        for name, path in _mapping(
            raw["supervised_label_roots"],
            f"{section}.supervised_label_roots",
        ).items()
    }
    sampling_weight = _positive_float(raw["sampling_weight"], f"{section}.sampling_weight")
    return RegionDatasetConfig(
        region=_string(raw["region"], f"{section}.region"),
        manifest_path=Path(_string(raw["manifest_path"], f"{section}.manifest_path")),
        statistics_dir=Path(_string(raw["statistics_dir"], f"{section}.statistics_dir")),
        patch_grid_path=Path(_string(raw["patch_grid_path"], f"{section}.patch_grid_path")),
        source_map=source_map,
        supervised_label_roots=label_roots,
        sampling_weight=sampling_weight,
    )


def _parse_data(value: Any, model: ModelConfig) -> DataConfig:
    raw = _strict(
        value,
        "data",
        allowed={"months", "datasets", "batch_size", "num_workers", "patch_size"},
        required={"months", "datasets"},
    )
    if not isinstance(raw["months"], list):
        raise ConfigError("data.months 必须是列表")
    months = [_string(month, f"data.months[{index}]") for index, month in enumerate(raw["months"])]
    if not months or any(not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month) for month in months):
        raise ConfigError("data.months 必须使用 YYYY-MM 且不能为空")
    if len(set(months)) != len(months) or months != sorted(months):
        raise ConfigError("data.months 包含重复或顺序冲突")
    if not isinstance(raw["datasets"], list) or not raw["datasets"]:
        raise ConfigError("data.datasets 必须是非空列表")
    datasets = [_parse_dataset(index, item) for index, item in enumerate(raw["datasets"])]
    regions = [dataset.region for dataset in datasets]
    if len(set(regions)) != len(regions):
        raise ConfigError("data.datasets 区域名称重复")
    return DataConfig(
        months=months,
        datasets=datasets,
        batch_size=_positive_int(raw.get("batch_size", 4), "data.batch_size"),
        num_workers=_non_negative_int(raw.get("num_workers", 8), "data.num_workers"),
        patch_size=_positive_int(raw.get("patch_size", 128), "data.patch_size"),
    )


def _validate_cross_contracts(
    model: ModelConfig,
    training: TrainingConfig,
    data: DataConfig,
) -> None:
    if model.num_months != len(data.months):
        raise ConfigError(
            f"月份冲突: model.num_months={model.num_months}, data.months={len(data.months)}"
        )
    first_year, first_month = (int(part) for part in data.months[0].split("-"))
    if (model.ref_year, model.ref_month) != (first_year, first_month):
        raise ConfigError("月份冲突: model.ref_year/ref_month 必须等于 data.months[0]")
    if not 1 <= model.ref_month <= 12:
        raise ConfigError("model.ref_month 必须位于 1..12")
    if training.warmup_epochs > training.epochs:
        raise ConfigError("training.warmup_epochs 不得超过 training.epochs")
    if not 0.0 <= training.semantic_probe_hard_negative_ratio <= 1.0:
        raise ConfigError("training.semantic_probe_hard_negative_ratio 必须位于 [0, 1]")
    unknown_task_weights = sorted(
        set(training.semantic_probe_task_weights) - set(training.semantic_probe_tasks)
    )
    unknown_pos_weights = sorted(
        set(training.semantic_probe_pos_weights) - set(training.semantic_probe_tasks)
    )
    if unknown_task_weights:
        raise ConfigError(
            "training.semantic_probe_task_weights 引用了未配置任务: "
            + ", ".join(unknown_task_weights)
        )
    if unknown_pos_weights:
        raise ConfigError(
            "training.semantic_probe_pos_weights 引用了未配置任务: "
            + ", ".join(unknown_pos_weights)
        )

    available_slots = set(model.input_sources) | {
        head.source for head in model.target_heads.values()
    }
    for dataset in data.datasets:
        unknown = sorted(set(dataset.source_map.values()) - available_slots)
        if unknown:
            raise ConfigError(
                f"data.datasets[{dataset.region}] source_map 引用了未知规范 source: "
                f"{', '.join(unknown)}"
            )

    for name, head in model.target_heads.items():
        source = model.input_sources.get(head.source)
        if source is not None and source.channels != head.channels:
            raise ConfigError(
                f"通道冲突: target {name!r} 为 {head.channels}，"
                f"source {head.source!r} 为 {source.channels}"
            )

    configured_labels = {
        name for dataset in data.datasets for name in dataset.supervised_label_roots
    }
    missing_labels = sorted(set(training.semantic_probe_tasks) - configured_labels)
    if missing_labels:
        raise ConfigError(
            f"semantic probe 缺少 supervised_label_roots: {', '.join(missing_labels)}"
        )


# V2 is intentionally a separate contract.  Existing V1 parsing remains available for
# frozen release artifacts, while every V2 entrypoint accepts only this schema.


@dataclass(frozen=True)
class V2PathsConfig:
    data_root: Path
    source_root: Path
    grid_package: Path
    product_roots: dict[str, Path] = field(default_factory=dict)
    auxiliary_roots: dict[str, Path] = field(default_factory=dict)
    legacy_unverified_roots: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class NetworkPolicyConfig:
    allow_remote_metadata: bool = False
    allow_remote_pixels: bool = False
    missing_local_observation: Literal["mask"] = "mask"


@dataclass(frozen=True)
class V2ProductConfig:
    role: Literal["dense", "highres", "target"]
    bands: tuple[str, ...]
    native_gsd_m: tuple[float, ...]
    stored_gsd_m: float
    dtype: str
    time_precision: Literal["exact", "day", "month", "static"]
    already_resampled: bool
    qa_available: bool

    def to_product_spec(self, product_id: str):
        from xuannv_embedding.data.contracts import ProductSpec

        return ProductSpec(
            product_id=product_id,
            role=self.role,
            bands=self.bands,
            native_gsd_m=self.native_gsd_m,
            stored_gsd_m=self.stored_gsd_m,
            dtype=self.dtype,
            time_precision=self.time_precision,
            already_resampled=self.already_resampled,
            qa_available=self.qa_available,
        )


@dataclass(frozen=True)
class V2TemporalConfig:
    mode: Literal["within_period", "causal_window", "centered_window"]
    dense_lookback_days: int
    highres_structure_days: int
    highres_appearance_days: int
    highres_structure_max_observations: int
    highres_appearance_max_observations: int


@dataclass(frozen=True)
class V2ModelConfig:
    embedding_dim: int
    stem_dim: int
    spatial_dim: int
    temporal_dim: int
    precision_dim: int
    num_blocks: int
    num_heads: int
    gradient_checkpointing: bool


@dataclass(frozen=True)
class V2TrainingConfig:
    epochs: int
    lr: float
    weight_decay: float
    batch_size: int
    gradient_accumulation_steps: int
    amp: bool
    reconstruction_weights: dict[str, float] = field(default_factory=dict)
    uniformity_weight: float = 0.0
    uniformity_warmup_epochs: int = 0
    uniformity_temperature: float = 2.0
    semantic_probe_weight: float = 0.0
    semantic_probe_warmup_epochs: int = 0
    semantic_probe_tasks: tuple[str, ...] = ()
    semantic_probe_hidden_dim: int = 64
    highres_detail_weight: float = 0.0


@dataclass(frozen=True)
class ValidationProfileConfig:
    records: int
    steps: int
    batch_size: int
    spatial_size: int = 128
    model_profile: Literal["production", "mini"] = "production"
    resume_steps: int = 1
    overfit_steps: int = 0


@dataclass(frozen=True)
class V2Config:
    schema_version: str
    paths: V2PathsConfig
    network_policy: NetworkPolicyConfig
    products: dict[str, V2ProductConfig]
    temporal: V2TemporalConfig
    model: V2ModelConfig
    training: V2TrainingConfig
    validation_profiles: dict[str, ValidationProfileConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "V2Config":
        config_path = Path(path)
        try:
            raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        except ConfigError:
            raise
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"无法读取 V2 配置 {config_path}: {exc}") from exc
        top = _strict(
            raw,
            "config",
            allowed={
                "schema_version",
                "paths",
                "network_policy",
                "products",
                "temporal",
                "model",
                "training",
                "validation_profiles",
            },
            required={
                "schema_version",
                "paths",
                "network_policy",
                "products",
                "temporal",
                "model",
                "training",
                "validation_profiles",
            },
        )
        _reject_base(top)
        schema_version = _string(top["schema_version"], "schema_version")
        if schema_version != "2":
            raise ConfigError("V2 runtime 要求 schema_version 为 '2'")
        return cls(
            schema_version=schema_version,
            paths=_parse_v2_paths(top["paths"]),
            network_policy=_parse_network_policy(top["network_policy"]),
            products=_parse_v2_products(top["products"]),
            temporal=_parse_v2_temporal(top["temporal"]),
            model=_parse_v2_model(top["model"]),
            training=_parse_v2_training(top["training"]),
            validation_profiles=_parse_validation_profiles(top["validation_profiles"]),
        )


def _parse_v2_paths(value: Any) -> V2PathsConfig:
    raw = _strict(
        value,
        "paths",
        allowed={
            "data_root",
            "source_root",
            "grid_package",
            "product_roots",
            "auxiliary_roots",
            "legacy_unverified_roots",
        },
        required={"data_root", "source_root", "grid_package"},
    )
    return V2PathsConfig(
        data_root=Path(_string(raw["data_root"], "paths.data_root")),
        source_root=Path(_string(raw["source_root"], "paths.source_root")),
        grid_package=Path(_string(raw["grid_package"], "paths.grid_package")),
        product_roots=_parse_path_mapping(raw.get("product_roots", {}), "paths.product_roots"),
        auxiliary_roots=_parse_path_mapping(
            raw.get("auxiliary_roots", {}), "paths.auxiliary_roots"
        ),
        legacy_unverified_roots=_parse_path_mapping(
            raw.get("legacy_unverified_roots", {}), "paths.legacy_unverified_roots"
        ),
    )


def _parse_path_mapping(value: Any, section: str) -> dict[str, Path]:
    raw = _mapping(value, section)
    return {
        _string(name, f"{section}.key"): Path(_string(path, f"{section}.{name}"))
        for name, path in raw.items()
    }


def _parse_network_policy(value: Any) -> NetworkPolicyConfig:
    raw = _strict(
        value,
        "network_policy",
        allowed={"allow_remote_metadata", "allow_remote_pixels", "missing_local_observation"},
        required={"allow_remote_metadata", "allow_remote_pixels", "missing_local_observation"},
    )
    allow_metadata = _boolean(raw["allow_remote_metadata"], "network_policy.allow_remote_metadata")
    allow_pixels = _boolean(raw["allow_remote_pixels"], "network_policy.allow_remote_pixels")
    if allow_pixels:
        raise ConfigError("全国本地 V2 配置禁止远程像元下载")
    missing = _string(raw["missing_local_observation"], "network_policy.missing_local_observation")
    if missing != "mask":
        raise ConfigError("network_policy.missing_local_observation 仅支持 'mask'")
    return NetworkPolicyConfig(allow_metadata, allow_pixels, "mask")


def _parse_v2_products(value: Any) -> dict[str, V2ProductConfig]:
    products = _mapping(value, "products")
    if not products:
        raise ConfigError("products 必须是非空 mapping")
    parsed: dict[str, V2ProductConfig] = {}
    for product_id, value in products.items():
        section = f"products.{product_id}"
        raw = _strict(
            value,
            section,
            allowed={
                "role",
                "bands",
                "native_gsd_m",
                "stored_gsd_m",
                "dtype",
                "time_precision",
                "already_resampled",
                "qa_available",
            },
            required={
                "role",
                "bands",
                "native_gsd_m",
                "stored_gsd_m",
                "dtype",
                "time_precision",
                "already_resampled",
                "qa_available",
            },
        )
        bands_raw = raw["bands"]
        gsd_raw = raw["native_gsd_m"]
        if not isinstance(bands_raw, list) or not bands_raw:
            raise ConfigError(f"{section}.bands 必须是非空列表")
        if not isinstance(gsd_raw, list) or not gsd_raw:
            raise ConfigError(f"{section}.native_gsd_m 必须是非空列表")
        bands = tuple(_string(band, f"{section}.bands") for band in bands_raw)
        native_gsd = tuple(_positive_float(gsd, f"{section}.native_gsd_m") for gsd in gsd_raw)
        if len(bands) != len(native_gsd):
            raise ConfigError(f"{section}.bands 与 native_gsd_m 数量必须一致")
        role = _string(raw["role"], f"{section}.role")
        if role not in {"dense", "highres", "target"}:
            raise ConfigError(f"{section}.role 非法: {role!r}")
        precision = _string(raw["time_precision"], f"{section}.time_precision")
        if precision not in {"exact", "day", "month", "static"}:
            raise ConfigError(f"{section}.time_precision 非法: {precision!r}")
        product = V2ProductConfig(
            role=role,
            bands=bands,
            native_gsd_m=native_gsd,
            stored_gsd_m=_positive_float(raw["stored_gsd_m"], f"{section}.stored_gsd_m"),
            dtype=_string(raw["dtype"], f"{section}.dtype"),
            time_precision=precision,
            already_resampled=_boolean(raw["already_resampled"], f"{section}.already_resampled"),
            qa_available=_boolean(raw["qa_available"], f"{section}.qa_available"),
        )
        product.to_product_spec(product_id)
        parsed[product_id] = product
    return parsed


def _parse_v2_temporal(value: Any) -> V2TemporalConfig:
    raw = _strict(
        value,
        "temporal",
        allowed={
            "mode",
            "dense_lookback_days",
            "highres_structure_days",
            "highres_appearance_days",
            "highres_structure_max_observations",
            "highres_appearance_max_observations",
        },
        required={
            "mode",
            "dense_lookback_days",
            "highres_structure_days",
            "highres_appearance_days",
            "highres_structure_max_observations",
            "highres_appearance_max_observations",
        },
    )
    mode = _string(raw["mode"], "temporal.mode")
    if mode not in {"within_period", "causal_window", "centered_window"}:
        raise ConfigError(f"temporal.mode 非法: {mode!r}")
    structure_days = _positive_int(raw["highres_structure_days"], "temporal.highres_structure_days")
    appearance_days = _positive_int(
        raw["highres_appearance_days"], "temporal.highres_appearance_days"
    )
    if appearance_days > structure_days:
        raise ConfigError("highres appearance 窗口不得超过 structure 窗口")
    return V2TemporalConfig(
        mode=mode,
        dense_lookback_days=_positive_int(
            raw["dense_lookback_days"], "temporal.dense_lookback_days"
        ),
        highres_structure_days=structure_days,
        highres_appearance_days=appearance_days,
        highres_structure_max_observations=_positive_int(
            raw["highres_structure_max_observations"],
            "temporal.highres_structure_max_observations",
        ),
        highres_appearance_max_observations=_positive_int(
            raw["highres_appearance_max_observations"],
            "temporal.highres_appearance_max_observations",
        ),
    )


def _parse_v2_model(value: Any) -> V2ModelConfig:
    raw = _strict(
        value,
        "model",
        allowed={
            "embedding_dim",
            "stem_dim",
            "spatial_dim",
            "temporal_dim",
            "precision_dim",
            "num_blocks",
            "num_heads",
            "gradient_checkpointing",
        },
        required={
            "embedding_dim",
            "stem_dim",
            "spatial_dim",
            "temporal_dim",
            "precision_dim",
            "num_blocks",
            "num_heads",
            "gradient_checkpointing",
        },
    )
    config = V2ModelConfig(
        embedding_dim=_positive_int(raw["embedding_dim"], "model.embedding_dim"),
        stem_dim=_positive_int(raw["stem_dim"], "model.stem_dim"),
        spatial_dim=_positive_int(raw["spatial_dim"], "model.spatial_dim"),
        temporal_dim=_positive_int(raw["temporal_dim"], "model.temporal_dim"),
        precision_dim=_positive_int(raw["precision_dim"], "model.precision_dim"),
        num_blocks=_positive_int(raw["num_blocks"], "model.num_blocks"),
        num_heads=_positive_int(raw["num_heads"], "model.num_heads"),
        gradient_checkpointing=_boolean(
            raw["gradient_checkpointing"], "model.gradient_checkpointing"
        ),
    )
    for name, dim in (
        ("spatial_dim", config.spatial_dim),
        ("temporal_dim", config.temporal_dim),
        ("precision_dim", config.precision_dim),
    ):
        if dim % config.num_heads != 0:
            raise ConfigError(f"model.{name} 必须能被 num_heads 整除")
    return config


def _parse_v2_training(value: Any) -> V2TrainingConfig:
    raw = _strict(
        value,
        "training",
        allowed={
            "epochs",
            "lr",
            "weight_decay",
            "batch_size",
            "gradient_accumulation_steps",
            "amp",
            "reconstruction_weights",
            "uniformity_weight",
            "uniformity_warmup_epochs",
            "uniformity_temperature",
            "semantic_probe_weight",
            "semantic_probe_warmup_epochs",
            "semantic_probe_tasks",
            "semantic_probe_hidden_dim",
            "highres_detail_weight",
        },
        required={
            "epochs",
            "lr",
            "weight_decay",
            "batch_size",
            "gradient_accumulation_steps",
            "amp",
        },
    )
    return V2TrainingConfig(
        epochs=_positive_int(raw["epochs"], "training.epochs"),
        lr=_positive_float(raw["lr"], "training.lr"),
        weight_decay=_non_negative_float(raw["weight_decay"], "training.weight_decay"),
        batch_size=_positive_int(raw["batch_size"], "training.batch_size"),
        gradient_accumulation_steps=_positive_int(
            raw["gradient_accumulation_steps"], "training.gradient_accumulation_steps"
        ),
        amp=_boolean(raw["amp"], "training.amp"),
        reconstruction_weights={
            _string(name, "training.reconstruction_weights.key"): _non_negative_float(
                weight, f"training.reconstruction_weights.{name}"
            )
            for name, weight in _mapping(
                raw.get("reconstruction_weights", {}), "training.reconstruction_weights"
            ).items()
        },
        uniformity_weight=_non_negative_float(
            raw.get("uniformity_weight", 0.0), "training.uniformity_weight"
        ),
        uniformity_warmup_epochs=_non_negative_int(
            raw.get("uniformity_warmup_epochs", 0), "training.uniformity_warmup_epochs"
        ),
        uniformity_temperature=_positive_float(
            raw.get("uniformity_temperature", 2.0), "training.uniformity_temperature"
        ),
        semantic_probe_weight=_non_negative_float(
            raw.get("semantic_probe_weight", 0.0), "training.semantic_probe_weight"
        ),
        semantic_probe_warmup_epochs=_non_negative_int(
            raw.get("semantic_probe_warmup_epochs", 0),
            "training.semantic_probe_warmup_epochs",
        ),
        semantic_probe_tasks=tuple(
            _string(task, "training.semantic_probe_tasks")
            for task in _string_list(
                raw.get("semantic_probe_tasks", []), "training.semantic_probe_tasks"
            )
        ),
        semantic_probe_hidden_dim=_non_negative_int(
            raw.get("semantic_probe_hidden_dim", 64), "training.semantic_probe_hidden_dim"
        ),
        highres_detail_weight=_non_negative_float(
            raw.get("highres_detail_weight", 0.0), "training.highres_detail_weight"
        ),
    )


def _parse_validation_profiles(value: Any) -> dict[str, ValidationProfileConfig]:
    profiles = _mapping(value, "validation_profiles")
    if not profiles:
        raise ConfigError("validation_profiles 必须是非空 mapping")
    result: dict[str, ValidationProfileConfig] = {}
    for name, profile_value in profiles.items():
        section = f"validation_profiles.{name}"
        raw = _strict(
            profile_value,
            section,
            allowed={
                "records",
                "steps",
                "batch_size",
                "spatial_size",
                "model_profile",
                "resume_steps",
                "overfit_steps",
            },
            required={"records", "steps", "batch_size"},
        )
        model_profile = _string(raw.get("model_profile", "production"), f"{section}.model_profile")
        if model_profile not in {"production", "mini"}:
            raise ConfigError(f"{section}.model_profile 非法: {model_profile!r}")
        result[name] = ValidationProfileConfig(
            records=_positive_int(raw["records"], f"{section}.records"),
            steps=_positive_int(raw["steps"], f"{section}.steps"),
            batch_size=_positive_int(raw["batch_size"], f"{section}.batch_size"),
            spatial_size=_positive_int(raw.get("spatial_size", 128), f"{section}.spatial_size"),
            model_profile=model_profile,
            resume_steps=_positive_int(raw.get("resume_steps", 1), f"{section}.resume_steps"),
            overfit_steps=_non_negative_int(
                raw.get("overfit_steps", 0), f"{section}.overfit_steps"
            ),
        )
    return result
