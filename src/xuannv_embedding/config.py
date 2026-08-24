from __future__ import annotations

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
    eval_every: int
    amp: bool = True
    gradient_checkpointing: bool = True
    log_every: int = 50
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
        schema_version = str(top["schema_version"])
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
    return PathsConfig(*(Path(raw[name]) for name in ("data_root", "output_root", "artifact_root")))


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
        name=str(raw["name"]),
        seed=int(raw.get("seed", 42)),
        output_dir=Path(raw["output_dir"]) if raw.get("output_dir") else None,
    )


def _parse_input_source(name: str, value: Any) -> InputSourceConfig:
    raw = _strict(
        value,
        f"model.input_sources.{name}",
        allowed={"channels", "role"},
        required={"channels", "role"},
    )
    role = str(raw["role"])
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
    loss_type = str(raw["loss_type"])
    if loss_type not in {"continuous", "categorical"}:
        raise ConfigError(f"model.target_heads.{name}.loss_type 非法: {loss_type!r}")
    weight = float(raw["weight"])
    if weight < 0:
        raise ConfigError(f"model.target_heads.{name}.weight 必须非负")
    return TargetHeadConfig(
        source=str(raw["source"]),
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
    temporal_fusion = str(raw.get("temporal_fusion", "concat"))
    time_attention_mode = str(raw.get("time_attention_mode", "full"))
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
        highres_fusion_to_embedding=bool(raw.get("highres_fusion_to_embedding", True)),
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
    dropout = {
        str(name): float(probability)
        for name, probability in _mapping(
            raw.get("modality_dropout_probs", {}),
            "training.input_masking.modality_dropout_probs",
        ).items()
    }
    unknown = sorted(set(dropout) - set(model.input_sources))
    if unknown:
        raise ConfigError(f"input masking 引用了未知 source: {', '.join(unknown)}")
    ratios = {
        "month_dropout_prob": float(raw.get("month_dropout_prob", 0.0)),
        "spatial_block_prob": float(raw.get("spatial_block_prob", 0.0)),
        "spatial_block_ratio": float(raw.get("spatial_block_ratio", 0.15)),
        **{f"modality_dropout_probs.{name}": probability for name, probability in dropout.items()},
    }
    for name, ratio in ratios.items():
        if not 0.0 <= ratio <= 1.0:
            raise ConfigError(f"training.input_masking.{name} 必须位于 [0, 1]")
    return InputMaskingConfig(
        enabled=bool(raw.get("enabled", False)),
        drop_availability_masks=bool(raw.get("drop_availability_masks", True)),
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
        "eval_every",
        "amp",
        "gradient_checkpointing",
        "log_every",
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
        "eval_every",
    }
    raw = _strict(value, "training", allowed=fields, required=required)
    tasks = [str(task) for task in raw.get("semantic_probe_tasks", [])]
    task_weights = {
        str(name): float(weight)
        for name, weight in _mapping(
            raw.get("semantic_probe_task_weights", {}),
            "training.semantic_probe_task_weights",
        ).items()
    }
    pos_weights = {
        str(name): float(weight)
        for name, weight in _mapping(
            raw.get("semantic_probe_pos_weights", {}),
            "training.semantic_probe_pos_weights",
        ).items()
    }
    return TrainingConfig(
        epochs=_positive_int(raw["epochs"], "training.epochs"),
        lr=float(raw["lr"]),
        weight_decay=float(raw["weight_decay"]),
        warmup_epochs=_non_negative_int(raw["warmup_epochs"], "training.warmup_epochs"),
        gradient_accumulation_steps=_positive_int(
            raw["gradient_accumulation_steps"],
            "training.gradient_accumulation_steps",
        ),
        save_every=_positive_int(raw["save_every"], "training.save_every"),
        eval_every=_positive_int(raw["eval_every"], "training.eval_every"),
        amp=bool(raw.get("amp", True)),
        gradient_checkpointing=bool(raw.get("gradient_checkpointing", True)),
        log_every=_non_negative_int(raw.get("log_every", 50), "training.log_every"),
        uniformity_weight=float(raw.get("uniformity_weight", 0.0)),
        uniformity_warmup_epochs=_non_negative_int(
            raw.get("uniformity_warmup_epochs", 0),
            "training.uniformity_warmup_epochs",
        ),
        uniformity_temperature=float(raw.get("uniformity_temperature", 2.0)),
        semantic_probe_weight=float(raw.get("semantic_probe_weight", 0.0)),
        semantic_probe_warmup_epochs=_non_negative_int(
            raw.get("semantic_probe_warmup_epochs", 0),
            "training.semantic_probe_warmup_epochs",
        ),
        semantic_probe_tasks=tasks,
        semantic_probe_task_weights=task_weights,
        semantic_probe_pos_weight=float(raw.get("semantic_probe_pos_weight", 1.0)),
        semantic_probe_pos_weights=pos_weights,
        semantic_probe_hidden_dim=int(raw.get("semantic_probe_hidden_dim", 64)),
        semantic_probe_hard_negative_ratio=float(
            raw.get("semantic_probe_hard_negative_ratio", 0.0)
        ),
        semantic_probe_hard_negative_weight=float(
            raw.get("semantic_probe_hard_negative_weight", 0.0)
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
        str(physical): str(canonical)
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
        str(name): Path(path)
        for name, path in _mapping(
            raw["supervised_label_roots"],
            f"{section}.supervised_label_roots",
        ).items()
    }
    sampling_weight = float(raw["sampling_weight"])
    if sampling_weight <= 0:
        raise ConfigError(f"{section}.sampling_weight 必须大于 0")
    return RegionDatasetConfig(
        region=str(raw["region"]),
        manifest_path=Path(raw["manifest_path"]),
        statistics_dir=Path(raw["statistics_dir"]),
        patch_grid_path=Path(raw["patch_grid_path"]),
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
    months = [str(month) for month in raw["months"]]
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
