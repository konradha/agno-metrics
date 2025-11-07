from dataclasses import dataclass
from typing import Literal, Sequence, Optional


@dataclass
class AGNOConfig:
    transform_type: Literal[
        "linear", "nonlinear", "linear_kernelonly", "nonlinear_kernelonly"
    ] = "linear"
    use_attn: bool = True
    attention_type: Literal["cosine", "dot_product"] = "cosine"
    coord_dim: int = 2
    mlp_layers: Sequence[int] = (128, 128)
    attention_dim: int = 64


@dataclass
class ReadoutConfig:
    method: Literal["mean", "max"] = "mean"
    hidden: int = 128
    out: int = 128


@dataclass
class MetricConfig:
    type: Literal["l2", "cosine", "mahalanobis"] = "l2"
    emb_dim: int = 128


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    batch_size: int = 8
    loss_mse: float = 1.0
    loss_corr: float = 0.5
    loss_triplet: float = 0.5
    margin: float = 0.2
