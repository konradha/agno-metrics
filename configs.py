from dataclasses import dataclass


@dataclass
class EntropyConfig:
    default_c: float = 2.0


@dataclass
class CalibratorConfig:
    kind: str = "isotonic"


@dataclass
class ExperimentConfig:
    entropy: EntropyConfig = EntropyConfig()
    calibrator: CalibratorConfig = CalibratorConfig()
    seed: int = 0
