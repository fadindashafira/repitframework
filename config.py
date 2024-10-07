from dataclasses import dataclass
from pathlib import Path



@dataclass
class BaseConfig:
    root_dir : str = Path(__file__).parent.resolve()
    dataloader_dir: str = Path(root_dir, "DataLoader")
    logs_dir: str = Path(root_dir, "logs")
    metrics_dir: str = Path(root_dir, "metrics")
    modelselector_dir: str = Path(root_dir, "ModelSelector")
    openfoam_dir: str = Path(root_dir, "OpenFOAM")
@dataclass
class ModelConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

if __name__ == "__main__":
    print(BaseConfig.root_dir)
    print(BaseConfig.openfoam_dir)
    BaseConfig.openfoam_dir = "/home/shilaj/ninelab"
    print(BaseConfig.openfoam_dir)
