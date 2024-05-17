import os

from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional, Union

# ## get root path ## #
this_file = Path(__file__)
this_studio_idx = [
    i for i, j in enumerate(this_file.parents) if j.name.endswith("this_studio")
][0]
this_studio = this_file.parents[this_studio_idx]

@dataclass
class Config:
    cache_dir: str = os.path.join(this_studio, "data")
    log_dir: str = os.path.join(this_studio, "logs")
    ckpt_dir: str = os.path.join(this_studio, "checkpoints")
    prof_dir: str = os.path.join(this_studio, "logs", "profiler")
    perf_dir: str = os.path.join(this_studio, "logs", "perf")
    seed: int = 42


@dataclass
class ModuleConfig:
    model_name: str = "textattack/bert-base-uncased-yelp-polarity" # change this to use a different pretrained checkpoint and tokenizer
    learning_rate: float = 5e-05
    finetuned: str = "checkpoints/textattack-bert-base-uncased-yelp-polarity_1xTesla-T4_LR5e-05_BS16_2024-01-30T20:39:34.235506.ckpt"

@dataclass
class DataModuleConfig:
    dataset_name: str = "imdb" # change this to use different dataset
    num_classes: int = 2
    batch_size: int = 12
    train_split: str = "train"
    test_split: str = "test"
    train_size: float = 0.8
    stratify_by_column: str = "label"
    num_workers: int = cpu_count()

@dataclass
class TrainerConfig:
    accelerator: str = "auto" # Trainer flag
    devices: Union[int, str] = "auto"  # Trainer flag
    strategy: str = "auto"  # Trainer flag
    precision: Optional[str] = "16-mixed"  # Trainer flag
    max_epochs: int = 1  # Trainer flag