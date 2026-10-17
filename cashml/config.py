import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    mode: str = 'mock'
    model_dir: Path = Path(r'C:\Users\Magzhan\cash-ml\models\Qwen3-VL-2B-Instruct')
    videos_dir: Path = ROOT / 'data/videos'
    labels_dir: Path = ROOT / 'data/open_labels'
    rules_file: Path = ROOT / 'rules.json'
    temp_dir: Path = ROOT / 'tmp'
    audit_dir: Path = ROOT / 'results/assessments'
    max_upload_mb: int = 512
    max_duration_sec: float = 150
    timeout_sec: float = 285
    window_sec: float = 12
    stride_sec: float = 10
    frames_per_window: int = 4
    refine_frames: int = 0
    max_side: int = 448
    max_new_tokens: int = 512
    merge_gap_sec: float = 4

    @classmethod
    def from_env(cls):
        values = {}
        defaults = cls()
        for name in cls.__dataclass_fields__:
            value = os.getenv('ML_' + name.upper())
            if value is not None:
                values[name] = type(getattr(defaults, name))(value)
        result = cls(**values)
        if result.mode not in ('mock', 'local'):
            raise ValueError('ML_MODE must be mock or local')
        for name in ('max_upload_mb', 'max_duration_sec', 'timeout_sec', 'window_sec',
                     'stride_sec', 'frames_per_window', 'max_side', 'max_new_tokens'):
            if getattr(result, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if (result.stride_sec > result.window_sec or result.frames_per_window > 16
                or not 0 <= result.refine_frames <= 16):
            raise ValueError('Invalid window sampling configuration')
        return result
