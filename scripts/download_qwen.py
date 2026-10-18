"""Download only public model artifacts; never upload local data."""
from pathlib import Path

from huggingface_hub import snapshot_download

if __name__ == '__main__':
    destination = Path(__file__).resolve().parents[1] / 'models/Qwen3-VL-2B-Instruct'
    print(snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir=str(destination),
                            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'],
                            max_workers=2))
