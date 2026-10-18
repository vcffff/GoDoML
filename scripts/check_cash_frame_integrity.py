"""Pixel integrity checks without exporting images to any service."""
import hashlib
import sys
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_cash_sft import read, write


def main():
    from PIL import Image, ImageStat
    rows = []
    for dataset in ['cash-lora', 'cash-diagnostic']:
        for example in read(ROOT / 'outputs' / dataset / 'examples.json'):
            hashes, sizes, deviations = [], [], []
            for path in example['images']:
                with Image.open(ROOT / path) as source:
                    image = source.convert('RGB')
                    hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
                    sizes.append(list(image.size))
                    deviations.append(max(ImageStat.Stat(image).stddev))
            stamps = example['timestamps']
            row = {'dataset': dataset, 'id': example['id'], 'frames': len(hashes),
                   'distinct_pixel_arrays': len(set(hashes)), 'sizes': sizes,
                   'minimum_channel_stddev': min(deviations),
                   'strictly_chronological': all(a < b for a, b in pairwise(stamps)),
                   'visual_semantics_checked': False}
            assert row['strictly_chronological'] and row['distinct_pixel_arrays'] > 1
            assert row['minimum_channel_stddev'] > 0
            rows.append(row)
    write(ROOT / 'outputs/cash-diagnostic/pixel_integrity.json', rows)
    print(f'{len(rows)} sequences: chronological, nonconstant pixels, multiple distinct images.')


if __name__ == '__main__':
    main()
