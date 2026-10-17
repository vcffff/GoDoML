"""Validate a saved response locally; emit technical validation status only."""
import argparse
from pathlib import Path

from cashml.schema import AnalysisResult

parser = argparse.ArgumentParser()
parser.add_argument('response', type=Path)
args = parser.parse_args()
AnalysisResult.model_validate_json(args.response.read_text(encoding='utf-8'))
print('PASS: response schema, verdict consistency, JPEG decoding and numeric bounds')
