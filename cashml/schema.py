import base64
import io
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Rule(StrictModel):
    title: str = Field(min_length=1)
    section: str = Field(min_length=1)
    url: str


class Evidence(StrictModel):
    timestamp_sec: float = Field(ge=0, allow_inf_nan=False)
    image_base64: str

    @field_validator('image_base64')
    @classmethod
    def jpeg(cls, value):
        data = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(data)) as image:
            if image.format != 'JPEG':
                raise ValueError('Evidence must be JPEG')
            image.verify()
        return value


class Violation(StrictModel):
    rule_id: str | None = Field(default=None, min_length=1)
    type: str = Field(min_length=1)
    timecode_sec: float = Field(ge=0, allow_inf_nan=False)
    description: str = Field(min_length=1, max_length=2000)
    evidence_frames: list[Evidence] = Field(min_length=1, max_length=3)
    rule: Rule


class AnalysisResult(StrictModel):
    verdict: Literal['violation', 'no_violation']
    violations: list[Violation]
    uncertain_rule_ids: list[str] | None = None

    @model_validator(mode='after')
    def consistent(self):
        if (self.verdict == 'violation') != bool(self.violations):
            raise ValueError('Inconsistent verdict')
        if self.verdict == 'no_violation' and self.uncertain_rule_ids:
            raise ValueError('Unresolved rules cannot produce a negative verdict')
        return self


TriState = bool | Literal['unknown']


class CriterionAssessment(StrictModel):
    type_id: str
    subject_id: str = Field(min_length=1, max_length=80, pattern=r'^[a-z0-9_-]+$')
    observed: TriState
    criterion_met: TriState
    supporting_timestamps_sec: list[float] = Field(default_factory=list, max_length=16)
    supporting_frame_indices: list[int] | None = Field(default=None, max_length=16)
    explanation: str = Field(min_length=1, max_length=1500)
    applicability: dict[str, TriState]
    evidence_basis: Literal['action', 'absence', 'unknown'] = 'unknown'

    @field_validator('supporting_timestamps_sec')
    @classmethod
    def finite_timestamps(cls, values):
        import math
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError('Invalid supporting timestamp')
        return values

    @model_validator(mode='after')
    def coherent(self):
        if self.observed is False and self.criterion_met is not False:
            raise ValueError('Absent object cannot meet criterion')
        if self.criterion_met is True and self.observed is not True:
            raise ValueError('Criterion requires an observed object/action')
        if self.criterion_met is True and not (self.supporting_timestamps_sec or self.supporting_frame_indices):
            raise ValueError('Positive criterion requires supporting frames')
        if self.criterion_met is True and self.subject_id in ('none', 'unknown'):
            raise ValueError('Positive events require a distinct anonymous anchor')
        return self


class WindowResult(StrictModel):
    assessments: list[CriterionAssessment] = Field(min_length=1, max_length=16)
