from fastapi import File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Union, Optional


class PredictRequest(BaseModel):
    image_id: str


class ClassifyDocumentResult(BaseModel):
    class_name: Union[str, int]
    prob: Optional[float]


class PredictResponse(BaseModel):
    result: ClassifyDocumentResult
    predict_time: float


# classfiy document
class ClassfifyDocumentRequest(PredictRequest):
    model_name: str = "classify-document"
    classes: List = []


class ClassfifyOrientationRequest(BaseModel):
    image_key: str
    model_name: str = 'classify-orientation-paddle'


class Coordinate(BaseModel):
    x: Union[str, float]
    y: Union[str, float]


class ClassfifyResnet18Request(PredictRequest):
    model_name: str  # no default model
    ROI: List[Coordinate] = Field(..., min_items=2, max_items=2)
    classes: List = []
