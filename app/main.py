import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.src.database import RedisCache
from app.src.models.classify_document import classify_document
from app.src.utils.image import bytes_to_cv2
from app.config import KNOWN_DOCUMENT_TYPES


app = FastAPI()


origins_regex = r"http://localhost:\d+"

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=origins_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=200)
async def health_check():
    return {"message": "I'm fine!"}


class PredictRequest(BaseModel):
    image_id: str


class ClassifyDocumentResult(BaseModel):
    class_name: str


class PredictResponse(BaseModel):
    result: ClassifyDocumentResult
    predict_time: float


@app.post("/predictor/classify-document", status_code=200, response_model=PredictResponse)
async def classify_document_api(predict_request: PredictRequest):
    tic = time.time()

    # get img bytes from redis
    img_key = predict_request.image_id
    img_bytes = RedisCache.get_file(img_key)
    img = bytes_to_cv2(img_bytes)

    # infer document
    classes = KNOWN_DOCUMENT_TYPES
    index = classify_document(img, 'classify-document')
    class_name = classes[index]

    toc = time.time() - tic

    # build response
    res = {
        "result": {
            "class_name": class_name
        },
        "predict_time": toc
    }

    return res


if __name__ == "__main__":
    import uvicorn
    import random
    uvicorn.run('main:app', host="0.0.0.0", port=random.randint(5500, 6000), reload=True)
