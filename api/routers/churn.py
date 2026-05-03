from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, Request

from api.schemas import ChurnRequest, ChurnResponse


router = APIRouter(tags=["churn"])


def _get_predictor(request: Request):
    predictor = getattr(request.app.state, "churn_predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Churn model is not loaded.")
    return predictor


@router.post("/predict", response_model=ChurnResponse)
def predict_churn(payload: ChurnRequest, request: Request) -> ChurnResponse:
    predictor = _get_predictor(request)
    try:
        result = predictor.predict_one(payload.model_dump())
        return ChurnResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Churn prediction failed: {exc}") from exc


@router.post("/batch-predict", response_model=List[ChurnResponse])
def batch_predict_churn(payload: List[ChurnRequest], request: Request) -> List[ChurnResponse]:
    predictor = _get_predictor(request)
    try:
        results = predictor.batch_predict([item.model_dump() for item in payload])
        return [ChurnResponse(**item) for item in results]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Batch churn prediction failed: {exc}",
        ) from exc


@router.get("/model-info")
def churn_model_info(request: Request) -> dict[str, object]:
    predictor = _get_predictor(request)
    return predictor.get_model_info()
