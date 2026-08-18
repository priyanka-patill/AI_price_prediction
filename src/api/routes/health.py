from fastapi import APIRouter
from pydantic import BaseModel
import os

router = APIRouter(prefix="/api", tags=["Health"])

class HealthResponse(BaseModel):
    status: str
    phase: str
    models_available: bool

@router.get("/health", response_model=HealthResponse, summary="Get System Health & Artifact Availability")
def get_health():
    models_exist = os.path.exists("models/lightgbm_7d.pkl") and os.path.exists("models/lightgbm_15d.pkl")
    return HealthResponse(
        status="healthy",
        phase="4",
        models_available=models_exist
    )
