from pydantic import BaseModel
from typing import List, Optional


# =========================
# Prediction schemas
# =========================

class PredictionItem(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    binary_label: str
    binary_confidence: float
    binary_confidence_level: str

    predicted_label: str
    confidence: float
    confidence_level: str

    top_predictions: List[PredictionItem]
    disease_info: dict
    sample_image_url: Optional[str] = None
    is_healthy: bool


# =========================
# Chat schemas
# =========================

class ChatRequest(BaseModel):
    message: str
    language: str = "en"
    disease_context: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    language: str
    disease_context: Optional[str] = None
    sample_image_url: Optional[str] = None


# =========================
# Auth schemas (keep them, even if unused)
# =========================

class UserRegister(BaseModel):
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"