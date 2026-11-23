"""
MindScore - FastAPI Backend
Predicts student mental health and wellbeing scores using a trained scikit-learn pipeline.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Literal, Optional
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"
INDEX_HTML = BASE_DIR / "index.html"
STYLE_CSS = BASE_DIR / "style.css"
SCRIPT_JS = BASE_DIR / "script.js"
MODEL_JS = BASE_DIR / "model.js"

# Dataset constants and limits
DATASET_MEAN = 6.23

RANGES = {
    "Age": [18.0, 24.0],
    "Avg_Daily_Usage_Hours": [1.0, 8.8],
    "Daily_Unlocks": [62.0, 273.0],
    "Study_Hours": [0.3, 8.3],
    "Physical_Activity_Hours": [0.0, 4.1],
    "Sleep_Hours_Per_Night": [3.6, 9.9],
}

CATEGORIES = {
    "Gender": ["Female", "Male"],
    "Academic_Level": ["Graduate", "High School", "Undergraduate"],
    "Stress_Level": ["Low", "Medium", "High", "Very High"],
    "Most_Used_Platform": [
        "Facebook",
        "Instagram",
        "KakaoTalk",
        "LINE",
        "LinkedIn",
        "Snapchat",
        "TikTok",
        "Twitter",
        "VKontakte",
        "WeChat",
        "WhatsApp",
        "YouTube",
    ],
    "Purpose_Of_Use": ["Education", "Entertainment", "Networking", "News"],
    "Grouped_country": [
        "Australia",
        "Canada",
        "France",
        "Germany",
        "India",
        "Mexico",
        "Other",
        "Spain",
        "Turkey",
        "UK",
        "USA",
    ],
}

DEFAULT_VALUES = {
    "Study_Hours": 2.8,
    "Age": 21,
    "Avg_Daily_Usage_Hours": 5.0,
    "Daily_Unlocks": 170,
    "Physical_Activity_Hours": 1.7,
    "Sleep_Hours_Per_Night": 6.6,
    "Stress_Level": "Medium",
    "Gender": "Female",
    "Academic_Level": "Undergraduate",
    "Most_Used_Platform": "Instagram",
    "Purpose_Of_Use": "Entertainment",
    "Grouped_country": "India",
}

# Global reference for loaded pipeline
ml_pipeline = None


def load_model():
    """Load the trained machine learning pipeline from disk."""
    global ml_pipeline
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    ml_pipeline = joblib.load(MODEL_PATH)
    return ml_pipeline


# Attempt immediate load on module import
try:
    load_model()
except Exception as exc:
    print(f"[MindScore Backend] Initial load notice: {exc}")


def get_model():
    """Safely obtain or initialize the model pipeline."""
    global ml_pipeline
    if ml_pipeline is None:
        return load_model()
    return ml_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager to load resources on startup."""
    try:
        get_model()
        print(f"[MindScore Backend] Model ready from {MODEL_PATH}")
    except Exception as exc:
        print(f"[MindScore Backend] Error during lifespan model load: {exc}")
    yield


app = FastAPI(
    title="MindScore Wellbeing Prediction API",
    description="FastAPI backend to predict student mental health wellbeing scores from daily habits.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class StudentFeatures(BaseModel):
    Study_Hours: float = Field(..., ge=0.0, le=24.0, description="Daily study hours")
    Age: int = Field(..., ge=10, le=120, description="Student age")
    Avg_Daily_Usage_Hours: float = Field(..., ge=0.0, le=24.0, description="Daily social media usage hours")
    Daily_Unlocks: int = Field(..., ge=0, le=1000, description="Daily phone unlocks count")
    Physical_Activity_Hours: float = Field(..., ge=0.0, le=24.0, description="Daily physical activity in hours")
    Sleep_Hours_Per_Night: float = Field(..., ge=0.0, le=24.0, description="Nightly sleep in hours")
    Stress_Level: Literal["Low", "Medium", "High", "Very High"] = Field(
        ..., description="Self-reported stress level"
    )
    Gender: Literal["Female", "Male"] = Field(..., description="Gender")
    Academic_Level: Literal["High School", "Undergraduate", "Graduate"] = Field(
        ..., description="Academic level"
    )
    Most_Used_Platform: str = Field(..., description="Most used app/platform")
    Purpose_Of_Use: str = Field(..., description="Main purpose of phone use")
    Grouped_country: str = Field(..., description="Country or grouped region")

    @field_validator("Most_Used_Platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        if v not in CATEGORIES["Most_Used_Platform"]:
            raise ValueError(f"Unknown platform '{v}'. Allowed: {CATEGORIES['Most_Used_Platform']}")
        return v

    @field_validator("Purpose_Of_Use")
    @classmethod
    def validate_purpose(cls, v: str) -> str:
        if v not in CATEGORIES["Purpose_Of_Use"]:
            raise ValueError(f"Unknown purpose '{v}'. Allowed: {CATEGORIES['Purpose_Of_Use']}")
        return v

    @field_validator("Grouped_country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        if v not in CATEGORIES["Grouped_country"]:
            raise ValueError(f"Unknown country '{v}'. Allowed: {CATEGORIES['Grouped_country']}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": DEFAULT_VALUES
        }
    }


class WhatIfItem(BaseModel):
    label: str
    delta: float
    new_score: float
    direction: Literal["pos", "neg", "zero"]


class PredictionResponse(BaseModel):
    score: float = Field(..., description="Estimated wellbeing score out of 10")
    band: str = Field(..., description="Qualitative category band")
    diff_from_mean: float = Field(..., description="Difference from survey average (6.23)")
    dataset_mean: float = Field(..., description="Survey benchmark score")
    what_if: List[WhatIfItem] = Field(..., description="Counterfactual habit recommendations")
    status: str = Field("success", description="Prediction status")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    model_type: Optional[str] = None


class MetadataResponse(BaseModel):
    categories: Dict[str, List[str]]
    ranges: Dict[str, List[float]]
    dataset_mean: float
    defaults: Dict[str, object]


def get_band(score: float) -> str:
    """Map numeric wellbeing score (0-10) to qualitative band."""
    if score < 5.0:
        return "Under pressure"
    if score < 6.5:
        return "Holding steady"
    if score < 8.0:
        return "Doing well"
    return "Thriving"


def clamp_val(val: float, key: str) -> float:
    """Clamp numeric values to dataset bounding limits."""
    if key in RANGES:
        lo, hi = RANGES[key]
        return max(lo, min(hi, val))
    return val


def compute_prediction_and_whatif(features_dict: dict, pipeline) -> PredictionResponse:
    """
    Run the scikit-learn pipeline on base input and calculate counterfactual 'what-if' suggestions
    in a single efficient batch prediction.
    """
    tweaks = [
        ("Sleep one more hour", "Sleep_Hours_Per_Night", +1.0),
        ("Use social media one hour less", "Avg_Daily_Usage_Hours", -1.0),
        ("Move half an hour more a day", "Physical_Activity_Hours", +0.5),
        ("Unlock your phone 30 fewer times", "Daily_Unlocks", -30.0),
    ]

    base_row = dict(features_dict)
    # Ensure clamped within limits for model accuracy
    for k in RANGES.keys():
        if k in base_row:
            base_row[k] = clamp_val(base_row[k], k)

    rows = [base_row]
    tweak_labels = []

    for label, col, d in tweaks:
        curr_val = base_row[col]
        new_val = clamp_val(curr_val + d, col)
        if new_val != curr_val:
            cand = dict(base_row)
            cand[col] = new_val
            rows.append(cand)
            tweak_labels.append(label)

    # Single batch prediction for optimal performance
    batch_df = pd.DataFrame(rows)
    raw_preds = pipeline.predict(batch_df)

    # Bound predictions between 0 and 10
    bounded_preds = [max(0.0, min(10.0, float(p))) for p in raw_preds]
    base_score = round(bounded_preds[0], 2)

    what_if_list: List[WhatIfItem] = []
    for label, new_score in zip(tweak_labels, bounded_preds[1:]):
        delta = round(new_score - base_score, 2)
        if delta > 0.04:
            direction = "pos"
        elif delta < -0.04:
            direction = "neg"
        else:
            direction = "zero"

        what_if_list.append(
            WhatIfItem(
                label=label,
                delta=delta,
                new_score=round(new_score, 2),
                direction=direction,
            )
        )

    # Sort what-if recommendations descending by delta impact
    what_if_list.sort(key=lambda item: item.delta, reverse=True)

    diff = round(base_score - DATASET_MEAN, 2)
    band = get_band(base_score)

    return PredictionResponse(
        score=base_score,
        band=band,
        diff_from_mean=diff,
        dataset_mean=DATASET_MEAN,
        what_if=what_if_list,
        status="success",
    )


# API Endpoints
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint to verify backend status and model readiness."""
    pipeline = None
    try:
        pipeline = get_model()
    except Exception:
        pass

    is_loaded = pipeline is not None
    model_type = type(pipeline).__name__ if is_loaded else None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        version="1.0.0",
        model_type=model_type,
    )


@app.get("/api/metadata", response_model=MetadataResponse, tags=["Metadata"])
def get_metadata():
    """Return all available categories, ranges, and defaults for the UI form."""
    return MetadataResponse(
        categories=CATEGORIES,
        ranges=RANGES,
        dataset_mean=DATASET_MEAN,
        defaults=DEFAULT_VALUES,
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(student: StudentFeatures):
    """
    Accepts student lifestyle and habit features, predicts wellbeing score (0-10),
    and returns counterfactual what-if habit analyses.
    """
    global ml_pipeline
    if ml_pipeline is None:
        try:
            ml_pipeline = load_model()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model could not be loaded: {exc}",
            )

    try:
        features_dict = student.model_dump()
        result = compute_prediction_and_whatif(features_dict, ml_pipeline)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(exc)}",
        )


@app.post("/api/what-if", response_model=List[WhatIfItem], tags=["Prediction"])
def what_if_analysis(student: StudentFeatures):
    """Calculates isolated habit changes and their expected impact on the score."""
    res = predict(student)
    return res.what_if


# Static File Endpoints
@app.get("/", include_in_schema=False)
def serve_index():
    """Serve the single-page frontend application."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/style.css", include_in_schema=False)
def serve_css():
    """Serve the CSS stylesheet."""
    if not STYLE_CSS.exists():
        raise HTTPException(status_code=404, detail="style.css not found")
    return FileResponse(STYLE_CSS, media_type="text/css")


@app.get("/script.js", include_in_schema=False)
def serve_js():
    """Serve the frontend JavaScript."""
    if not SCRIPT_JS.exists():
        raise HTTPException(status_code=404, detail="script.js not found")
    return FileResponse(SCRIPT_JS, media_type="application/javascript")


@app.get("/model.js", include_in_schema=False)
def serve_model_js():
    """Serve model.js if requested."""
    if MODEL_JS.exists():
        return FileResponse(MODEL_JS, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="model.js not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
