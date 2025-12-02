"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional


class HousePredictionRequest(BaseModel):
    """Request model for house price prediction"""
    bedrooms: float = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms")
    house_size: float = Field(..., gt=0, description="House size in square feet")
    city: str = Field(..., min_length=1, description="City name")
    acre_lot: Optional[float] = Field(None, ge=0, description="Lot size in acres")
    zip_code: Optional[str] = Field(None, description="Zip code")
    state: str = Field(default="California", description="State name")

    class Config:
        schema_extra = {
            "example": {
                "bedrooms": 3,
                "bathrooms": 2,
                "house_size": 1500,
                "city": "Los Angeles",
                "acre_lot": 0.25,
                "zip_code": "90001",
                "state": "California"
            }
        }


class HousePredictionResponse(BaseModel):
    """Response model for house price prediction"""
    predicted_price: float = Field(..., description="Predicted house price in dollars")
    model_version: str = Field(..., description="Model version used for prediction")
    model_type: str = Field(..., description="Type of model (sklearn or pytorch)")
    confidence_interval_lower: Optional[float] = Field(None, description="Lower bound of confidence interval")
    confidence_interval_upper: Optional[float] = Field(None, description="Upper bound of confidence interval")

    class Config:
        schema_extra = {
            "example": {
                "predicted_price": 450000.0,
                "model_version": "1.0",
                "model_type": "sklearn",
                "confidence_interval_lower": 400000.0,
                "confidence_interval_upper": 500000.0
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    model_type: str


class ModelInfoResponse(BaseModel):
    """Model metadata response"""
    model_type: str
    model_version: str
    training_date: str
    performance_metrics: dict
    features: list


