"""
FastAPI application for house price prediction
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from datetime import datetime
from typing import Optional

from .models import (
    HousePredictionRequest,
    HousePredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from .model_loader import ModelLoader
from .predictor import HousePricePredictor
from .logger import setup_logger

# Set up logging
logger = setup_logger("housing_api")

# Initialize FastAPI app
app = FastAPI(
    title="California Housing Price Prediction API",
    description="API for predicting house prices in California",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model loader and predictor
model_loader = ModelLoader()
predictor = None


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global predictor
    try:
        logger.info("Loading models on startup...")
        model_loader.load_all()
        predictor = HousePricePredictor(
            model_loader,
            training_data_path="california_housing.csv"
        )
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - Time: {process_time:.3f}s")
    
    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded(),
        model_version=model_loader.model_version,
        model_type=model_loader.model_type or "none"
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model metadata"""
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Load performance metrics from training log if available
    performance_metrics = {
        "model_type": model_loader.model_type,
        "version": model_loader.model_version
    }
    
    # Get feature list from preprocessor
    features = []
    if model_loader.preprocessor:
        # Extract feature names from preprocessor
        try:
            numerical_features = model_loader.preprocessor.named_transformers_['num'].feature_names_in_
            categorical_features = model_loader.preprocessor.named_transformers_['cat'].feature_names_in_
            features = list(numerical_features) + list(categorical_features)
        except:
            features = ["features_from_preprocessor"]
    
    return ModelInfoResponse(
        model_type=model_loader.model_type,
        model_version=model_loader.model_version,
        training_date=datetime.now().strftime("%Y-%m-%d"),  # Should be from model metadata
        performance_metrics=performance_metrics,
        features=features
    )


@app.post("/predict", response_model=HousePredictionResponse)
async def predict_price(request: HousePredictionRequest):
    """
    Predict house price based on input features
    
    Parameters:
    -----------
    request : HousePredictionRequest
        Input features for prediction
    
    Returns:
    --------
    HousePredictionResponse
        Predicted price and metadata
    """
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Log prediction request
        logger.info(
            f"Prediction request - Bedrooms: {request.bedrooms}, "
            f"Bathrooms: {request.bathrooms}, House Size: {request.house_size}, "
            f"City: {request.city}"
        )
        
        # Prepare input data
        input_data = {
            'bedrooms': request.bedrooms,
            'bathrooms': request.bathrooms,
            'house_size': request.house_size,
            'city': request.city,
            'acre_lot': request.acre_lot,
            'zip_code': request.zip_code,
            'state': request.state
        }
        
        # Make prediction
        start_time = time.time()
        predicted_price = predictor.predict(input_data)
        prediction_time = time.time() - start_time
        
        # Log prediction result
        logger.info(
            f"Prediction completed - Price: ${predicted_price:,.0f}, "
            f"Time: {prediction_time:.3f}s"
        )
        
        # Calculate confidence interval (simple approximation: ±15%)
        confidence_interval_lower = predicted_price * 0.85
        confidence_interval_upper = predicted_price * 1.15
        
        return HousePredictionResponse(
            predicted_price=predicted_price,
            model_version=model_loader.model_version,
            model_type=model_loader.model_type,
            confidence_interval_lower=confidence_interval_lower,
            confidence_interval_upper=confidence_interval_upper
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


