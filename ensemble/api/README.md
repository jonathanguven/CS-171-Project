# Housing Price Prediction API

FastAPI endpoint for predicting California house prices.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure trained models are in the `models/` directory:
   - `preprocessor.pkl`
   - `best_sklearn_model.pkl` (or `pytorch_model_full.pth`)

3. Run the API:
```bash
python -m api.app
# or
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Endpoints

### POST `/predict`
Predict house price based on input features.

**Request Body:**
```json
{
  "bedrooms": 3,
  "bathrooms": 2,
  "house_size": 1500,
  "city": "Los Angeles",
  "acre_lot": 0.25,
  "zip_code": "90001",
  "state": "California"
}
```

**Response:**
```json
{
  "predicted_price": 450000.0,
  "model_version": "1.0",
  "model_type": "sklearn",
  "confidence_interval_lower": 382500.0,
  "confidence_interval_upper": 517500.0
}
```

### GET `/health`
Health check endpoint.

### GET `/model/info`
Get model metadata and information.

## Logging

Logs are written to:
- Console (stdout)
- `logs/api_YYYYMMDD.log` files


