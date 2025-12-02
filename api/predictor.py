"""
Prediction logic for house price prediction
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional
from .model_loader import ModelLoader

logger = logging.getLogger(__name__)


class MLPRegressor(nn.Module):
    """PyTorch MLP model architecture"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()


class HousePricePredictor:
    """Predict house prices using loaded models"""
    
    def __init__(self, model_loader: ModelLoader, training_data_path: str = None):
        self.model_loader = model_loader
        self.training_data = None
        if training_data_path:
            self._load_training_data(training_data_path)
    
    def _load_training_data(self, path: str) -> None:
        """Load training data for feature reference"""
        try:
            self.training_data = pd.read_csv(path)
            logger.info("Training data loaded for feature reference")
        except Exception as e:
            logger.warning(f"Could not load training data: {e}")
    
    def _prepare_features(self, input_data: Dict) -> np.ndarray:
        """Prepare and transform input features"""
        # Create input DataFrame
        df_input = pd.DataFrame({
            'bed': [input_data['bedrooms']],
            'bath': [input_data['bathrooms']],
            'house_size': [input_data['house_size']],
            'city': [input_data['city']],
            'state': [input_data.get('state', 'California')],
            'zip_code': [str(input_data.get('zip_code', 'Unknown'))]
        })
        
        # Add derived features
        df_input['total_rooms'] = df_input['bed'] + df_input['bath']
        
        # Handle acre_lot
        if input_data.get('acre_lot') is not None:
            df_input['acre_lot'] = input_data['acre_lot']
            df_input['lot_size_per_sqft'] = input_data['acre_lot'] / (input_data['house_size'] / 43560)
        else:
            # Use median from training data if available
            if self.training_data is not None:
                df_input['acre_lot'] = self.training_data['acre_lot'].median()
                df_input['lot_size_per_sqft'] = self.training_data['lot_size_per_sqft'].median()
            else:
                df_input['acre_lot'] = 0.25  # Default
                df_input['lot_size_per_sqft'] = 0.0
        
        # Add date features (use median values if training data available)
        if self.training_data is not None and 'days_since_sale' in self.training_data.columns:
            df_input['days_since_sale'] = self.training_data['days_since_sale'].median()
            df_input['year_sold'] = self.training_data['year_sold'].median()
            df_input['month_sold'] = self.training_data['month_sold'].median()
        
        # Handle unknown city
        if self.training_data is not None:
            if input_data['city'] not in self.training_data['city'].unique():
                df_input['city'] = 'Other'
        
        # Get feature columns (should match training)
        numerical_cols = ['bed', 'bath', 'acre_lot', 'house_size', 'total_rooms', 'lot_size_per_sqft']
        if self.training_data is not None and 'days_since_sale' in self.training_data.columns:
            numerical_cols.extend(['days_since_sale', 'year_sold', 'month_sold'])
        
        categorical_cols = ['city', 'state', 'zip_code']
        
        # Select features
        feature_cols = numerical_cols.copy()
        input_features = df_input[feature_cols + categorical_cols]
        
        # Transform using preprocessor
        input_processed = self.model_loader.preprocessor.transform(input_features)
        
        return input_processed
    
    def predict_sklearn(self, input_data: Dict) -> float:
        """Predict using sklearn model"""
        input_processed = self._prepare_features(input_data)
        prediction = self.model_loader.sklearn_model.predict(input_processed)[0]
        return float(prediction)
    
    def predict_pytorch(self, input_data: Dict) -> float:
        """Predict using PyTorch model"""
        input_processed = self._prepare_features(input_data)
        
        # Load model
        config = self.model_loader.pytorch_model_config
        model = MLPRegressor(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            dropout_rate=config['dropout_rate']
        )
        model.load_state_dict(config['model_state_dict'])
        model.eval()
        
        # Predict
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_processed)
            prediction = model(input_tensor).item()
        
        return float(prediction)
    
    def predict(self, input_data: Dict, model_type: Optional[str] = None) -> float:
        """
        Make prediction using the specified model type
        
        Parameters:
        -----------
        input_data : Dict
            Input features
        model_type : str, optional
            Model type to use ('sklearn' or 'pytorch'). If None, uses default.
        
        Returns:
        --------
        float
            Predicted price
        """
        if model_type is None:
            model_type = self.model_loader.model_type
        
        if model_type == "sklearn":
            return self.predict_sklearn(input_data)
        elif model_type == "pytorch":
            return self.predict_pytorch(input_data)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


