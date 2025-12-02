"""
Model loading utilities
"""
import os
import joblib
import torch
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage trained models"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.preprocessor = None
        self.sklearn_model = None
        self.pytorch_model = None
        self.pytorch_model_config = None
        self.model_type = None
        self.model_version = "1.0"
        
    def load_preprocessor(self) -> None:
        """Load the preprocessor"""
        preprocessor_path = os.path.join(self.model_dir, "preprocessor.pkl")
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info("Preprocessor loaded successfully")
        else:
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    
    def load_sklearn_model(self) -> None:
        """Load the best sklearn model"""
        model_path = os.path.join(self.model_dir, "best_sklearn_model.pkl")
        if os.path.exists(model_path):
            self.sklearn_model = joblib.load(model_path)
            self.model_type = "sklearn"
            logger.info("Sklearn model loaded successfully")
        else:
            logger.warning(f"Sklearn model not found at {model_path}")
    
    def load_pytorch_model(self) -> None:
        """Load the PyTorch model"""
        model_path = os.path.join(self.model_dir, "pytorch_model_full.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.pytorch_model_config = checkpoint
            self.model_type = "pytorch"
            logger.info("PyTorch model config loaded successfully")
        else:
            logger.warning(f"PyTorch model not found at {model_path}")
    
    def load_all(self) -> None:
        """Load all models and preprocessor"""
        self.load_preprocessor()
        self.load_sklearn_model()
        self.load_pytorch_model()
        
        if self.model_type is None:
            raise ValueError("No model loaded. Please ensure at least one model file exists.")
    
    def get_model(self):
        """Get the active model"""
        if self.model_type == "sklearn":
            return self.sklearn_model
        elif self.model_type == "pytorch":
            return self.pytorch_model_config
        else:
            raise ValueError("No model loaded")
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return self.preprocessor is not None and self.model_type is not None


