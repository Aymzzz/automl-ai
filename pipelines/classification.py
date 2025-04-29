import os
import sys
from typing import Any, Dict
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm

# Fix import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.preprocess import preprocess_data
from modules.train import train_models
from modules.tune import tune_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def run_classification(data_path: str, target_column: str) -> Dict[str, Any]:
    """
    End-to-end classification pipeline:
    1. Preprocess data
    2. Train models
    3. Tune best model
    4. Save artifacts
    """
    try:
        # Create output directories
        Path("outputs/models").mkdir(parents=True, exist_ok=True)
        Path("outputs/reports").mkdir(parents=True, exist_ok=True)
        
        # 1. Preprocessing
        logger.info("Starting preprocessing...")
        X, y = preprocess_data(data_path, target_column)
        logger.info(f"Preprocessed data shape: {X.shape}")
        
        # 2. Model Training
        logger.info("Training models...")
        models = train_models(X, y)
        best_model_name, (model, score) = max(models.items(), key=lambda x: x[1][1])
        logger.info(f"Best model before tuning: {best_model_name} (score: {score:.4f})")
        
        # 3. Hyperparameter Tuning
        logger.info(f"Tuning {best_model_name}...")
        tuning_results = tune_model(model, X, y)
        
        # Update model with best parameters
        model.set_params(**tuning_results["best_params"])
        model.fit(X, y)  # Final training on full data
        
        # 4. Save artifacts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"outputs/models/{best_model_name}_{timestamp}.pkl"
        joblib.dump(model, model_path)
        
        # Prepare results
        results = {
            "best_model": best_model_name,
            "best_score": tuning_results["best_score"],
            "best_params": tuning_results["best_params"],
            "model_path": model_path,
            "feature_count": X.shape[1]
        }
        
        logger.info(f"Pipeline completed. Results:\n{results}")
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

# Test the pipeline
if __name__ == "__main__":
    print("Testing classification pipeline...")
    results = run_classification(
        data_path="data/titanic.csv",
        target_column="Survived"
    )
    
    print("\nFinal Results:")
    for key, value in results.items():
        print(f"{key}: {value}")