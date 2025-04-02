import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from tqdm import tqdm
import logging
import yaml
from typing import Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config() -> dict:
    """Load configuration from defaults.yaml."""
    try:
        with open("config/defaults.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def train_models(X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[object, float]]:
    """
    Train and evaluate all classification models in config.
    Returns: {model_name: (trained_model, avg_cross_val_score)}
    """
    try:
        config = load_config()
        models_to_train = config["models"]["classification"]
        metric = config["metrics"]["classification"]["primary"]
        
        results = {}
        for model_name in tqdm(models_to_train, desc="Training models"):
            try:
                # Instantiate the model
                model = eval(model_name)()  # e.g., LogisticRegression()
                
                # Evaluate using 5-fold cross-validation
                scores = cross_val_score(
                    model, X, y, 
                    scoring=metric, 
                    cv=5, 
                    n_jobs=-1  # Use all CPU cores
                )
                avg_score = np.mean(scores)
                results[model_name] = (model, avg_score)
                
                logger.info(f"{model_name}: {avg_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        return results
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

# Test the function
if __name__ == "__main__":
    # Sample test data (replace with your actual data)
    X_test = np.random.rand(100, 5)  # 100 samples, 5 features
    y_test = np.random.randint(0, 2, 100)  # Binary classification
    
    print("Testing train_models()...")
    models = train_models(X_test, y_test)
    
    # Print results
    print("\nResults:")
    for name, (model, score) in models.items():
        print(f"{name}: {score:.4f}")