import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from tqdm import tqdm
from modules.evaluate import evaluate_model
import logging
import yaml
from typing import Dict, Tuple, Any

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

def train_models(X: np.ndarray, y: np.ndarray, task_type: str = "classification") -> Dict[str, Tuple[Any, float]]:
    """
    Train multiple models and return the trained models along with their cross-validation scores.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from xgboost import XGBClassifier, XGBRegressor
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    models = {}
    
    if task_type == "classification":
        candidate_models = {
            "RandomForestClassifier": RandomForestClassifier(),
            "XGBClassifier": XGBClassifier()
        }
        scoring = "accuracy"
    
    elif task_type == "regression":
        candidate_models = {
            "RandomForestRegressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor()
        }
        scoring = "r2"
    
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    for name, model in candidate_models.items():
        logger.info(f"Training {name}...")
        score = cross_val_score(model, X, y, scoring=scoring, cv=3, n_jobs=-1).mean()
        models[name] = (model, score)
    
    return models


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