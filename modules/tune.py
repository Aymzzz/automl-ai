import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import logging
from tqdm import tqdm
from typing import Dict, Any
import yaml

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

def tune_model(model, X: np.ndarray, y: np.ndarray, task_type: str = "classification") -> Dict[str, Any]:
    """
    Optimize hyperparameters for a given model using Optuna.
    Returns: {'best_params': dict, 'best_score': float}
    """
    try:
        config = load_config()
        metric = config["metrics"][task_type]["primary"]
        
        # Define search space per model type
        def objective(trial):
            params = {}
            
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from xgboost import XGBClassifier, XGBRegressor

            if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
                params.update({
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
                })
            elif isinstance(model, (XGBClassifier, XGBRegressor)):
                params.update({
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                })
            else:  # Default space for other models
                params.update({
                    "C": trial.suggest_float("C", 0.1, 10, log=True) if hasattr(model, "C") else None,
                    "max_iter": trial.suggest_int("max_iter", 50, 200) if hasattr(model, "max_iter") else None
                })
            
            model.set_params(**{k: v for k, v in params.items() if v is not None})
            return cross_val_score(model, X, y, scoring=metric, cv=3, n_jobs=-1).mean()
        
        # Run optimization
        study = optuna.create_study(direction="maximize" if task_type == "classification" else "minimize")
        with tqdm(total=50, desc="Tuning") as pbar:
            def callback(study, trial):
                pbar.update(1)
            
            study.optimize(objective, n_trials=50, callbacks=[callback])
        
        # Flip negative scores for interpretability
        best_score = study.best_value
        if metric.startswith("neg_"):
            best_score = -best_score  # Convert to positive for RMSE/MAE

        logger.info(f"Best score: {best_score:.4f}")
        return {
            "best_params": study.best_params,
            "best_score": best_score
        }

    
    except Exception as e:
        logger.error(f"Tuning failed: {e}")
        raise

# Test the function
if __name__ == "__main__":
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    # Sample data
    X_test = np.random.rand(100, 5)
    y_test = np.random.randint(0, 2, 100)
    
    # Test with RandomForest
    print("Testing tune_model()...")
    model = RandomForestClassifier()
    result = tune_model(model, X_test, y_test)
    
    print("\nResults:")
    print(f"Best params: {result['best_params']}")
    print(f"Best score: {result['best_score']:.4f}")