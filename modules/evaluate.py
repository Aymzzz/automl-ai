# evaluate.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config() -> dict:
    """Load metrics configuration."""
    try:
        with open("config/defaults.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def evaluate_model(model, X: np.ndarray, y: np.ndarray, task_type: str = "classification") -> dict:
    """
    Evaluate a model on given data and return performance metrics.
    
    Args:
        model: Trained model.
        X (np.ndarray): Features.
        y (np.ndarray): Ground truth labels.
        task_type (str): "classification" or "regression".
    
    Returns:
        dict: Dictionary of metric_name -> value.
    """
    try:
        config = load_config()
        metrics_config = config["metrics"][task_type]
        primary_metric = metrics_config["primary"]
        secondary_metrics = metrics_config["secondary"]
        
        y_pred = model.predict(X)

        results = {}

        if task_type == "classification":
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]  # Probability for positive class
            else:
                y_prob = None

            for metric in [primary_metric] + secondary_metrics:
                if metric == "accuracy":
                    results["accuracy"] = accuracy_score(y, y_pred)
                elif metric == "f1_weighted":
                    results["f1_weighted"] = f1_score(y, y_pred, average="weighted")
                elif metric == "roc_auc" and y_prob is not None:
                    results["roc_auc"] = roc_auc_score(y, y_prob)
                else:
                    logger.warning(f"Metric {metric} not implemented for classification.")

        elif task_type == "regression":
            for metric in [primary_metric] + secondary_metrics:
                if metric == "neg_root_mean_squared_error":
                    results["rmse"] = np.sqrt(mean_squared_error(y, y_pred))
                elif metric == "r2":
                    results["r2"] = r2_score(y, y_pred)
                elif metric == "neg_mean_absolute_error":
                    results["mae"] = mean_absolute_error(y, y_pred)
                else:
                    logger.warning(f"Metric {metric} not implemented for regression.")
        
        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise