import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import yaml
import logging

from modules.report_generator import generate_pdf_report

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


def evaluate_model(model, X, y, task_type="classification", output_dir="outputs/reports"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{model.__class__.__name__}_{timestamp}"
    report_path = os.path.join(output_dir, f"report_{base_name}.txt")

    results = {
        "model_name": model.__class__.__name__,
        "timestamp": timestamp,
        "report_path": report_path,
    }

    if task_type == "classification":
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        # Save confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_path = os.path.join(output_dir, f"conf_matrix_{base_name}.png")
        plt.savefig(cm_path)
        plt.close()

        # Text report
        with open(report_path, "w") as f:
            f.write(f"Model: {model.__class__.__name__}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Confusion Matrix saved to: {cm_path}\n")

        # Update results
        results.update({
            "accuracy": acc,
            "plot_paths": [cm_path],
        })

    elif task_type == "regression":
        y_pred = model.predict(X)
        rmse = mean_squared_error(y, y_pred, squared=False)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Save scatter plot
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=y, y=y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        scatter_path = os.path.join(output_dir, f"scatter_{base_name}.png")
        plt.savefig(scatter_path)
        plt.close()

        # Text report
        with open(report_path, "w") as f:
            f.write(f"Model: {model.__class__.__name__}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"R²: {r2:.4f}\n")
            f.write(f"Prediction Plot saved to: {scatter_path}\n")

        # Update results
        results.update({
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "plot_paths": [scatter_path],
        })

    # Generate PDF report
    pdf_path = os.path.join(output_dir, f"{task_type}_report_{base_name}.pdf")
    metrics_to_include = {
        "Model": results["model_name"],
        "Accuracy" if task_type == "classification" else "R2": results.get("accuracy", results.get("r2")),
        "Report Text": f"Evaluation report for {task_type} task using {results['model_name']}"
    }

    generate_pdf_report("Model Report", metrics_to_include, results["plot_paths"], pdf_path)
    results["pdf_report"] = pdf_path

    return results
