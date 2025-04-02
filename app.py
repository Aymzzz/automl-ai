import fire
from pipelines.classification import run_classification

def automl(data_path: str, target: str, task: str = "classification"):
    if task == "classification":
        run_classification(data_path, target)
    # Add regression later

if __name__ == "__main__":
    fire.Fire(automl)  # CLI: python app.py --data_path=... --target=...