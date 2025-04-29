import fire
from pipelines.classification import run_classification
from pipelines.regression import run_regression

def automl(data_path: str, target: str, task: str = "classification"):
    if task == "classification":
        run_classification(data_path, target)
    elif task == "regression":
        run_regression(data_path, target)
    else:
        raise ValueError(f"Unsupported task type: {task}")

if __name__ == "__main__":
    fire.Fire(automl)
