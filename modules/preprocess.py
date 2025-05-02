import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest
import yaml
import logging
from tqdm import tqdm
from typing import Tuple, Union, Optional, List
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("preprocessing.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_config() -> dict:
    """Load and validate configuration."""
    try:
        with open("config/defaults.yaml", "r") as f:
            config = yaml.safe_load(f)
            # Set defaults if not specified
            config.setdefault("parallel", {"n_jobs": -1, "backend": "loky"})
            return config
    except Exception as e:
        logger.error(f"Config error: {e}", exc_info=True)
        raise

def _get_transformer_pipelines(config: dict) -> Tuple[Pipeline, Pipeline]:
    """Create numeric and categorical pipelines that work with both string and dict encoder configs."""
    try:
        # Numeric pipeline
        numeric_steps = [
            ("imputer", SimpleImputer(strategy=config["preprocessing"]["numeric"]["imputer"]))
        ]
        if config["preprocessing"]["numeric"]["scaler"] != "none":
            scaler = (
                StandardScaler() 
                if config["preprocessing"]["numeric"]["scaler"] == "standard" 
                else MinMaxScaler()
            )
            numeric_steps.append(("scaler", scaler))

        # Categorical pipeline - handles both string and dict encoder configs
        encoder_type = config["preprocessing"]["categorical"]["encoder"]
        
        # If encoder is specified as string (like in your YAML)
        if isinstance(encoder_type, str):
            if encoder_type == "onehot":
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            elif encoder_type == "label":
                encoder = LabelEncoder()
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")
        # If encoder is specified as dict (alternative format)
        else:
            encoder = OneHotEncoder(
                handle_unknown=encoder_type.get("handle_unknown", "ignore"),
                sparse_output=False
            )

        categorical_steps = [
            ("imputer", SimpleImputer(
                strategy=config["preprocessing"]["categorical"]["imputer"],
                fill_value="missing" 
                if config["preprocessing"]["categorical"]["imputer"] == "constant" 
                else None
            )),
            ("encoder", encoder)
        ]

        return Pipeline(numeric_steps), Pipeline(categorical_steps)
    
    except KeyError as e:
        logger.error(f"Missing config key: {e}")
        raise
    except Exception as e:
        logger.error(f"Pipeline creation failed: {e}")
        raise

def preprocess_data(
    file_path: str,
    target_column: Optional[str] = None,
    config_path: str = "config/defaults.yaml"
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Preprocessing pipeline that matches your YAML structure.
    
    Returns:
        Tuple (X_processed, y) if target_column provided, else X_processed
    """
    try:
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        
        # Load config
        config = load_config()
        
        # Separate features and target
        if target_column:
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            y = data[target_column]
            X = data.drop(columns=[target_column])
            
            # Encode target labels (if categorical)
            le = LabelEncoder()
            y = le.fit_transform(y)  # Convert labels to numerical
        else:
            X = data
            y = None
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        
        # Build pipelines
        numeric_pipe, categorical_pipe = _get_transformer_pipelines(config)
        
        # Combined preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_cols),
                ("cat", categorical_pipe, categorical_cols)
            ],
            remainder="drop"
        )
        
        # Apply transformations
        with tqdm(total=1, desc="Preprocessing") as pbar:
            X_processed = preprocessor.fit_transform(X)
            pbar.update(1)
        
        logger.info(f"Preprocessing complete. Final shape: {X_processed.shape}")
        return (X_processed, y) if target_column else X_processed
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        raise

# if __name__ == "__main__":
#     # Example usage
#     X, y = preprocess_data("data/titanic.csv", target_column="Survived")
#     print(f"Processed data shape: {X.shape}, Target shape: {y.shape if y is not None else 'None'}")
