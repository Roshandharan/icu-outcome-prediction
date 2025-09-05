\"\"\"
src package
------------

This package contains modules for the ICU Outcome Prediction project:
- preprocess.py   : Text and data preprocessing functions
- baseline.py     : Baseline ML model implementations
- embeddings.py   : Functions for clinical embeddings
- models.py       : PyTorch/LSTM/Transformer models
- train.py        : Training pipelines
- evaluate.py     : Evaluation metrics and utilities

Usage:
------
from src.preprocess import preprocess_text
from src.models import build_lstm_model
\"\"\"

from .preprocess import *
from .baseline import *
from .embeddings import *
from .models import *
from .train import *
from .evaluate import *

__all__ = [
    "preprocess",
    "baseline",
    "embeddings",
    "models",
    "train",
    "evaluate"
]

__version__ = "1.0.0"
