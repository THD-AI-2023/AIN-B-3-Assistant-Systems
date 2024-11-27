# data/data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy import stats

def preprocess_data(data):
    """
    Preprocesses the input data by handling missing values, removing outliers,
    and resetting the index.

    Parameters:
    - data (pd.DataFrame): The raw dataset.

    Returns:
    - pd.DataFrame: The preprocessed dataset.
    """
    # Handle missing values in 'bmi' by imputing with mean
    imputer = SimpleImputer(strategy="mean")
    data["bmi"] = imputer.fit_transform(data[["bmi"]])

    # Drop unnecessary columns
    data = data.drop(columns=["id"])

    # Outlier detection and removal using Z-score method
    numerical_cols = ["age", "avg_glucose_level", "bmi"]
    z_scores = np.abs(stats.zscore(data[numerical_cols]))
    threshold = 3
    outlier_indices = np.where(z_scores > threshold)[0]
    data = data.drop(data.index[outlier_indices])

    # Reset index after removing outliers
    data.reset_index(drop=True, inplace=True)

    return data
