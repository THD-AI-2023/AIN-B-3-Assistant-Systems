import pandas as pd
from sklearn.impute import SimpleImputer


def preprocess_data(data):
    """
    Preprocesses the input data by handling missing values, encoding categorical variables,
    and normalizing numerical features.

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

    # No need to encode or scale here; it will be handled in the pipeline
    return data
