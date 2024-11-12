import pandas as pd
import os


def load_data(filepath=None):
    """
    Loads the dataset from the specified filepath.

    Parameters:
    - filepath (str): Path to the CSV file containing the dataset.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    if filepath is None:
        filepath = os.path.join("data", "raw", "healthcare-dataset-stroke-data.csv")
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return pd.DataFrame()
