import pandas as pd

def load_data(filepath='data/raw/healthcare-dataset-stroke-data.csv'):
    """
    Loads the dataset from the specified filepath.

    Parameters:
    - filepath (str): Path to the CSV file containing the dataset.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        # TODO: Handle the case where the file does not exist
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        # TODO: Handle other potential exceptions
        print(f"An error occurred while loading data: {e}")
        return pd.DataFrame()
