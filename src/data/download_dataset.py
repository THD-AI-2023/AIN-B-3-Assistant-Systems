# import os
# from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(dataset_slug, download_path='data/raw'):
    """
    Downloads the specified Kaggle dataset to the designated path.

    Parameters:
    - dataset_slug (str): The dataset identifier in 'owner/dataset-name' format.
    - download_path (str): The directory where the dataset will be downloaded.
    """
    # TODO: Implement dataset downloading using Kaggle API
    pass

if __name__ == "__main__":
    dataset_slug = "fedesoriano/stroke-prediction-dataset"
    download_dataset(dataset_slug)
