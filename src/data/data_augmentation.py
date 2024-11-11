import pandas as pd
from faker import Faker
import numpy as np
import random


def augment_data(X, y, augmentation_factor=0.3):
    """
    Augments the training dataset with synthetic data generated using Faker.

    Parameters:
    - X (pd.DataFrame): The original training features.
    - y (pd.Series): The original training target.

    Returns:
    - augmented_X (pd.DataFrame): The augmented training features.
    - augmented_y (pd.Series): The augmented training target.
    """
    # Reset indices of X and y to ensure alignment
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Combine X and y into a single DataFrame
    data = X.copy()
    data['stroke'] = y

    fake = Faker()
    num_new_samples = int(len(data) * augmentation_factor)
    synthetic_data = []

    # Generate realistic ranges based on existing data
    age_min, age_max = data['age'].min(), data['age'].max()
    glucose_min, glucose_max = data['avg_glucose_level'].min(), data['avg_glucose_level'].max()
    bmi_min, bmi_max = data['bmi'].min(), data['bmi'].max()

    for _ in range(num_new_samples):
        gender = random.choice(['Male', 'Female'])
        age = random.uniform(age_min, age_max)
        hypertension = random.choice([0, 1])
        heart_disease = random.choice([0, 1])
        ever_married = 'Yes' if age >= 18 else 'No'
        work_type = random.choice(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
        Residence_type = random.choice(['Urban', 'Rural'])
        avg_glucose_level = random.uniform(glucose_min, glucose_max)
        bmi = random.uniform(bmi_min, bmi_max)
        smoking_status = random.choice(['never smoked', 'formerly smoked', 'smokes', 'Unknown'])
        stroke = random.choice([0, 1])  # Adjust probability if needed

        synthetic_data.append({
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status,
            'stroke': stroke
        })

    synthetic_df = pd.DataFrame(synthetic_data)

    augmented_data = pd.concat([data, synthetic_df], ignore_index=True)

    # Check for NaN values in 'stroke' column and drop them if any
    if augmented_data['stroke'].isnull().any():
        augmented_data = augmented_data.dropna(subset=['stroke'])

    # Ensure 'stroke' column is of integer type
    augmented_data['stroke'] = augmented_data['stroke'].astype(int)

    # Separate features and target
    augmented_X = augmented_data.drop(columns=['stroke'])
    augmented_y = augmented_data['stroke']

    return augmented_X, augmented_y
