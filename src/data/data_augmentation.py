import pandas as pd
from faker import Faker
import numpy as np
import random


def augment_data(data, augmentation_factor=0.3):
    """
    Augments the dataset with synthetic data generated using Faker.

    Parameters:
    - data (pd.DataFrame): The original dataset.
    - augmentation_factor (float): The fraction of the original data to generate.

    Returns:
    - pd.DataFrame: The augmented dataset.
    """
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
        stroke = random.choice([0, 1])  # You may adjust the probability if needed

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

    return augmented_data
