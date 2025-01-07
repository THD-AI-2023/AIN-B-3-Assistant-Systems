![Assistance Systems Project Banner](./img/ASP_Banner.png)

# **Data**

Understanding the dataset is essential for building an effective recommendation system. This section provides detailed information about the data used in the **Assistance Systems Project**, including the original data source, detailed feature descriptions, target variables, data preprocessing steps, visualizations, data augmentation methods, and the impact on model training.

---

## **1. Original Data Source**

- **Dataset:** Stroke Prediction Dataset
- **Source:** [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
- **Description:** This dataset is used to predict whether a patient is likely to get a stroke based on input parameters like gender, age, various diseases, and smoking status.

---

## **2. Feature and Target Variables**

### **Features**

1. **gender**: Gender of the patient.
   - **Type:** Categorical (Nominal)
   - **Categories:** 'Male', 'Female', 'Other'
   - **Distribution:** Majority are 'Male' or 'Female', very few 'Other'

2. **age**: Age of the patient in years.
   - **Type:** Numerical (Continuous)
   - **Range:** 0 - 82
   - **Distribution:** Skewed towards older ages due to stroke risk increasing with age.

3. **hypertension**: Indicates if the patient has hypertension.
   - **Type:** Categorical (Binary)
   - **Values:** 0 (No), 1 (Yes)
   - **Description:** 0 if the patient doesn't have hypertension, 1 if the patient has hypertension.

4. **heart_disease**: Indicates if the patient has heart disease.
   - **Type:** Categorical (Binary)
   - **Values:** 0 (No), 1 (Yes)
   - **Description:** 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease.

5. **ever_married**: Marital status of the patient.
   - **Type:** Categorical (Binary)
   - **Values:** 'No', 'Yes'
   - **Description:** Indicates if the patient has ever been married.

6. **work_type**: Type of occupation of the patient.
   - **Type:** Categorical (Nominal)
   - **Categories:** 'children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'
   - **Description:** Indicates the type of work the patient does.

7. **Residence_type**: Type of residence of the patient.
   - **Type:** Categorical (Binary)
   - **Values:** 'Rural', 'Urban'
   - **Description:** Indicates whether the patient lives in a rural or urban area.

8. **avg_glucose_level**: Average glucose level in the blood.
   - **Type:** Numerical (Continuous)
   - **Range:** Varies widely
   - **Units:** mg/dL
   - **Description:** Higher levels may indicate diabetes or other conditions affecting stroke risk.

9. **bmi**: Body Mass Index of the patient.
   - **Type:** Numerical (Continuous)
   - **Range:** Varies
   - **Units:** kg/m²
   - **Description:** BMI is a measure of body fat based on height and weight.

10. **smoking_status**: Smoking status of the patient.
    - **Type:** Categorical (Nominal)
    - **Categories:** 'formerly smoked', 'never smoked', 'smokes', 'Unknown'
    - **Description:** Indicates the patient's smoking habits.

### **Target Variable**

- **stroke**: Indicates if the patient has had a stroke.
  - **Type:** Categorical (Binary)
  - **Values:** 0 (No), 1 (Yes)
  - **Description:** 0 if the patient didn't have a stroke, 1 if the patient had a stroke.

---

## **3. Data Preprocessing**

### **Handling Missing Values**

- **BMI**: Missing values in the 'bmi' column were imputed using the mean BMI.

  - **Rationale:** BMI is an important feature for predicting stroke risk. Imputing missing values with the mean allows us to retain these records without introducing significant bias.

### **Feature Encoding**

- **Categorical Variables**: Categorical features were encoded using **One-Hot Encoding** to convert them into numerical format suitable for machine learning models.

  - **Variables Encoded:**
    - **gender**
    - **ever_married**
    - **work_type**
    - **Residence_type**
    - **smoking_status**

- **Binary Variables**: Variables like 'hypertension' and 'heart_disease' were already numerical, with values 0 and 1.

### **Dropping Unnecessary Columns**

- **id**: The 'id' column was dropped as it does not contribute to the prediction.

### **Feature Scaling**

- **Numerical Variables**: Features like 'age', 'avg_glucose_level', and 'bmi' were scaled using **Standardization** (z-score normalization) to bring them to a similar scale.

---

## **4. Data Visualization**

To understand the distribution and relationships within the data, various visualizations were created using **Altair** in the application.

### **Correlation Heatmap**

- **Purpose:** To identify correlations between features and the target variable.
- **Description:** The heatmap visualizes the Pearson correlation coefficients between numerical variables.
- **Insights:** Strong correlations were observed between 'age' and 'stroke', and moderate correlations between 'heart_disease' and 'stroke'.

![Correlation Heatmap](./img/correlation_heatmap.png)

### **Age Distribution by Stroke**

- **Purpose:** To observe how age distribution varies between patients who have had a stroke and those who haven't.
- **Description:** Histogram showing the age distribution, differentiated by 'stroke' status.
- **Insights:** Patients who had a stroke tend to be older.

![Age Distribution](./img/age_distribution_by_stroke.png)

### **BMI Distribution**

- **Purpose:** To examine the distribution of BMI values in the dataset.
- **Description:** Histogram showing the frequency of different BMI values.
- **Insights:** Most patients have a BMI in the range of 20-35 kg/m².

![BMI Distribution](./img/bmi_distribution.png)

### **Smoking Status vs Stroke**

- **Purpose:** To analyze the relationship between smoking status and stroke occurrence.
- **Description:** Bar chart showing the count of patients with different smoking statuses, separated by 'stroke' status.
- **Insights:** Smoking status 'Unknown' is prevalent, indicating missing data. Smoking may have an impact on stroke risk.

![Smoking Status vs Stroke](./img/smoking_status_vs_stroke.png)

*(Note: Visualizations are generated within the Data Analysis section of the application.)*

---

## **5. Outlier Handling**

Outliers were detected and removed using the **Z-score method**. For detailed information, please refer to the [Outlier Detection and Handling](Outlier_Handling) section.

---

## **6. Data Augmentation and Fake Data Generation**

### **Approach**

To improve model performance and address class imbalance (since stroke occurrences are less frequent), we augmented the dataset by adding **30% realistic synthetic data**.

- **Method Used**: Data augmentation using the **Faker** library and **SMOTE (Synthetic Minority Over-sampling Technique)** algorithm for generating synthetic samples.

- **Variables Synthesized**: All features were considered to maintain consistency and realism in the synthetic data.

### **Implementation Details**

#### **Using Faker Library**

- **Purpose**: To generate realistic fake data for various features.
- **Process**:
  - Randomly generate values for categorical features based on their distributions.
  - Generate numerical values within realistic ranges observed in the dataset.
  - Ensure consistency between related features (e.g., 'ever_married' is 'Yes' for ages above 18).

#### **Using SMOTE**

- **Purpose**: To address class imbalance by generating synthetic samples of the minority class ('stroke' = 1).
- **Process**:
  - Apply SMOTE on the training data after splitting to prevent data leakage.
  - Generate synthetic samples until the desired balance is achieved.

### **Implementation Code Snippet**

Using Faker:

```python
from faker import Faker
import pandas as pd
import random

fake = Faker()
synthetic_data = []

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
```

Using SMOTE:

```python
from imblearn.over_sampling import SMOTE

# Assuming 'X_train' and 'y_train' are the training features and target
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### **Influence on Model Training**

- **Balanced Dataset**: The class imbalance was significantly reduced, providing a more balanced dataset for training.

- **Improved Model Performance**: Models trained on the augmented data showed better recall and precision for the minority class (stroke cases), leading to more reliable predictions.

- **Reduced Overfitting**: The additional data helped in reducing overfitting by providing more varied samples.

- **Enhanced Generalization**: The models generalized better to unseen data due to the diversity introduced by synthetic samples.

---

## **7. Effect on the Training of the Model**

Adding synthetic data and handling outliers had a significant positive impact on the model's performance:

- **Accuracy Improvement**: An increase in overall accuracy was observed across multiple models.

- **Better Generalization**: The models performed better on the test set, indicating improved generalization.

- **Enhanced Recall and Precision**: The ability to correctly predict stroke cases (positive class) improved, with higher recall and precision scores.

- **Model Evaluation Metrics**: Detailed evaluation metrics for each model are provided in the application, showing comparisons between models trained on real data vs. augmented data.

---

**Conclusion**

By thoroughly understanding and preprocessing the data, including handling missing values, removing outliers, and augmenting the dataset with synthetic data, we enhanced the model's ability to provide accurate and reliable health recommendations. The comprehensive approach to data management was instrumental in building effective machine learning models for the project.

---

### **Note:** For additional details and insights, please refer to the application's Data Analysis section, where you can interact with the data, apply filters, and view updated visualizations in real-time.

---