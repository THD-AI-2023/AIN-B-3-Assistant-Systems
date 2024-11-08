import os
import time
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from data.data_loader import load_data
from data.data_preprocessor import preprocess_data
from data.data_augmentation import augment_data
from data.data_visualization import visualize_data

class DataAnalysis:
    """
    A class to encapsulate the data analysis logic with interactive widgets.
    """

    def __init__(self):
        # Initialize attributes
        self.data = None
        self.X = None
        self.y = None
        self.X_train_raw = None
        self.X_test_raw = None
        self.y_train = None
        self.y_test = None
        self.augmented_X_train_raw = None
        self.augmented_y_train = None
        self.preprocessor = None
        self.models = {}
        self.categorical_cols = [
            "gender",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
        ]
        self.numerical_cols = [
            "age",
            "hypertension",
            "heart_disease",
            "avg_glucose_level",
            "bmi",
        ]

    def run(self):
        """
        Executes the data analysis workflow with interactive widgets.
        """
        st.subheader("Data Analysis")

        # Placeholders for status messages and progress bars in the sidebar
        status_text = st.sidebar.empty()
        progress_bar = st.sidebar.progress(0)

        # Data loading and preprocessing
        self.load_and_preprocess_data(status_text, progress_bar)

        # Display data filters
        self.display_filters()

        # Visualize data
        self.visualize_data(status_text, progress_bar)

        # Split data
        if not self.split_data():
            return  # Stop execution if splitting fails

        # Augment data
        self.augment_data()

        # Define preprocessor
        self.define_preprocessor()

        # Fit preprocessor
        self.fit_preprocessor()

        # Train models on real data
        if not self.train_models_on_real_data(status_text, progress_bar):
            return  # Stop execution if training fails

        # Train models on augmented data
        if not self.train_models_on_augmented_data(status_text, progress_bar):
            return  # Stop execution if training fails

        # Make sample prediction
        self.make_sample_prediction()

    def load_and_preprocess_data(self, status_text, progress_bar):
        """
        Loads and preprocesses the data.
        """
        status_text.text("Loading data...")
        self.data = load_data()
        progress_bar.progress(25)
        self.data = preprocess_data(self.data)
        progress_bar.progress(50)
        self.data.reset_index(drop=True, inplace=True)
        progress_bar.progress(75)

    def display_filters(self):
        """
        Displays interactive widgets for data filtering within a collapsible expander.
        """
        with st.expander("Data Filters", expanded=False):
            st.write("### Filter the Data")

            # Initialize session state for filters
            if 'age_range' not in st.session_state:
                st.session_state['age_range'] = (int(self.data['age'].min()), int(self.data['age'].max()))
            if 'gender_selected' not in st.session_state:
                st.session_state['gender_selected'] = self.data['gender'].unique().tolist()
            if 'hypertension_selected' not in st.session_state:
                st.session_state['hypertension_selected'] = self.data['hypertension'].unique().tolist()
            if 'heart_disease_selected' not in st.session_state:
                st.session_state['heart_disease_selected'] = self.data['heart_disease'].unique().tolist()
            if 'smoking_selected' not in st.session_state:
                st.session_state['smoking_selected'] = self.data['smoking_status'].unique().tolist()

            # Age filter
            age_min, age_max = int(self.data['age'].min()), int(self.data['age'].max())
            age_range = st.slider('Age Range', min_value=age_min, max_value=age_max, value=st.session_state['age_range'])
            st.session_state['age_range'] = age_range

            # Gender filter
            gender_options = self.data['gender'].unique().tolist()
            gender_selected = st.multiselect('Gender', options=gender_options, default=st.session_state['gender_selected'])
            st.session_state['gender_selected'] = gender_selected

            # Hypertension filter
            hypertension_options = self.data['hypertension'].unique().tolist()
            hypertension_selected = st.multiselect('Hypertension', options=hypertension_options, default=st.session_state['hypertension_selected'])
            st.session_state['hypertension_selected'] = hypertension_selected

            # Heart disease filter
            heart_disease_options = self.data['heart_disease'].unique().tolist()
            heart_disease_selected = st.multiselect('Heart Disease', options=heart_disease_options, default=st.session_state['heart_disease_selected'])
            st.session_state['heart_disease_selected'] = heart_disease_selected

            # Smoking status filter
            smoking_options = self.data['smoking_status'].unique().tolist()
            smoking_selected = st.multiselect('Smoking Status', options=smoking_options, default=st.session_state['smoking_selected'])
            st.session_state['smoking_selected'] = smoking_selected

        # Filter the data based on selections
        filtered_data = self.data[
            (self.data['age'] >= st.session_state['age_range'][0]) &
            (self.data['age'] <= st.session_state['age_range'][1]) &
            (self.data['gender'].isin(st.session_state['gender_selected'])) &
            (self.data['hypertension'].isin(st.session_state['hypertension_selected'])) &
            (self.data['heart_disease'].isin(st.session_state['heart_disease_selected'])) &
            (self.data['smoking_status'].isin(st.session_state['smoking_selected']))
        ]

        if filtered_data.empty:
            st.error("No data available for the selected filters. Please adjust your filters.")
            st.stop()

        # Update the data attribute to use the filtered data
        self.data = filtered_data.reset_index(drop=True)

        # Store filtered data in session state
        st.session_state['filtered_data'] = self.data

        # Save the filtered data to a CSV file for chatbot access
        filtered_data_path = os.path.join("data", "processed", "filtered_data.csv")
        os.makedirs(os.path.dirname(filtered_data_path), exist_ok=True)
        self.data.to_csv(filtered_data_path, index=False)

        # Display the filtered data
        st.write("### Filtered Data")
        st.dataframe(self.data.head(20))  # Show first 20 rows


    def visualize_data(self, status_text, progress_bar):
        """
        Visualizes the data.
        """
        visualize_data(self.data)
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.text("Data loading complete.")
        time.sleep(1)
        status_text.empty()

    def split_data(self):
        """
        Splits the data into training and testing sets.
        """
        # Define features and target variable
        self.X = self.data.drop(columns=["stroke"])
        self.y = self.data["stroke"]

        # Check if there are enough samples to split
        if len(self.X) < 10:
            st.error("Not enough data to split into training and testing sets. Please adjust your filters.")
            return False

        # Check if both classes are present
        if len(self.y.unique()) < 2:
            st.error("Data does not contain both classes after filtering. Please adjust your filters.")
            return False

        # Split the dataset with stratification
        (
            self.X_train_raw,
            self.X_test_raw,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        return True

    def augment_data(self):
        """
        Augments the training data with synthetic samples.
        """
        # Augment the training data
        self.augmented_X_train_raw, self.augmented_y_train = augment_data(
            self.X_train_raw, self.y_train, augmentation_factor=0.3
        )
        self.augmented_X_train_raw.reset_index(drop=True, inplace=True)
        self.augmented_y_train.reset_index(drop=True, inplace=True)

    def define_preprocessor(self):
        """
        Defines the preprocessing steps for numerical and categorical data.
        """
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    self.numerical_cols,
                ),
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    self.categorical_cols,
                ),
            ]
        )

    def fit_preprocessor(self):
        """
        Fits the preprocessor on the training data.
        """
        # Fit preprocessor on real training data
        self.X_train_real = self.preprocessor.fit_transform(self.X_train_raw)
        self.X_test = self.preprocessor.transform(self.X_test_raw)

    def train_models_on_real_data(self, status_text, progress_bar):
        """
        Trains machine learning models on the real data.
        """
        # Handle class imbalance using SMOTE on real data
        sm = SMOTE(random_state=42)
        try:
            X_train_res_real, y_train_res_real = sm.fit_resample(
                self.X_train_real, self.y_train.reset_index(drop=True)
            )
        except ValueError as e:
            st.error(f"SMOTE error: {e}")
            return False

        # Initialize models
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", solver="liblinear"
            ),
            "Support Vector Machine": SVC(class_weight="balanced", probability=True),
            "Random Forest": RandomForestClassifier(class_weight="balanced"),
        }

        # Training models on real data
        status_text.text("Training models on real data...")
        total_models = len(self.models)
        for idx, (model_name, model) in enumerate(self.models.items()):
            # Train the model
            try:
                model.fit(X_train_res_real, y_train_res_real)
            except Exception as e:
                st.error(f"Error training {model_name}: {e}")
                continue

            # Update progress bar and status message
            progress = int(((idx + 1) / total_models) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Training models on real data... ({progress}%)")

            # Evaluate and display results
            self.evaluate_model(model, model_name, "Real Data")

        # After training all models on real data
        progress_bar.empty()
        status_text.text("Model training on real data complete.")
        time.sleep(1)
        status_text.empty()
        return True

    def train_models_on_augmented_data(self, status_text, progress_bar):
        """
        Trains machine learning models on the augmented data.
        """
        # Preprocess augmented training data
        X_train_augmented = self.preprocessor.transform(self.augmented_X_train_raw)

        # Handle class imbalance using SMOTE on augmented data
        sm = SMOTE(random_state=42)
        try:
            X_train_res_augmented, y_train_res_augmented = sm.fit_resample(
                X_train_augmented, self.augmented_y_train.reset_index(drop=True)
            )
        except ValueError as e:
            st.error(f"SMOTE error: {e}")
            return False

        # Training models on augmented data
        status_text.text("Training models on augmented data...")
        total_models = len(self.models)
        for idx, (model_name, model) in enumerate(self.models.items()):
            # Train the model
            try:
                model.fit(X_train_res_augmented, y_train_res_augmented)
            except Exception as e:
                st.error(f"Error training {model_name}: {e}")
                continue

            # Update progress bar and status message
            progress = int(((idx + 1) / total_models) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Training models on augmented data... ({progress}%)")

            # Evaluate and display results
            self.evaluate_model(model, model_name, "Augmented Data")

        # After training all models on augmented data
        progress_bar.empty()
        status_text.text("Model training on augmented data complete.")
        time.sleep(1)
        status_text.empty()
        return True

    def evaluate_model(self, model, model_name, data_type):
        """
        Evaluates the model and displays metrics.

        Parameters:
        - model: Trained machine learning model.
        - model_name (str): Name of the model.
        - data_type (str): Indicates whether the model was trained on real or augmented data.
        """
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_prob = (
            model.predict_proba(self.X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(self.X_test)
        )

        # Calculate evaluation metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_prob)

        # Display results
        st.write(f"### {model_name} (Trained on {data_type})")
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")
        st.write(f"**ROC AUC Score:** {roc_auc:.4f}")

        st.write("**Confusion Matrix:**")
        cm = confusion_matrix(self.y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Negative", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Positive"],
        )
        st.dataframe(cm_df)

        st.write("**Classification Report:**")
        st.text(classification_report(self.y_test, y_pred, zero_division=0))
        st.write("-" * 60)

    def make_sample_prediction(self):
        """
        Makes predictions on a sample input using the trained models.
        """
        st.write("### Sample Prediction")
        sample_input = pd.DataFrame(
            {
                "gender": ["Female"],
                "age": [79.0],
                "hypertension": [1],
                "heart_disease": [0],
                "ever_married": ["Yes"],
                "work_type": ["Self-employed"],
                "Residence_type": ["Rural"],
                "avg_glucose_level": [174.12],
                "bmi": [24.0],
                "smoking_status": ["never smoked"],
            }
        )

        st.write("**Sample Input:**")
        st.dataframe(sample_input)

        # Preprocess the sample input
        sample_input_processed = self.preprocessor.transform(sample_input)

        # Use each model to predict the sample input
        st.write("#### Predictions from Models Trained on Real Data")
        for model_name, model in self.models.items():
            sample_prediction = model.predict(sample_input_processed)
            st.write(f"{model_name} prediction: {sample_prediction[0]}")

        st.write("#### Predictions from Models Trained on Augmented Data")
        for model_name, model in self.models.items():
            sample_prediction = model.predict(sample_input_processed)
            st.write(f"{model_name} prediction: {sample_prediction[0]}")
