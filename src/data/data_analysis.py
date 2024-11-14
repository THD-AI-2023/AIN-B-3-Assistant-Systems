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
import joblib


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

        # Load and preprocess data
        status_text.text("Loading data...")
        self.load_and_preprocess_data()
        progress_bar.progress(20)

        # Display and apply data filters
        filters_changed = self.display_filters()
        progress_bar.progress(30)

        # Visualize data
        self.visualize_data()
        progress_bar.progress(50)

        # Split data
        if not self.split_data():
            return  # Stop execution if splitting fails
        progress_bar.progress(60)

        # Augment data
        self.augment_data()
        progress_bar.progress(65)

        # Define preprocessor
        self.define_preprocessor()
        progress_bar.progress(70)

        # Fit preprocessor and track creation
        status_text.text("Fitting preprocessor and saving to preprocessor.pkl...")
        self.fit_preprocessor()
        progress_bar.progress(75)
        time.sleep(1)

        # Check if filters have changed to determine if retraining is needed
        if filters_changed:
            st.info("Data filters have changed. Retraining models...")
            self.delete_existing_models()
            self.train_models(status_text, progress_bar)
        else:
            # Check if models are trained
            if not self.models_trained():
                st.info("Models are not trained yet. Training now...")
                self.train_models(status_text, progress_bar)
            else:
                st.success("Models are already trained and loaded.")
                self.load_trained_models()

        # Make sample prediction
        self.make_sample_prediction()
        progress_bar.progress(100)
        status_text.text("Data analysis workflow complete.")
        time.sleep(1)
        status_text.empty()

    def load_and_preprocess_data(self):
        """
        Loads and preprocesses the data.
        """
        self.data = load_data()
        if self.data.empty:
            st.error("Failed to load data.")
            st.stop()
        self.data = preprocess_data(self.data)
        self.data.reset_index(drop=True, inplace=True)

    def display_filters(self):
        """
        Displays interactive widgets for data filtering within a collapsible expander.
        Returns True if filters have changed since last run, False otherwise.
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
            gender_selected = st.multiselect('Gender', options=self.data['gender'].unique().tolist(), default=st.session_state['gender_selected'])
            hypertension_selected = st.multiselect('Hypertension', options=self.data['hypertension'].unique().tolist(), default=st.session_state['hypertension_selected'])
            heart_disease_selected = st.multiselect('Heart Disease', options=self.data['heart_disease'].unique().tolist(), default=st.session_state['heart_disease_selected'])
            smoking_selected = st.multiselect('Smoking Status', options=self.data['smoking_status'].unique().tolist(), default=st.session_state['smoking_selected'])

            current_filters = {
                'age_range': age_range,
                'gender_selected': tuple(sorted(gender_selected)),
                'hypertension_selected': tuple(sorted(hypertension_selected)),
                'heart_disease_selected': tuple(sorted(heart_disease_selected)),
                'smoking_selected': tuple(sorted(smoking_selected)),
            }

            # Determine if filters have changed
            previous_filters = st.session_state.get('previous_filters', None)
            filters_changed = previous_filters != current_filters
            st.session_state['previous_filters'] = current_filters

            # Update session state
            st.session_state['age_range'] = age_range
            st.session_state['gender_selected'] = gender_selected
            st.session_state['hypertension_selected'] = hypertension_selected
            st.session_state['heart_disease_selected'] = heart_disease_selected
            st.session_state['smoking_selected'] = smoking_selected

        # Filter the data based on selections
        filtered_data = self.data[
            (self.data['age'] >= current_filters['age_range'][0]) &
            (self.data['age'] <= current_filters['age_range'][1]) &
            (self.data['gender'].isin(current_filters['gender_selected'])) &
            (self.data['hypertension'].isin(current_filters['hypertension_selected'])) &
            (self.data['heart_disease'].isin(current_filters['heart_disease_selected'])) &
            (self.data['smoking_status'].isin(current_filters['smoking_selected']))
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

        return filters_changed

    def visualize_data(self):
        """
        Visualizes the data.
        """
        visualize_data(self.data)

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
        try:
            self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
            )
            return True
        except ValueError as e:
            st.error(f"Error during train-test split: {e}")
            return False

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
                    OneHotEncoder(handle_unknown="ignore", sparse=False),
                    self.categorical_cols,
                ),
            ]
        )

    def fit_preprocessor(self):
        """
        Fits the preprocessor on the training data and saves it.
        """
        # Fit preprocessor on real training data
        self.X_train_real = self.preprocessor.fit_transform(self.X_train_raw)
        self.X_test = self.preprocessor.transform(self.X_test_raw)

        # Save the preprocessor to disk
        preprocessor_path = os.path.join("models", "data_analysis", "preprocessor.pkl")
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        joblib.dump(self.preprocessor, preprocessor_path)

    def models_trained(self):
        """
        Checks if data analysis models are already trained and loaded.
        """
        model_files = [
            "Logistic_Regression_real.pkl",
            "Support_Vector_Machine_real.pkl",
            "Random_Forest_real.pkl",
            "Logistic_Regression_augmented.pkl",
            "Support_Vector_Machine_augmented.pkl",
            "Random_Forest_augmented.pkl",
        ]
        for model_file in model_files:
            model_path = os.path.join("models", "data_analysis", model_file)
            if not os.path.exists(model_path):
                return False
        return True

    def load_trained_models(self):
        """
        Loads trained data analysis models from disk.
        """
        model_names = [
            "Logistic_Regression_real.pkl",
            "Support_Vector_Machine_real.pkl",
            "Random_Forest_real.pkl",
            "Logistic_Regression_augmented.pkl",
            "Support_Vector_Machine_augmented.pkl",
            "Random_Forest_augmented.pkl",
        ]
        for model_name in model_names:
            model_path = os.path.join("models", "data_analysis", model_name)
            try:
                self.models[model_name] = joblib.load(model_path)
            except Exception as e:
                st.error(f"Error loading model {model_name}: {e}")

    def train_models(self, status_text, progress_bar):
        """
        Trains machine learning models on real and augmented data, saves them, and displays evaluation metrics.
        """
        # Handle class imbalance using SMOTE on real data
        sm = SMOTE(random_state=42)
        try:
            X_train_res_real, y_train_res_real = sm.fit_resample(
                self.X_train_real, self.y_train.reset_index(drop=True)
            )
        except ValueError as e:
            st.error(f"SMOTE error on real data: {e}")
            return

        # Initialize models
        self.models = {
            "Logistic_Regression_real.pkl": LogisticRegression(
                max_iter=1000, class_weight="balanced", solver="liblinear"
            ),
            "Support_Vector_Machine_real.pkl": SVC(
                class_weight="balanced", probability=True, random_state=42
            ),
            "Random_Forest_real.pkl": RandomForestClassifier(
                class_weight="balanced", random_state=42
            ),
        }

        # Training models on real data
        status_text.text("Training models on real data...")
        total_models = len(self.models)
        for idx, (model_name, model) in enumerate(self.models.items()):
            try:
                model.fit(X_train_res_real, y_train_res_real)
                st.write(f"Trained {model_name.split('_')[0]} on real data.")
                # Save the model
                model_path = os.path.join("models", "data_analysis", model_name)
                joblib.dump(model, model_path)
                st.write(f"Saved {model_name} to disk.")
                # Evaluate model
                self.evaluate_model(model, model_name, self.X_test, self.y_test, data_type="Real Data")
            except Exception as e:
                st.error(f"Error training {model_name}: {e}")
                continue
            # Update progress
            progress = 75 + int(((idx + 1) / total_models) * 10)  # Next 10% for real data models
            progress_bar.progress(progress)

        # Handle class imbalance using SMOTE on augmented data
        sm_aug = SMOTE(random_state=42)
        try:
            X_train_res_augmented, y_train_res_augmented = sm_aug.fit_resample(
                self.preprocessor.transform(self.augmented_X_train_raw),
                self.augmented_y_train.reset_index(drop=True),
            )
        except ValueError as e:
            st.error(f"SMOTE error on augmented data: {e}")
            return

        # Initialize models for augmented data
        augmented_models = {
            "Logistic_Regression_augmented.pkl": LogisticRegression(
                max_iter=1000, class_weight="balanced", solver="liblinear"
            ),
            "Support_Vector_Machine_augmented.pkl": SVC(
                class_weight="balanced", probability=True, random_state=42
            ),
            "Random_Forest_augmented.pkl": RandomForestClassifier(
                class_weight="balanced", random_state=42
            ),
        }

        # Training models on augmented data
        status_text.text("Training models on augmented data...")
        total_augmented_models = len(augmented_models)
        for idx, (model_name, model) in enumerate(augmented_models.items()):
            try:
                model.fit(X_train_res_augmented, y_train_res_augmented)
                st.write(f"Trained {model_name.split('_')[0]} on augmented data.")
                # Save the model
                model_path = os.path.join("models", "data_analysis", model_name)
                joblib.dump(model, model_path)
                st.write(f"Saved {model_name} to disk.")
                # Add to main models dict
                self.models[model_name] = model
                # Evaluate model
                self.evaluate_model(model, model_name, self.X_test, self.y_test, data_type="Augmented Data")
            except Exception as e:
                st.error(f"Error training {model_name}: {e}")
                continue
            # Update progress
            progress = 85 + int(((idx + 1) / total_augmented_models) * 10)  # Next 10% for augmented data models
            progress_bar.progress(progress)

        progress_bar.progress(95)
        status_text.text("Model training complete.")
        time.sleep(1)
        status_text.empty()

    def evaluate_model(self, model, model_name, X_test, y_test, data_type=""):
        """
        Evaluates the model and displays metrics.
        """
        try:
            y_pred = model.predict(X_test)
            y_prob = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else model.decision_function(X_test)
            )

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)

            # Display results
            st.write(f"### {model_name.split('_')[0]} Evaluation ({data_type})")
            st.write(f"**Accuracy:** {accuracy:.4f}")
            st.write(f"**Precision:** {precision:.4f}")
            st.write(f"**Recall:** {recall:.4f}")
            st.write(f"**F1 Score:** {f1:.4f}")
            st.write(f"**ROC AUC Score:** {roc_auc:.4f}")

            st.write("**Confusion Matrix:**")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"],
            )
            st.dataframe(cm_df)

            st.write("**Classification Report:**")
            st.text(classification_report(y_test, y_pred, zero_division=0))
            st.write("-" * 60)

            # Save evaluation metrics to CSV
            eval_dir = os.path.join("models", "data_analysis", "evaluations")
            os.makedirs(eval_dir, exist_ok=True)
            eval_path = os.path.join(eval_dir, "model_evaluations.csv")

            eval_data = {
                "Model": model_name,
                "Data_Type": data_type,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1,
                "ROC_AUC_Score": roc_auc
            }
            eval_df = pd.DataFrame([eval_data])

            if os.path.exists(eval_path):
                existing_eval_df = pd.read_csv(eval_path)
                existing_eval_df = existing_eval_df[
                    ~((existing_eval_df['Model'] == model_name) & (existing_eval_df['Data_Type'] == data_type))
                ]
                eval_df = pd.concat([existing_eval_df, eval_df], ignore_index=True)
            eval_df.to_csv(eval_path, index=False)

        except Exception as e:
            st.error(f"Error evaluating {model_name}: {e}")

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
                "avg_glucose_level": [174.12],
                "bmi": [24.0],
                "ever_married": ["Yes"],
                "work_type": ["Self-employed"],
                "Residence_type": ["Rural"],
                "smoking_status": ["never smoked"],
            }
        )

        st.write("**Sample Input:**")
        st.dataframe(sample_input)

        # Preprocess the sample input
        try:
            sample_input_processed = self.preprocessor.transform(sample_input)
        except Exception as e:
            st.error(f"Error preprocessing sample input: {e}")
            return

        # Use each model to predict the sample input
        st.write("#### Predictions from Models Trained on Real Data")
        real_model_names = ["Logistic_Regression_real.pkl", "Support_Vector_Machine_real.pkl", "Random_Forest_real.pkl"]
        for model_name in real_model_names:
            model = self.models.get(model_name)
            if model:
                try:
                    sample_prediction = model.predict(sample_input_processed)[0]
                    st.write(f"{model_name.split('_')[0]} prediction: {sample_prediction}")
                except Exception as e:
                    st.error(f"Error making prediction with {model_name}: {e}")

        st.write("#### Predictions from Models Trained on Augmented Data")
        augmented_model_names = ["Logistic_Regression_augmented.pkl", "Support_Vector_Machine_augmented.pkl", "Random_Forest_augmented.pkl"]
        for model_name in augmented_model_names:
            model = self.models.get(model_name)
            if model:
                try:
                    sample_prediction = model.predict(sample_input_processed)[0]
                    st.write(f"{model_name.split('_')[0]} prediction: {sample_prediction}")
                except Exception as e:
                    st.error(f"Error making prediction with {model_name}: {e}")

    def delete_existing_models(self):
        """
        Deletes all existing data analysis trained models to allow retraining with new filters.
        """
        data_analysis_model_dir = os.path.join("models", "data_analysis")
        if os.path.exists(data_analysis_model_dir):
            for file in os.listdir(data_analysis_model_dir):
                file_path = os.path.join(data_analysis_model_dir, file)
                try:
                    if os.path.isfile(file_path) and file != "preprocessor.pkl":
                        os.remove(file_path)
                        st.write(f"Deleted existing data analysis model: {file}")
                except Exception as e:
                    st.error(f"Error deleting model {file}: {e}")
        else:
            st.warning("Data analysis models directory does not exist. No models to delete.")
