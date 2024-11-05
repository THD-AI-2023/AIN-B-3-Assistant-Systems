import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from data.data_loader import load_data
from data.data_preprocessor import preprocess_data
from data.data_augmentation import augment_data
from data.data_visualization import visualize_data
from chatbot.rasa_chatbot import Chatbot


def main():
    st.set_page_config(page_title="Project Apero", layout="wide")

    st.title("Project Apero (Assistance Systems Project)")

    # Sidebar for navigation
    menu = ["Home", "Data Analysis", "Recommendations", "Chatbot"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Welcome to Project Apero")
        st.write("Use the sidebar to navigate through the application.")

    elif choice == "Data Analysis":
        st.subheader("Data Analysis")

        # Sidebar progress bar for data loading
        data_load_bar = st.sidebar.progress(0)
        st.sidebar.write("Loading data...")

        # Load and preprocess data
        data = load_data()
        data_load_bar.progress(25)
        data = preprocess_data(data)
        data_load_bar.progress(50)
        data.reset_index(drop=True, inplace=True)  # Reset index
        data_load_bar.progress(75)

        # Display the preprocessed data
        st.write("### Preprocessed Data")
        # st.dataframe(data.head(20))  # Show 20 rows
        st.dataframe(data)

        # Visualize data
        visualize_data(data)
        data_load_bar.progress(100)
        data_load_bar.empty()
        st.sidebar.write("Data loaded and preprocessed.")

        # Define features and target variable
        X = data.drop(columns=["stroke"])
        y = data["stroke"]

        # Split the dataset into training and testing sets with stratification
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Augment the training data
        augmented_X_train_raw, augmented_y_train = augment_data(
            X_train_raw, y_train, augmentation_factor=0.3
        )
        augmented_X_train_raw.reset_index(drop=True, inplace=True)
        augmented_y_train.reset_index(drop=True, inplace=True)

        # Define categorical and numerical columns
        categorical_cols = [
            "gender",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
        ]
        numerical_cols = [
            "age",
            "hypertension",
            "heart_disease",
            "avg_glucose_level",
            "bmi",
        ]

        # Define preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    numerical_cols,
                ),
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    categorical_cols,
                ),
            ]
        )

        # Fit preprocessor on real training data
        X_train_real = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        # Handle class imbalance using SMOTE on real data
        sm = SMOTE(random_state=42)
        X_train_res_real, y_train_res_real = sm.fit_resample(
            X_train_real, y_train.reset_index(drop=True)
        )

        # Initialize models with class weights to handle imbalance
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", solver="liblinear"
            ),
            "Support Vector Machine": SVC(class_weight="balanced", probability=True),
            "Random Forest": RandomForestClassifier(class_weight="balanced"),
        }

        # Sidebar progress bar for model training on real data
        st.sidebar.write("Training models on real data...")
        model_train_bar_real = st.sidebar.progress(0)
        total_models = len(models)

        # Train and evaluate models on real data only
        st.write("## Models Trained on Real Data Only")
        for idx, (model_name, model) in enumerate(models.items()):
            st.write(f"### {model_name}")
            # Train the model
            model.fit(X_train_res_real, y_train_res_real)

            # Update progress bar
            model_train_bar_real.progress(int(((idx + 1) / total_models) * 100))

            # Make predictions
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

        st.sidebar.write("Model training on real data completed.")

        # Preprocess augmented training data
        X_train_augmented = preprocessor.transform(augmented_X_train_raw)

        # Handle class imbalance using SMOTE on augmented data
        sm = SMOTE(random_state=42)
        X_train_res_augmented, y_train_res_augmented = sm.fit_resample(
            X_train_augmented, augmented_y_train.reset_index(drop=True)
        )

        # Sidebar progress bar for model training on augmented data
        st.sidebar.write("Training models on augmented data...")
        model_train_bar_aug = st.sidebar.progress(0)

        # Train and evaluate models on augmented data
        st.write("## Models Trained on Augmented Data")
        for idx, (model_name, model) in enumerate(models.items()):
            st.write(f"### {model_name}")
            # Train the model
            model.fit(X_train_res_augmented, y_train_res_augmented)

            # Update progress bar
            model_train_bar_aug.progress(int(((idx + 1) / total_models) * 100))

            # Make predictions
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

        st.sidebar.write("Model training on augmented data completed.")

        # Sample input for prediction
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
        sample_input_processed = preprocessor.transform(sample_input)

        # Use each model to predict the sample input
        st.write("#### Predictions from Models Trained on Real Data")
        for model_name, model in models.items():
            sample_prediction = model.predict(sample_input_processed)
            st.write(f"{model_name} prediction: {sample_prediction[0]}")

        st.write("#### Predictions from Models Trained on Augmented Data")
        for model_name, model in models.items():
            sample_prediction = model.predict(sample_input_processed)
            st.write(f"{model_name} prediction: {sample_prediction[0]}")

    elif choice == "Recommendations":
        st.subheader("Personalized Recommendations")
        # TODO: Implement the recommendation system interface
        st.write("Recommendation system interface will be implemented here.")

    elif choice == "Chatbot":
        st.subheader("Chatbot Assistance")
        if "key" not in st.session_state:
            st.session_state["key"] = os.urandom(24).hex()
        rasa_server_url = os.getenv(
            "RASA_SERVER", "http://localhost:5005/webhooks/rest/webhook"
        )
        chatbot = Chatbot(rasa_url=rasa_server_url, session_id=st.session_state["key"])
        chatbot.run()


if __name__ == "__main__":
    main()
