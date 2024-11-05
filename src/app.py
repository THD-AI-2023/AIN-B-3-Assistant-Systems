# src/app.py

import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data.data_loader import load_data
from data.data_preprocessor import preprocess_data
from models.recommendation_model import RecommendationModel
from chatbot.rasa_chatbot import Chatbot

def main():
    st.set_page_config(page_title="Project Apero", layout="wide")

    st.title("Project Apero (Assistance Systems Project)")

    # Sidebar for navigation
    menu = ["Home", "Data Analysis", "Recommendations", "Chatbot"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Load and preprocess data
    data = load_data()
    processed_data = preprocess_data(data)

    # Initialize recommendation model
    rec_model = RecommendationModel(processed_data)
    rec_model.train_models()

    if choice == "Home":
        st.subheader("Welcome to Project Apero")
        st.write("Use the sidebar to navigate through the application.")

    elif choice == "Data Analysis":
        data = pd.read_csv("data/processed/cleaned_file.csv")
        st.dataframe(data)
        
        # Define features and target variable
        X = data.drop(columns=['id', 'stroke'])  # Drop non-predictor columns
        y = data['stroke'] # predictor column

        # Convert categorical variables to dummy/indicator variables (as in the mapping)
        X = pd.get_dummies(X, drop_first=True)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # test on 20% of the dataset

        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Support Vector Machine': SVC()
        }

        # Train and evaluate each model
        for model_name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            print(y_pred) # shows how basically the entirety of the testing set shows a value of 0, even when it should be 1

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Print results
            print(f"{model_name} Accuracy: {accuracy:.4f}")
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
            print("=" * 60)  # Separator between models

        # A sample row which "should" output 1, but outputs 0
        new_row = pd.DataFrame({
            'gender': ['Female'],
            'age': [79.0],
            'hypertension': [1],
            'heart_disease': [0],
            'ever_married': ['Yes'],
            'work_type': ['Self-employed'],
            'Residence_type': ['Rural'],
            'avg_glucose_level': [174.12],
            'bmi': [24.0],
            'smoking_status': ['never smoked']
        })

        # Apply the same encoding to the new row as was applied to X
        new_row = pd.get_dummies(new_row, drop_first=True)

        # Align the new row DataFrame with the training set to ensure the same columns
        new_row = new_row.reindex(columns=X.columns, fill_value=0)

        # Use each model to predict the new row
        for model_name, model in models.items():
            new_prediction = model.predict(new_row)
            print(f"{model_name} prediction for the new row: {new_prediction[0]}")

    elif choice == "Recommendations":
        st.subheader("Personalized Recommendations")
        # TODO: Implement the recommendation system interface
        st.write("Recommendation system interface will be implemented here.")

    elif choice == "Chatbot":
        st.subheader("Chatbot Assistance")
        if "key" not in st.session_state:
            st.session_state["key"] = os.urandom(24).hex()
        rasa_server_url = os.getenv("RASA_SERVER", "http://rasa:5005/webhooks/rest/webhook")
        chatbot = Chatbot(rasa_url=rasa_server_url, session_id=st.session_state["key"])
        chatbot.run()

if __name__ == "__main__":
    main()
