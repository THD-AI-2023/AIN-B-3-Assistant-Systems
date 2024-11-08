from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd
import os

class ActionShowDataAnalysis(Action):
    def name(self) -> Text:
        return "action_show_data_analysis"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Load the filtered data
        data_path = os.path.join("data", "processed", "filtered_data.csv")
        if not os.path.exists(data_path):
            dispatcher.utter_message(text="Filtered data is not available.")
            return []

        data = pd.read_csv(data_path)

        # Provide correlation matrix
        correlation = data.corr().to_string()
        dispatcher.utter_message(text=f"Here are the data correlations:\n```\n{correlation}\n```")

        # Provide descriptive statistics
        descriptive = data.describe().to_string()
        dispatcher.utter_message(text=f"Here are the descriptive statistics:\n```\n{descriptive}\n```")

        return []

class ActionGenerateRecommendation(Action):
    def name(self) -> Text:
        return "action_generate_recommendation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Extract slots
        age = tracker.get_slot("age")
        hypertension = tracker.get_slot("hypertension")
        gender = tracker.get_slot("gender")
        bmi = tracker.get_slot("bmi")
        heart_disease = tracker.get_slot("heart_disease")

        # Load the trained model
        model_path = os.path.join("models", "best_model.pkl")
        if not os.path.exists(model_path):
            dispatcher.utter_message(text="Recommendation model is not available.")
            return []

        model = joblib.load(model_path)

        # Create input DataFrame
        input_data = pd.DataFrame({
            'age': [age],
            'hypertension': [hypertension],
            'gender': [gender],
            'bmi': [bmi],
            'heart_disease': [heart_disease]
        })

        # Preprocess input data
        # Load the preprocessor
        preprocessor_path = os.path.join("models", "preprocessor.pkl")
        if not os.path.exists(preprocessor_path):
            dispatcher.utter_message(text="Preprocessor is not available.")
            return []

        preprocessor = joblib.load(preprocessor_path)
        input_data_processed = preprocessor.transform(input_data)

        # Generate prediction
        prediction = model.predict(input_data_processed)[0]
        recommendation = f"Based on your data, we predict a stroke risk of {prediction}. Please consult a healthcare professional for personalized advice."

        dispatcher.utter_message(text=recommendation)
        return []
