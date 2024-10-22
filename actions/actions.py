from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import pandas as pd
import joblib
import os

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

        # TODO: Implement logic to generate recommendations based on user data
        # Example:
        # Load the trained model
        model_path = os.path.join("models", "recommendation_model.pkl")
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

        # Generate prediction
        prediction = model.predict(input_data)[0]
        recommendation = "Based on your data, we recommend the following services: ..."

        dispatcher.utter_message(text=recommendation)
        return []

class ActionShowDataAnalysis(Action):
    def name(self) -> Text:
        return "action_show_data_analysis"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # TODO: Implement logic to fetch and display data analysis results
        # Example:
        # Load preprocessed data
        data_path = os.path.join("data", "processed", "processed_data.csv")
        if not os.path.exists(data_path):
            dispatcher.utter_message(text="Data analysis results are not available.")
            return []

        data = pd.read_csv(data_path)

        # Provide correlation matrix
        correlation = data.corr().to_string()
        dispatcher.utter_message(text=f"Here are the data correlations:\n```\n{correlation}\n```")

        # Provide descriptive statistics
        descriptive = data.describe().to_string()
        dispatcher.utter_message(text=f"Here are the descriptive statistics:\n```\n{descriptive}\n```")

        return []
