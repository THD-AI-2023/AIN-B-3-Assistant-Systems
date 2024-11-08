import os
import pandas as pd
import joblib
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionShowDataAnalysis(Action):
    def name(self) -> str:
        return "action_show_data_analysis"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:

        # Load the filtered data
        data_path = os.path.join("data", "processed", "filtered_data.csv")
        if not os.path.exists(data_path):
            dispatcher.utter_message(text="Filtered data is not available.")
            return []

        data = pd.read_csv(data_path)

        # Select only numerical columns for correlation
        numerical_columns = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]
        data_numeric = data[numerical_columns]

        # Compute correlation matrix
        try:
            correlation = data_numeric.corr().to_string()
            dispatcher.utter_message(text=f"Here are the data correlations:\n```\n{correlation}\n```")
        except Exception as e:
            dispatcher.utter_message(text=f"Error generating correlation matrix: {e}")

        # Provide descriptive statistics
        try:
            descriptive = data_numeric.describe().to_string()
            dispatcher.utter_message(text=f"Here are the descriptive statistics:\n```\n{descriptive}\n```")
        except Exception as e:
            dispatcher.utter_message(text=f"Error generating descriptive statistics: {e}")

        # Provide model evaluation summaries if available
        evaluation_path = os.path.join("models", "data_analysis", "evaluations", "model_evaluations.csv")
        if os.path.exists(evaluation_path):
            try:
                evaluations = pd.read_csv(evaluation_path)
                evaluations_str = evaluations.to_string(index=False)
                dispatcher.utter_message(text=f"**Model Evaluations:**\n```\n{evaluations_str}\n```")
            except Exception as e:
                dispatcher.utter_message(text=f"Error loading model evaluations: {e}")
        else:
            dispatcher.utter_message(text="Model evaluations are not available.")

        return []

class ActionGenerateRecommendation(Action):
    def name(self) -> str:
        return "action_generate_recommendation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:

        # Extract slots
        age = tracker.get_slot("age")
        hypertension = tracker.get_slot("hypertension")
        gender = tracker.get_slot("gender")
        bmi = tracker.get_slot("bmi")
        heart_disease = tracker.get_slot("heart_disease")

        # Validate extracted slots
        if None in [age, hypertension, gender, bmi, heart_disease]:
            dispatcher.utter_message(text="I'm missing some information to provide a recommendation. Please provide all required details.")
            return []

        # Load the best performing data analysis model
        # Assuming Random Forest on augmented data is the best model
        model_path = os.path.join("models", "data_analysis", "Random_Forest_augmented.pkl")
        if not os.path.exists(model_path):
            dispatcher.utter_message(text="Recommendation model is not available.")
            return []

        try:
            model = joblib.load(model_path)
        except Exception as e:
            dispatcher.utter_message(text=f"Error loading recommendation model: {e}")
            return []

        # Load preprocessor
        preprocessor_path = os.path.join("models", "data_analysis", "preprocessor.pkl")
        if not os.path.exists(preprocessor_path):
            dispatcher.utter_message(text="Preprocessor is not available.")
            return []

        try:
            preprocessor = joblib.load(preprocessor_path)
        except Exception as e:
            dispatcher.utter_message(text=f"Error loading preprocessor: {e}")
            return []

        # Create input DataFrame based on available slots
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'avg_glucose_level': [100.0],  # Placeholder, need to collect or set default
            'bmi': [bmi],
            'ever_married': ["Yes"],  # Default value or collect from user
            'work_type': ["Private"],  # Default value or collect from user
            'Residence_type': ["Urban"],  # Default value or collect from user
            'smoking_status': ["never smoked"],  # Default value or collect from user
        })

        # Preprocess the input data
        try:
            input_data_processed = preprocessor.transform(input_data)
        except Exception as e:
            dispatcher.utter_message(text=f"Error preprocessing input data: {e}")
            return []

        # Generate prediction
        try:
            prediction = model.predict(input_data_processed)[0]
            prediction_proba = model.predict_proba(input_data_processed)[0][1]
        except Exception as e:
            dispatcher.utter_message(text=f"Error generating prediction: {e}")
            return []

        # Generate recommendation based on prediction
        if prediction == 1:
            recommendation = (
                f"Based on your data, we predict a high risk of stroke ({prediction_proba*100:.2f}%). "
                "Please consult a healthcare professional for personalized advice."
            )
        else:
            recommendation = (
                f"Based on your data, we predict a low risk of stroke ({(1-prediction_proba)*100:.2f}%). "
                "Continue maintaining a healthy lifestyle."
            )

        dispatcher.utter_message(text=recommendation)
        return []
