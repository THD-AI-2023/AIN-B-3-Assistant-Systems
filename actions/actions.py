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

        # Load the models
        models = {}
        model_names = [
            "Logistic_Regression_real.pkl",
            "Support_Vector_Machine_real.pkl",
            "Random_Forest_real.pkl",
            "Logistic_Regression_augmented.pkl",
            "Support_Vector_Machine_augmented.pkl",
            "Random_Forest_augmented.pkl",
        ]
        missing_models = []
        for model_name in model_names:
            model_path = os.path.join("models", model_name)
            if os.path.exists(model_path):
                try:
                    models[model_name] = joblib.load(model_path)
                except Exception as e:
                    dispatcher.utter_message(text=f"Error loading model {model_name}: {e}")
            else:
                missing_models.append(model_name)

        if missing_models:
            dispatcher.utter_message(text=f"The following models are missing: {', '.join(missing_models)}.")
            return []

        # Provide correlation matrix
        try:
            correlation = data.corr().to_string()
            dispatcher.utter_message(text=f"Here are the data correlations:\n```\n{correlation}\n```")
        except Exception as e:
            dispatcher.utter_message(text=f"Error generating correlation matrix: {e}")

        # Provide descriptive statistics
        try:
            descriptive = data.describe().to_string()
            dispatcher.utter_message(text=f"Here are the descriptive statistics:\n```\n{descriptive}\n```")
        except Exception as e:
            dispatcher.utter_message(text=f"Error generating descriptive statistics: {e}")

        evaluation_path = os.path.join("models", "model_evaluations.csv")
        if os.path.exists(evaluation_path):
            try:
                evaluations = pd.read_csv(evaluation_path)
                dispatcher.utter_message(text=f"**Model Evaluations:**\n```\n{evaluations.to_string(index=False)}\n```")
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

        # Load the best performing model
        # Assuming Random Forest on augmented data is the best model
        model_path = os.path.join("models", "Random_Forest_augmented.pkl")
        if not os.path.exists(model_path):
            dispatcher.utter_message(text="Recommendation model is not available.")
            return []

        try:
            model = joblib.load(model_path)
        except Exception as e:
            dispatcher.utter_message(text=f"Error loading recommendation model: {e}")
            return []

        # Load preprocessor
        preprocessor_path = os.path.join("models", "preprocessor.pkl")
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
