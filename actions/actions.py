import os
import json
import pandas as pd
import joblib
import logging
from typing import List, Dict, Text, Any
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class ActionShowDataAnalysis(Action):
    def name(self) -> str:
        return "action_show_data_analysis"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # Load the filtered data
        data_path = os.path.join("data", "processed", "filtered_data.csv")
        if not os.path.exists(data_path):
            dispatcher.utter_message(
                text="I'm sorry, but I couldn't find the data analysis results."
            )
            return []

        data = pd.read_csv(data_path)
        numerical_columns = [
            "age",
            "hypertension",
            "heart_disease",
            "avg_glucose_level",
            "bmi",
            "stroke",
        ]
        data_numeric = data[numerical_columns]

        # Compute key findings
        try:
            correlations = data_numeric.corr()
            stroke_correlations = correlations["stroke"].drop("stroke")
            strong_correlations = stroke_correlations.abs().sort_values(ascending=False)
            top_features = strong_correlations.head(3).index.tolist()

            message = "Based on our data analysis, the factors that most strongly correlate with stroke are:\n"

            for feature in top_features:
                corr_value = stroke_correlations[feature]
                explanation = f"- **{feature.capitalize()}** with a correlation coefficient of {corr_value:.2f}\n"
                message += explanation

            explanations = {
                "age": "As age increases, the risk of stroke tends to increase.",
                "hypertension": "Having hypertension can increase the risk of stroke.",
                "heart_disease": "Heart disease is associated with a higher risk of stroke.",
                "avg_glucose_level": "Higher average glucose levels can increase stroke risk.",
                "bmi": "Higher BMI can be associated with increased stroke risk.",
            }

            for feature in top_features:
                explanation = explanations.get(feature)
                if explanation:
                    message += f"{explanation}\n"

            dispatcher.utter_message(text=message.strip())

        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            dispatcher.utter_message(
                text="I'm sorry, but I couldn't generate the data analysis summary."
            )
            return []

        # Provide model evaluation summaries
        evaluation_path = os.path.join(
            "models", "data_analysis", "evaluations", "model_evaluations.csv"
        )
        if os.path.exists(evaluation_path):
            try:
                evaluations = pd.read_csv(evaluation_path)
                evaluations = evaluations.drop_duplicates(subset=['Model', 'Data_Type'], keep='last')

                best_model = evaluations.sort_values("F1_Score", ascending=False).iloc[0]
                model_name = (
                    best_model["Model"].split(".")[0].replace("_", " ").title()
                )
                accuracy = best_model["Accuracy"]
                f1_score = best_model["F1_Score"]

                message = (
                    f"The best performing model is **{model_name}** "
                    f"with an accuracy of **{accuracy:.2f}** and an F1 score of **{f1_score:.2f}**."
                )
                dispatcher.utter_message(text=message)

            except Exception as e:
                logger.error(f"Error reading model evaluations: {e}")
                dispatcher.utter_message(
                    text="I'm sorry, but I couldn't retrieve the model evaluations."
                )
        else:
            dispatcher.utter_message(text="Model evaluations are not available.")

        return []

class ActionGenerateRecommendation(Action):
    def name(self) -> Text:
        return "action_generate_recommendation"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # Get the session_id from the tracker
        session_id = tracker.sender_id

        # Read user data from file
        user_data_file = os.path.join("data", "user_data", f"{session_id}.json")
        try:
            with open(user_data_file, "r") as f:
                user_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading user data: {e}")
            dispatcher.utter_message(
                text=(
                    "I'm sorry, but I don't have your data to generate a recommendation. "
                    "Please make sure you've provided your health information on the home page."
                )
            )
            return []

        # Extract user data
        age = user_data.get("age")
        gender = user_data.get("gender")
        hypertension = user_data.get("hypertension")
        heart_disease = user_data.get("heart_disease")
        bmi = user_data.get("bmi")
        avg_glucose_level = user_data.get("avg_glucose_level")
        ever_married = user_data.get("ever_married")
        work_type = user_data.get("work_type")
        Residence_type = user_data.get("Residence_type")
        smoking_status = user_data.get("smoking_status")

        # Check if all necessary data is present
        if None in [age, gender, hypertension, heart_disease, bmi, avg_glucose_level, ever_married, work_type, Residence_type, smoking_status]:
            dispatcher.utter_message(
                text=(
                    "Some of your health information is missing. "
                    "Please ensure all fields are filled on the home page."
                )
            )
            return []

        # Convert categorical data if necessary
        hypertension = int(hypertension)
        heart_disease = int(heart_disease)

        # Load the best performing model
        model_path = os.path.join(
            "models", "data_analysis", "Logistic_Regression_real.pkl"
        )
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            dispatcher.utter_message(
                text="I'm sorry, but I'm unable to generate a recommendation at this time."
            )
            return []

        try:
            model = joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            dispatcher.utter_message(
                text="I'm sorry, but I'm unable to generate a recommendation at this time."
            )
            return []

        # Prepare input data
        input_data = pd.DataFrame(
            {
                "gender": [gender],
                "age": [float(age)],
                "hypertension": [hypertension],
                "heart_disease": [heart_disease],
                "avg_glucose_level": [float(avg_glucose_level)],
                "bmi": [float(bmi)],
                "ever_married": [ever_married],
                "work_type": [work_type],
                "Residence_type": [Residence_type],
                "smoking_status": [smoking_status],
            }
        )

        # Preprocess input data
        preprocessor_path = os.path.join(
            "models", "data_analysis", "preprocessor.pkl"
        )
        if not os.path.exists(preprocessor_path):
            logger.error(f"Preprocessor file not found at {preprocessor_path}")
            dispatcher.utter_message(
                text="I'm sorry, but I'm unable to process your data at this time."
            )
            return []

        try:
            preprocessor = joblib.load(preprocessor_path)
            input_data_processed = preprocessor.transform(input_data)
        except Exception as e:
            logger.error(f"Error preprocessing input data: {e}")
            dispatcher.utter_message(
                text="I'm sorry, but I'm unable to process your data at this time."
            )
            return []

        # Generate prediction
        try:
            prediction = model.predict(input_data_processed)[0]
            prediction_proba = model.predict_proba(input_data_processed)[0][1]
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            dispatcher.utter_message(
                text="I'm sorry, but I'm unable to generate a recommendation at this time."
            )
            return []

        # Provide recommendation
        if prediction == 1:
            risk_level = "high"
            recommendation = (
                f"Based on your data (Age: {age}, Gender: {gender}, BMI: {bmi}), you may have an elevated risk of stroke. "
                "It's important to consult a healthcare professional for personalized advice."
            )
        else:
            risk_level = "low"
            recommendation = (
                f"Based on your data (Age: {age}, Gender: {gender}, BMI: {bmi}), your risk of stroke appears to be low. "
                "Maintaining a healthy lifestyle can help keep it that way."
            )

        # Additional advice
        dispatcher.utter_message(text=recommendation)

        if risk_level == "high":
            dispatcher.utter_message(
                text="Would you like some tips on how to reduce your stroke risk?"
            )
            return [SlotSet("risk_level", risk_level)]
        else:
            return []

class ActionProvideStrokeRiskReductionAdvice(Action):
    def name(self) -> str:
        return "action_provide_stroke_risk_reduction_advice"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: dict,
    ) -> list:
        # Provide lifestyle recommendations
        message = (
            "Here are some recommendations to help reduce your stroke risk:\n"
            "1. **Healthy Diet:** Eat plenty of fruits, vegetables, and whole grains.\n"
            "2. **Regular Exercise:** Aim for at least 150 minutes of moderate aerobic activity each week.\n"
            "3. **Avoid Smoking:** If you smoke, consider quitting.\n"
            "4. **Limit Alcohol:** Drink alcohol in moderation.\n"
            "5. **Monitor Health Indicators:** Keep an eye on blood pressure, cholesterol, and blood sugar levels.\n"
            "6. **Stress Management:** Practice relaxation techniques like meditation or yoga.\n"
            "Always consult with a healthcare professional for personalized advice."
        )
        dispatcher.utter_message(text=message)
        return []

class ActionFallback(Action):
    def name(self) -> str:
        return "action_default_fallback"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: dict,
    ) -> list:
        dispatcher.utter_message(
            text="I'm sorry, I didn't understand that. Could you please rephrase?"
        )
        return []
