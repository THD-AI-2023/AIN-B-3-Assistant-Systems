import os
import json
import pandas as pd
import joblib
import logging
from typing import List, Dict, Text, Any
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class ActionSaveName(Action):
    """Custom action to extract the user's name from the message and set the 'name' slot."""

    def name(self) -> str:
        return "action_save_name"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Retrieve the recognized 'name' entity (if any)
        name_entity = next(tracker.get_latest_entity_values("name"), None)

        if name_entity:
            return [SlotSet("name", name_entity)]
        else:
            # If no name was recognized, inform the user
            dispatcher.utter_message(text="I didn't catch your name. Could you please repeat it?")
            return []


class ActionSaveHealthInfo(Action):
    """
    Extract partial user health info from 'inform' intent, check if user data file
    already exists, and if so, ask for overwrite confirmation.
    """

    def name(self) -> str:
        return "action_save_health_info"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        session_id = tracker.sender_id
        user_data_file = os.path.join("data", "user_data", f"{session_id}.json")

        # Parse recognized entities from the last user message
        age = None
        gender = None
        hypertension = None
        heart_disease = None
        bmi = None

        for ent in tracker.latest_message.get("entities", []):
            if ent["entity"] == "age":
                age = ent["value"]
            elif ent["entity"] == "gender":
                gender = ent["value"]
            elif ent["entity"] == "hypertension":
                hypertension = 1 if ent["value"] == "yes" else 0
            elif ent["entity"] == "heart_disease":
                heart_disease = 1 if ent["value"] == "yes" else 0
            elif ent["entity"] == "bmi":
                bmi = ent["value"]

        # Build a partial dict with new data
        updated_data = {}
        if age is not None:
            updated_data["age"] = float(age)
        if gender is not None:
            updated_data["gender"] = gender
        if hypertension is not None:
            updated_data["hypertension"] = hypertension
        if heart_disease is not None:
            updated_data["heart_disease"] = heart_disease
        if bmi is not None:
            updated_data["bmi"] = float(bmi)

        if not updated_data:
            dispatcher.utter_message(
                text="I didn't detect any new health information to save."
            )
            return []

        # Check if user_data_file already exists
        if os.path.exists(user_data_file):
            # We have existing data, ask for overwrite confirmation
            dispatcher.utter_message(response="utter_ask_overwrite_data")
            return [
                SlotSet("pending_update", True),
                SlotSet("user_new_data", updated_data),
            ]
        else:
            # No file => directly save the info
            try:
                with open(user_data_file, "w") as f:
                    json.dump(updated_data, f)
                dispatcher.utter_message(text="Your health info has been saved!")
            except Exception as e:
                logger.error(f"Error saving user data: {e}")
                dispatcher.utter_message(
                    text="Sorry, I couldn't save your info due to an error."
                )
            return []


class ActionConfirmOverwrite(Action):
    """
    Overwrites the existing user_data.json with the new partial info
    stored in the 'user_new_data' slot.
    """

    def name(self) -> str:
        return "action_confirm_overwrite"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        session_id = tracker.sender_id
        user_data_file = os.path.join("data", "user_data", f"{session_id}.json")

        new_data = tracker.get_slot("user_new_data")
        if not new_data:
            dispatcher.utter_message(text="I have no new data to save, sorry.")
            return [SlotSet("pending_update", False), SlotSet("user_new_data", None)]

        # Attempt to read existing data
        try:
            with open(user_data_file, "r") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = {}

        # Overwrite or merge existing data
        existing_data.update(new_data)

        # Save
        try:
            with open(user_data_file, "w") as f:
                json.dump(existing_data, f)
            dispatcher.utter_message(response="utter_overwrite_confirmed")
        except Exception as e:
            logger.error(f"Error overwriting user data: {e}")
            dispatcher.utter_message(
                text="I encountered an error while overwriting your information."
            )

        return [
            SlotSet("pending_update", False),
            SlotSet("user_new_data", None),
        ]


class ActionCancelOverwrite(Action):
    """
    Cancels overwriting existing data. We do nothing but reset relevant slots.
    """

    def name(self) -> str:
        return "action_cancel_overwrite"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_overwrite_cancelled")
        return [
            SlotSet("pending_update", False),
            SlotSet("user_new_data", None),
        ]


class ActionSaveName(Action):
    """Custom action to extract the user's name from the message and set the 'name' slot."""

    def name(self) -> str:
        return "action_save_name"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Retrieve the recognized 'name' entity (if any)
        name_entity = next(tracker.get_latest_entity_values("name"), None)

        if name_entity:
            return [SlotSet("name", name_entity)]
        else:
            # If no name was recognized, inform the user
            dispatcher.utter_message(text="I didn't catch your name. Could you please repeat it?")
            return []

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

            message = (
                "Based on our data analysis, the factors that most strongly "
                "correlate with stroke are:\n"
            )

            for feature in top_features:
                corr_value = stroke_correlations[feature]
                explanation = (
                    f"- **{feature.capitalize()}** with a correlation coefficient "
                    f"of {corr_value:.2f}\n"
                )
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
                # Remove duplicated rows on (Model, Data_Type), keep the last occurrence
                evaluations = evaluations.drop_duplicates(
                    subset=["Model", "Data_Type"], keep="last"
                )

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
        if None in [
            age,
            gender,
            hypertension,
            heart_disease,
            bmi,
            avg_glucose_level,
            ever_married,
            work_type,
            Residence_type,
            smoking_status,
        ]:
            dispatcher.utter_message(
                text=(
                    "Some of your health information is missing. "
                    "Please ensure all fields are filled on the home page."
                )
            )
            return []

        # Convert categorical data if necessary
        try:
            hypertension = int(hypertension)
            heart_disease = int(heart_disease)
        except ValueError:
            dispatcher.utter_message(text="Invalid numeric value in user data.")
            return []

        # Load the best performing model (example path)
        # NOTE: Adjust to your best model's name if different
        model_path = os.path.join(
            "models", "data_analysis", "Logistic_Regression_real.pkl"
        )
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            dispatcher.utter_message(
                text="I'm sorry, but I'm unable to generate a recommendation at this time."
            )
            return []

        # Load the preprocessor
        preprocessor_path = os.path.join("models", "data_analysis", "preprocessor.pkl")
        if not os.path.exists(preprocessor_path):
            logger.error(f"Preprocessor file not found at {preprocessor_path}")
            dispatcher.utter_message(
                text="I'm sorry, but I'm unable to process your data at this time."
            )
            return []

        try:
            # Load model and preprocessor
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)

            # Prepare input data
            input_data = pd.DataFrame({
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
            })

            # Transform via preprocessor
            input_data_processed = preprocessor.transform(input_data)

            # Use probability for stroke
            stroke_probability = model.predict_proba(input_data_processed)[0][1]

            # Threshold logic (optional)
            if stroke_probability < 0.33:
                risk_label = "low"
            elif stroke_probability < 0.66:
                risk_label = "moderate"
            else:
                risk_label = "high"

            # Provide a response with the probability
            message = (
                f"Your estimated stroke probability is {stroke_probability:.2f}. "
                f"This corresponds to a '{risk_label}' risk level."
            )
            dispatcher.utter_message(text=message)

            # Optionally offer advice if risk is high
            if risk_label == "high":
                dispatcher.utter_message(
                    text="Would you like some tips on how to reduce your stroke risk?"
                )
                return [SlotSet("risk_level", risk_label)]
            else:
                return []

        except Exception as e:
            logger.error(f"Error generating probability-based recommendation: {e}")
            dispatcher.utter_message(
                text="I'm sorry, but I'm unable to generate a recommendation at this time."
            )
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
