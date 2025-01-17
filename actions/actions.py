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
    """Extract the user's name from the message and set the 'name' slot."""

    def name(self) -> str:
        return "action_save_name"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        name_entity = next(tracker.get_latest_entity_values("name"), None)
        if name_entity:
            return [SlotSet("name", name_entity)]
        dispatcher.utter_message(
            text="I didn't catch your name. Could you please repeat it?"
        )
        return []


class ActionSaveHealthInfo(Action):
    """
    Extract partial user health info from 'inform' or from any user message 
    containing recognized age, gender, BMI, etc., then store or update the user_data JSON.
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

        # Parse recognized entities
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
                # ‘yes’ => 1, ‘no’ => 0
                if ent["value"] == "yes":
                    hypertension = 1
                elif ent["value"] == "no":
                    hypertension = 0
            elif ent["entity"] == "heart_disease":
                if ent["value"] == "yes":
                    heart_disease = 1
                elif ent["value"] == "no":
                    heart_disease = 0
            elif ent["entity"] == "bmi":
                bmi = ent["value"]

        # Load existing data if any
        if os.path.exists(user_data_file):
            try:
                with open(user_data_file, "r") as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = {}
        else:
            existing_data = {}

        # Update or add new fields
        if age is not None:
            existing_data["age"] = float(age)
        if gender is not None:
            existing_data["gender"] = gender
        if hypertension is not None:
            existing_data["hypertension"] = hypertension
        if heart_disease is not None:
            existing_data["heart_disease"] = heart_disease
        if bmi is not None:
            existing_data["bmi"] = float(bmi)

        # Save updated info
        try:
            os.makedirs(os.path.dirname(user_data_file), exist_ok=True)
            with open(user_data_file, "w") as f:
                json.dump(existing_data, f)
            dispatcher.utter_message(text="Your health info has been saved/updated.")
        except Exception as e:
            logger.error(f"Error saving user data: {e}")
            dispatcher.utter_message(
                text="Sorry, I couldn't save your info due to an internal error."
            )

        return []


class ActionConfirmOverwrite(Action):
    """(Optional) If you track user data overwrite confirmations—currently not used in this revision."""

    def name(self) -> str:
        return "action_confirm_overwrite"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        # Not strictly necessary in this revised version
        return []


class ActionCancelOverwrite(Action):
    """(Optional) If user declines overwriting. Not used in this revision."""

    def name(self) -> str:
        return "action_cancel_overwrite"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        return []


class ActionShowDataAnalysis(Action):
    """Summarizes correlation results and best model metrics."""

    def name(self) -> str:
        return "action_show_data_analysis"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        # Load data
        data_path = os.path.join("data", "processed", "filtered_data.csv")
        if not os.path.exists(data_path):
            dispatcher.utter_message(
                text="I'm sorry, but I couldn't find the data analysis results."
            )
            return []

        try:
            data = pd.read_csv(data_path)
            numerical_cols = [
                "age", "hypertension", "heart_disease", 
                "avg_glucose_level", "bmi", "stroke"
            ]
            corr = data[numerical_cols].corr()
            stroke_correlations = corr["stroke"].drop("stroke")
            # sort by absolute correlation
            strong_corr = stroke_correlations.abs().sort_values(ascending=False)
            top_feats = strong_corr.head(3).index.tolist()

            # Build message
            message = (
                "Based on our data analysis, the factors that most strongly correlate with stroke are:\n\n"
            )
            for feat in top_feats:
                corr_val = stroke_correlations[feat]
                feat_display = feat.replace("_", " ").title()
                message += f"- **{feat_display}** with a correlation coefficient of {corr_val:.2f}\n"

            # Explanations only for those top feats
            explanation_map = {
                "age": "As age increases, the risk of stroke tends to increase.",
                "heart_disease": "Heart disease is associated with a higher risk of stroke.",
                "hypertension": "Having hypertension can increase the risk of stroke.",
                "bmi": "As BMI increases, stroke risk can also increase.",
                "avg_glucose_level": "Higher glucose levels may indicate diabetes or other conditions affecting stroke risk.",
            }
            for feat in top_feats:
                if feat in explanation_map:
                    message += explanation_map[feat] + "\n"

            dispatcher.utter_message(message.strip())

        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            dispatcher.utter_message(
                text="I'm sorry, but I couldn't generate the data analysis summary."
            )
            return []

        # Provide best model info
        eval_path = os.path.join("models", "data_analysis", "evaluations", "model_evaluations.csv")
        if os.path.exists(eval_path):
            try:
                df_eval = pd.read_csv(eval_path)
                df_eval = df_eval.sort_values("F1_Score", ascending=False)
                best = df_eval.iloc[0]
                model_name = best["Model"].split(".")[0].replace("_", " ").title()
                acc = best["Accuracy"]
                f1 = best["F1_Score"]
                msg = (
                    f"The best performing model is **{model_name}** "
                    f"with an accuracy of **{acc:.2f}** and an F1 score of **{f1:.2f}**."
                )
                dispatcher.utter_message(text=msg)
            except Exception as e:
                logger.error(f"Error reading model evaluations: {e}")
                dispatcher.utter_message(
                    text="I'm sorry, but I couldn't retrieve the model evaluations."
                )
        else:
            dispatcher.utter_message(text="Model evaluations are not available.")

        return []


class ActionExplainCorrelation(Action):
    """
    Provides explanations for how a given factor (e.g., 'hypertension', 'heart_disease')
    correlates with stroke, along with correlation coefficients.
    """

    def name(self) -> str:
        return "action_explain_correlation"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        # OPTIONAL: If you want to handle two factors -> correlation with each other
        # check how many 'factor' entities were extracted
        factors = tracker.latest_message.get("entities", [])
        # If user asked about 2 different factors, do some extra logic, else fallback
        if len(factors) >= 2:
            dispatcher.utter_message(
                text=(
                    "I currently provide factor-to-stroke correlation, not factor-to-factor. "
                    "Try: 'How does hypertension affect stroke risk?'"
                )
            )
            return []

        factor = next(tracker.get_latest_entity_values("factor"), None)
        if not factor:
            dispatcher.utter_message(
                text="Which factor are you curious about? (e.g. 'hypertension')"
            )
            return []

        # normalizing factor
        factor_lower = factor.lower().strip()
        correlation_info = {
            "hypertension": {
                "coef": 0.12,
                "explanation": (
                    "Having hypertension increases stress on blood vessels, "
                    "leading to an elevated risk of stroke."
                ),
            },
            "heart disease": {
                "coef": 0.13,
                "explanation": (
                    "Heart disease is associated with a higher stroke risk."
                ),
            },
            "bmi": {
                "coef": 0.10,
                "explanation": (
                    "A higher BMI may indicate overweight or obesity, "
                    "which contributes to increased stroke risk."
                ),
            },
            "age": {
                "coef": 0.24,
                "explanation": (
                    "As age increases, the risk of stroke tends to increase."
                ),
            },
            "avg_glucose_level": {
                "coef": 0.15,
                "explanation": (
                    "Elevated glucose levels can indicate diabetes or pre-diabetes, "
                    "which increases stroke risk over time."
                ),
            },
        }

        # a quick normalization if user typed "heart_disease"
        if factor_lower in ["heart_disease", "heart disease"]:
            factor_lower = "heart disease"

        if factor_lower not in correlation_info:
            dispatcher.utter_message(
                text=(
                    "I don't have correlation info for that factor. "
                    "Try 'hypertension', 'heart disease', 'bmi', 'age', or 'avg_glucose_level'."
                )
            )
            return []

        coef = correlation_info[factor_lower]["coef"]
        explanation = correlation_info[factor_lower]["explanation"]
        message = (
            f"{explanation} In our dataset, '{factor_lower}' showed a correlation "
            f"coefficient of about {coef:.2f} with stroke."
        )
        dispatcher.utter_message(text=message)
        return []


class ActionGenerateRecommendation(Action):
    """
    Generates a health recommendation based on user-provided data.
    Checks data in user_data/<session_id>.json, loads a model, 
    then outputs stroke probability with a relevant message.
    """

    def name(self) -> Text:
        return "action_generate_recommendation"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        session_id = tracker.sender_id
        user_data_file = os.path.join("data", "user_data", f"{session_id}.json")

        if not os.path.exists(user_data_file):
            dispatcher.utter_message(
                text=(
                    "I don't have enough data on file. "
                    "Please provide your age, gender, bmi, etc. in the Personalized Recommendations form."
                )
            )
            return []

        # Load user data
        try:
            with open(user_data_file, "r") as f:
                user_data = json.load(f)
        except Exception as e:
            dispatcher.utter_message(
                text="I couldn't read your health info. Please re-enter your data."
            )
            return []

        needed_fields = [
            "age", "gender", "hypertension", "heart_disease", 
            "bmi", "avg_glucose_level", "ever_married", 
            "work_type", "Residence_type", "smoking_status"
        ]
        missing = [fld for fld in needed_fields if fld not in user_data]
        if missing:
            dispatcher.utter_message(
                text=(
                    f"I'm missing some details: {missing}. "
                    "Please fill them out or provide them in a message, e.g. 'I am 45 with BMI 26...'"
                )
            )
            return []

        model_path = os.path.join("models", "data_analysis", "Random_Forest_augmented.pkl")
        preprocessor_path = os.path.join("models", "data_analysis", "preprocessor.pkl")
        if not os.path.exists(model_path):
            dispatcher.utter_message(
                text="No trained model found. Please train one first."
            )
            return []
        if not os.path.exists(preprocessor_path):
            dispatcher.utter_message(
                text="Preprocessor not found. Please ensure it is trained."
            )
            return []

        try:
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
        except Exception as e:
            dispatcher.utter_message(
                text="I encountered an error loading the model."
            )
            return []

        # Build input
        input_data = pd.DataFrame([{
            "age": float(user_data["age"]),
            "gender": user_data["gender"],
            "hypertension": int(user_data["hypertension"]),
            "heart_disease": int(user_data["heart_disease"]),
            "avg_glucose_level": float(user_data["avg_glucose_level"]),
            "bmi": float(user_data["bmi"]),
            "ever_married": user_data["ever_married"],
            "work_type": user_data["work_type"],
            "Residence_type": user_data["Residence_type"],
            "smoking_status": user_data["smoking_status"]
        }])

        # Transform and predict
        try:
            input_data_processed = preprocessor.transform(input_data)
            stroke_prob = model.predict_proba(input_data_processed)[0][1]
        except Exception as e:
            dispatcher.utter_message(
                text="Sorry, I had trouble computing your stroke probability."
            )
            return []

        # Determine risk
        if stroke_prob < 0.33:
            risk_label = "low"
        elif stroke_prob < 0.66:
            risk_label = "moderate"
        else:
            risk_label = "high"

        # Return personalized text
        message = (
            f"Based on your data (Age: {user_data['age']}, Gender: {user_data['gender']}, BMI: {user_data['bmi']}), "
            f"your estimated stroke probability is {stroke_prob:.2f}. "
            f"This corresponds to a '{risk_label}' risk level."
        )
        dispatcher.utter_message(text=message)

        if risk_label == "high":
            dispatcher.utter_message(
                text=(
                    "You seem to have a high risk of stroke. Would you like some tips on reducing this risk?"
                )
            )
            return [SlotSet("risk_level", "high")]

        return []


class ActionProvideStrokeRiskReductionAdvice(Action):
    def name(self) -> str:
        return "action_provide_stroke_risk_reduction_advice"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict
    ) -> list:
        message = (
            "Here are some recommendations to help reduce your stroke risk:\n\n"
            "1. **Healthy Diet:** Eat plenty of fruits, vegetables, and whole grains.\n"
            "2. **Regular Exercise:** Aim for at least 150 minutes of moderate aerobic activity each week.\n"
            "3. **Avoid Smoking:** If you smoke, consider quitting.\n"
            "4. **Limit Alcohol:** Drink alcohol in moderation.\n"
            "5. **Monitor Health Indicators:** Keep an eye on blood pressure, cholesterol, and blood sugar.\n"
            "6. **Stress Management:** Practice relaxation techniques like meditation or yoga.\n\n"
            "Always consult with a healthcare professional for personalized advice."
        )
        dispatcher.utter_message(text=message)
        return []


class ActionFallback(Action):
    def name(self) -> str:
        return "action_default_fallback"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict
    ) -> list:
        dispatcher.utter_message(
            text="I'm sorry, I didn't understand that. Could you please rephrase?"
        )
        return []
