import os
import json
import pandas as pd
import joblib
import logging
from typing import List, Dict, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class ActionShowDataAnalysis(Action):
    def name(self) -> str:
        return "action_show_data_analysis"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:

        # Load the filtered data
        data_path = os.path.join("data", "processed", "filtered_data.csv")
        if not os.path.exists(data_path):
            dispatcher.utter_message(text="I'm sorry, but I couldn't find the data analysis results.")
            return []

        data = pd.read_csv(data_path)
        numerical_columns = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]
        data_numeric = data[numerical_columns]

        # Compute key findings
        try:
            correlations = data_numeric.corr()
            strong_correlations = correlations['stroke'].abs().sort_values(ascending=False)
            strong_correlations = strong_correlations[strong_correlations.index != 'stroke']
            top_features = strong_correlations.head(3).index.tolist()

            message = "Based on our data analysis, the factors that most strongly correlate with stroke are "
            factors = ', '.join(top_features)
            message += f"{factors}. "

            explanations = {
                'age': "As age increases, the risk of stroke tends to increase.",
                'hypertension': "Having hypertension can increase the risk of stroke.",
                'heart_disease': "Heart disease is associated with a higher risk of stroke.",
                'avg_glucose_level': "Higher average glucose levels can increase stroke risk.",
                'bmi': "Higher BMI can be associated with increased stroke risk."
            }

            for feature in top_features:
                explanation = explanations.get(feature)
                if explanation:
                    message += f"{explanation} "

            dispatcher.utter_message(text=message.strip())

        except Exception as e:
            dispatcher.utter_message(text="I'm sorry, but I couldn't generate the data analysis summary.")

        # Provide model evaluation summaries
        evaluation_path = os.path.join("models", "data_analysis", "evaluations", "model_evaluations.csv")
        if os.path.exists(evaluation_path):
            try:
                evaluations = pd.read_csv(evaluation_path)
                best_model = evaluations.sort_values('F1_Score', ascending=False).iloc[0]
                model_name = best_model['Model'].split('.')[0]
                accuracy = best_model['Accuracy']
                f1_score = best_model['F1_Score']

                message = f"The best performing model is {model_name} with an accuracy of {accuracy:.2f} and an F1 score of {f1_score:.2f}."
                dispatcher.utter_message(text=message)

            except Exception as e:
                dispatcher.utter_message(text="I'm sorry, but I couldn't retrieve the model evaluations.")

        else:
            dispatcher.utter_message(text="Model evaluations are not available.")

        return []

class ActionGenerateRecommendation(Action):
    def name(self) -> Text:
        return "action_generate_recommendation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, any]) -> List[Dict[Text, any]]:

        # Get the session_id from the tracker
        session_id = tracker.sender_id

        # Read user data from file
        user_data_file = os.path.join("data", "user_data", f"{session_id}.json")
        try:
            with open(user_data_file, 'r') as f:
                user_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading user data: {e}")
            dispatcher.utter_message(text="I'm sorry, but I don't have your data to generate a recommendation. Please make sure you've provided your health information on the home page.")
            return []

        # Extract user data
        age = user_data.get('age')
        gender = user_data.get('gender')
        hypertension = user_data.get('hypertension')
        heart_disease = user_data.get('heart_disease')
        bmi = user_data.get('bmi')

        # Check if all necessary data is present
        if None in [age, gender, hypertension, heart_disease, bmi]:
            dispatcher.utter_message(text="Some of your health information is missing. Please ensure all fields are filled on the home page.")
            return []

        # Convert categorical data if necessary
        hypertension = int(hypertension)
        heart_disease = int(heart_disease)

        # Load the best performing model
        model_path = os.path.join("models", "data_analysis", "Random_Forest_augmented.pkl")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            dispatcher.utter_message(text="I'm sorry, but I'm unable to generate a recommendation at this time.")
            return []

        try:
            model = joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            dispatcher.utter_message(text="I'm sorry, but I'm unable to generate a recommendation at this time.")
            return []

        # Prepare input data
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [float(age)],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'avg_glucose_level': [100.0],  # Placeholder value
            'bmi': [float(bmi)],
            'ever_married': ["Yes" if float(age) >= 18 else "No"],
            'work_type': ["Private"],
            'Residence_type': ["Urban"],
            'smoking_status': ["never smoked"],
        })

        # Preprocess input data
        preprocessor_path = os.path.join("models", "data_analysis", "preprocessor.pkl")
        if not os.path.exists(preprocessor_path):
            logger.error(f"Preprocessor file not found at {preprocessor_path}")
            dispatcher.utter_message(text="I'm sorry, but I'm unable to process your data at this time.")
            return []

        try:
            preprocessor = joblib.load(preprocessor_path)
            input_data_processed = preprocessor.transform(input_data)
        except Exception as e:
            logger.error(f"Error preprocessing input data: {e}")
            dispatcher.utter_message(text="I'm sorry, but I'm unable to process your data at this time.")
            return []

        # Generate prediction
        try:
            prediction = model.predict(input_data_processed)[0]
            prediction_proba = model.predict_proba(input_data_processed)[0][1]
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            dispatcher.utter_message(text="I'm sorry, but I'm unable to generate a recommendation at this time.")
            return []

        # Provide recommendation
        if prediction == 1:
            recommendation = (
                f"Based on your data (Age: {age}, Gender: {gender}, BMI: {bmi}), you may have an elevated risk of stroke. "
                "It's important to consult a healthcare professional for personalized advice."
            )
        else:
            recommendation = (
                f"Based on your data (Age: {age}, Gender: {gender}, BMI: {bmi}), your risk of stroke appears to be low. "
                "Maintaining a healthy lifestyle can help keep it that way."
            )

        dispatcher.utter_message(text=recommendation)
        return []

class ActionExplainBMIEffect(Action):
    def name(self) -> str:
        return "action_explain_bmi_effect"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:
        bmi = tracker.get_slot("bmi")
        if bmi:
            bmi = float(bmi)
            if bmi < 18.5:
                interpretation = "underweight"
            elif 18.5 <= bmi < 25:
                interpretation = "normal weight"
            elif 25 <= bmi < 30:
                interpretation = "overweight"
            else:
                interpretation = "obese"
            message = (
                f"Your BMI is {bmi:.1f}, which is considered {interpretation}. "
                "A higher BMI can increase the risk of stroke. "
                "Maintaining a healthy BMI through balanced diet and regular exercise can help reduce this risk."
            )
        else:
            message = "I'm sorry, I don't have your BMI information. Could you please provide it?"
            return [FollowupAction(name="action_listen")]
        dispatcher.utter_message(text=message)
        return []

class ActionProvideStrokeRiskReductionAdvice(Action):
    def name(self) -> str:
        return "action_provide_stroke_risk_reduction_advice"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:
        # Provide lifestyle recommendations
        message = (
            "Here are some recommendations to help reduce your stroke risk:\n"
            "1. **Healthy Diet:** Eat plenty of fruits, vegetables, and whole grains.\n"
            "2. **Regular Exercise:** Aim for at least 150 minutes of moderate aerobic activity each week.\n"
            "3. **Avoid Smoking:** If you smoke, consider quitting.\n"
            "4. **Limit Alcohol:** Drink alcohol in moderation.\n"
            "5. **Monitor Health Indicators:** Keep an eye on blood pressure, cholesterol, and blood sugar levels.\n"
            "6. **Stress Management:** Practice relaxation techniques like meditation or yoga.\n"
            "Consult with a healthcare professional for personalized advice."
        )
        dispatcher.utter_message(text=message)
        return []

