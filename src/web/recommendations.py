import os
import json
import pandas as pd
import streamlit as st
import joblib

def run():
    st.subheader("Personalized Recommendations")
    st.write("Please enter your personal health information to receive personalized recommendations.")

    # Initialize session state for user data
    if 'user_data' not in st.session_state:
        st.session_state['user_data'] = {}

    with st.form(key='user_data_form'):
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        hypertension = st.selectbox("Do you have hypertension?", options=["No", "Yes"])
        heart_disease = st.selectbox("Do you have heart disease?", options=["No", "Yes"])
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
        ever_married = st.selectbox("Have you ever been married?", options=["No", "Yes"])
        work_type = st.selectbox("Work Type", options=['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
        Residence_type = st.selectbox("Residence Type", options=['Urban', 'Rural'])
        smoking_status = st.selectbox("Smoking Status", options=['never smoked', 'formerly smoked', 'smokes', 'Unknown'])
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.session_state['user_data'] = {
            'age': age,
            'gender': gender,
            'hypertension': 1 if hypertension == "Yes" else 0,
            'heart_disease': 1 if heart_disease == "Yes" else 0,
            'bmi': bmi,
            'avg_glucose_level': avg_glucose_level,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'smoking_status': smoking_status
        }
        st.success("Your information has been saved.")

        # Save user data to a file
        user_data_file = os.path.join("data", "user_data", f"{st.session_state['session_id']}.json")
        os.makedirs(os.path.dirname(user_data_file), exist_ok=True)
        with open(user_data_file, 'w') as f:
            json.dump(st.session_state['user_data'], f)

    # Display stored user data
    if st.session_state['user_data']:
        st.write("### Your Current Health Information:")
        st.json(st.session_state['user_data'])

        # Personalized Recommendations
        st.write("### Recommendations")
        try:
            # Load the preprocessor
            preprocessor_path = os.path.join("models", "data_analysis", "preprocessor.pkl")
            if not os.path.exists(preprocessor_path):
                st.error("Preprocessor not found. Please ensure the preprocessor is trained and available.")
            else:
                preprocessor = joblib.load(preprocessor_path)

                # Load the trained model (assuming Random Forest is the best model)
                model_path = os.path.join("models", "data_analysis", "Random_Forest_augmented.pkl")
                if not os.path.exists(model_path):
                    st.error("Recommendation model not found. Please ensure the model is trained and available.")
                else:
                    model = joblib.load(model_path)

                    # Prepare user data for prediction
                    user_data = pd.DataFrame([st.session_state['user_data']])
                    user_data_processed = preprocessor.transform(user_data)

                    # Make prediction
                    prediction = model.predict(user_data_processed)[0]
                    prediction_proba = model.predict_proba(user_data_processed)[0][1]

                    # Display recommendations based on prediction
                    if prediction == 1:
                        risk_level = "High"
                        st.markdown(f"**Risk Level:** {risk_level}")
                        st.markdown("""
                        Based on your provided information, you have a **high risk of stroke**. Here are some recommendations:

                        1. **Consult a Healthcare Professional:** It's important to seek medical advice for personalized care.
                        2. **Manage Blood Pressure:** Maintain a healthy blood pressure through diet, exercise, and medication if prescribed.
                        3. **Maintain a Healthy BMI:** Work towards achieving and maintaining a healthy Body Mass Index.
                        4. **Quit Smoking:** If you smoke, consider quitting to reduce your risk.
                        5. **Regular Physical Activity:** Engage in regular exercise to improve overall health.
                        """)
                        st.write(prediction_proba, prediction)
                    else:
                        risk_level = "Low"
                        st.markdown(f"**Risk Level:** {risk_level}")
                        st.markdown("""
                        Based on your provided information, you have a **low risk of stroke**. Here are some recommendations to maintain your health:

                        1. **Balanced Diet:** Continue to eat a balanced diet rich in fruits, vegetables, and whole grains.
                        2. **Regular Exercise:** Maintain regular physical activity to keep your BMI in a healthy range.
                        3. **Avoid Smoking:** Continue to avoid smoking to sustain your low-risk status.
                        4. **Monitor Health Indicators:** Keep an eye on your blood pressure and glucose levels.
                        5. **Stress Management:** Practice stress-reducing techniques such as meditation or yoga.
                        """)
                        st.write(prediction_proba, prediction)
        except Exception as e:
            st.error(f"An error occurred while generating recommendations: {e}")
