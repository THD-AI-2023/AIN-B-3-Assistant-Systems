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

    # ======== User Input Form ========
    with st.form(key='user_data_form'):
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        hypertension = st.selectbox("Do you have hypertension?", options=["No", "Yes"])
        heart_disease = st.selectbox("Do you have heart disease?", options=["No", "Yes"])
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
        ever_married = st.selectbox("Have you ever been married?", options=["No", "Yes"])
        work_type = st.selectbox("Work Type", options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        Residence_type = st.selectbox("Residence Type", options=["Urban", "Rural"])
        smoking_status = st.selectbox("Smoking Status", options=["never smoked", "formerly smoked", "smokes", "Unknown"])
        submitted = st.form_submit_button("Submit")

    if submitted:
        # Convert Yes/No to numeric
        hypertension_val = 1 if hypertension == "Yes" else 0
        heart_disease_val = 1 if heart_disease == "Yes" else 0

        st.session_state['user_data'] = {
            'age': age,
            'gender': gender,
            'hypertension': hypertension_val,
            'heart_disease': heart_disease_val,
            'bmi': bmi,
            'avg_glucose_level': avg_glucose_level,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'smoking_status': smoking_status
        }
        st.success("Your information has been saved.")

        # Save user data to a file for possible access by Rasa or other modules
        user_data_file = os.path.join("data", "user_data", f"{st.session_state['session_id']}.json")
        os.makedirs(os.path.dirname(user_data_file), exist_ok=True)
        with open(user_data_file, 'w') as f:
            json.dump(st.session_state['user_data'], f)

    # If we already have user data in session, show it and generate predictions
    if st.session_state['user_data']:
        st.write("### Your Current Health Information:")
        st.json(st.session_state['user_data'])

        # ========= Generate Probability-Based Recommendations =========
        st.write("### Recommendations")
        try:
            # Load the preprocessor
            preprocessor_path = os.path.join("models", "data_analysis", "preprocessor.pkl")
            if not os.path.exists(preprocessor_path):
                st.error("Preprocessor not found. Please ensure the preprocessor is trained and available.")
                return

            preprocessor = joblib.load(preprocessor_path)

            # Load a trained model (example: best-performing augmented RF)
            model_path = os.path.join("models", "data_analysis", "Random_Forest_augmented.pkl")
            if not os.path.exists(model_path):
                st.error("Recommendation model not found. Please ensure the model is trained and available.")
                return

            model = joblib.load(model_path)

            # Prepare user data for prediction
            user_data_df = pd.DataFrame([st.session_state['user_data']])
            user_data_processed = preprocessor.transform(user_data_df)

            # Use probability (predict_proba)
            stroke_probability = model.predict_proba(user_data_processed)[0][1]

            # Basic thresholding (optional)
            if stroke_probability < 0.33:
                risk_label = "Low"
            elif stroke_probability < 0.66:
                risk_label = "Moderate"
            else:
                risk_label = "High"

            # Display the stroke probability
            st.markdown(f"**Stroke Probability:** {stroke_probability:.2f} (range: 0.0 to 1.0)")
            st.markdown(f"**Risk Level:** {risk_label}")

            # Provide short recommendations
            if risk_label == "High":
                st.markdown(
                    """
                    Based on your data, you have a **high** probability of stroke.  
                    **Recommendations**:
                    1. Consult a healthcare professional for a personalized plan.  
                    2. Manage hypertension (if applicable) and blood pressure.  
                    3. Adopt a healthier diet and regular exercise regimen.  
                    4. Quit smoking if you smoke.  
                    """
                )
            elif risk_label == "Moderate":
                st.markdown(
                    """
                    You have a **moderate** risk of stroke.  
                    **Recommendations**:
                    1. Maintain a balanced diet and moderate exercise.  
                    2. Monitor blood pressure and cholesterol levels.  
                    3. Follow up with healthcare providers for routine checks.  
                    """
                )
            else:
                st.markdown(
                    """
                    Your stroke risk probability is **relatively low**.  
                    **Recommendations**:
                    1. Balanced diet with plenty of fruits and vegetables.  
                    2. Regular physical activity.  
                    3. Avoid smoking and reduce alcohol consumption.  
                    """
                )

        except Exception as e:
            st.error(f"An error occurred while generating recommendations: {e}")
