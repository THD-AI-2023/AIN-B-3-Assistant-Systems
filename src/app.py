import os
import json
import pandas as pd
import streamlit as st
from chatbot.rasa_chatbot import Chatbot
from data.data_analysis import DataAnalysis

def main():
    st.set_page_config(page_title="Assistance Systems Project", layout="wide")

    st.title("Assistance Systems Project")

    # Sidebar for navigation
    menu = ["Home", "Data Analysis", "Recommendations", "Chatbot"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Initialize DataAnalysis instance in session state
    if "data_analysis" not in st.session_state:
        st.session_state["data_analysis"] = DataAnalysis()

    data_analysis = st.session_state["data_analysis"]

    # Initialize session ID
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = os.urandom(24).hex()

    # Display saved data analysis model files in the sidebar
    with st.sidebar.expander("Data Analysis Models", expanded=True):
        data_analysis_model_dir = os.path.join("models", "data_analysis")
        if os.path.exists(data_analysis_model_dir):
            model_files = os.listdir(data_analysis_model_dir)
            if model_files:
                st.write("**Available Data Analysis Models:**")
                for model_file in model_files:
                    st.write(f"- {model_file}")
            else:
                st.write("No data analysis models found.")
        else:
            st.write("Data analysis models directory does not exist.")

    # Display chatbot model files in the sidebar
    with st.sidebar.expander("Chatbot Models", expanded=True):
        chatbot_model_dir = os.path.join("models", "chatbot")
        if os.path.exists(chatbot_model_dir):
            chatbot_model_files = os.listdir(chatbot_model_dir)
            if chatbot_model_files:
                st.write("**Available Chatbot Models:**")
                for model_file in chatbot_model_files:
                    st.write(f"- {model_file}")
            else:
                st.write("No chatbot models found.")
        else:
            st.write("Chatbot models directory does not exist.")

    # Display model evaluations in the sidebar
    with st.sidebar.expander("Model Evaluations", expanded=False):
        eval_dir = os.path.join("models", "data_analysis", "evaluations")
        eval_file = os.path.join(eval_dir, "model_evaluations.csv")
        if os.path.exists(eval_file):
            try:
                evaluations = pd.read_csv(eval_file)
                st.write("**Data Analysis Model Evaluations:**")
                st.dataframe(evaluations)
            except Exception as e:
                st.write(f"Error loading model evaluations: {e}")
        else:
            st.write("Model evaluations are not available.")

    if choice == "Home":
        st.subheader("Welcome to the Assistance Systems Project")
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
            st.write("Your current health information:")
            st.json(st.session_state['user_data'])

    elif choice == "Data Analysis":
        data_analysis.run()

    elif choice == "Recommendations":
        st.subheader("Personalized Recommendations")
        st.write("This section will display personalized recommendations based on your data filters.")
        # TODO: Implement recommendation system interface here if needed.

    elif choice == "Chatbot":
        st.subheader("Chatbot Assistance")
        rasa_server_url = os.getenv(
            "RASA_SERVER", "http://localhost:5005/webhooks/rest/webhook"
        )
        chatbot = Chatbot(rasa_url=rasa_server_url, session_id=st.session_state["session_id"])
        chatbot.run()

if __name__ == "__main__":
    main()
