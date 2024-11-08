import os
import pandas as pd
import streamlit as st
from chatbot.rasa_chatbot import Chatbot
from data.data_analysis import DataAnalysis

def main():
    st.set_page_config(page_title="Project Apero", layout="wide")

    st.title("Project Apero (Assistance Systems Project)")

    # Sidebar for navigation
    menu = ["Home", "Data Analysis", "Recommendations", "Chatbot"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Initialize DataAnalysis instance in session state
    if "data_analysis" not in st.session_state:
        st.session_state["data_analysis"] = DataAnalysis()

    data_analysis = st.session_state["data_analysis"]

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
        st.subheader("Welcome to Project Apero")
        st.write("Use the sidebar to navigate through the application.")

    elif choice == "Data Analysis":
        data_analysis.run()

    elif choice == "Recommendations":
        st.subheader("Personalized Recommendations")
        st.write("This section will display personalized recommendations based on your data filters.")
        # TODO: Implement recommendation system interface here if needed.

    elif choice == "Chatbot":
        st.subheader("Chatbot Assistance")
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = os.urandom(24).hex()
        rasa_server_url = os.getenv(
            "RASA_SERVER", "http://localhost:5005/webhooks/rest/webhook"
        )
        chatbot = Chatbot(rasa_url=rasa_server_url, session_id=st.session_state["session_id"])
        chatbot.run()

if __name__ == "__main__":
    main()
