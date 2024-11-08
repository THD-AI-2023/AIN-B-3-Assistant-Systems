import os
import streamlit as st
from chatbot.rasa_chatbot import Chatbot
from data.data_analysis import DataAnalysis

def main():
    st.set_page_config(page_title="Project Apero", layout="wide")

    st.title("Project Apero (Assistance Systems Project)")

    # Sidebar for navigation
    menu = ["Home", "Data Analysis", "Recommendations", "Chatbot"]
    choice = st.sidebar.selectbox("Menu", menu)

    if "data_analysis" not in st.session_state:
        st.session_state["data_analysis"] = DataAnalysis()

    data_analysis = st.session_state["data_analysis"]

    # Display saved model files in the sidebar
    with st.sidebar.expander("Saved Models", expanded=True):
        model_dir = "models"
        if os.path.exists(model_dir):
            model_files = os.listdir(model_dir)
            if model_files:
                st.write("**Available Models:**")
                for model_file in model_files:
                    st.write(f"- {model_file}")
            else:
                st.write("No models found.")
        else:
            st.write("Models directory does not exist.")

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
