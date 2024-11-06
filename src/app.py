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

    if choice == "Home":
        st.subheader("Welcome to Project Apero")
        st.write("Use the sidebar to navigate through the application.")

    elif choice == "Data Analysis":
        data_analysis = DataAnalysis()
        data_analysis.run()

    elif choice == "Recommendations":
        st.subheader("Personalized Recommendations")
        # TODO: Implement the recommendation system interface
        st.write("Recommendation system interface will be implemented here.")

    elif choice == "Chatbot":
        st.subheader("Chatbot Assistance")
        if "key" not in st.session_state:
            st.session_state["key"] = os.urandom(24).hex()
        rasa_server_url = os.getenv(
            "RASA_SERVER", "http://localhost:5005/webhooks/rest/webhook"
        )
        chatbot = Chatbot(rasa_url=rasa_server_url, session_id=st.session_state["key"])
        chatbot.run()

if __name__ == "__main__":
    main()
