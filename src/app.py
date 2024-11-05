# src/app.py

import os
import pandas as pd
import streamlit as st
from data.data_loader import load_data
from data.data_preprocessor import preprocess_data
from models.recommendation_model import RecommendationModel
from chatbot.rasa_chatbot import Chatbot
import os

def main():
    st.set_page_config(page_title="Project Apero", layout="wide")

    st.title("Project Apero (Assistance Systems Project)")

    # Sidebar for navigation
    menu = ["Home", "Data Analysis", "Recommendations", "Chatbot"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Load and preprocess data
    data = load_data()
    processed_data = preprocess_data(data)

    # Initialize recommendation model
    rec_model = RecommendationModel(processed_data)
    rec_model.train_models()

    if choice == "Home":
        st.subheader("Welcome to Project Apero")
        st.write("Use the sidebar to navigate through the application.")

    elif choice == "Data Analysis":
        st.subheader("Data Analysis & Visualization")
        df = load_data("data/raw/healthcare-dataset-stroke-data.csv")
        st.dataframe(df)

    elif choice == "Recommendations":
        st.subheader("Personalized Recommendations")
        # TODO: Implement the recommendation system interface
        st.write("Recommendation system interface will be implemented here.")

    elif choice == "Chatbot":
        st.subheader("Chatbot Assistance")
        if "key" not in st.session_state:
            st.session_state["key"] = os.urandom(24).hex()
        rasa_server_url = os.getenv("RASA_SERVER", "http://rasa:5005/webhooks/rest/webhook")
        chatbot = Chatbot(rasa_url=rasa_server_url, session_id=st.session_state["key"])
        chatbot.run()

if __name__ == "__main__":
    main()
