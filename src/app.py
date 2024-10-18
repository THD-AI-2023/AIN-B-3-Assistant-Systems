import streamlit as st
from data.data_loader import load_data
from data.data_preprocessor import preprocess_data
from models.recommendation_model import RecommendationModel
# from chatbot.rasa_chatbot import Chatbot

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
    # TODO: Implement the training of recommendation models
    # rec_model.train_models()

    if choice == "Home":
        st.subheader("Welcome to the Project Apero")
        st.write("Use the sidebar to navigate through the application.")

    elif choice == "Data Analysis":
        st.subheader("Data Analysis & Visualization")
        # TODO: Add data visualization components using Streamlit

    elif choice == "Recommendations":
        st.subheader("Personalized Recommendations")
        # TODO: Implement the recommendation system interface

    elif choice == "Chatbot":
        st.subheader("Chatbot Assistance")
        # TODO: Integrate the Rasa chatbot with the Streamlit frontend
        # chatbot = Chatbot()
        # chatbot.run()
        st.write("Chatbot integration will be implemented here.")

if __name__ == "__main__":
    main()
