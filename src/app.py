import os
import streamlit as st

def main():
    st.set_page_config(page_title="Assistance Systems Project", layout="wide")
    st.title("Assistance Systems Project")

    # Sidebar for navigation
    menu = ["Home", "Data Analysis", "Personalized Recommendations", "Chatbot"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Initialize session state variables if needed
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = os.urandom(24).hex()

    # Navigation
    if choice == "Home":
        from pages.home import run as run_home
        run_home()
    elif choice == "Data Analysis":
        from pages.data_analysis_page import run as run_data_analysis
        run_data_analysis()
    elif choice == "Personalized Recommendations":
        from pages.recommendations import run as run_recommendations
        run_recommendations()
    elif choice == "Chatbot":
        from pages.chatbot_page import run as run_chatbot
        run_chatbot()
        
if __name__ == "__main__":
    main()
