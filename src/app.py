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

    # Navigation
    if choice == "Home":
        from web.home import run as run_home
        run_home()
    elif choice == "Data Analysis":
        from web.data_analysis_page import run as run_data_analysis
        run_data_analysis()
    elif choice == "Personalized Recommendations":
        from web.recommendations import run as run_recommendations
        run_recommendations()
    elif choice == "Chatbot":
        from web.chatbot_page import run as run_chatbot
        run_chatbot()
        
if __name__ == "__main__":
    main()
