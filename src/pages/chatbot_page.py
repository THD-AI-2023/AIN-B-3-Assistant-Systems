import os
import streamlit as st
from chatbot.rasa_chatbot import Chatbot

def run():
    st.subheader("Chatbot Assistance")
    rasa_server_url = os.getenv("RASA_SERVER", "http://rasa:5005")
    chatbot = Chatbot(rasa_url=rasa_server_url, session_id=st.session_state["session_id"])

    # Display chatbot status
    with st.spinner("Checking chatbot status..."):
        model_ready, status = chatbot.is_model_ready()

    if model_ready:
        st.success("Chatbot is ready to assist you.")
        chatbot.run()
    else:
        if status == "no_model":
            st.error("No chatbot model is loaded. Please train a model using `rasa train`.")
            st.markdown("""
                **No model loaded in the Rasa server.**

                To train and load a model, please follow these steps:

                1. **Access the Rasa Server Container:**
                    ```bash
                    docker exec -it rasa_server bash
                    ```

                2. **Train the Rasa Model:**
                    Inside the container, execute:
                    ```bash
                    rasa train
                    ```

                3. **Restart Services:**
                    Exit the container and restart the Docker services:
                    ```bash
                    exit
                    docker-compose down
                    docker-compose up --build -d
                    ```

                After completing these steps, refresh the Streamlit app to interact with the chatbot.
            """)
        elif status == "model_not_loaded":
            st.warning("Chatbot model is loading. Please wait a moment and try again.")
            st.markdown("""
                The chatbot model is currently loading. This may take a few moments. Please wait and try again shortly.
            """)
        else:
            st.error("Chatbot is unavailable.")
            st.markdown("""
                **Chatbot is not available at the moment.**

                Please ensure that the Rasa server is running correctly and that a model is trained and loaded.
            """)
