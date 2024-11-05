# src/chatbot/rasa_chatbot.py

import streamlit as st
import requests
import os

class Chatbot:
    def __init__(self, rasa_url=None, session_id=None):
        """
        Initializes the Chatbot instance.

        Parameters:
        - rasa_url (str): The endpoint URL for the Rasa REST webhook.
        - session_id (str): A unique identifier for the user session.
        """
        self.rasa_url = rasa_url or os.getenv("RASA_SERVER", "http://localhost:5005/webhooks/rest/webhook")
        self.session_id = session_id or "user"

    def send_message(self, message):
        """
        Sends a message to the Rasa chatbot and retrieves the response.

        Parameters:
        - message (str): The user's input message.

        Returns:
        - list: A list of responses from the chatbot.
        """
        payload = {"sender": self.session_id, "message": message}
        try:
            response = requests.post(self.rasa_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the chatbot: {e}")
            return [{"text": "I'm sorry, but I'm unable to process your request at this time."}]

    def run(self):
        """
        Runs the chatbot interface within the Streamlit app.
        """
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        st.markdown("### Chatbot")

        for message in st.session_state["messages"]:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Enter your message:"):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            responses = self.send_message(prompt)
            for resp in responses:
                with st.chat_message("assistant"):
                    st.markdown(resp.get("text", "No response from chatbot."))
                st.session_state["messages"].append({"role": "assistant", "content": resp.get("text", "")})
