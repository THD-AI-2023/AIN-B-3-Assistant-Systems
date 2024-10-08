import streamlit as st
import openai

# Set the page title
st.title("Project Apero: ChatGPT-like Application")

# Load the OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Enter your message:"):
    # Add user message to session state
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response from OpenAI
    with st.chat_message("assistant"):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=st.session_state["messages"]
            )
            assistant_message = response['choices'][0]['message']['content']
            st.markdown(assistant_message)
        except openai.OpenAIError as e:
            st.error(f"An error occurred: {e}")
            assistant_message = (
                "I'm sorry, but I'm unable to process your request at this time."
            )

    # Add assistant message to session state
    st.session_state["messages"].append(
        {"role": "assistant", "content": assistant_message}
    )
