import os
import streamlit as st
from PIL import Image

def run():
    st.subheader("Welcome to the Assistance Systems Project")

    banner_path = os.path.join("docs", "img", "ASP_Banner.png")
    if os.path.exists(banner_path):
        try:
            image = Image.open(banner_path)
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading banner image: {e}")
    else:
        st.warning(f"Banner image not found. Please ensure '{banner_path}' exists in the 'docs' directory.")


    st.write("""
    **Assistance Systems Project** is a comprehensive web application designed to provide personalized health recommendations and data insights. Navigate through the sidebar to explore different functionalities:

    - **Data Analysis:** Explore and visualize health-related data.
    - **Personalized Recommendations:** Receive tailored health recommendations based on your personal information.
    - **Chatbot:** Interact with our intelligent chatbot for assistance and information.

    Get started by navigating to the **Personalized Recommendations** page to input your health data and receive customized advice.
    """)
