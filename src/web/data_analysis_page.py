import streamlit as st
from data.data_analysis import DataAnalysis

def run():
    st.subheader("Data Analysis")

    # Initialize DataAnalysis instance in session state
    if "data_analysis" not in st.session_state:
        st.session_state["data_analysis"] = DataAnalysis()

    data_analysis = st.session_state["data_analysis"]

    # Run the data analysis workflow
    data_analysis.run()
