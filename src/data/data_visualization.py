import streamlit as st
import pandas as pd
import altair as alt

def visualize_data(data):
    """
    Generates and displays data visualizations using Streamlit components.

    Parameters:
    - data (pd.DataFrame): The preprocessed dataset.
    """
    st.write("### Data Visualization")

    # Correlation Heatmap
    st.write("#### Correlation Heatmap")
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    corr = data[numeric_cols].corr().stack().reset_index()
    corr.columns = ['Variable1', 'Variable2', 'Correlation']
    heatmap = alt.Chart(corr).mark_rect().encode(
        x=alt.X('Variable1:O', title=None),
        y=alt.Y('Variable2:O', title=None),
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
        tooltip=['Variable1', 'Variable2', 'Correlation']
    ).properties(
        width=600,
        height=600
    )
    st.altair_chart(heatmap, use_container_width=True)

    # Distribution of Target Variable
    st.write("#### Distribution of Stroke Variable")
    stroke_counts = data['stroke'].value_counts().reset_index()
    stroke_counts.columns = ['stroke', 'count']
    stroke_counts['stroke'] = stroke_counts['stroke'].astype(str)
    stroke_bar = alt.Chart(stroke_counts).mark_bar().encode(
        x=alt.X('stroke:N', axis=alt.Axis(title='Stroke')),
        y=alt.Y('count:Q', axis=alt.Axis(title='Count')),
        tooltip=['stroke', 'count']
    ).properties(
        width=600
    )
    st.altair_chart(stroke_bar, use_container_width=True)

    # Age Distribution by Stroke
    st.write("#### Age Distribution by Stroke")
    age_hist = alt.Chart(data).mark_bar(opacity=0.7).encode(
        x=alt.X('age:Q', bin=alt.Bin(maxbins=30), title='Age'),
        y=alt.Y('count()', stack=None, title='Count'),
        color=alt.Color('stroke:N', legend=alt.Legend(title='Stroke')),
        tooltip=['count()']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(age_hist, use_container_width=True)

    # BMI Distribution
    st.write("#### BMI Distribution")
    bmi_hist = alt.Chart(data).mark_bar().encode(
        x=alt.X('bmi:Q', bin=alt.Bin(maxbins=30), title='BMI'),
        y=alt.Y('count()', title='Count'),
        tooltip=['count()']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(bmi_hist, use_container_width=True)

    # Smoking Status vs Stroke
    st.write("#### Smoking Status vs Stroke")
    smoking_counts = data.groupby(['smoking_status', 'stroke']).size().reset_index(name='count')
    smoking_counts['stroke'] = smoking_counts['stroke'].astype(str)
    smoking_bar = alt.Chart(smoking_counts).mark_bar().encode(
        x=alt.X('smoking_status:N', axis=alt.Axis(title='Smoking Status')),
        y=alt.Y('count:Q', axis=alt.Axis(title='Count')),
        color=alt.Color('stroke:N', legend=alt.Legend(title='Stroke')),
        tooltip=['smoking_status', 'stroke', 'count']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(smoking_bar, use_container_width=True)
