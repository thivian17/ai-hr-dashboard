import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Title
st.title("AI Readiness Dashboard for HR")
st.write("Upload survey responses to auto-assign clusters and generate personalized training plans.")

# File uploader
uploaded_file = st.file_uploader("Upload the survey response CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = ['Timestamp', 'Comfort', 'Creativity', 'Confidence', 'Awareness', 'Risk', 'Email']

    # Feature engineering
    df['AI_ACCEPTANCE_SCORE'] = df[['Comfort', 'Creativity', 'Confidence']].mean(axis=1)
    df['AWARE_X_CONFIDENCE'] = df['Confidence'] * df['Awareness']
    features = df[['Comfort', 'Creativity', 'Confidence', 'Awareness', 'Risk', 'AI_ACCEPTANCE_SCORE', 'AWARE_X_CONFIDENCE']]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Training plan logic
    def get_recommendation(cluster):
        if cluster == 0:
            return "Foundational AI training, low-stakes tasks, peer mentorship"
        else:
            return "Advanced AI tools, ethics training, AI leadership role"

    df['Training_Plan'] = df['Cluster'].apply(get_recommendation)

    # Display results
    st.success("Survey responses processed successfully!")
    st.dataframe(df[['Email', 'Cluster', 'AI_ACCEPTANCE_SCORE', 'Training_Plan']])

    # Download option
    csv = df[['Email', 'Cluster', 'AI_ACCEPTANCE_SCORE', 'Training_Plan']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Employee Profiles CSV",
        data=csv,
        file_name='employee_profiles.csv',
        mime='text/csv')
