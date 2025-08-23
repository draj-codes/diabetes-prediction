import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import random
import time
import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Header with image
st.image("https://user-images.githubusercontent.com/103222259/255432423-8157a6d0-8d82-43b4-894f-53cd2125c891.png")

st.header("Diabetes Prediction Using Machine Learning")

data_info = '''The Diabetes prediction dataset is a collection of medical and demographic data from patients, along with their diabetes status (positive or negative). The data includes features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. This dataset can be used to build machine learning models to predict diabetes in patients based on their medical history and demographic information. This can be useful for healthcare professionals in identifying patients who may be at risk of developing diabetes and in developing personalized treatment plans. Additionally, the dataset can be used by researchers to explore the relationships between various medical and demographic factors and the likelihood of developing diabetes.'''
st.subheader(data_info)

# Load model safely with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load("diabetes_model_XGB.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.error("Failed to load the diabetes prediction model. Please check if the model file exists.")
    st.stop()

# Sample data for demonstration (since we removed kagglehub dependency)
# In production, you might want to include a small sample dataset or load from a different source
sample_data = pd.DataFrame({
    'age': [25, 45, 65],
    'hypertension': [0, 1, 1],
    'heart_disease': [0, 0, 1],
    'bmi': [22.5, 28.3, 32.1],
    'HbA1c_level': [5.2, 6.8, 7.5],
    'blood_glucose_level': [85, 140, 180],
    'gender_encoded': [1, 0, 1],
    'smoking_No_Info': [0, 0, 0],
    'smoking_current': [0, 1, 0],
    'smoking_former': [0, 0, 1],
    'smoking_never': [1, 0, 0]
})

st.sidebar.header("Select feature to predict Diabetes")
st.sidebar.image("https://www.eresvihda.es/wp-content/uploads/2023/10/Diabetes.gif")

# Sidebar input sliders with better ranges
col = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
       'blood_glucose_level', 'gender_encoded',
       'smoking_No_Info', 'smoking_current', 'smoking_former', 'smoking_never']

# Define reasonable ranges for each feature
feature_ranges = {
    'age': (18, 100),
    'hypertension': (0, 1),
    'heart_disease': (0, 1),
    'bmi': (15.0, 50.0),
    'HbA1c_level': (3.5, 15.0),
    'blood_glucose_level': (50, 300),
    'gender_encoded': (0, 1),
    'smoking_No_Info': (0, 1),
    'smoking_current': (0, 1),
    'smoking_former': (0, 1),
    'smoking_never': (0, 1)
}

all_values = []
random.seed(57)

for i in col:
    min_val, max_val = feature_ranges[i]
    if i in ['hypertension', 'heart_disease', 'gender_encoded', 'smoking_No_Info', 'smoking_current', 'smoking_former', 'smoking_never']:
        # Binary features
        var = st.sidebar.selectbox(f'Select {i}', [0, 1], index=random.randint(0, 1))
    else:
        # Continuous features
        var = st.sidebar.slider(f'Select {i} value',
                                float(min_val), float(max_val),
                                float(random.uniform(min_val, max_val)))
    all_values.append(var)

# Convert to DataFrame for prediction
final_value = pd.DataFrame([all_values], columns=col)

# Predict button
if st.button("Predict Diabetes", type="primary"):
    # Predict
    try:
        ans = model.predict(final_value)[0]
        
        # Progress bar animation
        progress_bar = st.progress(0)
        placeholder = st.empty()
        placeholder.subheader('Predicting Diabetes')
        
        place = st.empty()
        place.image('https://media0.giphy.com/media/aPBXEeY01Dfp9tjyqi/giphy.gif', width=200)
        
        for i in range(100):
            time.sleep(0.05)
            progress_bar.progress(i + 1)
        
        # Show result
        if ans == 0:
            placeholder.empty()
            place.empty()
            st.success('✅ No Diabetes Detected')
            st.balloons()
        else:
            placeholder.empty()
            place.empty()
            st.warning('⚠️ Diabetes Risk Detected')
            
        # Show input values
        st.subheader("Input Values Used:")
        input_df = pd.DataFrame([all_values], columns=col)
        st.dataframe(input_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.markdown("---")
st.markdown("Design By : Divyanshu Raj")