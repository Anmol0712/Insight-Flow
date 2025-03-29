import streamlit as st 
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# Loading the model
with open('rf_recommendation_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title('🏥Personalized Healthcare Recommendation')
st.divider()
st.write("##### Provide your health details to receive personalized recommendation")

# Input Fields
age = st.number_input('Age', min_value=1, max_value=120)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0)
HbA1c = st.number_input('HbA1c Level', min_value=3.0, max_value=10.0)
blood_glucose = st.number_input('Blood Glucose Level', min_value=50, max_value=300)
hypertension = st.selectbox('Hypertension', [0, 1])
heart_disease = st.selectbox('Heart Disease', [0, 1])
gender = st.selectbox("Gender", ['Select Gender','Male','Female','Other'])
smoking_history = st.selectbox('Smoking History', ['Never', 'No Info', 'Current', 'Former', 'Ever', 'Not current'])

st.divider()

input_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'HbA1c_level': [HbA1c],
    'blood_glucose_level': [blood_glucose],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'gender': [gender],
    'smoking_history': [smoking_history]
})

if st.button('Predict Risk Level'):
    prediction = model.predict(input_data)
    risk = 'High Risk' if prediction[0] == 1 else 'Low Risk'
    st.write(f'### Prediction: {risk}')

    if risk == 'High Risk':
        st.write('### Advice:')
        st.write('''
                 - Maintain a healthy diet and exercise regularly.
                 - Monitor blood glucose levels frequently.
                 - Consult a healthcare professional for personalized guidance.
        ''')
    else:
        st.write('### Advice:')
        st.write('''
                 - Keep up the healthy lifestyle.
                 - Continue regular check-ups to stay informed about your health.
        ''')

# Use this command in terminal to run the app - 'streamlit run .\PHR_App.py'