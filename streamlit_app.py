import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
diabetes_model = pickle.load(open('calories_xgb_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Set the title and some introduction text
st.title("🔥 Diabetes Prediction App 🔥")
st.markdown("""
Welcome to the Diabetes Prediction App! This app will help you predict whether a person is diabetic based on certain medical features.
Fill in the details below and hit 'Get Results' to see the prediction.
""")

# Add a background color or image (optional)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F0F8FF;
    }
    </style>
    """, unsafe_allow_html=True
)

# Add a logo or image (optional)
st.image("your_image_logo.png", width=200)

# Getting the input data from USER
st.subheader("Enter the following details:")

Pregnancies = st.number_input("Number of Pregnancies", min_value=0)
Glucose = st.number_input("Level of Glucose", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Level of Insulin", min_value=0)
BMI = st.number_input("Body Mass Index", min_value=0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
Age = st.number_input("Age", min_value=0)

# Code for prediction
diab_diagnosis = ''

# Add a space between inputs and button
st.write("\n")
st.write("### Click below to get the results:")

# Prediction
if st.button("Get Results", key="predict_button"):
    # Create a NumPy array from the inputs
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    # Apply the scaler transformation
    input_data_scaled = scaler.transform(input_data)
    
    # Make the prediction
    diab_pred = diabetes_model.predict(input_data_scaled)

    # Display the result with a nice format
    if diab_pred[0] == 0:
        diab_diagnosis = "❌ The person is NOT diabetic."
        st.success(diab_diagnosis)  # Use success for non-diabetic
    else:
        diab_diagnosis = "✅ The person is diabetic."
        st.error(diab_diagnosis)  # Use error for diabetic
