import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
diabetes_model = pickle.load(open('calories_xgb_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
st.markdown('<img src="your_image_logo.png" class="logo">', unsafe_allow_html=True)
# Set the title and some introduction text
st.title("🔥 Diabetes Prediction App 🔥")

# Add the logo image (make sure the image is in the same directory as the script or provide the correct path)


st.markdown("""
Welcome to the Diabetes Prediction App! This app will help you predict whether a person is diabetic based on certain medical features.
Fill in the details below and hit 'Get Results' to see the prediction.
""", unsafe_allow_html=True)

# Add custom styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
    }
    .stTitle, .stHeader, .stSubheader, .stText, .stMarkdown {
        color: #ffffff;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stNumberInput>div>label {
        color: #ffffff;
    }
    .stTextInput>div>label {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True
)

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

    # Ensure the input data is a 2D array (1 row, 8 columns/features)
    st.write("Shape of input data:", input_data.shape)  # This will show (1, 8)

    # Apply the scaler transformation
    try:
        # Ensure the input data is 2D before transforming
        input_data_scaled = scaler.transform(input_data)  # Apply scaling
    except Exception as e:
        st.write("Error in scaling the input data:", e)
    
    # Make the prediction
    try:
        diab_pred = diabetes_model.predict(input_data_scaled)
        # Display the result
        if diab_pred[0] == 0:
            diab_diagnosis = "❌ The person is NOT diabetic."
            st.success(diab_diagnosis)  # Use success for non-diabetic
        else:
            diab_diagnosis = "✅ The person is diabetic."
            st.error(diab_diagnosis)  # Use error for diabetic
    except Exception as e:
        st.write("Error in prediction:", e)
