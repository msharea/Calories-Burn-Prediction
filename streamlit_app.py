import streamlit as st
import numpy as np
import pickle

# تحميل النموذج والمقياس
model = pickle.load(open('calories_xgb_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title('Calories Burn Prediction Web App')
st.write('This is a web app to predict the calories burned based on input features.')
st.write("Enter your details below to predict the calories burned during exercise.")

# استلام المدخلات من المستخدم
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.slider('Age', 10, 90)
Weight = st.slider('Weight in kg', 30, 200)
Height = st.slider('Height in cm', 100, 250)  # إضافة الطول
Duration = st.slider('Duration in minutes', 10, 300)
HeartRate = st.slider('Heart Rate', 60, 200)

# تحويل الجنس إلى قيم رقمية
Gender = 1 if Gender == 'Female' else 0

# عندما يضغط المستخدم على "Predict Calories"
if st.button("Predict Calories"):
    try:
        # التأكد من جميع المدخلات
        if not (Age and Weight and Height and Duration and HeartRate):
            st.warning("Please enter all the details.")
        else:
            # تضمين جميع المدخلات التي سيتم تمريرها للنموذج
            input_data = np.array([[Gender, Age, Weight, Height, Duration, HeartRate]])

            # تطبيق التحويل باستخدام المقياس
            input_data_scaled = scaler.transform(input_data)

            # إجراء التنبؤ
            prediction = model.predict(input_data_scaled)

            # عرض النتيجة
            st.success(f"Estimated Calories Burned: {prediction[0]:.2f} kcal")
    
    except Exception as e:
        st.error(f"Error in prediction: {e}")
