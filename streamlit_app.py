import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# تحميل النموذج والمقياس المحفوظين
@st.cache
def load_model():
    model = joblib.load("calories_xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# تحميل البيانات
df = pd.read_csv('calories_burn.csv')

# عرض المعلومات حول البيانات
st.write("## بيانات التدريب:")
st.write(df.head())

# إظهار المعلومات
st.write("## معلومات حول البيانات:")
st.write(df.info())

# إظهار الرسومات البيانية
st.write("## توزيع السعرات الحرارية:")
plt.figure(figsize=(8, 5))
sns.histplot(df["Calories"], bins=30, kde=True, color="red")
plt.title("Calories Distribution")
st.pyplot()

st.write("## Heatmap للبيانات:")
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap of Feature Correlations")
st.pyplot()

# تحضير البيانات
df.dropna(inplace=True)
df.duplicated().sum()

encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])
df.drop(columns=["User_ID"], inplace=True)

# تحضير المدخلات
X = df.drop(columns=["Calories"])
y = df["Calories"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تطبيع البيانات
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# التنبؤ باستخدام النموذج المدرب
y_pred = model.predict(X_test_scaled)

# تقييم النموذج
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"## تقييم النموذج:")
st.write(f" MAE: {mae:.2f}")
st.write(f" MSE: {mse:.2f}")
st.write(f" R² Score: {r2:.2f}")

# إدخال بيانات جديدة للتنبؤ
st.write("## إدخال بيانات جديدة:")
weight = st.number_input('الوزن (بالكيلوغرام)', min_value=0, max_value=200, value=70)
height = st.number_input('الطول (بالمتر)', min_value=0, max_value=2.5, value=1.75)
age = st.number_input('العمر (بالسنوات)', min_value=10, max_value=100, value=25)
activity_level = st.selectbox('مستوى النشاط', ['قليل', 'معتدل', 'مرتفع'])

activity_factor = {'قليل': 1.2, 'معتدل': 1.55, 'مرتفع': 1.9}
activity = activity_factor[activity_level]

# تحضير المدخلات
input_data = pd.DataFrame({
    'Weight': [weight],
    'Height': [height],
    'Age': [age],
    'Activity': [activity]
})

input_data_scaled = scaler.transform(input_data)

# توقع السعرات الحرارية
if st.button("تنبؤ بالسعرات الحرارية"):
    prediction = model.predict(input_data_scaled)
    st.write(f"توقع السعرات الحرارية اليومية: {prediction[0]:.2f} كيلو كالوري")
