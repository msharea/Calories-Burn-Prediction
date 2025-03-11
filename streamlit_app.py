# -*- coding: utf-8 -*-


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

df = pd.read_csv('calories_burn.csv')

df.info()

df.isnull().sum()

df.dropna(inplace=True)

df.duplicated().sum()

encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

df.drop(columns=["User_ID"], inplace=True)

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap of Feature Correlations")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["Calories"], bins=30, kde=True, color="red")
plt.title("Calories Distribution")
plt.show()

def remove_outliers(column, factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    df[column] = np.clip(df[column], lower_bound, upper_bound)

outlier_columns = ["Height", "Weight", "Heart_Rate", "Body_Temp", "Calories"]
for col in outlier_columns:
    remove_outliers(col, factor=1.5)

X = df.drop(columns=["Calories"])
y = df["Calories"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb = XGBRegressor(random_state=42)

param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.07, 0.1],
    "n_estimators": [50, 100, 150],
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

best_xgb = grid_search.best_estimator_

y_pred = best_xgb.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f" MAE: {mae:.2f}")
print(f" MSE: {mse:.2f}")
print(f" R² Score: {r2:.2f}")

joblib.dump(best_xgb, "calories_xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
