import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    data = pd.read_csv("Crop_recommendation.csv")  
    return data

df = load_data()

X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

st.title("🌾 AI-Powered Crop Recommendation System")

st.write("Enter the following soil and climate parameters:")

N = st.slider("Nitrogen (N)", 0, 150, 50)
P = st.slider("Phosphorous (P)", 0, 150, 50)
K = st.slider("Potassium (K)", 0, 150, 50)
temperature = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)
humidity = st.slider("Humidity (%)", 10.0, 90.0, 50.0)
ph = st.slider("pH level", 4.0, 9.0, 6.5)
rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

if st.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    st.success(f"✅ Recommended Crop: **{prediction[0].capitalize()}**")

    acc = accuracy_score(y_test, model.predict(X_test))
    st.info(f"🔍 Model Accuracy (test set): {acc:.2f}")
