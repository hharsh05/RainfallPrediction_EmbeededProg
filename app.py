import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Initialize and fit the model and scaler (example training data)
training_data = pd.DataFrame({
    'day': [1, 2, 3, 4],
    'pressure': [1012, 1010, 1011, 1009],
    'temperature': [20, 21, 19, 22],
    'humidity': [30, 40, 50, 35],
    'cloud': [0, 1, 0, 1],
    'sunshine': [5, 6, 2, 4],
    'wind_direction': [180, 90, 270, 360],
    'wind_speed': [5, 7, 3, 4]
})
target_data = pd.Series([0, 1, 0, 1])

# Initialize scaler and logistic regression model
scaler = StandardScaler()
model = LogisticRegression()
scaler.fit(training_data)
model.fit(scaler.transform(training_data), target_data)

# Streamlit app
st.title("Weather Prediction")

# User input
day = st.number_input("Day:", min_value=1, max_value=31)
pressure = st.number_input("Pressure (hPa):")
temperature = st.number_input("Temperature (°C):")
humidity = st.number_input("Humidity (%):")
cloud = st.number_input("Cloud (0-1):")
sunshine = st.number_input("Sunshine (hours):")
wind_direction = st.number_input("Wind Direction (degrees):")
wind_speed = st.number_input("Wind Speed (km/h):")

if st.button("Predict"):
    user_input = pd.DataFrame([[day, pressure, temperature, humidity, cloud, 
                                 sunshine, wind_direction, wind_speed]], 
                               columns=['day', 'pressure', 'temperature', 'humidity', 
                                        'cloud', 'sunshine', 'wind_direction', 'wind_speed'])

    # Standardize user input
    user_input_scaled = scaler.transform(user_input)

    # Make prediction
    prediction = model.predict(user_input_scaled)[0]

    # Interpret the prediction
    if prediction == 1:
        msg_body = "🌧️ Rain predicted today! Please carry an umbrella ☂️ and stay safe."
        st.success(msg_body)
    else:
        msg_body = "☀️ No rain expected today! It’s a good day to plan outdoor activities."
        st.success(msg_body)

# Run your Streamlit app
