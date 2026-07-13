import streamlit as st
import pickle
import pandas as pd

def crop_recommendation():
    with open('RecommendationModel.pkl', 'rb') as file:
        model = pickle.load(file)

    st.title("🌱 Crop Recommendation System")
    st.markdown("Enter the soil and weather conditions below to find the best crop to cultivate.")

    st.markdown("### Soil Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        nitrogen = st.number_input("🧪 Nitrogen (N)", min_value=0)
    with col2:
        phosphorus = st.number_input("🧪 Phosphorus (P)", min_value=0)
    with col3:
        potassium = st.number_input("🧪 Potassium (K)", min_value=0)

    st.markdown("### Weather & Environment")
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0)
    with col5:
        humidity = st.number_input("💧 Humidity (%)", min_value=0.0)
    with col6:
        ph_value = st.number_input("⚗️ pH Value", min_value=0.0)
    with col7:
        rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Crop"):
        if nitrogen == 0 and phosphorus == 0 and potassium == 0 and temperature == 0 and humidity == 0 and rainfall == 0:
            st.warning("⚠️ Please enter realistic values! Soil and weather metrics cannot all be absolute zero.")
        else:
            input_data = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]],
                                    columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

            prediction = model.predict(input_data)[0]

            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #666;">Recommended Crop</h3>
                    <h1 style="color: #4CAF50; font-size: 3em; margin: 0;">{prediction.capitalize()}</h1>
                </div>
            """, unsafe_allow_html=True)
            return prediction