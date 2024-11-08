import streamlit as st
import pickle
import pandas as pd

def crop_recommendation():
    with open('RecommendationModel.pkl', 'rb') as file:
        model = pickle.load(file)

    st.title("Crop Sense")

    st.sidebar.header("Input Features")
    nitrogen = st.sidebar.number_input("Nitrogen", min_value=0)
    phosphorus = st.sidebar.number_input("Phosphorus", min_value=0)
    potassium = st.sidebar.number_input("Potassium", min_value=0)
    temperature = st.sidebar.number_input("Temperature", min_value=0)
    humidity = st.sidebar.number_input("Humidity", min_value=0)
    ph_value = st.sidebar.number_input("pH Value", min_value=0)
    rainfall = st.sidebar.number_input("Rainfall", min_value=0)

    if st.button("Predict Crop"):
        input_data = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]],
                                columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

        prediction = model.predict(input_data)[0]

        st.subheader("Predicted Crop:")
        st.write(prediction)
        return prediction