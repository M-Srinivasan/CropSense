# main.py
import streamlit as st
import Crop_Recommendation  # Import the crop recommendation app module
import Yield_Pred  # Import the crop production prediction app module

def main():
    st.sidebar.title("Navigation")
    app_selection = st.sidebar.radio("Choose an App", ["Crop Recommendation", "Crop Yield Prediction"])

    if app_selection == "Crop Recommendation":
        Crop_Recommendation.crop_recommendation()
    elif app_selection == "Crop Production Prediction":
        Yield_Pred.yield_pred()

if __name__ == "__main__":
    main()
