# main.py
import streamlit as st
import Crop_Recommendation  # Import the crop recommendation app module
import Yield_Pred  # Import the crop production prediction app module

def main():
    # Page Config MUST be the first Streamlit command
    st.set_page_config(page_title="CropSense", page_icon="🌾", layout="wide")

    # Custom CSS for rich aesthetics
    st.markdown("""
        <style>
        .main {
            background-color: #f7f9fc;
        }
        h1 {
            color: #2e7d32;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-align: center;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            transition: 0.3s;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("🌾 Navigation")
    st.sidebar.markdown("Welcome to **CropSense**, an intelligent agricultural assistant.")
    app_selection = st.sidebar.radio("Choose a Tool", ["Crop Recommendation", "Crop Yield Prediction"])

    if app_selection == "Crop Recommendation":
        Crop_Recommendation.crop_recommendation()
    elif app_selection == "Crop Yield Prediction":
        Yield_Pred.yield_pred()

if __name__ == "__main__":
    main()
