import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def yield_pred():
    # Load and preprocess data
    @st.cache_data
    def load_data():
        try:
            # Update to the correct path of your local file
            data = pd.read_csv('crop_production.csv')
            data = data.dropna()
            data = data.drop(columns=['Crop_Year'])

            # Encoding categorical features
            label_encoders = {}
            for col in ['State_Name', 'District_Name', 'Crop', 'Season']:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le  # Save encoder for inverse transformation

            return data, label_encoders

        except FileNotFoundError:
            st.error("File not found. Please ensure the CSV file is in the correct location.")
            return None, None

    data, label_encoders = load_data()
    if data is None:
        return  # Exit if data could not be loaded

    # Prepare features and target variable
    X = data.drop(columns=['Production'])
    y = np.log1p(data['Production']) # Log transformation to handle massive outliers and prevent negative predictions

    # Split the dataset into training and testing sets
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    @st.cache_resource
    def train_yield_model_v2(_xtrain, _ytrain): # Renamed to bust cache
        # Define models
        random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                     max_depth=5, alpha=10, n_estimators=100)
        gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

        # Ensemble model
        ensemble_model = VotingRegressor([('random_forest', random_forest_model), ('gradient', gb_model)])

        # Train the ensemble model
        ensemble_model.fit(_xtrain, _ytrain)
        return ensemble_model

    ensemble_model = train_yield_model_v2(xtrain, ytrain)

    # Streamlit app layout
    st.title("📈 Crop Yield Prediction")
    st.markdown("Estimate your total production volume based on location, crop type, and area.")

    st.markdown("### Location & Season")
    col1, col2, col3 = st.columns(3)
    with col1:
        state_name = st.selectbox("📍 Select State", options=label_encoders['State_Name'].classes_)
    with col2:
        district_name = st.selectbox("📍 Select District", options=label_encoders['District_Name'].classes_)
    with col3:
        season_name = st.selectbox("🌤️ Select Season", options=label_encoders['Season'].classes_)
        
    st.markdown("### Crop Details")
    col4, col5 = st.columns(2)
    with col4:
        crop_name = st.selectbox("🌾 Select Crop", options=label_encoders['Crop'].classes_)
    with col5:
        area = st.number_input("📏 Area (Hectares)", min_value=0.0)

    # Encode user inputs using the label encoders
    input_data = pd.DataFrame({
        'State_Name': [label_encoders['State_Name'].transform([state_name])[0]],
        'District_Name': [label_encoders['District_Name'].transform([district_name])[0]],
        'Season': [label_encoders['Season'].transform([season_name])[0]],
        'Crop': [label_encoders['Crop'].transform([crop_name])[0]],
        'Area': [area]
    })

    st.markdown("<br>", unsafe_allow_html=True)

    # Predict production using the ensemble model
    if st.button("Predict Yield"):
        if area <= 0:
            st.warning("⚠️ You cannot grow crops on 0 Hectares of land! Please enter a valid area greater than 0.")
        else:
            try:
                log_prediction = ensemble_model.predict(input_data)
                final_prediction = np.expm1(log_prediction[0]) # Convert back from log scale
                
                # Calculate R2 Score for display
                r2_ensemble = r2_score(ytest, ensemble_model.predict(xtest))
                
                # Dynamic text for R2
                if r2_ensemble > 0.8:
                    r2_text = "Highly Reliable"
                    r2_color = "#4CAF50" # Green
                elif r2_ensemble > 0.6:
                    r2_text = "Moderately Reliable"
                    r2_color = "#FF9800" # Orange
                else:
                    r2_text = "Poor Reliability"
                    r2_color = "#F44336" # Red
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="color: #666;">Estimated Production</h3>
                            <h1 style="color: #4CAF50; font-size: 2.5em; margin: 0;">{final_prediction:,.2f}</h1>
                            <p style="color: #999; margin: 0;">Tonnes (or appropriate unit)</p>
                        </div>
                    """, unsafe_allow_html=True)
                with c2:
                     st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="color: #666;">Model Accuracy (R²)</h3>
                            <h1 style="color: {r2_color}; font-size: 2.5em; margin: 0;">{r2_ensemble:.2f}</h1>
                            <p style="color: #999; margin: 0;">{r2_text}</p>
                        </div>
                    """, unsafe_allow_html=True)
                     
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
