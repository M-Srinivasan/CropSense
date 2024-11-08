import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def yield_pred():
    # Load and preprocess data
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
    y = data['Production']

    # Split the dataset into training and testing sets
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                 max_depth=5, alpha=10, n_estimators=100)
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # Ensemble model
    ensemble_model = VotingRegressor([('random_forest', random_forest_model), ('gradient', gb_model)])

    # Train the ensemble model
    ensemble_model.fit(xtrain, ytrain)

    # Streamlit app layout
    st.title("Crop Production Prediction")
    st.write("Enter the following details to predict crop production:")

    # User input for features
    state_name = st.selectbox("Select State", options=label_encoders['State_Name'].classes_)
    district_name = st.selectbox("Select District", options=label_encoders['District_Name'].classes_)
    crop_name = st.selectbox("Select Crop", options=label_encoders['Crop'].classes_)
    season_name = st.selectbox("Select Season", options=label_encoders['Season'].classes_)

    # Encode user inputs using the label encoders
    input_data = pd.DataFrame({
        'State_Name': [label_encoders['State_Name'].transform([state_name])[0]],
        'District_Name': [label_encoders['District_Name'].transform([district_name])[0]],
        'Crop': [label_encoders['Crop'].transform([crop_name])[0]],
        'Season': [label_encoders['Season'].transform([season_name])[0]]
    })

    # Predict production using the ensemble model
    if st.button("Predict"):
        try:
            prediction = ensemble_model.predict(input_data)
            st.success(f"Predicted Crop Production: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Display R² score of the model
    r2_ensemble = r2_score(ytest, ensemble_model.predict(xtest))
    st.write(f"Ensemble R² Score: {r2_ensemble:.2f}")
