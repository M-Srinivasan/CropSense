# CropSense
# Crop-Yield-Prediction: A Sustainable Development Research Project

  This research project leverages machine learning algorithms to predict crop yield and recommend suitable crops for a given district based on factors like temperature, rainfall, and land area. By applying machine learning techniques, the project aims to enhance agricultural productivity and sustainability in India, a country where agriculture remains a vital yet underpaid occupation. The models developed in this project help farmers make informed decisions about which crops to plant for maximum yield, ultimately boosting income and reducing risks associated with crop selection.

# Project Overview

  Agriculture in India faces numerous challenges, including uncertain crop outcomes and suboptimal crop choices. Machine learning offers a transformative opportunity to improve yield predictions and crop recommendations. This project focuses on predicting crop yield using various machine learning models, evaluating their performance based on the R-squared (R²) metric. By considering key factors such as temperature, rainfall, and land area, these models guide farmers in selecting the right crops to maximize yield and improve profitability.

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **Streamlit** - For building the interactive web application
- **Machine Learning**: Scikit-learn (Sklearn), XGBoost
- **Deep Learning**: TensorFlow
- **Data Manipulation & Visualization**: Pandas, NumPy, Matplotlib

## 📊 Model Performance

- **Crop Recommendation System**:
  - **Random Forest Classifier**: 100% Accuracy
  - **Voting Classifier (Ensemble)**: 98% Accuracy
  - **Logistic Regression**: 97% Accuracy
  - **Ridge Classifier**: 66% Accuracy

- **Yield Prediction System**: 
  - **Voting Regressor (Ensemble)**: R² Score of 0.8439
  - Incorporates Random Forest Regressor and Gradient Boosting Regressor to achieve robust continuous predictions on crop production.

## 🧠 Approach & Models

1. **Ensemble Voting Architectures**:
   The core of both recommendation and yield systems relies on Ensembles. For recommendation, predictions from Logistic Regression, Random Forest, and Ridge classifiers are combined to reach a stable 98% accuracy. For yield prediction, a Voting Regressor leverages Random Forest and Gradient Boosting.

2. **Random Forest**:
   Leveraged utilizing bagging (Bootstrap Aggregating). Multiple decision trees are trained on random data subsets, reducing variance and mitigating overfitting. Demonstrated 100% validation accuracy on the recommendation dataset.

3. **Gradient Boosting / XGBoost**:
   Used in yield prediction to sequentially correct errors from previous models, employing gradient descent to minimize the loss function. This makes it highly effective for extracting complex patterns from large-scale agricultural data.

4. **Linear Models (Logistic Regression & Ridge)**:
   Employed for classification tasks in crop recommendation, offering strong baseline performances (97% and 66% respectively) and contributing to the stability of the ensemble model.

## 💻 Interactive Streamlit Interface

The project features a sleek, user-friendly **Streamlit** dashboard. Farmers and stakeholders can effortlessly input environmental and geographic parameters (e.g., temperature, rainfall, area) to receive instant, real-time predictions. The interface abstracts away the technical complexity, providing clear, actionable insights and visualizations.

### Navigation
- **Crop Recommendation:** Suggests the most suitable crop based on localized conditions.
- **Crop Yield Prediction:** Estimates the expected yield in metric tons per hectare to aid in financial planning.

## 🚀 How to Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/M-Srinivasan/CropSense.git
   cd CropSense
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit application:
   ```bash
   streamlit run main.py
   ```

## 🌱 Conclusion

By integrating predictive modeling with an accessible web interface, CropSense bridges the gap between advanced data science and practical farming. The project provides data-driven decision-making tools that promote agricultural sustainability and maximize crop productivity.
