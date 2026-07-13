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

- **Crop Recommendation System**: Achieved an impressive **98% validation accuracy** (benchmarked against PyCaret which reached up to 99.35%).
- **Yield Prediction System**: Delivered a robust **R² score of 0.84**, trained on a comprehensive agricultural dataset comprising 246,091 records.

## 🧠 Approach & Models

1. **Random Forest Classifier/Regressor**:
   Leveraged as an ensemble method utilizing bagging (Bootstrap Aggregating). Multiple decision trees are trained on random data subsets, reducing variance and mitigating overfitting. It excels at handling missing data and outliers while processing large datasets efficiently.

2. **Extreme Gradient Boosting (XGBoost)**:
   An advanced ensemble technique based on gradient-boosted decision trees. It sequentially corrects errors from previous models, employing gradient descent to minimize the loss function. This makes it highly effective for extracting complex patterns from large-scale agricultural data.

3. **Logistic Regression**:
   Employed primarily for experimentation and baseline comparisons. It uses L2 regularization to model categorical outcomes for crop recommendation tasks.

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
