import streamlit as st
import joblib
import pandas as pd

# Load the trained model and feature names
model, feature_names = joblib.load("breast_cancer_features.pkl")

# List of important features based on the provided data
important_features = [
    "concave_points1", "concave_points2", "symmetry3", "area3", "compactness2", 
    "texture1", "concave_points3", "concavity3", "area2", "compactness1", 
    "fractal_dimension1", "texture3", "symmetry2", "concavity2", "radius2", 
    "smoothness1", "fractal_dimension2", "perimeter3", "symmetry1", 
    "smoothness3", "fractal_dimension3", "smoothness2", "perimeter2", 
    "concavity1", "compactness3", "perimeter1"
]

# App title and introduction
st.markdown(
    """
    <h1 style='
        text-align: center; 
        color: #1E88E5; 
        font-family: Arial, sans-serif; 
        font-size: 2.5rem; 
        font-weight: bold;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);'>
        Breast Cancer Prediction System
    </h1>
    <h3 style='text-align: center; color: #555; font-family: Arial, sans-serif;'>Predict whether a tumor is benign or malignant based on medical data.</h3>
    """, unsafe_allow_html=True)

# Sidebar with app information
st.sidebar.title("ℹ️ About the App")
st.sidebar.info(
    """
    This predictive application leverages a machine learning model trained on breast cancer data to determine whether a tumor is **benign** or **malignant**.
    To use the app, input the values of the important tumor features and press **Predict** to get the result.
    """
)

# Feature input section
st.markdown("## 🔍 **Input Feature Values**")
st.write("Please enter the values of the following tumor features. These values are critical for determining the likelihood of cancer presence.")

# Create input fields for the important features only
input_values = {}
for feature in important_features:
    # Input fields for each feature
    input_values[feature] = st.number_input(f"Enter {feature}", min_value=0.0, value=0.0, step=0.1)

# When the user clicks 'Predict', perform the prediction
if st.button("Predict"):
    # Ensure input data is in the same order and structure as the model expects
    input_data = pd.DataFrame([input_values], columns=important_features)

    # Make the prediction using the model
    prediction = model.predict(input_data)

    # Display the result
    if prediction[0] == 1:
        st.success("The tumor is **Malignant**.")
    else:
        st.success("The tumor is **Benign**.")
