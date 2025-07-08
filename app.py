import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Load the saved model, scaler, and selector
try:
    model = load('wdbc_random_forest_model.pkl')
    scaler = load('wdbc_scaler.pkl')
    selector = load('wdbc_selector.pkl')
except FileNotFoundError as e:
    st.error(f"Error: File not found - {e}")
    st.stop()

# Define feature columns (must match training data)
feature_columns = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Streamlit app
st.title("Breast Cancer Detection App")
st.write("Upload a file (CSV, Excel, or JSON) or enter values manually to predict breast cancer.")

# Input method selection
input_method = st.radio("Choose input method:", ("Manual", "Upload File"))

if input_method == "Manual":
    features = [st.number_input(f"{col}:", value=0.0, step=0.01) for col in feature_columns]
    input_df = pd.DataFrame([features], columns=feature_columns)
else:
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "json"])
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        try:
            if file_type in ["csv"]:
                input_df = pd.read_csv(uploaded_file)
            elif file_type in ["xlsx", "xls"]:
                input_df = pd.read_excel(uploaded_file)
            elif file_type in ["json"]:
                input_df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file type. Please use CSV, Excel, or JSON.")
                st.stop()

            # Validate and align columns
            if len(input_df.columns) != 30 or not all(col in input_df.columns for col in feature_columns):
                st.error("File must contain exactly 30 columns matching the feature names.")
            else:
                input_df = input_df[feature_columns]
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()

if 'input_df' in locals() and st.button("Predict Only"):
    try:
        input_scaled = scaler.transform(input_df)
        input_selected = selector.transform(input_scaled)
        prediction = model.predict(input_selected)
        prediction_prob = model.predict_proba(input_selected)

        st.write("### Prediction Results")
        for i, (pred, prob) in enumerate(zip(prediction, prediction_prob)):
            st.write(f"Sample {i+1}: **Predicted: {'Malignant' if pred == 1 else 'Benign'}**, "
                     f"Probabilities: Benign={prob[0]:.4f}, Malignant={prob[1]:.4f}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

if 'input_df' in locals():
    if st.button("Predict +", key="predict_button"):
        try:
            input_scaled = scaler.transform(input_df)
            input_selected = selector.transform(input_scaled)
            prediction = model.predict(input_selected)
            prediction_prob = model.predict_proba(input_selected)

            st.write("### Prediction Results")
            results = []
            for i, (pred, prob) in enumerate(zip(prediction, prediction_prob)):
                result = {"Sample": f"Sample {i+1}", "Prediction": "Malignant" if pred == 1 else "Benign",
                          "Benign_Prob": prob[0], "Malignant_Prob": prob[1]}
                st.write(f"Sample {i+1}: **Predicted: {'Malignant' if pred == 1 else 'Benign'}**, "
                         f"Probabilities: Benign={prob[0]:.4f}, Malignant={prob[1]:.4f}")
                st.bar_chart({"Benign": prob[0], "Malignant": prob[1]})
                results.append(result)
            results_df = pd.DataFrame(results)
            st.download_button("Download Results", results_df.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")    

# Optional: Add model info or instructions
# st.write("**Note**: Ensure feature values are in the same range as the training data. Model: Random Forest.")