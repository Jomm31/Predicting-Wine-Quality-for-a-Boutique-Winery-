import streamlit as st
import pandas as pd
import joblib

# Load the balanced trained model
model = joblib.load("random_forest_balanced_compressed.joblib")


# Imputation values (replace with the real ones from training if available)
median_citric_acid = 0.26
mean_density = 0.996783
mean_pH = 3.310454

# Page config
st.set_page_config(page_title="🍷 Wine Quality Predictor", layout="centered")

st.title("🍇 Wine Quality Predictor (Balanced Multi-class)")

st.markdown("""
This application predicts the **wine quality rating (3–8)** based on its chemical properties.  

👉 According to the company:  
- **Good Quality** = rating **7–10**  
- **Not Good** = rating **below 7**  
""")

# Input fields (organized into columns for cleaner UI)
col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, format="%.2f")
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, format="%.2f")
    citric_acid = st.number_input("Citric Acid", min_value=0.0, format="%.2f")
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, format="%.2f")

with col2:
    chlorides = st.number_input("Chlorides", min_value=0.0, format="%.3f")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, format="%.1f")
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, format="%.1f")
    density = st.number_input("Density", min_value=0.0, format="%.4f")

with col3:
    pH = st.number_input("pH", min_value=0.0, format="%.2f")
    sulphates = st.number_input("Sulphates", min_value=0.0, format="%.2f")
    alcohol = st.number_input("Alcohol", min_value=0.0, format="%.2f")

# Prediction button
if st.button("🔮 Predict Wine Quality"):
    # Create DataFrame
    input_df = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])

    # Handle missing values with training imputations
    input_df["citric acid"] = input_df["citric acid"].fillna(median_citric_acid)
    input_df["density"] = input_df["density"].fillna(mean_density)
    input_df["pH"] = input_df["pH"].fillna(mean_pH)

    # Predictions
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    # Confidence score for predicted class
    confidence = prediction_proba[prediction - 3]  # adjust index since classes start at 3

    # Class interpretation
    quality_class = "🍷 Good Quality" if prediction >= 7 else "❌ Not Good Quality"

    # Show results
    st.subheader("Prediction Result")
    st.success(f"**Predicted Quality Rating:** {prediction}/10")
    st.info(f"**Classification:** {quality_class}")
    st.write(f"**Confidence Score:** {confidence:.2f}")

    # Show all class probabilities as a bar chart
    st.subheader("Class Probabilities")
    prob_df = pd.DataFrame({
        "Quality Rating": list(range(3, 9)),
        "Probability": prediction_proba
    })
    st.bar_chart(prob_df.set_index("Quality Rating"))



