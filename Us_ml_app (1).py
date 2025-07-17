import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load('kmeans_model.pkl')

st.title("Country Classification: Foreign Aid Predictor")

st.markdown("Enter the following national statistics to determine aid priority:")

# User inputs
life_expec = st.number_input("Life Expectancy", min_value=0.0, step=0.1)
health = st.number_input("Health Spending (% of GDP)", min_value=0.0, step=0.1)
child_mort = st.number_input("Child Mortality (per 1000 births)", min_value=0.0, step=0.1)
total_fer = st.number_input("Total Fertility Rate", min_value=0.0, step=0.1)

imports = st.number_input("Imports (% of GDP)", min_value=0.0, step=0.1)
exports = st.number_input("Exports (% of GDP)", min_value=0.0, step=0.1)

income = st.number_input("Income per person", min_value=0.0, step=10.0)
gdpp = st.number_input("GDP per capita", min_value=0.0, step=10.0)
inflation = st.number_input("Inflation rate (%)", min_value=0.0, step=0.1)

if st.button("Predict Aid Need"):
    # Create feature dataframe without scaling
    input_data = pd.DataFrame({
        'Health': [(life_expec / 65) + (health / 6) - (child_mort / 40) - (total_fer / 2.5)],
        'Trade': [(imports / 40) + (exports / 40)],
        'Finance': [(income / 15000) + (gdpp / 15000) - (inflation / 10)]
    })

    # Make prediction using the model (without scaling)
    prediction = model.predict(input_data)[0]

    # Map predictions to classes
    class_mapping = {
        0: "Not a priority",
        1: "Requires foreign aid",
        2: "Does NOT require foreign aid"
    }

    st.subheader(f"Prediction: **{class_mapping[prediction]}**")
