import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Simple ML app using Logistic Regression")

# Upload dataset
uploaded_file = st.file_uploader("Upload credit card dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Features & target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    st.write("### Model Performance")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(model, "fraud_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    st.success("Model trained and saved!")

    # Prediction section
    st.write("### Predict a New Transaction")

    input_data = []
    for col in X.columns:
        val = st.number_input(col, value=0.0)
        input_data.append(val)

    if st.button("Predict Fraud"):
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction")
        else:
            st.success("✅ Legitimate Transaction")
