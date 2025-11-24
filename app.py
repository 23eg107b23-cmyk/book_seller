import streamlit as st
import joblib

# Load Model & Vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

st.title("📚 Amazon Bestseller Prediction App")
st.write("Enter the book name or details and find out if it can become a Bestseller!")

# Single input
book_input = st.text_input("📌 Enter Book Title / Book Details")

if st.button("Predict Bestseller"):
    if book_input.strip() == "":
        st.warning("Please enter a book name or details.")
    else:
        # Transform input
        text_features = vectorizer.transform([book_input])

        # Predict
        prediction = model.predict(text_features)[0]
        
        # Probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(text_features)[0][1]
        else:
            probability = 0.0

        # Output
        st.subheader("✅ Prediction Result:")
        if prediction == 1:
            st.success(f"🔥 Likely to be a **BESTSELLER!** (Confidence: {probability*100:.2f}%)")
        else:
            st.error(f"📉 Not likely to be a **bestseller** (Confidence: {probability*100:.2f}%)")
