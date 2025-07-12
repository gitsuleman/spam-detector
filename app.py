# app.py
import streamlit as st
import pandas as pd
import pickle

# Load model and accuracy
with open('spam_model.pkl', 'rb') as f:
    model, accuracy = pickle.load(f)

st.title("📧 Spam Detection App")
st.write("Built with Streamlit & Machine Learning")
st.markdown(f"**Model Accuracy:** `{accuracy*100:.2f}%`")

# Tabs for single/bulk detection
tab1, tab2 = st.tabs(["🔍 Single Message", "📁 Bulk Messages"])

# -------------------- TAB 1: Single Prediction --------------------
with tab1:
    user_input = st.text_area("Enter your message here")

    if st.button("Predict", key="predict_single"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            prediction = model.predict([user_input])[0]
            if prediction == 1:
                st.error("🚫 This message is SPAM.")
            else:
                st.success("✅ This message is NOT SPAM.")

# -------------------- TAB 2: Bulk Prediction from File --------------------
with tab2:
    st.write("Upload a CSV file with a column `message`")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'message' not in df.columns:
                st.error("❌ CSV must have a 'message' column.")
            else:
                df['prediction'] = model.predict(df['message'])
                df['prediction'] = df['prediction'].map({1: 'SPAM', 0: 'NOT SPAM'})
                st.success("✅ Predictions completed")
                st.write(df[['message', 'prediction']])
                st.download_button("📥 Download Results as CSV", df.to_csv(index=False), "results.csv", "text/csv")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
