# import pickle
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# # model load
# model = pickle.load(open("model.pkl", "rb"))

# @app.route("/")
# def home():
#     return "Model is running!"

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.json["features"]
#     prediction = model.predict([data])
#     return jsonify({"prediction": int(prediction[0])})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=10000)


import streamlit as st
import joblib
import numpy as np

# 1. Model Load karein
# Tasalli kar lein ke aapki file ka naam exactly 'model.pkl' hi hai
model = joblib.load('model.pkl')

st.set_page_config(page_title="Random Forest Model", page_icon="🌲")

st.title("🌲 Random Forest Model Deployment")
st.write("Niche diye gaye sliders se parameters change karein aur prediction check karein.")

# 2. 5 Features ke liye Sliders
st.sidebar.header("Input Parameters")

f1 = st.sidebar.slider("Feature 1", 0.0, 100.0, 75.0)
f2 = st.sidebar.slider("Feature 2", 0.0, 100.0, 50.0)
f3 = st.sidebar.slider("Feature 3", 0.0, 100.0, 50.0)
f4 = st.sidebar.slider("Feature 4", 0.0, 100.0, 50.0)
f5 = st.sidebar.slider("Feature 5", 0.0, 100.0, 50.0)

# 3. Prediction logic
if st.button("Predict"):
    # 5 features ko model ke format mein dalein
    input_data = np.array([[f1, f2, f3, f4, f5]])
    
    try:
        prediction = model.predict(input_data)
        st.success(f"Model ki Prediction hai: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Tip: Check karein ke aapka model 5 features hi accept karta hai.")

st.markdown("---")
st.info("Teacher's Task: Is interface ke zariye aap live parameters test kar saktay hain.")
