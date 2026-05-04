import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ---------- MUST BE FIRST STREAMLIT COMMAND ----------
st.set_page_config(page_title="Fake Profile Detector", page_icon="🕵️")

# ---------- CONFIG ----------
IMG_SIZE = (224, 224)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("../models/fake_profile_detector.h5")

model = load_model()

# ---------- UI ----------
st.title("🕵️ Fake Profile Detector")

st.markdown("""
This app detects whether a profile image is **real or fake** using a deep learning model (**MobileNetV2**).

📌 Upload an image to see the prediction.
""")

uploaded_file = st.file_uploader(
    "Upload Profile Image",
    type=["jpg", "png", "jpeg"]
)

# ---------- MAIN LOGIC ----------
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array)[0][0]

    score = float(prediction)

    st.write(f"### Prediction Score: {score:.4f}")

    if score > 0.5:
        st.success(f"✅ Real Profile (Confidence: {score:.2f})")
    else:
        st.error(f"❌ Fake Profile (Confidence: {1 - score:.2f})")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Built using Transfer Learning (MobileNetV2) + Streamlit")