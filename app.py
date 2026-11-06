import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Intel Image Classifier", page_icon="🏞️")


@st.cache_resource
def load_bundle():
    return joblib.load("models/best_model.joblib")


bundle = load_bundle()
model = bundle["model"]
kmeans = bundle["kmeans"]
classes = bundle["classes"]
img_size = bundle["img_size"]
k = bundle["k"]
sift = cv2.SIFT_create()


def preprocess(img_bgr):
    img = cv2.resize(img_bgr, (img_size, img_size))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def extract_features(gray):
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return None, 0
    words = kmeans.predict(descriptors.astype(np.float32))
    hist = np.bincount(words, minlength=k).astype(float)
    return (hist / hist.sum()).reshape(1, -1), len(keypoints)


st.title("Intel Image Classifier")
st.write("Upload a photo and the model will predict one of: " + ", ".join(classes) + ".")
st.caption(f"Model: {bundle['model_name']} | test accuracy: {bundle['test_accuracy']:.2%} | SIFT + Bag of Visual Words")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read this image. Please try another file.")
    else:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_container_width=True)
        features, n_keypoints = extract_features(preprocess(img_bgr))

        if features is None:
            st.warning("No SIFT keypoints were found in this image, so it cannot be classified.")
        else:
            probs = model.predict_proba(features)[0]
            best = int(np.argmax(probs))
            st.success(f"Prediction: **{classes[best]}**  (confidence: {probs[best]:.1%})")
            st.caption(f"{n_keypoints} SIFT keypoints used")
            chart = pd.DataFrame({"class": classes, "probability": probs}).set_index("class")
            st.bar_chart(chart)
