import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import urllib.request

# ========================================
# AUTO-DOWNLOAD MODEL + OPTIONAL FILES
# ========================================
os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/sipakmed_best_2.keras"
HISTORY_PATH = "models/history.pkl"
CM_PATH = "models/confusion_matrix.pkl"

# Model download (only runs once)
if not os.path.exists(MODEL_PATH):
    with st.spinner("First launch – downloading trained model (~4.5 MB)..."):
        url = "https://github.com/Kamunyamike/Automated-Cervical-Cancer-Detection-via-Deep-Learning/releases/download/Trained_model_for_Streamlit_deployment/sipakmed_best_2.keras"
        urllib.request.urlretrieve(url, MODEL_PATH)
        st.success("Model downloaded successfully!")

# Optional: download history & confusion matrix if you upload them later to the same release
for file_path, asset_name in [
    (HISTORY_PATH, "history.pkl"),
    (CM_PATH, "confusion_matrix.pkl")
]:
    if not os.path.exists(file_path):
        url = f"https://github.com/Kamunyamike/Automated-Cervical-Cancer-Detection-via-Deep-Learning/releases/download/Trained_model_for_Streamlit_deployment/{asset_name}"
        try:
            urllib.request.urlretrieve(url, file_path)
        except:
            pass  # Silently skip if not uploaded yet

# Load model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

CLASS_LABELS = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]

# ========================================
# Helper Functions
# ========================================
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def grad_cam(model, img_array, layer_name="conv2d_15"):  #last conv layer = best for Grad-CAM:
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], weights.numpy())
    cam = cv2.resize(cam, (64, 64))
    cam = np.maximum(cam, 0)
    heatmap = cam / cam.max() if cam.max() > 0 else cam
    return heatmap

def overlay_gradcam(img, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.resize(img, (64,64)), 0.6, heatmap, 0.4, 0)
    return overlay

# ========================================
# Streamlit UI
# ========================================
st.set_page_config(page_title="Cervical Cytology Classifier", layout="wide")
st.title("Automated Cervical Cytology Classification")
st.markdown("Upload Pap smear cell images to classify into 5 categories using a deep learning model.")

# Sidebar
st.sidebar.header("About this App")
st.sidebar.info("""
Developed for the **16th KEMRI Annual Scientific & Health Conference**  
**Authors:** Mikenickson Wanjohi et al.  
**Dataset:** SIPaKMeD  
**Model:** Custom CNN (TensorFlow/Keras)  
**Test Accuracy:** 91.79% | **Validation Accuracy:** 92.27%
""")

tab1, tab2 = st.tabs(["Classifier", "Dashboard"])

# -----------------------------
# Tab 1: Classifier
# -----------------------------
with tab1:
    st.success("Model loaded successfully! Ready for predictions.")
    uploaded_files = st.file_uploader("Upload Pap Smear Images", type=["jpg", "png", "jpeg", "bmp"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.image(img, caption="Uploaded Image", width=700)
            processed_img = preprocess_image(img)
            preds = model.predict(processed_img, verbose=0)
            pred_idx = np.argmax(preds)
            pred_class = CLASS_LABELS[pred_idx]
            confidence_score = float(preds[0][pred_idx])        # ← Python float
            confidence_percent = confidence_score * 100

            st.subheader(f"Prediction: **{pred_class}**")
            st.progress(confidence_score)
            st.write(f"Confidence: **{confidence_percent:.2f}%**")

            # Grad-CAM
            try:
                heatmap = grad_cam(model, processed_img)
                overlay = overlay_gradcam(img, heatmap)
                st.markdown("**Grad-CAM Heatmap (regions the model focused on):**")
                st.image(overlay, use_column_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM not available: {e}")

# -----------------------------
# Tab 2: Dashboard
# -----------------------------
with tab2:
    st.subheader("Model Performance Dashboard")

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            history = pickle.load(f)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(history['accuracy'], label='Train Accuracy')
        ax[0].plot(history['val_accuracy'], label='Val Accuracy')
        ax[0].set_title("Accuracy")
        ax[0].set_xlabel("Epochs")
        ax[0].legend()
        ax[1].plot(history['loss'], label='Train Loss')
        ax[1].plot(history['val_loss'], label='Val Loss')
        ax[1].set_title("Loss")
        ax[1].set_xlabel("Epochs")
        ax[1].legend()
        st.pyplot(fig)
    else:
        st.info("Training history (history.pkl) not found yet. Upload it to your GitHub release to display curves.")

    if os.path.exists(CM_PATH):
        with open(CM_PATH, "rb") as f:
            cm, labels = pickle.load(f)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)
    else:
        st.info("Confusion matrix (confusion_matrix.pkl) not found yet. Upload it to the release to show here.")



