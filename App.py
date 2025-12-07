import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "models/sipakmed_best_2.keras"  # or "models/best_model.h5"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please place your trained model there.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_LABELS = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]

# -----------------------------
# Helper Functions
# -----------------------------
def preprocess_image(img):
    """Resize and normalize image for model input"""
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def grad_cam(model, img_array, layer_name="conv2d_7"):
    """Generate Grad-CAM heatmap"""
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], weights.numpy())
    cam = cv2.resize(cam, (64, 64))
    cam = np.maximum(cam, 0)
    heatmap = cam / cam.max()
    return heatmap

def overlay_gradcam(img, heatmap):
    """Overlay Grad-CAM heatmap on original image"""
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.resize(img, (64,64)), 0.6, heatmap, 0.4, 0)
    return overlay

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Cervical Cytology Classifier", layout="wide")

st.title("🧬 Automated Cervical Cytology Classification")
st.markdown("Upload Pap smear cell images to classify into 5 categories using a deep learning model.")

# Sidebar
st.sidebar.header("About this App")
st.sidebar.info("""
Developed for the **16th KEMRI Annual Scientific & Health Conference**  
**Authors:** Mikenickson Wanjohi et al.  
**Dataset:** SIPaKMeD  
**Model:** Custom CNN (TensorFlow/Keras)  
**Accuracy:** ~95.5% Test Accuracy  
""")

# Tabs for functionality
tab1, tab2 = st.tabs(["🔍 Classifier", "📊 Dashboard"])

# -----------------------------
# Tab 1: Classifier
# -----------------------------
with tab1:
    uploaded_files = st.file_uploader("Upload Pap Smear Images", type=["jpg", "png", "jpeg","bmp"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Preprocess
            processed_img = preprocess_image(img)

            # Prediction
            preds = model.predict(processed_img)
            pred_class = CLASS_LABELS[np.argmax(preds)]
            confidence = np.max(preds)

            st.subheader(f"Prediction: {pred_class}")
            st.write(f"Confidence: {confidence:.2f}")

            # Grad-CAM
            try:
                heatmap = grad_cam(model, processed_img)
                overlay = overlay_gradcam(img, heatmap)
                st.markdown("**Grad-CAM Visualization:**")
                st.image(overlay, caption="Highlighted Regions", use_column_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM could not be generated: {e}")

# -----------------------------
# Tab 2: Dashboard
# -----------------------------
with tab2:
    st.subheader("Model Performance Dashboard")

    # Load training history if saved as pickle
    history_path = "models/history.pkl"
    if os.path.exists(history_path):
        with open(history_path, "rb") as f:
            history = pickle.load(f)

        # Accuracy plot
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        ax[0].plot(history['accuracy'], label='Train Acc')
        ax[0].plot(history['val_accuracy'], label='Val Acc')
        ax[0].set_title("Training vs Validation Accuracy")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend()

        # Loss plot
        ax[1].plot(history['loss'], label='Train Loss')
        ax[1].plot(history['val_loss'], label='Val Loss')
        ax[1].set_title("Training vs Validation Loss")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss")
        ax[1].legend()

        st.pyplot(fig)
    else:
        st.info("Training history not found. Save it as 'models/history.pkl' to view curves.")

    # Confusion matrix if saved
    cm_path = "models/confusion_matrix.pkl"
    if os.path.exists(cm_path):
        with open(cm_path, "rb") as f:
            cm, class_labels = pickle.load(f)

        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
    else:
        st.info("Confusion matrix not found. Save it as 'models/confusion_matrix.pkl' to view.")