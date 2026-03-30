import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Layer, Conv2D, concatenate
import cv2
import requests

# ---------- CUSTOM LAYER ----------
class fire_module(Layer):
    def __init__(self, squeeze, expand, **kwargs):
        super().__init__(**kwargs)
        self.squeeze = squeeze
        self.expand = expand
        self.conv_squeeze = Conv2D(squeeze, (1,1), activation='tanh', padding='same')
        self.conv_expand1 = Conv2D(expand, (1,1), activation='tanh', padding='same')
        self.conv_expand3 = Conv2D(expand, (3,3), activation='tanh', padding='same')
    def call(self, inputs):
        s = self.conv_squeeze(inputs)
        e1 = self.conv_expand1(s)
        e3 = self.conv_expand3(s)
        return concatenate([e1, e3])
    def get_config(self):
        config = super().get_config()
        config.update({'squeeze': self.squeeze, 'expand': self.expand})
        return config

# ---------- PREPROCESSING (EXACT MATCH TO COLAB) ----------
def reduce_noise(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def extract_roi(image, roi_size=160):
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    half = roi_size // 2
    roi = image[cy-half:cy+half, cx-half:cx+half]
    if roi.size == 0:
        return image
    roi = reduce_noise(roi)
    return roi

def preprocess_image(uploaded_file):
    # Step 1: Load image as RGB and resize to 224x224 (same as training)
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img).astype(np.uint8)   # shape (224,224,3)

    # Step 2: Extract central 160x160 ROI
    roi = extract_roi(img_np, roi_size=160)

    # Step 3: Resize ROI back to 224x224 (same as training)
    roi_resized = cv2.resize(roi, (224, 224))

    # Step 4: Apply EfficientNet preprocessing
    roi_resized = preprocess_input(roi_resized.astype(np.float32))

    # Step 5: Add batch dimension
    roi_batch = np.expand_dims(roi_resized, axis=0)

    # Also return the ROI image for display (after resizing to 224x224)
    roi_display = Image.fromarray(roi_resized.astype(np.uint8))
    return roi_batch, roi_display

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    url = "https://huggingface.co/spaces/M-Parames01/thermal-defect-model/resolve/main/fusion_resume_model_1.keras?download=true"
    response = requests.get(url)
    with open("fusion_resume_model_1.keras", "wb") as f:
        f.write(response.content)
    model = tf.keras.models.load_model("fusion_resume_model_1.keras", custom_objects={'fire_module': fire_module}, compile=False)
    return model

model = load_model()
CLASS_NAMES = ['No_Defect', 'Minor_Defect', 'Major_Defect']

# ---------- STREAMLIT UI ----------
st.title("🔥 Thermal Image Defect Detection")
st.markdown("Upload a thermal image to classify as **No Defect**, **Minor Defect**, or **Major Defect**.")

uploaded_file = st.file_uploader("Choose a thermal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    original = Image.open(uploaded_file).convert('RGB')
    st.image(original, caption="📸 Original Image", width=300)

    # Preprocess
    roi_batch, roi_display = preprocess_image(uploaded_file)

    # Display preprocessed image (what the model actually sees)
    st.image(roi_display, caption="🛠 Preprocessed Image (Resize → Crop → Noise Reduction → Resize)", width=300)

    # Predict
    pred = model.predict(roi_batch)
    pred_class = CLASS_NAMES[np.argmax(pred[0])]
    confidence = np.max(pred[0])

    st.success(f"**Prediction:** {pred_class}")
    st.metric("Confidence", f"{confidence:.2%}")

    # Optional: bar chart
    prob_dict = {CLASS_NAMES[i]: float(pred[0][i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(prob_dict)
