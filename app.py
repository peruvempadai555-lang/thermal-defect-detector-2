import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
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
    # 1. Load, resize to 224x224
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img).astype(np.uint8)

    # 2. ROI extraction & resize back to 224x224
    roi = extract_roi(img_np, roi_size=160)
    roi_resized = cv2.resize(roi, (224, 224))

    # 3. First preprocess_input
    x = preprocess_input(roi_resized.astype(np.float32))

    # 4. Per‑image min‑max normalization to [0,1]
    x_min, x_max = x.min(), x.max()
    if x_max - x_min > 1e-8:
        x_norm = (x - x_min) / (x_max - x_min)
    else:
        x_norm = x - x_min

    # 5. Scale to 0‑255 and convert to uint8
    x_uint8 = (x_norm * 255).astype(np.uint8)

    # 6. Second preprocess_input
    x_final = preprocess_input(x_uint8.astype(np.float32))

    # 7. Batch dimension
    x_batch = np.expand_dims(x_final, axis=0)

    # For display (optional)
    roi_display = Image.fromarray(roi_resized)
    return x_batch, roi_display

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    url = "https://huggingface.co/spaces/M-Parames01/thermal-defect-model/resolve/main/fusion_resume_model_1.keras?download=true"
    response = requests.get(url)
    with open("fusion_resume_model_1.keras", "wb") as f:
        f.write(response.content)
    model = tf.keras.models.load_model(
        "fusion_resume_model_1.keras",
        custom_objects={'fire_module': fire_module},
        compile=False
    )
    return model

model = load_model()
CLASS_NAMES = ['No_Defect', 'Minor_Defect', 'Major_Defect']

# ---------- STREAMLIT UI ----------
st.title("🔥 Thermal Image Defect Detection")
st.markdown("Upload a thermal image to classify as **No Defect**, **Minor Defect**, or **Major Defect**.")

uploaded_file = st.file_uploader("Choose a thermal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original = Image.open(uploaded_file).convert('RGB')
    st.image(original, caption="📸 Original Image", width=300)

    roi_batch, roi_display = preprocess_image(uploaded_file)
    st.image(roi_display, caption="🛠 ROI (after crop & resize)", width=300)

    pred = model.predict(roi_batch)
    pred_class = CLASS_NAMES[np.argmax(pred[0])]
    confidence = np.max(pred[0])

    st.success(f"**Prediction:** {pred_class}")
    st.metric("Confidence", f"{confidence:.2%}")

    prob_dict = {CLASS_NAMES[i]: float(pred[0][i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(prob_dict)
