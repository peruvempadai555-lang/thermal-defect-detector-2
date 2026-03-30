import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
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

# ---------- PREPROCESSING FUNCTIONS ----------
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

# ---------- LOAD MODEL ----------
url = "https://huggingface.co/spaces/M-Parames01/thermal-defect-model/resolve/main/fusion_resume_model_1.keras?download=true"
response = requests.get(url)
with open("fusion_resume_model_1.keras", "wb") as f:
    f.write(response.content)

model = tf.keras.models.load_model("fusion_resume_model_1.keras", custom_objects={'fire_module': fire_module}, compile=False)

CLASS_NAMES = ['No_Defect', 'Minor_Defect', 'Major_Defect']

# ---------- UI ----------
st.title("🔥 Thermal Image Defect Detection")
uploaded_file = st.file_uploader("Choose a thermal image...", type=["jpg","jpeg","png"])

if uploaded_file:
    # Original
    original = Image.open(uploaded_file).convert('RGB')
    st.image(original, caption="Original Image", width=300)

    # Convert to numpy for OpenCV
    img_np = np.array(original)

    # ROI + Denoise
    roi = extract_roi(img_np, roi_size=160)
    roi_resized = cv2.resize(roi, (224, 224))

    # Display preprocessed image
    preprocessed_pil = Image.fromarray(roi_resized)
    st.image(preprocessed_pil, caption="Preprocessed Image (ROI + Noise Reduced)", width=300)

    # Prepare for model
    img_array = img_to_array(preprocessed_pil)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(pred[0])]
    confidence = np.max(pred[0])

    st.success(f"Prediction: {pred_class}")
    st.metric("Confidence", f"{confidence:.2%}")
    
