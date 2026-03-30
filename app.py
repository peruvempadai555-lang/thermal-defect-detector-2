import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Layer, Conv2D, concatenate
import cv2
import requests

# ---------- CUSTOM LAYER (your fire_module) ----------
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

# ---------- PREPROCESSING FUNCTIONS (EXACTLY AS IN YOUR COLAB) ----------
def reduce_noise(image):
    """Bilateral filter for noise reduction (same as training)"""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def extract_roi(image, roi_size=160):
    """Extract central region of size roi_size x roi_size"""
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    half = roi_size // 2
    roi = image[cy-half:cy+half, cx-half:cx+half]
    if roi.size == 0:
        return image
    roi = reduce_noise(roi)
    return roi

# ---------- LOAD MODEL FROM HUGGING FACE ----------
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
    # 1. Load original image (RGB)
    original = Image.open(uploaded_file).convert('RGB')
    st.image(original, caption="📸 Original Image", width=300)

    # 2. Convert to numpy array for OpenCV processing
    img_np = np.array(original)

    # 3. Apply ROI extraction + noise reduction
    roi = extract_roi(img_np, roi_size=160)

    # 4. Resize to 224x224 (model input size)
    roi_resized = cv2.resize(roi, (224, 224))

    # 5. Display preprocessed image (for user transparency)
    preprocessed_pil = Image.fromarray(roi_resized)
    st.image(preprocessed_pil, caption="🛠 Preprocessed Image (ROI + Noise Reduction)", width=300)

    # 6. Prepare for model: convert to array, apply EfficientNet scaling
    img_array = img_to_array(preprocessed_pil)        # shape (224,224,3)
    img_array = preprocess_input(img_array)           # scale to [-1, 1] (EfficientNet)
    img_array = np.expand_dims(img_array, axis=0)     # add batch dimension

    # 7. Predict
    pred = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(pred[0])]
    confidence = np.max(pred[0])

    # 8. Show results
    st.success(f"**Prediction:** {pred_class}")
    st.metric("Confidence", f"{confidence:.2%}")

    # Optional: Show probability bar chart
    prob_dict = {CLASS_NAMES[i]: float(pred[0][i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(prob_dict)
    
