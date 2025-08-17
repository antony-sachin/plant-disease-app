import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ----- CONFIG -----
MODEL_FILENAME = 'PlantDNet.h5'
IMAGE_SIZE = (64, 64)
DISEASE_CLASSES = [
    'Pepper Bell Bacterial Spot',
    'Pepper Bell Healthy',
    'Potato Early Blight',
    'Potato Late Blight',
    'Potato Healthy',
    'Tomato Bacterial Spot',
    'Tomato Early Blight',
    'Tomato Late Blight',
    'Tomato Leaf Mold',
    'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites',
    'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus',
    'Tomato Mosaic Virus',
    'Tomato Healthy'
]
st.set_page_config(
    page_title="Plant Disease Diagnosis",
    layout="centered",
    page_icon="🌱"
)

# ----- TITLE & HEADER -----
st.markdown("""
    <div style='text-align:center'>
        <h1 style='color:#26734d;'>🌿 Plant Village Disease Diagnoser</h1>
        <h3 style='color:#206040; font-weight:400; margin-top:-10px;'>
            Detect plant diseases from images using Artificial Intelligence
        </h3>
        <p style='font-size:17px;'>Upload a clear leaf photo and get instant disease diagnosis along with class confidence scores.</p>
    </div>
""", unsafe_allow_html=True)

# ----- LOAD MODEL (CACHED) -----
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_FILENAME, compile=False)

model = load_model()

# ----- IMAGE PREDICTION LOGIC -----
def model_predict(img_file, model):
    """Preprocesses the uploaded image and returns model probabilities."""
    img = image.load_img(img_file, target_size=IMAGE_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32') / 255.
    preds = model.predict(x)
    return preds

def get_predicted_class(preds):
    a = preds[0]
    ind = np.argmax(a)
    return DISEASE_CLASSES[ind], a

# ----- SIDEBAR & INFO -----
with st.sidebar:
    st.header("🧑‍🔬 About This Tool")
    st.markdown("""
    - **Model:** Deep Learning (DenseNet120/Custom CNN)
    - **Input:** JPG, JPEG, or PNG leaf image
    - **Output:** Most probable disease class and confidence scores  
    - **Usage:** Just upload and hit *Diagnose*
    - **Tip:** For best results, use a clear closeup of a single leaf.
    """)
    st.info("🌱 Made for farmers, researchers, and students.")

# ----- FILE UPLOAD -----
uploaded_file = st.file_uploader(
    "Upload a leaf image (JPG or PNG, up to 5MB)", 
    type=["jpg", "jpeg", "png"],
    help="Photo should clearly show the affected plant leaf"
)

# ----- MAIN LOGIC -----
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Leaf Image', use_container_width=True)
    st.write("---")

    col1, col2 = st.columns([1,2])
    with col1:
        diagnose = st.button("🔬 Diagnose", type='primary', use_container_width=True)
    with col2:
        st.write("*AI will analyze the image for plant diseases.*")

    if diagnose:
        with st.spinner('Analyzing image…'):
            preds = model_predict(uploaded_file, model)
            pred_label, prob_vector = get_predicted_class(preds)
        st.success(f"🩺 **Predicted Disease:** {pred_label}")

        # Show a progress bar for the top score
        score = float(prob_vector[np.argmax(prob_vector)])
        st.progress(score)
        st.caption(f"Model confidence: {score:0.2%}")

        # Show confidence table
        st.markdown("#### 🔎 Class Confidence Scores:")
        prob_dict = {DISEASE_CLASSES[i]: float(prob_vector[i]) for i in range(len(DISEASE_CLASSES))}
        # Sort and display probabilities as a DataFrame
        import pandas as pd
        df_prob = pd.DataFrame(
            list(prob_dict.items()),
            columns=['Disease Class', 'Probability']
        ).sort_values(by='Probability', ascending=False).reset_index(drop=True)
        st.dataframe(
            df_prob.style
            .bar(subset=['Probability'], color='#1b9966', vmin=0, vmax=1)
            .format({'Probability': '{:.2%}'})
        )
else:
    st.info("⬆️ Upload a plant leaf image to begin.")

# ----- FOOTER -----
st.markdown("""
---
<div style='text-align:center; color: #888; font-size:15px;'>
    © 2025 Plant Village AI | Powered by Deep Learning, Streamlit & Your Model
</div>
""", unsafe_allow_html=True)
