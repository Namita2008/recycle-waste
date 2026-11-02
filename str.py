
import json
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ EcoSort AI")
st.subheader("Recycle Waste Classification")
st.write("Upload a waste image to identify its category.")

# Load class names and model
@st.cache_resource
def load_ai():
    with open("classes.json", "r") as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model("cat.keras")
    return model, class_names

try:
    model, class_names = load_ai()

    # Upload image
    uploaded_file = st.file_uploader(
        "Choose a waste image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")

        st.image(
            img,
            caption="Uploaded Waste Image",
            width=350
        )

        if st.button("Identify Waste ♻️"):
            # Image preprocessing
            img = img.resize((64, 64))
            x = tf.keras.utils.img_to_array(img)
            x = tf.expand_dims(x, 0)

            # AI prediction
            p = model.predict(x, verbose=0)
            scores = p[0].flatten()

            if len(scores) != len(class_names):
                st.error("Model classes do not match classes.json")
            else:
                index = int(np.argmax(scores))
                predicted_class = class_names[index]

                st.success(
                    f"Predicted Waste Category: {predicted_class}"
                )

                st.write("Model score:", f"{scores[index]:.2%}")

                if scores[index] < 0.60:
                    st.warning(
                        "Prediction uncertain. Please verify manually."
                    )

                st.info(
                    "Follow your local waste collection guidelines."
                )

except Exception as e:
    st.error(f"Error: {e}")
