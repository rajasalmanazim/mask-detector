import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Face Mask Detector", layout="wide")
st.title("Face Mask Detection (YOLOv8)")
st.write("Upload an image, adjust the confidence threshold, and see detections.")


@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()

conf = st.slider("Confidence threshold", 0.05, 1.0, 0.25, 0.05)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    result = model.predict(np.array(img), conf=conf)[0]
    plotted = result.plot()[:, :, ::-1]

    c1, c2 = st.columns(2)
    c1.image(img, caption="Original", use_container_width=True)
    c2.image(plotted, caption="Detections", use_container_width=True)

    st.subheader("Detected objects")
    boxes = result.boxes
    if len(boxes) == 0:
        st.info("No objects detected. Try lowering the threshold.")
    else:
        rows = [
            {"Class": model.names[int(b.cls[0])],
             "Confidence": f"{float(b.conf[0]):.2%}"}
            for b in boxes
        ]
        st.table(rows)
        st.caption(f"Total objects: {len(boxes)}")