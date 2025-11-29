import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="YOLOv11 Object Detection", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("Ultralytics YOLOv11 Object Detection")
st.markdown("Upload an image or use the camera to detect objects.")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    source = st.radio("Select Source", ["Image Upload", "Camera"])

def process_image(image):
    # Run inference
    results = model.predict(image, conf=confidence_threshold)
    
    # Plot results on the image
    # results[0].plot() returns a BGR numpy array, convert to RGB for Streamlit
    res_plotted = results[0].plot()[:, :, ::-1]
    
    return res_plotted, results[0]

if source == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
        if st.button("Detect Objects"):
            with st.spinner("Running detection..."):
                result_img, results = process_process = process_image(image)
                
                with col2:
                    st.subheader("Detected Objects")
                    st.image(result_img, use_container_width=True)
                
                # Show detections in text format
                st.subheader("Detection Results")
                for box in results.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    conf = float(box.conf[0])
                    st.write(f"- **{label}**: {conf:.2f} confidence")

elif source == "Camera":
    camera_input = st.camera_input("Take a picture")
    
    if camera_input is not None:
        image = Image.open(camera_input)
        
        with st.spinner("Running detection..."):
            result_img, results = process_image(image)
            
            st.subheader("Detected Objects")
            st.image(result_img, use_container_width=True)
            
            # Show detections in text format
            with st.expander("See detection details"):
                for box in results.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    conf = float(box.conf[0])
                    st.write(f"- **{label}**: {conf:.2f} confidence")

st.markdown("---")
st.markdown("Powered by [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)")
