import os
import joblib

from PIL import Image

import torch
from ultralytics import YOLO
from transformers import ViTForImageClassification

from scripts.model import predict_yolo_cls_svc_single_image

import streamlit as st

import warnings
warnings.filterwarnings("ignore")



# Streamlit app for detecting Bangkok Metro Station Signage using YOLOv8s and SVC + CLS model

# Load model
@st.cache_resource
def load_yolo_cls_svc_model():
    '''Load the YOLOv8s [Single-label and Multi-label] and SVC + CLS model trained for Bangkok Metro Station Signage detection.'''

    # Load YOLOv8s [Single-label and Multi-label] models
    yolo_model_multi_labels_path = os.path.join("models", "deep_learning_yolo_multi_labels", "best.pt")
    yolo_model_single_label_path = os.path.join("models", "deep_learning_yolo_single_label", "best.pt")

    yolo_model_multi_labels = YOLO(yolo_model_multi_labels_path)
    yolo_model_single_label = YOLO(yolo_model_single_label_path)

    # Load SVC + CLS Model
    svc_model_path = os.path.join("models", "classical_machine_learning", "best_svc_model.pkl")
    svc_model = joblib.load(svc_model_path)

    # Load ViT Model/Backbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vit_model_path = os.path.join("models", "deep_learning_vit", "checkpoint-1990")
    vit_model = ViTForImageClassification.from_pretrained(vit_model_path).to(device)
    vit_model.eval()
    vit_backbone = vit_model.vit

    # Load ViT Transform
    vit_transform_path = os.path.join("models", "deep_learning_vit", "inference_transform.pkl")
    vit_transform = joblib.load(vit_transform_path)

    # Load id2label mapping
    id2label_path = os.path.join("models", "deep_learning_yolo_multi_labels", "id2label.pkl")
    id2label = joblib.load(id2label_path)

    return yolo_model_multi_labels, yolo_model_single_label, svc_model, vit_backbone, vit_transform, id2label



def run():
    '''Run the Streamlit app for detecting Bangkok Metro Station Signage.'''

    # Streamlit page title
    st.title("Bangkok Metro Station Signage Detection")
    st.markdown('**This is a demo application for identifying metro station name and signage from images containing metro station signage, for 8 skytrain stations in BTS Silom Line (Dark Green Line without extension:).**')
    hide_footer_style = """<style>.reportview-container .main footer {visibility: hidden;}"""
    st.markdown(hide_footer_style, unsafe_allow_html=True)

    # Load the YOLOv8s [Single-label and Multi-label] and SVC + CLS Models, together with ViT Backbone for CLS token extraction from ViT Model
    yolo_model_multi_labels, yolo_model_single_label, svc_model, vit_backbone, vit_transform, id2label = load_yolo_cls_svc_model()

    # File uploader
    uploaded_file = st.file_uploader("Upload an image with BTS Silom Line station signage", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.subheader("Uploaded Image")
        # Open uploaded_image and convert it to RGB
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # Predict metro station and draw bounding boxes in the uploaded image
        new_resulting_image_path, cropped_image_path, svc_pred = predict_yolo_cls_svc_single_image(yolo_model_multi_labels, yolo_model_single_label, svc_model, vit_backbone, vit_transform, id2label, uploaded_image)

        # Display the detection result
        st.subheader("Detection Result")
        st.write("Detected Station:")
        if svc_pred:
            st.write(f"{svc_pred}")
        else:
            st.write("No station detected.")

        # Display the resulting image with bounding boxes
        st.image(new_resulting_image_path, caption="Detected Metro Station Signage", use_container_width=True)

    else:
        st.warning("Please upload an image file to proceed.")



if __name__ == "__main__":
    run()
