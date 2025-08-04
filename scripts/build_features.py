import numpy as np

import os

import cv2
from PIL import Image

import torch
from skimage.color import rgb2gray
from skimage.feature import hog



def build_features_svc():
    '''Build features and labels for SVC model from cropped images.'''

    splits = ["train", "valid", "test"]

    # Target size for HOG feature extraction
    target_size = (128, 64)
    num_scales = 1

    # Minimum crop dimension to ensure HOG feature extraction is valid
    min_crop_dim = 16

    features_by_split = {}
    labels_by_split = {}

    # Iterate through each split (train, valid, test)
    for split in splits:
        split_crop_dir = os.path.join("data", "outputs_multi_labels", split, "crops")
        split_features = []
        split_labels = []

        # Loop through each label folder in the split crop directory
        for label_folder in os.listdir(split_crop_dir):
            label_path = os.path.join(split_crop_dir, label_folder)
            if not os.path.isdir(label_path):
                continue
            
            # Iterate through each image in the label folder
            for img_file in os.listdir(label_path):
                # Read image files
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                for scale in np.linspace(1.0, 2.0, num_scales):
                    # Calculate the new dimensions and crop the image
                    sw = int(w * scale)
                    sh = int(h * scale)
                    cx, cy = w // 2, h // 2

                    x1 = max(cx - sw // 2, 0)
                    y1 = max(cy - sh // 2, 0)
                    x2 = min(cx + sw // 2, w)
                    y2 = min(cy + sh // 2, h)

                    crop = img[y1:y2, x1:x2]
                    if crop.shape[0] < min_crop_dim or crop.shape[1] < min_crop_dim:
                        # Skip crops that are too small
                        continue

                    try:
                        # Resize, convert to grayscale, and extract HOG features
                        hog_feat = extract_hog_feature(crop, target_size)

                        # Append the feature and label
                        split_features.append(hog_feat)
                        split_labels.append(label_folder)
                    except Exception as e:
                        continue
                      
        features_by_split[split] = split_features
        labels_by_split[split] = split_labels
        print(f"[RESULT] {split}: {len(split_features)} samples")

    return features_by_split, labels_by_split



def extract_hog_feature(image, target_size=(128, 64)):
    '''Extract HOG feature from an image, which is resized to the target size, and ready for SVC model.'''

    # Resize the image to the target size
    resized = cv2.resize(image, target_size)

    # Convert the resized image to grayscale
    gray = rgb2gray(resized)

    # Extract HOG features from the grayscale image
    hog_feat = hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    return hog_feat



def extract_cls_token(image_path, vit_backbone, inference_transform):
    '''Extract CLS tokens from Trained Vision Transformer, from the last hidden state'''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = Image.open(image_path).convert("RGB")
    inputs = inference_transform(image).unsqueeze(0).to(device)
    
    # Get CLS tokens
    with torch.no_grad():
        outputs = vit_backbone(inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]
    
    return cls_token.cpu().numpy().flatten()