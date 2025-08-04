import numpy as np

import os
import shutil
import joblib
import hashlib

import cv2
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imgaug import augmenters as iaa
from torchvision import transforms
from transformers import AutoImageProcessor
from datasets import load_dataset

from scripts.build_features import extract_cls_token



def is_image_corrupted(filepath):
    '''Check if an image file is corrupted.'''
    try:
        with Image.open(filepath) as img:
            # Verify image integrity
            img.verify()
        return False
    except Exception:
        return True



def get_file_hash(filepath):
    '''Return SHA256 hash of file content.'''
    hash_func = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()



def remove_corrupted_duplicates_and_rename_images(station_folder_name):
    '''Remove corrupted images, duplicates, and rename images in the dataset.'''

    seen_hashes = set()
    copied_count = 0

    for station in station_folder_name:
        seen_hashes = set()
        image_id = 0

        station_path = os.path.join("data", "raw", station)
        if not os.path.isdir(station_path):
            continue
        dst_station_path = os.path.join("data", "processed_before_roboflow", station)
        os.makedirs(dst_station_path, exist_ok=True)
        for fname in os.listdir(station_path):
            src_img_path = os.path.join(station_path, fname)
            if not os.path.isfile(src_img_path):
                continue
            # Check for corrupted files
            if is_image_corrupted(src_img_path):
                print(f"Corrupted: {src_img_path}")
                continue
            # Check for duplicates
            file_hash = get_file_hash(src_img_path)
            if file_hash in seen_hashes:
                print(f"Duplicate: {src_img_path}")
                continue
            seen_hashes.add(file_hash)
            # Copy to destination folder
            image_id += 1
            ext = os.path.splitext(fname)[1]
            new_fname = f"{station}_{image_id:03d}{ext}"
            dst_img_path = os.path.join(dst_station_path, new_fname)
            shutil.copy2(src_img_path, dst_img_path)
            copied_count += 1

    print(f"{copied_count} unique, valid images are processed and copied to data/processed_before_roboflow folder.")
    return



def split_dataset(station_folder_name):
    '''Split the dataset into training, validation, and test sets, from the dataset that is in YOLO format with images and labels.'''

    input_base_dir = os.path.join("data", "processed")
    images_dir = os.path.join(input_base_dir, "train", "images")
    labels_dir = os.path.join(input_base_dir, "train", "labels")
    output_base_dir = os.path.join("data", "outputs_multi_labels")

    # Set the split ratio for train, validation, and test sets
    split_ratio = (0.8, 0.1, 0.1)  # train, valid, test

    # Load image-label pairs and classes
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg") or f.endswith(".png")]
    samples = []
    classes_per_image = []

    # Iterate through image files and their corresponding label files
    for img_file in image_files:
        label_file = img_file.rsplit('.', 1)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        if not os.path.exists(label_path):
            continue
        # Read the label file to get class IDs
        with open(label_path, "r") as f:
            class_ids = [line.strip().split()[0] for line in f if line.strip()]
        if class_ids:
            samples.append((img_file, label_file))
            classes_per_image.append(class_ids[0])  # Use first class ID (or a representative one)

    # Split data stratified by class
    train_val_files, test_files = train_test_split(samples, test_size=split_ratio[2], stratify=classes_per_image, random_state=42)
    train_val_classes = [label[0] for _, label in train_val_files]
    train_files, val_files = train_test_split(train_val_files, test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]), stratify=train_val_classes, random_state=42)

    splits = {
        "train": train_files,
        "valid": val_files,
        "test": test_files
    }

    # Copy files to split folders
    for split_name, files in splits.items():
        # Create directories for images and labels in the output base directory
        img_out = os.path.join(output_base_dir, split_name, "images")
        lbl_out = os.path.join(output_base_dir, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        # Copy images and labels to the respective split directories
        for img_file, label_file in files:
            shutil.copy(os.path.join(images_dir, img_file), os.path.join(img_out, img_file))
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(lbl_out, label_file))

    for fname in os.listdir(input_base_dir):
        # Copy any additional files (including data.yaml) to the output base directory
        input_path = os.path.join(input_base_dir, fname)
        output_path = os.path.join(output_base_dir, fname)
        if os.path.isfile(input_path):
            shutil.copy2(input_path, output_path)

    print("Stratified split complete.")

    return



def convert_yolo_to_bbox(x_center, y_center, width, height, img_w, img_h):
    '''Convert YOLO format to image crops'''
    x1 = int((x_center - width / 2) * img_w)
    y1 = int((y_center - height / 2) * img_h)
    x2 = int((x_center + width / 2) * img_w)
    y2 = int((y_center + height / 2) * img_h)
    return max(x1, 0), max(y1, 0), min(x2, img_w - 1), min(y2, img_h - 1)



def crop_augment_images_for_svc(label_map):
    '''Crop images in the training, validation, and test sets for SVC model. This uses YOLO format annotations to crop and get images of metro station signage.'''

    # Minimum input size before resize (must be at least HOG-safe)
    min_width, min_height = 16, 16

    # Define augmentation pipeline
    augmenter = iaa.Sequential([
        iaa.Affine(
            rotate=(-10, 10),
            scale=(0.9, 1.1),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
        ),
        iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)),
        iaa.Multiply((0.8, 1.2)),  # brightness
        iaa.LinearContrast((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0.0, 1.0))
    ])

    # Crop images for each split
    splits = ["train", "valid", "test"]
    for split in splits:
        split_dir = os.path.join("data", "outputs_multi_labels", split)
        image_dir = os.path.join(split_dir, "images")
        label_dir = os.path.join(split_dir, "labels")
        output_crop_dir = os.path.join(split_dir, "crops")

        # Create label folders
        for label in label_map.values():
            os.makedirs(os.path.join(output_crop_dir, label), exist_ok=True)

        # Count skipped images per label
        skipped_counts = {label: 0 for label in label_map.values()}
        total_counts = {label: 0 for label in label_map.values()}

        crop_count = 0
        for img_file in os.listdir(image_dir):
            if not img_file.endswith(".jpg"):
                continue
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))

            image = cv2.imread(img_path)
            if image is None or not os.path.exists(label_path):
                continue
            
            # Read image dimensions
            img_h, img_w = image.shape[:2]

            # Process each label file
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    x1, y1, x2, y2 = convert_yolo_to_bbox(x, y, w, h, img_w, img_h)
                    crop = image[y1:y2, x1:x2]
                    label = label_map[int(class_id)]
                    total_counts[label] += 1
                    ch, cw = crop.shape[:2]
                    if ch < min_height or cw < min_width:
                        skipped_counts[label] += 1
                        continue
                    cropped_image_name = f"crop_{crop_count}.jpg"
                    crop_path = os.path.join(output_crop_dir, label, cropped_image_name)
                    cv2.imwrite(crop_path, crop)
                    crop_count += 1
                    
                    # Generate 5 augmented versions of cropped images
                    for i in range(5):
                        augmented = augmenter(image=crop)
                        save_name = f"{os.path.splitext(cropped_image_name)[0]}_aug{i}.jpg"
                        cv2.imwrite(os.path.join(output_crop_dir, label, save_name), augmented)

        # Print summary for the split
        print(f"[{split}] Extracted {crop_count} cropped signage images.")
        print(f"[{split}] Skipped images (too small):")

        for label in label_map.values():
            total = total_counts[label]
            skipped = skipped_counts[label]
            print(f"  {label}: {skipped} skipped")
        
    return



def split_dataset_for_svc(features_by_split, labels_by_split):
    '''Split the dataset into training, validation, and test sets for SVC model.'''
    
    X_train = features_by_split["train"]
    y_train = labels_by_split["train"]
    X_valid = features_by_split["valid"]
    y_valid = labels_by_split["valid"]
    X_test = features_by_split["test"]
    y_test = labels_by_split["test"]

    # Combine training and validation sets for cross-validation
    X_train_full = X_train + X_valid
    y_train_full = y_train + y_valid

    # Scale X_train and X_test
    hog_scaler = StandardScaler()
    X_train_full_scaled = hog_scaler.fit_transform(X_train_full)
    X_test_scaled = hog_scaler.transform(X_test)

    # Save the scaler
    hog_scaler_path = os.path.join("models", "classical_machine_learning", "hog_scaler.pkl")
    joblib.dump(hog_scaler, hog_scaler_path)

    return X_train_full_scaled, y_train_full, X_test_scaled, y_test, hog_scaler



def prepare_vit_data():
    '''Prepare training data for Vision Transformer, in HuggingFace format'''

    # Load dataset
    crop_data_dir = os.path.join("data", "outputs_multi_labels")

    dataset = load_dataset(
        "imagefolder",
        data_files={
            "train": f"{crop_data_dir}/train/crops/**",
            "validation": f"{crop_data_dir}/valid/crops/**",
            "test": f"{crop_data_dir}/test/crops/**"
        }
    )

    # Apply transformation
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    # Same as train_transform, but excludes random augmentation, to prevent CLS features extracted from ViT model vary randomly across runs
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    # Preprocess and save pixel_values before training
    def preprocess_examples(examples):
        examples['pixel_values'] = [train_transform(img) for img in examples['image']]
        return examples

    dataset = dataset.map(preprocess_examples, batched=True)
    dataset.set_format(type='torch', columns=['pixel_values', 'label'])

    # Label mapping
    id2label = {i: label for i, label in enumerate(dataset['train'].features['label'].names)}
    label2id = {label: i for i, label in id2label.items()}

    # Save id2label
    id2label_path = os.path.join("models", "deep_learning_yolo_multi_labels", "id2label.pkl")
    joblib.dump(id2label, id2label_path)

    # Save inference_transform
    inference_transform_path = os.path.join("models", "deep_learning_vit", "inference_transform.pkl")
    joblib.dump(inference_transform, inference_transform_path)

    return dataset, id2label, label2id, inference_transform, processor



def prepare_svc_lr_rf_cls_features(vit_backbone, inference_transform):
    '''Prepare training data (CLS Features) for SVC, LR, RF models'''

    # Directories for the three splits
    base_dir = os.path.join("data", "outputs_multi_labels")
    splits = ["train", "valid", "test"]

    # Dictionaries to store features and labels for each split
    X_data = {}
    y_data = {}

    # Loop through all splits, to create train and test set
    for split in splits:
        print(f"Processing {split} split...")
        split_dir = os.path.join(base_dir, split, "crops")
        X_split, y_split = [], []

        # Get sorted station names to keep label order consistent
        station_names = sorted(os.listdir(split_dir))

        for label_idx, station_name in enumerate(station_names):
            station_dir = os.path.join(split_dir, station_name)

            for img_file in os.listdir(station_dir):
                # Get CLS tokens for X, and label for y
                img_path = os.path.join(station_dir, img_file)
                features = extract_cls_token(img_path, vit_backbone, inference_transform)
                X_split.append(features)
                y_split.append(label_idx)

        # Convert to NumPy arrays
        X_data[split] = np.array(X_split)
        y_data[split] = np.array(y_split)

        print(f"{split} split -> Features: {X_data[split].shape}, Labels: {len(y_data[split])}")

    # Merge train and validation sets
    X_train_full = np.vstack([X_data["train"], X_data["valid"]])
    y_train_full = np.hstack([y_data["train"], y_data["valid"]])

    X_test = X_data["test"]
    y_test = y_data["test"]

    # Scale X with standard scaler
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    scaler_path = os.path.join("models", "classical_machine_learning", "vit_feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    print("Training set shape:", X_train_full_scaled.shape)
    print("Training labels shape:", y_train_full.shape)

    print("Test set shape:", X_test_scaled.shape)
    print("Test labels shape:", y_test.shape)

    return X_train_full_scaled, y_train_full, X_test_scaled, y_test



def convert_labels_to_single_class_all_splits():
    '''Copy YOLO multi-label training data and convert it into single-label data'''
    
    # source_root="data/outputs_multi_labels"
    source_root = os.path.join("data", "outputs_multi_labels")
    target_root = os.path.join("data", "outputs_single_label")

    splits = ["train", "valid", "test"]

    # Loop through all splits to convert each from multi-label to single-label
    for split in splits:
        src_label_dir = os.path.join(source_root, split, "labels")
        tgt_label_dir = os.path.join(target_root, split, "labels")
        tgt_image_dir = os.path.join(target_root, split, "images")
        src_image_dir = os.path.join(source_root, split, "images")

        os.makedirs(tgt_label_dir, exist_ok=True)
        os.makedirs(tgt_image_dir, exist_ok=True)

        # Point to the same image files (no copying)
        for fname in os.listdir(src_label_dir):
            src_label_path = os.path.join(src_label_dir, fname)
            tgt_label_path = os.path.join(tgt_label_dir, fname)

            with open(src_label_path, "r") as f_in, open(tgt_label_path, "w") as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # Skip malformed lines
                    # Force class to 0, keep bbox
                    f_out.write(f"0 {' '.join(parts[1:])}\n")

        # Create symbolic links to the same images (to save space)
        for fname in os.listdir(src_image_dir):
            src_img_path = os.path.join(src_image_dir, fname)
            tgt_img_path = os.path.join(tgt_image_dir, fname)
            if not os.path.exists(tgt_img_path):
                os.symlink(os.path.abspath(src_img_path), tgt_img_path)