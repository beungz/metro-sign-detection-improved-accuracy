import numpy as np
import pandas as pd

import os
import shutil
import joblib
from datetime import datetime
import re
from collections import Counter

import cv2
from PIL import Image

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import torch
from torchvision import transforms
import torch.nn.functional as F

from transformers import ViTForImageClassification, TrainingArguments, Trainer, AutoImageProcessor, default_data_collator
from evaluate import load

from ultralytics import YOLO

from scripts.build_features import extract_hog_feature, extract_cls_token



# NN-based Deep Learning Model: YOLOv8

def train_yolo(epochs=200, img_size=960, batchsize=16, device='cuda', single_label=False):
    '''Train a YOLOv8 model via transfer learning and return the trained model.'''
    torch.cuda.empty_cache()

    # Path for single-label and multi-labels model
    if single_label:
        yaml_path = os.path.join("data", "outputs_single_label", "data.yaml")
        project_path = os.path.join("models", "deep_learning_yolo_single_label")
        deep_learning_path = "deep_learning_yolo_single_label"
        data_outputs_path = "outputs_single_label"
    else:
        yaml_path = os.path.join("data", "outputs_multi_labels", "data.yaml")
        project_path = os.path.join("models", "deep_learning_yolo_multi_labels")        
        deep_learning_path = "deep_learning_yolo_multi_labels"
        data_outputs_path = "outputs_multi_labels"

    # Load the pretrained YOLOv8s model
    model = YOLO("yolov8s.pt")

    # Transfer learning with training dataset
    # Augmentations are automatically applied by YOLOv8 to training
    model.train(
        data=yaml_path,  # Path of dataset YAML file
        epochs=epochs,                  # Number of training epochs
        imgsz=img_size,                 # Image size
        batch=batchsize,                # Batch size
        device=device,                  # GPU or CPU
        augment=True,                   # Enable data augmentation
        save=True,                      # Save the model after training
        project=project_path,           # Main directory for saving the model
        name="training_results"         # Directory name for saving the results
    )

    # Copy the best model from its original path to the final path
    best_model_original_path = os.path.join("models", deep_learning_path, "training_results", "weights", "best.pt")
    best_model_final_path = os.path.join("models", deep_learning_path, "best.pt")
    shutil.copyfile(best_model_original_path, best_model_final_path)

    # Validate the model on the test dataset
    metrics = model.val(
        data=yaml_path,  # Path of dataset YAML file
        split='test',                   # Use the test split for final evaluation
        imgsz=img_size,                 # Image size
        device=device,                  # GPU or CPU
        project=project_path,           # Main directory for saving the model
        name="test_results"             # Directory name for saving the results
    )

    # Print metrics
    print(metrics)

    # Compute y_pred and y_true for sklearn-style metrics
    y_test = []
    y_pred = []

    test_images_dir = os.path.join("data", data_outputs_path, "test", "images")
    test_labels_dir = os.path.join("data", data_outputs_path, "test", "labels")

    for image_name in os.listdir(test_images_dir):
        if not image_name.endswith(('.jpg', '.png', '.jpeg')):
            continue

        # Image path and label path
        img_path = os.path.join(test_images_dir, image_name)
        label_path = os.path.join(test_labels_dir, os.path.splitext(image_name)[0] + ".txt")

        # Get ground truth labels
        with open(label_path, "r") as f:
            for line in f:
                cls_id = int(line.split()[0])
                y_test.append(cls_id)

        # Run inference
        results = model.predict(img_path, imgsz=img_size)
        boxes = results[0].boxes

        # Take top-1 prediction per ground-truth box
        if boxes is not None and len(boxes.cls) > 0:
            pred_classes = boxes.cls.cpu().numpy().astype(int)
            pred_count = len(pred_classes)
            # Append predicted class
            y_pred.extend(pred_classes[:len(open(label_path).readlines())])
        else:
            # No prediction — use -1 or dummy class (optional)
            y_pred.extend([-1] * len(open(label_path).readlines()))

    min_len = min(len(y_test), len(y_pred))
    y_test = np.array(y_test[:min_len])
    y_pred = np.array(y_pred[:min_len])

    # Evaluate model
    test_accuracy = accuracy_score(y_test, y_pred)
    print("\nTest Accuracy: {:.6f}".format(test_accuracy))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, metrics



def predict_yolo_single_image(model, image):
    '''Predict metro station name(s) and bounding box(es) around station signage(s) in the input image using the trained YOLOv8 model.'''

    # Get the model's predictions and save the resulting image with bounding boxes
    results = model.predict(source=image, save=True, project="data", name="demo_predictions")
    
    # Extract all predicted class_ids (station names) from the results
    result = results[0]
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    
    # Map class_ids to station names, then remove duplicates
    station_names = [model.names[class_id] for class_id in class_ids]
    station_names_unique = list(set(station_names))
    
    # Rename the saved image (default name = image0.jpg) with a timestamp
    result_dir = results[0].save_dir
    original_path = os.path.join(result_dir, "image0.jpg")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_image_name = f"detected_{timestamp}.jpg"
    new_resulting_image_path = os.path.join(result_dir, new_image_name)
    os.rename(original_path, new_resulting_image_path)

    return station_names_unique, new_resulting_image_path



# Classical Machine Learning Model: Support Vector Classifier (SVC)

def train_svc_hog(X_train_full_scaled, y_train_full, X_test_scaled, y_test, param_grid):
    '''Train a Support Vector Classifier (SVC) model with hyperparameter tuning using GridSearchCV and evaluate its performance.'''

    # Initiate SVC model
    model = SVC(probability=True)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1, return_train_score=True)

    # Fit the model to the training set
    grid_search.fit(X_train_full_scaled, y_train_full)

    # The optimal parameters
    print("Best parameters found: ", grid_search.best_params_)
    cv_accuracy = grid_search.best_score_
    print("Best Cross-validation Accuracy: {:.6f}".format(cv_accuracy))

    # Evaluate the best model
    best_model = grid_search.best_estimator_

    # Predict class labels for test set
    y_pred = best_model.predict(X_test_scaled)

    # Evaluate model
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy: {:.6f}".format(test_accuracy))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save the best model
    best_model_final_path = os.path.join("models", "classical_machine_learning", "best_svc_hog_model.pkl")
    joblib.dump(best_model, best_model_final_path)

    return best_model, cv_accuracy, test_accuracy



def predict_svc_hog_single_image(model, cropped_signage_image):
    '''Predict metro station name in the cropped signage image using the trained SVC model.'''
    # Extract HOG features from the cropped signage image
    X_before_scaled = extract_hog_feature(cropped_signage_image, target_size=(128, 64))

    # Load the scaler
    scaler_path = os.path.join("models", "classical_machine_learning", "hog_scaler.pkl")
    scaler = joblib.load(scaler_path)

    # Scale the features
    X_scaled = scaler.transform(X_before_scaled.reshape(1, -1))

    # Predict station name
    y_pred = model.predict(X_scaled)
    station_name = y_pred[0] if len(y_pred) == 1 else y_pred.tolist()
    
    return station_name.replace("_", " ")



# Naive Model

def predict_naive_evaluate_test_set(options, y_train_full, y_test):
    '''Predict metro station name with a naive approach based on the input options.
        options:1 always predicts 'Siam', which has the highest ridership, according to the statistics from BTS, the metro operator. 
        options:2 randomly predicts one of the 8 stations, based on the frequency of the stations in the training set.'''

    if options == 1:
        # Always predict 'Siam' as it has the highest ridership, according to the statistics from BTS, the metro operator. 
        y_pred = ['Siam'] * len(y_test)

        # Evaluate the naive model
        test_accuracy = accuracy_score(y_test, y_pred)
        print("Test Accuracy: {:.6f}".format(test_accuracy))
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    elif options == 2:
        # Calcuate the frequency distribution of stations in the training set

        class_counter = Counter(y_train_full)
        total = sum(class_counter.values())
        classes = list(class_counter.keys())
        probabilities = [class_counter[c] / total for c in classes]

        print("Station Frequency Distribution:\n")
        for cls, prob in zip(classes, probabilities):
            print(f"  {cls}: {prob*100:.2f}%")

        # Predict by randomly choosing a station based on the frequency distribution
        np.random.seed(42)
        y_pred_random = np.random.choice(classes, size=len(y_test), p=probabilities)

        # Evaluate the naive model
        test_accuracy = accuracy_score(y_test, y_pred_random)
        print("\nTest Accuracy: {:.6f}".format(test_accuracy))
        print("Classification Report:\n", classification_report(y_test, y_pred_random))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_random))

    return test_accuracy



def train_vit(dataset, id2label, label2id):
    '''Train Vision Transformer Model, with cropped images'''

    # Output folder for ViT checkpoint
    output_dir = os.path.join("models", "deep_learning_vit")

    # Define ViT model
    vit_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True 
    )

    # Define training arguments for ViT Model
    training_args = TrainingArguments(
        output_dir=output_dir, 
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=1500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True
    )

    # Set compute metrics to be accuracy
    accuracy = load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        return accuracy.compute(predictions=predictions, references=labels)

    # Train the ViT model
    trainer = Trainer(
        model=vit_model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
        data_collator=default_data_collator  # important for batched tensors
    )

    trainer.train()

    # Make prediction
    predictions = trainer.predict(dataset['test'])
    logits, labels, metrics = predictions
    print("Metrics:", metrics)
    # Convert logits to predicted classes
    predicted_labels = np.argmax(logits, axis=-1)

    # Evaluate model
    test_accuracy = accuracy_score(labels, predicted_labels)
    print("\nTest Accuracy: {:.6f}".format(test_accuracy))
    print("Classification Report:\n", classification_report(labels, predicted_labels, target_names=list(id2label.values())))
    print("Confusion Matrix:\n", confusion_matrix(labels, predicted_labels))

    return vit_model, test_accuracy



def predict_vit_single_image(vit_model, inference_transform, id2label, image_path, device="cuda"):
    '''Predict metro station name from the input image using the trained Vision Transformer model.'''
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")

    # Apply transformation
    inputs = inference_transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Move model to device and set to eval mode
    vit_model.to(device)
    vit_model.eval()

    # Get probability and prediction from the model
    with torch.no_grad():
        outputs = vit_model(inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Get top-1 prediction
    top1_idx = torch.argmax(probabilities).item()
    station_name = id2label[top1_idx].replace("_", " ")

    return station_name



def train_svc_lr_rf_cls_features(model_type, X_train_full_scaled, y_train_full, X_test_scaled, y_test, param_grid):
    '''Train a Support Vector Classifier (SVC), Logistics Regression (LR), or Random Forest (RF) model with CLS Features, 
        and hyperparameter tuning using GridSearchCV and evaluate its performance.'''

    # Initiate model
    if model_type == "svc":
        model = SVC(probability=True)
    elif model_type == "lr":
        model = LogisticRegression(multi_class='multinomial')
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=300, max_depth=None)
    else:
        print("Please enter correct model type: svc, lr, or rf")
        return

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), # cv=5
        scoring='accuracy', 
        verbose=2, 
        n_jobs=-1, 
        return_train_score=True
    )

    # Fit the model to the training set
    grid_search.fit(X_train_full_scaled, y_train_full)

    # The optimal parameters
    print("Best parameters found: ", grid_search.best_params_)
    cv_accuracy = grid_search.best_score_
    print("Best Cross-validation Accuracy: {:.6f}".format(cv_accuracy))

    # Evaluate the best model
    best_model = grid_search.best_estimator_

    # Predict class labels for test set
    y_pred = best_model.predict(X_test_scaled)

    # Evaluate model
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy: {:.6f}".format(test_accuracy))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save the best model
    best_model_final_path = os.path.join("models", "classical_machine_learning", "best_" + model_type + "_model.pkl")
    joblib.dump(best_model, best_model_final_path)

    return best_model, cv_accuracy, test_accuracy



def predict_svc_lr_rf_single_image(model, vit_backbone, inference_transform, id2label, cropped_image_path):
    '''Predict metro station name in the cropped signage image using the trained SVC/LR/RF model and CLS Features.'''

    # Extract CLS token features
    feature_vector = extract_cls_token(cropped_image_path, vit_backbone, inference_transform)  # shape: (768,)

    # Reshape for model prediction
    X_before_scaled = feature_vector.reshape(1, -1)

    # Load the scaler
    scaler_path = os.path.join("models", "classical_machine_learning", "vit_feature_scaler.pkl")
    scaler = joblib.load(scaler_path)

    # Scale the features
    X_scaled = scaler.transform(X_before_scaled)

    # Predict station name
    y_pred = model.predict(X_scaled)
    station_id = y_pred[0] if len(y_pred) == 1 else y_pred.tolist()
    station_name = id2label[station_id].replace("_", " ")
    
    return station_name



def predict_single_image(yolo_model_multi_labels, yolo_model_single_label, vit_model, svc_model, lr_model, rf_model, svc_hog_model, vit_backbone, vit_transform, id2label, image_path):
    '''Predict metro station name(s) and bounding box(es) around station signage(s) in the input image, using all models.'''
    
    # Set a path to prediction results; delete existing ones
    pred_yolo_multi_labels_path = os.path.join("data", "pred_yolo_multi_labels")
    pred_yolo_single_label_path = os.path.join("data", "pred_yolo_single_label")
    
    if os.path.exists(pred_yolo_multi_labels_path) and os.path.isdir(pred_yolo_multi_labels_path):
        shutil.rmtree(pred_yolo_multi_labels_path)
    if os.path.exists(pred_yolo_single_label_path) and os.path.isdir(pred_yolo_single_label_path): 
        shutil.rmtree(pred_yolo_single_label_path)
    
    # Get the Yolo model [Multi-label]'s bounding box prediction, save the resulting image with bounding boxes, and cropped images
    results = yolo_model_multi_labels.predict(source=image_path, save=True, save_crop=True, project="data", name="pred_yolo_multi_labels")
    
    # Extract all predicted class_ids (station names) from the results
    result = results[0]
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    # Map class_ids to station names, then remove duplicates
    station_names = [yolo_model_multi_labels.names[class_id] for class_id in class_ids]
    station_names_unique = list(set(station_names))

    # Initiate the predictions for all models
    yolo_pred = None
    vit_pred = None
    svc_pred = None
    lr_pred = None
    rf_pred = None
    svc_hog_pred = None

    yolo_station_detected = False

    if len(station_names_unique) > 0:
        yolo_pred = station_names_unique[0]
        yolo_station_detected = True

    # Get the Yolo model [Single-label]'s bounding box prediction, save the resulting image with bounding boxes, and cropped images
    results_single_label = yolo_model_single_label.predict(source=image_path, save=True, save_crop=True, project="data", name="pred_yolo_single_label")
    
    result_single_label = results_single_label[0]
    class_ids_single_label = result_single_label.boxes.cls.cpu().numpy().astype(int)
    signage_names = [yolo_model_single_label.names[class_id_single_label] for class_id_single_label in class_ids_single_label]
    signage_names_unique = list(set(signage_names))

    yolo_signage_detected = False

    if len(signage_names_unique) > 0:
        yolo_signage_detected = True

    result_dir = None
    crops_root = None
    cropped_image_path = None
    new_resulting_image_path = None

    if yolo_station_detected or yolo_signage_detected:

        # Get a path to cropped images
        if yolo_signage_detected:
            result_dir = results_single_label[0].save_dir
            crops_root = os.path.join(result_dir, "crops", "signage")
            cropped_image_file = os.listdir(crops_root)[0]
            cropped_image_path = os.path.join(crops_root, cropped_image_file)
        else:
            result_dir = results[0].save_dir
            print(result_dir)
            crops_root = os.path.join(result_dir, "crops", yolo_pred)
            cropped_image_file = os.listdir(crops_root)[0]
            cropped_image_path = os.path.join(crops_root, cropped_image_file)
        
        # Get prediction from ViT, SVC, LR, RF, SVC+HOG Model
        vit_pred = predict_vit_single_image(vit_model, vit_transform, id2label, cropped_image_path)
        svc_pred = predict_svc_lr_rf_single_image(svc_model, vit_backbone, vit_transform, id2label, cropped_image_path)
        lr_pred = predict_svc_lr_rf_single_image(lr_model, vit_backbone, vit_transform, id2label, cropped_image_path)
        rf_pred = predict_svc_lr_rf_single_image(rf_model, vit_backbone, vit_transform, id2label, cropped_image_path)

        cropped_img = cv2.imread(cropped_image_path)
        svc_hog_pred = predict_svc_hog_single_image(svc_hog_model, cropped_img)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_image_name = f"detected_{timestamp}.jpg"
        new_resulting_image_path = os.path.join(crops_root, new_image_name)
        os.rename(cropped_image_path, new_resulting_image_path)

    print("Prediction:")
    print(f"YOLOv8: {yolo_pred}")
    print(f"ViT Transformer: {vit_pred}")
    print(f"Support Vector Classifier: {svc_pred}")
    print(f"Logistic Regression: {lr_pred}")
    print(f"Random Forest: {rf_pred}")
    print(f"Support Vector Classifier + HOG: {svc_hog_pred}")

    return new_resulting_image_path, cropped_image_path, yolo_pred, vit_pred, svc_pred, lr_pred, rf_pred, svc_hog_pred



def evaluate_test_set(
    yolo_model_multi_labels,
    yolo_model_single_label,
    test_images_dir,
    vit_backbone,
    vit_transform,
    vit_model,
    svc_model,
    lr_model,
    rf_model,
    svc_hog_model,
    hog_scaler,
    id2label,
    batch_size=16,
    device="cuda"
):
    '''Reevaluate all models, using the same starting test set, for apple-to-apple comparison'''
    
    project_multi_labels_path = os.path.join("data", "eval_yolo_multi_labels")
    project_single_label_path = os.path.join("data", "eval_yolo_single_label")

    # Delete existing prediction folders if they exist
    project_multi_labels_predict_path = os.path.join(project_multi_labels_path, "predict")
    project_single_label_predict_path = os.path.join(project_single_label_path, "predict")

    if os.path.exists(project_multi_labels_predict_path) and os.path.isdir(project_multi_labels_predict_path):
        shutil.rmtree(project_multi_labels_predict_path)
    if os.path.exists(project_single_label_predict_path) and os.path.isdir(project_single_label_predict_path): 
        shutil.rmtree(project_single_label_predict_path)

    # Make prediction using Yolo Model [Multi-label]
    results_multi_labels = yolo_model_multi_labels.predict(
        source=test_images_dir,
        save=True,
        save_crop=True,
        project=project_multi_labels_path,
        name="",
        exist_ok=True
    )

    # Make prediction using Yolo Model [Single-label]
    results_single_label = yolo_model_single_label.predict(
        source=test_images_dir,
        save=True,
        save_crop=True,
        project=project_single_label_path,
        name="",
        exist_ok=True
    )

    # Prepare ground truth mapping from YOLO labels
    labels_dir = os.path.join(os.path.dirname(test_images_dir), "labels")
    y_true_dict = {}
    for lbl_file in os.listdir(labels_dir):
        if lbl_file.endswith(".txt"):
            image_name = lbl_file.replace(".txt", ".jpg")
            with open(os.path.join(labels_dir, lbl_file), "r") as f:
                lines = f.readlines()
                cls_ids = [int(line.split()[0]) for line in lines]
            y_true_dict[image_name] = cls_ids

    # Collect all cropped image paths from Yolo Model [Multi-label]
    crops_root_multi_labels = os.path.join(project_multi_labels_path, "predict", "crops")
    all_crops_multi_labels = []
    for class_folder in os.listdir(crops_root_multi_labels):
        class_path = os.path.join(crops_root_multi_labels, class_folder)
        if os.path.isdir(class_path):
            for crop_file in os.listdir(class_path):
                crop_path = os.path.join(class_path, crop_file)
                all_crops_multi_labels.append(crop_path)

    # Collect all cropped image paths from Yolo Model [Single-label]
    crops_root_single_label = os.path.join(project_single_label_path, "predict", "crops")
    all_crops_single_label = []
    for class_folder in os.listdir(crops_root_single_label):
        class_path = os.path.join(crops_root_single_label, class_folder)
        if os.path.isdir(class_path):
            for crop_file in os.listdir(class_path):
                crop_path = os.path.join(class_path, crop_file)
                all_crops_single_label.append(crop_path)

    # Build test result dataframe
    df_test_result = pd.DataFrame({
        'image': list(y_true_dict.keys()),
        'y_true': [labels[0] for labels in y_true_dict.values()]
    })

    # Add predictions from Yolo Model [Multi-label and Single-label] to the test result dataframe
    df_test_result = add_yolo_preds_and_crops(df_test_result, all_crops_multi_labels, id2label, single_label=False)
    df_test_result = add_yolo_preds_and_crops(df_test_result, all_crops_single_label, id2label, single_label=True)

    # Add YOLO confidence/prediction probability from YOLO [Multi-label] results
    yolo_conf_map = {}
    for r in results_multi_labels:
        img_name = os.path.basename(r.path)
        if len(r.boxes) > 0:
            # Take max confidence
            conf = float(r.boxes.conf.max().item())
        else:
            conf = 0.0
        yolo_conf_map[img_name] = conf

    df_test_result['yolo_prob'] = df_test_result['image'].map(yolo_conf_map).fillna(0.0)


    # Set ViT Model to eval mode
    vit_backbone.eval()
    vit_model.eval()

    # Initialize predictions/probabilities for all models
    vit_preds, svc_preds, lr_preds, rf_preds, svc_hog_preds = [], [], [], [], []
    vit_probs, svc_probs, lr_probs, rf_probs, svc_hog_probs = [], [], [], [], []

    # Process predictions from all models (except YOLO) in multiple batches
    for i in range(0, len(df_test_result), batch_size):
        batch_df = df_test_result.iloc[i:i + batch_size]

        # Set invalid mask, for images that both single-label and multi-label models cannot detect
        invalid_mask = (batch_df['yolo_preds_single_label'] == -1) & (batch_df['yolo_preds'] == -1)

        if invalid_mask.all():
            # Assign -1 predictions and 0.0 probability for all rows that are invalid
            batch_vit_preds = [-1] * len(batch_df)
            batch_svc_preds = [-1] * len(batch_df)
            batch_lr_preds = [-1] * len(batch_df)
            batch_rf_preds = [-1] * len(batch_df)
            batch_svc_hog_preds = [-1] * len(batch_df)

            batch_vit_probs = [0.0] * len(batch_df)
            batch_svc_probs = [0.0] * len(batch_df)
            batch_lr_probs = [0.0] * len(batch_df)
            batch_rf_probs = [0.0] * len(batch_df)
            batch_svc_hog_probs = [0.0] * len(batch_df)
        else:
            valid_rows = batch_df[~invalid_mask]

            # Load images for valid rows
            images = []
            hog_features = []

            # Get cropped image path from Yolo Model [Single-label] first. If not available, then get it from Yolo Model [Multi-label]
            # Also generate HOG features from cropped images
            for row in valid_rows[['cropped_image_single_label', 'cropped_image']].itertuples(index=False):
                img_path_single_label, img_path = row
                try:
                    if img_path_single_label != None:
                        img = Image.open(img_path_single_label).convert("RGB")

                        image_for_hog = cv2.imread(img_path_single_label)
                        hog_feat = extract_hog_feature(image_for_hog, (128, 64))
                        hog_features.append(hog_feat)
                    else:
                        img = Image.open(img_path).convert("RGB")

                        image_for_hog = cv2.imread(img_path)
                        hog_feat = extract_hog_feature(image_for_hog, (128, 64))
                        hog_features.append(hog_feat)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    img = Image.new("RGB", (224, 224), (0, 0, 0))
                images.append(img)

            # Apply transformation on cropped images
            inputs = torch.stack([vit_transform(img) for img in images]).to(device)

            # Apply HOG scaler on HOG Features
            hog_features_scaled = hog_scaler.transform(hog_features)

            # Get predictions and probabilities from all models (except YOLO)
            with torch.no_grad():
                outputs = vit_backbone(inputs)
                cls_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                logits = vit_model(inputs).logits
                preds_vit = logits.argmax(-1).cpu().numpy()
                vit_prob_values = F.softmax(logits, dim=-1).max(dim=-1)[0].cpu().numpy()

                svc_proba = svc_model.predict_proba(cls_features)
                lr_proba = lr_model.predict_proba(cls_features)
                rf_proba = rf_model.predict_proba(cls_features)
                svc_hog_proba = svc_hog_model.predict_proba(hog_features_scaled)

                preds_svc = svc_proba.argmax(axis=1)
                preds_lr = lr_proba.argmax(axis=1)
                preds_rf = rf_proba.argmax(axis=1)
                preds_svc_hog = svc_hog_proba.argmax(axis=1)

                svc_prob_values = svc_proba.max(axis=1)
                lr_prob_values = lr_proba.max(axis=1)
                rf_prob_values = rf_proba.max(axis=1)
                svc_hog_prob_values = svc_hog_proba.max(axis=1)

            # Fill predictions/probabilities to invalid rows
            batch_vit_preds, batch_svc_preds, batch_lr_preds, batch_rf_preds, batch_svc_hog_preds = [], [], [], [], []
            batch_vit_probs, batch_svc_probs, batch_lr_probs, batch_rf_probs, batch_svc_hog_probs = [], [], [], [], []
            valid_idx = 0
            for is_invalid in invalid_mask:
                if is_invalid:
                    batch_vit_preds.append(-1)
                    batch_svc_preds.append(-1)
                    batch_lr_preds.append(-1)
                    batch_rf_preds.append(-1)
                    batch_svc_hog_preds.append(-1)

                    batch_vit_probs.append(0.0)
                    batch_svc_probs.append(0.0)
                    batch_lr_probs.append(0.0)
                    batch_rf_probs.append(0.0)
                    batch_svc_hog_probs.append(0.0)
                else:
                    batch_vit_preds.append(int(preds_vit[valid_idx]))
                    batch_svc_preds.append(int(preds_svc[valid_idx]))
                    batch_lr_preds.append(int(preds_lr[valid_idx]))
                    batch_rf_preds.append(int(preds_rf[valid_idx]))
                    batch_svc_hog_preds.append(int(preds_svc_hog[valid_idx]))

                    batch_vit_probs.append(float(vit_prob_values[valid_idx]))
                    batch_svc_probs.append(float(svc_prob_values[valid_idx]))
                    batch_lr_probs.append(float(lr_prob_values[valid_idx]))
                    batch_rf_probs.append(float(rf_prob_values[valid_idx]))
                    batch_svc_hog_probs.append(float(svc_hog_prob_values[valid_idx]))
                    valid_idx += 1

        vit_preds.extend(batch_vit_preds)
        svc_preds.extend(batch_svc_preds)
        lr_preds.extend(batch_lr_preds)
        rf_preds.extend(batch_rf_preds)
        svc_hog_preds.extend(batch_svc_hog_preds)

        vit_probs.extend(batch_vit_probs)
        svc_probs.extend(batch_svc_probs)
        lr_probs.extend(batch_lr_probs)
        rf_probs.extend(batch_rf_probs)
        svc_hog_probs.extend(batch_svc_hog_probs)

    # Add predictions and probabilities to test result dataframe
    df_test_result['vit_preds'] = vit_preds
    df_test_result['svc_preds'] = svc_preds
    df_test_result['lr_preds'] = lr_preds
    df_test_result['rf_preds'] = rf_preds
    df_test_result['svc_hog_preds'] = svc_hog_preds

    df_test_result['vit_prob'] = vit_probs
    df_test_result['svc_prob'] = svc_probs
    df_test_result['lr_prob'] = lr_probs
    df_test_result['rf_prob'] = rf_probs
    df_test_result['svc_hog_prob'] = svc_hog_probs

    # Accuracy calculation (include -1 as incorrect)
    mask = np.ones(len(df_test_result), dtype=bool)
    print("\n--- ACCURACY REPORT ---")
    print(f"YOLOv8 Accuracy: {accuracy_score(df_test_result.loc[mask, 'y_true'], df_test_result.loc[mask, 'yolo_preds']):.4f}")
    print(f"ViT Accuracy:    {accuracy_score(df_test_result.loc[mask, 'y_true'], df_test_result.loc[mask, 'vit_preds']):.4f}")
    print(f"SVC Accuracy:    {accuracy_score(df_test_result.loc[mask, 'y_true'], df_test_result.loc[mask, 'svc_preds']):.4f}")
    print(f"LR Accuracy:     {accuracy_score(df_test_result.loc[mask, 'y_true'], df_test_result.loc[mask, 'lr_preds']):.4f}")
    print(f"RF Accuracy:     {accuracy_score(df_test_result.loc[mask, 'y_true'], df_test_result.loc[mask, 'rf_preds']):.4f}")
    print(f"SVC HOG Accuracy:    {accuracy_score(df_test_result.loc[mask, 'y_true'], df_test_result.loc[mask, 'svc_hog_preds']):.4f}")
    
    # Save test result dataframe to file
    df_test_result_path = os.path.join("models", "test_set_evaluation.csv")
    df_test_result.to_csv(df_test_result_path)

    return df_test_result



def normalize_folder_name(name):
    '''Lowercase, replace underscores with spaces, and strip extra spaces'''
    return name.lower().replace('_', ' ').strip()



def build_crop_lookup(all_crops, id2label, single_label=False):
    '''Build a lookup dictionary from original image name prefix (without .rf.<hash>.jpg) to cropped image path and predicted label'''
    # Reverse map for folder name to label integer
    foldername_to_label = {v.replace('_', ' ').lower(): k for k, v in id2label.items()}

    lookup = {}
    seen = set()
    for crop_path in all_crops:
        # Extract folder name (prediction name)
        parts = crop_path.replace('\\', '/').split('/')
        pred_folder = parts[-2]

        if not single_label:
            pred_folder_norm = normalize_folder_name(pred_folder)

            # Get predicted label int from folder name
            yolo_pred_label = foldername_to_label.get(pred_folder_norm, None)
            if yolo_pred_label is None:
                # Try also exact match with underscores to cover cases like 'Sala Daeng'
                yolo_pred_label = foldername_to_label.get(pred_folder.replace(' ', '_').lower(), None)
        else:
            yolo_pred_label = None
            if pred_folder == "signage":
                yolo_pred_label = 0

        # Extract cropped image filename
        crop_filename = parts[-1]
        crop_basename = crop_filename
        # Remove trailing digits before extension, e.g. 'Chong_Nonsi_008_png2.jpg' > 'Chong_Nonsi_008_png.jpg'
        crop_basename = re.sub(r'(\d+)(?=\.jpg$)', '', crop_basename)

        # Store in lookup using crop_basename as key
        key = crop_basename.lower()

        if key not in seen:
            lookup[key] = (crop_path, yolo_pred_label)
            seen.add(key)

    return lookup



def find_crop_info(image_name, crop_lookup):
    '''Given an original image filename from df, find matching crop info from lookup'''
    
    # Remove the '.rf.<hash>.jpg' part from image_name
    # For example, 'Saint_Louis_063_png.rf.d2eaffa29c87977a84a17fce8eddf048.jpg' > 'Saint_Louis_063_png.jpg'
    prefix = image_name.split('.rf.')[0] + '.jpg'
    prefix_lower = prefix.lower()

    return crop_lookup.get(prefix_lower, (None, None))



def add_yolo_preds_and_crops(df, all_crops, id2label, single_label=False):
    '''Add Yolo Model's prediction to the test result dataframe (df)'''

    # Build a lookup dictionary from original image name prefix (without .rf.<hash>.jpg) to cropped image path and predicted label
    crop_lookup = build_crop_lookup(all_crops, id2label, single_label)

    cropped_images = []
    yolo_preds = []

    for image_name in df['image']:
        # Find matching crop info from lookup, from a given original image filename
        crop_path, pred_label = find_crop_info(image_name, crop_lookup)
        cropped_images.append(crop_path)

        # If pred_label is None, replace with -1
        yolo_preds.append(pred_label if pred_label is not None else -1)

    # Add path to cropped images, and YOLO predictions to the test result dataframe
    if not single_label:
        df['cropped_image'] = cropped_images
        df['yolo_preds'] = pd.Series(yolo_preds, dtype='Int64')  # pandas nullable int dtype
    else:
        df['cropped_image_single_label'] = cropped_images
        df['yolo_preds_single_label'] = pd.Series(yolo_preds, dtype='Int64')  # pandas nullable int dtype

    return df



def predict_yolo_cls_svc_single_image(yolo_model_multi_labels, yolo_model_single_label, svc_model, vit_backbone, vit_transform, id2label, input_image):
    '''Predict metro station name(s) and bounding box(es) around station signage(s) in the input image, using YOLOv8s and SVC + CLS Features.'''
    
    # Set a path to prediction results; delete existing ones
    pred_yolo_multi_labels_path = os.path.join("data", "pred_yolo_multi_labels")
    pred_yolo_single_label_path = os.path.join("data", "pred_yolo_single_label")
    
    if os.path.exists(pred_yolo_multi_labels_path) and os.path.isdir(pred_yolo_multi_labels_path):
        shutil.rmtree(pred_yolo_multi_labels_path)
    if os.path.exists(pred_yolo_single_label_path) and os.path.isdir(pred_yolo_single_label_path): 
        shutil.rmtree(pred_yolo_single_label_path)
    
    # Get the Yolo model [Multi-label]'s bounding box prediction, save the resulting image with bounding boxes, and cropped images
    results = yolo_model_multi_labels.predict(source=input_image, save=True, save_crop=True, project="data", name="pred_yolo_multi_labels")
    
    # Extract all predicted class_ids (station names) from the results
    result = results[0]
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    # Map class_ids to station names, then remove duplicates
    station_names = [yolo_model_multi_labels.names[class_id] for class_id in class_ids]
    station_names_unique = list(set(station_names))

    # Initiate the predictions for all models
    yolo_pred = None
    svc_pred = None

    yolo_station_detected = False

    if len(station_names_unique) > 0:
        yolo_pred = station_names_unique[0]
        yolo_station_detected = True

    # Get the Yolo model [Single-label]'s bounding box prediction, save the resulting image with bounding boxes, and cropped images
    results_single_label = yolo_model_single_label.predict(source=input_image, save=True, save_crop=True, project="data", name="pred_yolo_single_label")
    
    result_single_label = results_single_label[0]
    class_ids_single_label = result_single_label.boxes.cls.cpu().numpy().astype(int)
    signage_names = [yolo_model_single_label.names[class_id_single_label] for class_id_single_label in class_ids_single_label]
    signage_names_unique = list(set(signage_names))

    yolo_signage_detected = False

    if len(signage_names_unique) > 0:
        yolo_signage_detected = True

    result_dir = None
    crops_root = None
    cropped_image_path = None
    new_resulting_image_path = None

    if yolo_station_detected or yolo_signage_detected:

        # Get a path to cropped images
        if yolo_signage_detected:
            result_dir = results_single_label[0].save_dir
            crops_root = os.path.join(result_dir, "crops", "signage")
            cropped_image_file = os.listdir(crops_root)[0]
            cropped_image_path = os.path.join(crops_root, cropped_image_file)
        else:
            result_dir = results[0].save_dir
            print(result_dir)
            crops_root = os.path.join(result_dir, "crops", yolo_pred)
            cropped_image_file = os.listdir(crops_root)[0]
            cropped_image_path = os.path.join(crops_root, cropped_image_file)
        
        # Get prediction from SVC + CLS Features)
        svc_pred = predict_svc_lr_rf_single_image(svc_model, vit_backbone, vit_transform, id2label, cropped_image_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_image_name = f"detected_{timestamp}.jpg"
        new_resulting_image_path = os.path.join(crops_root, new_image_name)
        os.rename(cropped_image_path, new_resulting_image_path)

    print("Prediction:")
    print(f"YOLOv8s + Support Vector Classifier (CLS Features): {svc_pred}")

    return new_resulting_image_path, cropped_image_path, svc_pred