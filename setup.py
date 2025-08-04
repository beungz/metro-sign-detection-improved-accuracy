from scripts.make_dataset import remove_corrupted_duplicates_and_rename_images, split_dataset, crop_augment_images_for_svc, split_dataset_for_svc
from scripts.make_dataset import prepare_vit_data, prepare_svc_lr_rf_cls_features, convert_labels_to_single_class_all_splits
from scripts.build_features import build_features_svc
from scripts.model import train_yolo, train_svc_hog, predict_naive_evaluate_test_set, train_vit, train_svc_lr_rf_cls_features, evaluate_test_set

import os

from torchvision import transforms



def main():
    '''Main function to run the entire pipeline for detecting Bangkok Metro Station Signage.'''
    # A. Get Dataset
    print("A. Get Dataset")

    # Step A1: Gather the raw dataset from Google Maps
    # Gather images with metro station signages from Google Maps/Photos (approx 100 photos per station) and save them in the data/raw folder.

    # List of labels/folders corresponding to YOLO class IDs (folder names have underscores while YOLO class IDs do not)
    label_map = {
        0: 'Chong_Nonsi', 
        1: 'National_Stadium',
        2: 'Ratchadamri', 
        3: 'Saint_Louis', 
        4: 'Sala_Daeng', 
        5: 'Saphan_Taksin', 
        6: 'Siam', 
        7: 'Surasak'
        }
    
    # List of station folder names corresponding to the 8 stations in the BTS Silom Line (Dark Green Line without extension)
    station_folder_name = list(label_map.values())

    # Step A2: Remove corrupted images, duplicates, and rename images, then save them in the data/processed_before_roboflow folder
    # This step was done before uploading to Roboflow, so it should be commented out if the annotated dataset exported from Roboflow is ready for the next step.
    print("Step A2: Remove corrupted images, duplicates, and rename images.")
    remove_corrupted_duplicates_and_rename_images(station_folder_name)


    # Step A3: Upload the dataset to Roboflow, annotate the images, draw bounding boxes manually, and then export the dataset in YOLO format.
    # Put the exported dataset in the data/processed folder.


    # Step A4: Split the dataset into training, validation, and test sets, then save them in the data/outputs_multi_labels folder.
    print("Step A4: Split the dataset into training, validation, and test sets.")
    split_dataset(station_folder_name)


    # Step A5: Crop the images in the training, validation, and test sets to create a new dataset with only the metro station signage. This is for the SVC model.
    print("Step A5: Crop the images for SVC model.")
    crop_augment_images_for_svc(label_map)


    # Step A6: Build features and labels for SVC model
    print("Step A6: Build features and labels for SVC model.")
    features_by_split, labels_by_split = build_features_svc()


    # Step A7: Prepare the training and test data for SVC + HOG model
    print("Step A7: Prepare the training and test data for SVC model.")
    svc_X_train_full_scaled, svc_y_train_full, svc_X_test_scaled, svc_y_test, hog_scaler = split_dataset_for_svc(features_by_split, labels_by_split)


    # Step A8: Copy YOLO multi-label training data and convert it into single-label.
    print("Step A8: Copy YOLO multi-label training data and convert it into single-label.")
    convert_labels_to_single_class_all_splits()


    # Step A9: Prepare data for Vision Transformer Model training.
    print("Step A9: Prepare data for Vision Transformer Model training.")
    dataset, id2label, label2id, inference_transform, processor = prepare_vit_data()



    # B. Train models
    print("B. Train models")

    # Step B1: Train YOLOv8 model [Multi-label] for detecting Bangkok Metro Station Signage
    print("Step B1: Train YOLOv8 model [Multi-label] for detecting Bangkok Metro Station Signage")
    best_yolo_model_multi_labels, yolo_metrics_multi_labels = train_yolo(epochs=100, img_size=960, batchsize=16, device='cuda', single_label=False)


    # Step B2: Train YOLOv8 model [Single-label] for detecting Bangkok Metro Station Signage
    print("Step B2: Train YOLOv8 model [Single-label] for detecting Bangkok Metro Station Signage")
    best_yolo_model_single_label, yolo_metrics_single_label = train_yolo(epochs=100, img_size=960, batchsize=16, device='cuda', single_label=True)


    # Step B3: Train Vision Transformer Model
    print("Step B3: Train Vision Transformer Model")
    best_vit_model, vit_test_accuracy = train_vit(dataset, id2label, label2id)


    # Step B4: Extract CLS features from Vision Transformer
    print("Step B4: Extract CLS features from Vision Transformer")

    best_vit_model.eval().to("cuda")
    vit_backbone = best_vit_model.vit

    X_train_full_scaled, y_train_full, X_test_scaled, y_test = prepare_svc_lr_rf_cls_features(vit_backbone, inference_transform)


    # Step B5: Train Support Vector Classifier Model, using CLS Features 
    print("Step B5: Train Support Vector Classifier Model, using CLS Features")
    svc_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    best_svc_model, svc_cv_accuracy, svc_test_accuracy = train_svc_lr_rf_cls_features("svc", X_train_full_scaled, y_train_full, X_test_scaled, y_test, svc_param_grid)


    # Step B6: Train Logistic Regression Model, using CLS Features 
    print("Step B6: Train Logistic Regression Model, using CLS Features")
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [500, 1000]
    }

    best_lr_model, lr_cv_accuracy, lr_test_accuracy = train_svc_lr_rf_cls_features("lr", X_train_full_scaled, y_train_full, X_test_scaled, y_test, lr_param_grid)


    # Step B7: Train Random Forest Model, using CLS Features 
    print("Step B7: Train Random Forest Model, using CLS Features ")
    rf_param_grid = {
        'n_estimators': [100, 300, 500],     # Number of trees
        'max_depth': [None, 20, 50, 100],    # None = fully grown trees
        'min_samples_split': [2, 5, 10],     # Minimum samples for a split
        'min_samples_leaf': [1, 2, 4],       # Minimum samples at a leaf node
        'max_features': ['sqrt', 'log2']     # Feature selection per split
    }

    best_rf_model, rf_cv_accuracy, rf_test_accuracy = train_svc_lr_rf_cls_features("rf", X_train_full_scaled, y_train_full, X_test_scaled, y_test, rf_param_grid)


    # Step B8: Train Support Vector Classifier Model, using HOG Features
    print("Step B8: Train Support Vector Classifier Model, using HOG Features")

    # Define the list of hyperparameter for cross validation
    svc_hog_param_grid = {
        'kernel': ['rbf'],  # Type of kernel
        'C': [0.1, 1, 10],  # Regularization parameter
        'gamma': ['scale', 'auto'],  # Kernel coefficient for ‘rbf’, ‘poly’, and ‘sigmoid’
        'tol': [1e-3, 1e-4]  # Tolerance for stopping criteria
    }

    best_svc_hog_model, svc_hog_cv_accuracy, svc_hog_test_accuracy = train_svc_hog(svc_X_train_full_scaled, svc_y_train_full, svc_X_test_scaled, svc_y_test, svc_hog_param_grid)


    # Step B9: Reevaluate all models, using the same starting test set, for apple-to-apple comparison
    print("Step B9: Reevaluate all models, using the same starting test set, for apple-to-apple comparison")

    test_images_dir = os.path.join("data", "outputs_multi_labels", "test", "images")

    df_test_result = evaluate_test_set(
        best_yolo_model_multi_labels,
        best_yolo_model_single_label,
        test_images_dir,
        vit_backbone,
        inference_transform,
        best_vit_model,
        best_svc_model,
        best_lr_model,
        best_rf_model,
        best_svc_hog_model,
        hog_scaler,
        id2label,
        batch_size=16,
        device="cuda"
    )


    # Step B10: Evaluate naive model with option 1 
    print("Step B10: Evaluate naive model with option 1")
    # Always predicts 'Siam', which has the highest ridership, according to the statistics from BTS, the metro operator.
    naive_1_accuracy = predict_naive_evaluate_test_set(1, svc_y_train_full, svc_y_test)


    # Step B11: Evaluate naive model with option 2
    print("Step B11: Evaluate naive model with option 2")
    # Randomly predicts one of the 8 stations, based on the frequency of the stations in the training set.
    naive_2_accuracy = predict_naive_evaluate_test_set(2, svc_y_train_full, svc_y_test)



if __name__ == "__main__":
    main()
