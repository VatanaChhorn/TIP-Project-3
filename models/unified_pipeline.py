#!/usr/bin/env python3
"""
Unified Pipeline for Microsoft Malware Dataset
Trains both RF and SVM models using the same .bytes files
"""

import os
import glob
import pandas as pd
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from collections import Counter
import time

# Constants
DATA_DIR = "../data/microsoft/train"
LABELS_FILE = "../data/microsoft/trainLabels.csv"
OUTPUT_DIR = "../models_saved"

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output paths
RF_MODEL_PATH = os.path.join(OUTPUT_DIR, "rf_model.pkl")
SVM_MODEL_PATH = os.path.join(OUTPUT_DIR, "svm_model.pkl")
RF_SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler_rf.pkl")
SVM_SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler_svm.pkl")
RF_ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_rf.pkl")
SVM_ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_svm.pkl")
STATIC_FEATURES_CSV = os.path.join(OUTPUT_DIR, "static_features.csv")
IMAGE_FEATURES_CSV = os.path.join(OUTPUT_DIR, "image_features.csv")

# Feature extraction for RF model
def extract_static_features(file_path):
    """Extract static byte features from .bytes file for RF model"""
    try:
        with open(file_path, 'r', errors='ignore') as file:
            tokens = []
            for line in file:
                parts = line.strip().split()
                if len(parts) > 1:  # Skip empty lines
                    values = parts[1:]  # Skip address
                    tokens.extend(values)
                
            token_counts = Counter(tokens)
            features = [len(tokens)]  # Total token count
            for i in range(256):
                byte_token = f'{i:02X}'
                features.append(token_counts.get(byte_token, 0))
            return features
    except Exception as e:
        print(f"Error extracting static features from {file_path}: {e}")
        return None

# Feature extraction for SVM model
def convert_to_image_features(file_path):
    """Convert .bytes file to image features for SVM model"""
    try:
        with open(file_path, 'r', errors='ignore') as f:
            content = f.read()
        
        # Convert content to a 2D representation
        bytes_values = []
        for line in content.split('\n'):
            parts = line.strip().split()
            if len(parts) > 1:  # Skip empty lines
                row = []
                for byte in parts[1:]:  # Skip address
                    try:
                        value = int(byte, 16)
                        row.append(value)
                    except ValueError:
                        row.append(0)  # Use 0 for invalid values
                bytes_values.extend(row)
        
        # Create a square image
        size = int(np.sqrt(len(bytes_values)))
        if size < 64:
            size = 64  # Minimum size
        
        # Pad or truncate to size²
        if len(bytes_values) < size * size:
            bytes_values.extend([0] * (size * size - len(bytes_values)))
        elif len(bytes_values) > size * size:
            bytes_values = bytes_values[:size * size]
        
        # Reshape and resize to 64x64
        image = np.array(bytes_values).reshape(size, size).astype(np.uint8)
        image = cv2.resize(image, (64, 64))
        
        return image.flatten()
    except Exception as e:
        print(f"Error converting to image features {file_path}: {e}")
        return None

def main():
    print("=== Starting Unified Training Pipeline ===")
    start_time = time.time()
    
    # Load labels
    print("Loading class labels...")
    labels_df = pd.read_csv(LABELS_FILE)
    file_labels = dict(zip(labels_df['Id'], labels_df['Class']))
    
    # Get all .bytes files
    print("Finding all .bytes files...")
    bytes_files = glob.glob(os.path.join(DATA_DIR, "*.bytes"))
    print(f"Found {len(bytes_files)} .bytes files")
    
    # Extract features
    print("Extracting features for both models...")
    static_features = []  # For RF
    image_features = []   # For SVM
    file_ids = []         # File IDs
    labels = []           # Class labels
    
    total_files = len(bytes_files)
    for i, file_path in enumerate(bytes_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{total_files}...")
            
        file_id = os.path.basename(file_path).split('.')[0]
        
        # Skip if no label found
        if file_id not in file_labels:
            print(f"No label found for {file_id}, skipping")
            continue
            
        # Extract features for RF model
        static_feature = extract_static_features(file_path)
        
        # Extract features for SVM model
        image_feature = convert_to_image_features(file_path)
        
        if static_feature is not None and image_feature is not None:
            static_features.append(static_feature)
            image_features.append(image_feature)
            file_ids.append(file_id)
            labels.append(file_labels[file_id])
    
    print(f"Successfully processed {len(static_features)} files")
    
    # Convert to numpy arrays
    static_features = np.array(static_features)
    image_features = np.array(image_features)
    labels = np.array(labels)
    
    print(f"Static feature shape: {static_features.shape}")
    print(f"Image feature shape: {image_features.shape}")
    
    # Save features to CSV
    print("Saving features to CSV...")
    
    # Static features for RF
    static_df = pd.DataFrame(static_features)
    # Rename columns
    static_df.columns = ['total_count'] + [f'byte_{i:02X}' for i in range(256)]
    # Add ID and label columns
    static_df['Id'] = file_ids
    static_df['Class'] = labels
    # Save to CSV
    static_df.to_csv(STATIC_FEATURES_CSV, index=False)
    print(f"Static features saved to {STATIC_FEATURES_CSV}")
    
    # Image features for SVM (just save a subset due to high dimensionality)
    image_df = pd.DataFrame({
        'Id': file_ids,
        'Label': labels
    })
    # Add first 100 image features as columns (full set would be too large)
    for i in range(min(100, image_features.shape[1])):
        image_df[f'pixel_{i}'] = image_features[:, i]
    # Save to CSV
    image_df.to_csv(IMAGE_FEATURES_CSV, index=False)
    print(f"Image features (subset) saved to {IMAGE_FEATURES_CSV}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f"Classes: {label_encoder.classes_}")
    
    # Split data for both models
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        static_features, encoded_labels, test_size=0.2, random_state=42)
    
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
        image_features, encoded_labels, test_size=0.2, random_state=42)
    
    # Normalize features
    print("Normalizing features...")
    
    # MinMaxScaler for RF (static features)
    rf_scaler = MinMaxScaler()
    X_train_rf_scaled = rf_scaler.fit_transform(X_train_rf)
    X_test_rf_scaled = rf_scaler.transform(X_test_rf)
    
    # StandardScaler for SVM (image features)
    svm_scaler = StandardScaler()
    X_train_svm_scaled = svm_scaler.fit_transform(X_train_svm)
    X_test_svm_scaled = svm_scaler.transform(X_test_svm)
    
    # Train Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    rf_model.fit(X_train_rf_scaled, y_train_rf)
    rf_accuracy = rf_model.score(X_test_rf_scaled, y_test_rf)
    print(f"RF Accuracy: {rf_accuracy:.4f}")
    
    # Train SVM
    print("Training SVM model...")
    svm_model = SVC(kernel="linear", probability=True, C=1.0, random_state=42)
    svm_model.fit(X_train_svm_scaled, y_train_svm)
    svm_accuracy = svm_model.score(X_test_svm_scaled, y_test_svm)
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    
    # Save models and preprocessing tools
    print("Saving models and scalers...")
    joblib.dump(rf_model, RF_MODEL_PATH)
    joblib.dump(svm_model, SVM_MODEL_PATH)
    joblib.dump(rf_scaler, RF_SCALER_PATH)
    joblib.dump(svm_scaler, SVM_SCALER_PATH)
    joblib.dump(label_encoder, RF_ENCODER_PATH)
    joblib.dump(label_encoder, SVM_ENCODER_PATH)
    
    elapsed_time = time.time() - start_time
    print(f"=== Pipeline completed in {elapsed_time:.2f} seconds ===")
    print(f"Models saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 