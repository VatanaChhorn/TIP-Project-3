#!/usr/bin/env python3
"""
Monitor Training Script
This script monitors the training progress and provides detailed information
about the models and features.
"""

import os
import pandas as pd
import numpy as np
import joblib

# Paths to model files
OUTPUT_DIR = "../models_saved"
RF_MODEL_PATH = os.path.join(OUTPUT_DIR, "rf_model.pkl")
SVM_MODEL_PATH = os.path.join(OUTPUT_DIR, "svm_model.pkl")
RF_SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler_rf.pkl")
SVM_SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler_svm.pkl")
RF_ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_rf.pkl")
SVM_ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_svm.pkl")
STATIC_FEATURES_CSV = os.path.join(OUTPUT_DIR, "static_features.csv")
IMAGE_FEATURES_CSV = os.path.join(OUTPUT_DIR, "image_features.csv")

def check_files_exist():
    """Check if all required files exist"""
    files = [
        RF_MODEL_PATH, SVM_MODEL_PATH,
        RF_SCALER_PATH, SVM_SCALER_PATH,
        RF_ENCODER_PATH, SVM_ENCODER_PATH
    ]
    
    all_exist = True
    print("Checking for model files...")
    for file in files:
        exists = os.path.exists(file)
        print(f"- {os.path.basename(file)}: {'✓' if exists else '✗'}")
        all_exist = all_exist and exists
    
    return all_exist

def load_features():
    """Load feature data if available"""
    features = {}
    
    if os.path.exists(STATIC_FEATURES_CSV):
        print(f"Loading static features from {os.path.basename(STATIC_FEATURES_CSV)}...")
        static_df = pd.read_csv(STATIC_FEATURES_CSV)
        features['static'] = static_df
        print(f"- Shape: {static_df.shape}")
        print(f"- Classes: {static_df['Class'].unique()}")
        print(f"- Sample count per class: \n{static_df['Class'].value_counts()}")
    else:
        print(f"Static features file not found: {STATIC_FEATURES_CSV}")
    
    if os.path.exists(IMAGE_FEATURES_CSV):
        print(f"\nLoading image features from {os.path.basename(IMAGE_FEATURES_CSV)}...")
        image_df = pd.read_csv(IMAGE_FEATURES_CSV)
        features['image'] = image_df
        print(f"- Shape: {image_df.shape}")
        print(f"- Classes: {image_df['Label'].unique()}")
        print(f"- Sample count per class: \n{image_df['Label'].value_counts()}")
    else:
        print(f"Image features file not found: {IMAGE_FEATURES_CSV}")
    
    return features

def load_models():
    """Load trained models and preprocessing tools"""
    models = {}
    
    try:
        print("\nLoading RF model and tools...")
        models['rf_model'] = joblib.load(RF_MODEL_PATH)
        models['rf_scaler'] = joblib.load(RF_SCALER_PATH)
        models['rf_encoder'] = joblib.load(RF_ENCODER_PATH)
        print("- RF model loaded successfully")
        
        if hasattr(models['rf_model'], 'feature_importances_'):
            print("- RF feature importances available")
            models['rf_feature_importances'] = models['rf_model'].feature_importances_
    except Exception as e:
        print(f"Error loading RF model: {e}")
    
    try:
        print("\nLoading SVM model and tools...")
        models['svm_model'] = joblib.load(SVM_MODEL_PATH)
        models['svm_scaler'] = joblib.load(SVM_SCALER_PATH)
        models['svm_encoder'] = joblib.load(SVM_ENCODER_PATH)
        print("- SVM model loaded successfully")
        
        if hasattr(models['svm_model'], 'coef_'):
            print("- SVM coefficients available")
            models['svm_coefficients'] = models['svm_model'].coef_
    except Exception as e:
        print(f"Error loading SVM model: {e}")
    
    return models

def main():
    print("=== Model Training Monitor ===\n")
    
    # Check if files exist
    files_exist = check_files_exist()
    if not files_exist:
        print("\nSome model files are missing. Please run the training pipeline first.")
        return
    
    # Load feature data
    features = load_features()
    
    # Load models
    models = load_models()
    
    # Print model information
    if 'rf_model' in models:
        rf_model = models['rf_model']
        print("\nRandom Forest Model Information:")
        print(f"- n_estimators: {rf_model.n_estimators}")
        print(f"- max_depth: {rf_model.max_depth}")
        print(f"- n_classes: {len(rf_model.classes_)}")
        print(f"- Classes: {models['rf_encoder'].classes_}")
    
    if 'svm_model' in models:
        svm_model = models['svm_model']
        print("\nSVM Model Information:")
        print(f"- Kernel: {svm_model.kernel}")
        print(f"- C: {svm_model.C}")
        print(f"- n_classes: {len(svm_model.classes_)}")
        print(f"- Classes: {models['svm_encoder'].classes_}")
    
    print("\n=== Monitor Complete ===")

if __name__ == "__main__":
    main() 