import joblib
import os

# Load all models and scalers once
rf_model = joblib.load('../models_saved/rf_model.pkl')
svm_model = joblib.load('../models_saved/svm_model.pkl')
scaler_rf = joblib.load('../models_saved/scaler_rf.pkl')
scaler_svm = joblib.load('../models_saved/scaler_svm.pkl')
encoder_rf = joblib.load('../models_saved/encoder_rf.pkl')
encoder_svm = joblib.load('../models_saved/encoder_svm.pkl')

def load_models():
    return rf_model, svm_model, scaler_rf, scaler_svm, encoder_rf, encoder_svm
