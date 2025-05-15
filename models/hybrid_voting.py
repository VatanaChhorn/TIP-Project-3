import joblib
import numpy as np
import pandas as pd

# Load saved models and preprocessors
rf_model = joblib.load('../models_saved/rf_model.pkl')
svm_model = joblib.load('../models_saved/svm_model.pkl')

scaler_rf = joblib.load('../models_saved/scaler_rf.pkl')
scaler_svm = joblib.load('../models_saved/scaler_svm.pkl')

encoder_rf = joblib.load('../models_saved/encoder_rf.pkl')
encoder_svm = joblib.load('../models_saved/encoder_svm.pkl')

def hybrid_predict(static_features, dynamic_features):
    # Preprocess static features (Microsoft data)
    # Convert to DataFrame with correct feature names
    static_df = pd.DataFrame([static_features], columns=scaler_rf.feature_names_in_)
    static_scaled = scaler_rf.transform(static_df)
    rf_probs = rf_model.predict_proba(static_scaled)[0]
    rf_confidence = max(rf_probs)
    rf_label_index = np.argmax(rf_probs)
    rf_label = encoder_rf.classes_[rf_label_index]

    # Preprocess dynamic features (Malimg image data)
    dynamic_df = pd.DataFrame([dynamic_features], columns=scaler_svm.feature_names_in_)
    dynamic_scaled = scaler_svm.transform(dynamic_df)
    svm_probs = svm_model.predict_proba(dynamic_scaled)[0]
    svm_confidence = max(svm_probs)
    svm_label_index = np.argmax(svm_probs)
    svm_label = encoder_svm.inverse_transform([svm_label_index])[0]

    # Weighted Hybrid Voting (60% RF + 40% SVM)
    final_rf_weight = 0.6 * rf_confidence
    final_svm_weight = 0.4 * svm_confidence

    if final_rf_weight >= final_svm_weight:
        final_label = rf_label
        final_confidence = final_rf_weight
    else:
        final_label = svm_label
        final_confidence = final_svm_weight

    return {
        "rf_label": rf_label,
        "rf_confidence": round(rf_confidence, 2),
        "svm_label": svm_label,
        "svm_confidence": round(svm_confidence, 2),
        "final_label": final_label,
        "final_confidence": round(final_confidence, 2)
    }


