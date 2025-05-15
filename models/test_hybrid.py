import pandas as pd
from hybrid_voting import hybrid_predict

# Load sample static features (Microsoft dataset)
static_df = pd.read_csv('../data/microsoft/static_features.csv')
sample_static = static_df.drop(['Id'], axis=1).iloc[0].values

# Load sample dynamic features (Malimg dataset)
dynamic_df = pd.read_csv('../data/malimg/image_features.csv')
sample_dynamic = dynamic_df.drop(['FileID', 'Label'], axis=1).iloc[0].values

# Run Hybrid Prediction
result = hybrid_predict(sample_static, sample_dynamic)

# Display Result
print("Hybrid Prediction Result:")
print(f"  RF Prediction       : {result['rf_label']} ({result['rf_confidence']:.2f})")
print(f"  SVM Prediction      : {result['svm_label']} ({result['svm_confidence']:.2f})")
print(f"  Final Decision      : {result['final_label']} ({result['final_confidence']:.2f})")
