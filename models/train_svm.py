import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Path to the image features you created
FEATURES_CSV = '../data/malimg/image_features.csv'
MODEL_OUTPUT = '../models_saved/svm_model.pkl'
SCALER_OUTPUT = '../models_saved/scaler_svm.pkl'
ENCODER_OUTPUT = '../models_saved/encoder_svm.pkl'

# Load data
df = pd.read_csv(FEATURES_CSV)

X = df.drop(['FileID', 'Label'], axis=1)
y = df['Label']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train SVM
svm = SVC(kernel="linear", probability=True, C=1.0)
svm.fit(X_train, y_train)

# Save the model and preprocessing tools
joblib.dump(svm, MODEL_OUTPUT)
joblib.dump(scaler, SCALER_OUTPUT)
joblib.dump(le, ENCODER_OUTPUT)

print(f"SVM model saved to {MODEL_OUTPUT}")
