import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Path to the feature CSV you created
FEATURES_CSV = '../data/microsoft/static_features.csv'
MODEL_OUTPUT = '../models_saved/rf_model.pkl'
SCALER_OUTPUT = '../models_saved/scaler_rf.pkl'
ENCODER_OUTPUT = '../models_saved/encoder_rf.pkl'

# Load data
df = pd.read_csv(FEATURES_CSV)

# Assume you have 'Id' and 'Class' in trainlabels.csv
labels_df = pd.read_csv('../data/microsoft/trainLabels.csv')

# Merge features with labels
df = df.merge(labels_df, left_on='Id', right_on='Id')

X = df.drop(['Id', 'Class'], axis=1)
y = df['Class']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf.fit(X_train, y_train)

# Save the model and preprocessing tools
joblib.dump(rf, MODEL_OUTPUT)
joblib.dump(scaler, SCALER_OUTPUT)
joblib.dump(le, ENCODER_OUTPUT)

print(f"Random Forest model saved to {MODEL_OUTPUT}")
