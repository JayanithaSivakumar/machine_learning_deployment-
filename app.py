from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Initialize Flask app
app = Flask(__name__)

# --- Load Model and Preprocessing Objects ---
# Load the trained Random Forest model
rf_classifier_resampled = joblib.load('random_forest_model.joblib')

# Re-initialize and fit LabelEncoders for binary features and the target 'Disease'
# This ensures the API has the same transformations as the training pipeline.
original_df = pd.read_csv('/content/Disease_symptom_and_patient_profile_dataset.csv')

disease_le = LabelEncoder()
y_original = disease_le.fit_transform(original_df['Disease'])

binary_le = {}
binary_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Outcome Variable']
for col in binary_cols:
    le = LabelEncoder()
    le.fit(original_df[col])
    binary_le[col] = le

# Re-initialize and fit the ColumnTransformer for OneHotEncoding
onehot_cols = ['Blood Pressure', 'Cholesterol Level']

# Create a temporary DataFrame to fit the preprocessor
df_features_for_preprocessor_fit = original_df.drop('Disease', axis=1)
for col in binary_cols:
    df_features_for_preprocessor_fit[col] = binary_le[col].transform(df_features_for_preprocessor_fit[col])

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), onehot_cols)
    ],
    remainder='passthrough'
)
preprocessor.fit(df_features_for_preprocessor_fit)

# Store the column order of X_train from the original training for consistent input to the model
# This assumes X_train.columns was globally defined after the last successful training run.
# If not, you'd need to re-run the full preprocessing flow to get X_train.columns.
# For robustness, we can derive it from the preprocessor if the original X_train is not available.

# Let's derive it safely assuming X_processed_df from cell e4a8e37a was correct.
# Original feature columns that will be passed through (after binary encoding)
original_feature_cols_for_ordering = [col for col in df_features_for_preprocessor_fit.columns if col not in onehot_cols]
onehot_feature_names_for_ordering = preprocessor.named_transformers_['onehot'].get_feature_names_out(onehot_cols)
MODEL_INPUT_COLUMNS = list(onehot_feature_names_for_ordering) + list(original_feature_cols_for_ordering)

print("Model, encoders, and preprocessor loaded/initialized successfully.")
