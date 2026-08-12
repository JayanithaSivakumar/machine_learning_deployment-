from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
encoder = LabelEncoder()

# Fit using your original training data
encoder.fit(your_training_data)

joblib.dump(encoder, "label_encoder.joblib")

# --- Load the trained model and label encoders ---
try:
    best_rf_model = joblib.load('random_forest_model_tuned.joblib')
    label_encoders = joblib.load('label_encoders.joblib')
    
    # Get the feature names used during training
    # Assuming X_train_resampled is still available in the kernel or recreate from original X and drop 'Disease'
    # If X is not available, you would need to manually list them or save them with the model.
    # For this example, let's assume X is still in scope or infer from X_test/X_train in kernel state
    # If not, you might need to load an example dataframe first.
    # For robust deployment, it's better to save X.columns as part of the model bundle.
    # Let's use X.columns from the current kernel state for this example.
    
    # This part needs to be careful if X is not available during runtime
    # For demonstration, we'll use a hardcoded list of columns or try to infer from loaded data
    # In a real app, save X.columns alongside the model.
    # For this notebook, we'll assume X is in the kernel, otherwise, one needs to reconstruct it.
    # As per context, X is available in kernel state.
    
    # Reconstruct X_train.columns (or load if persisted)
    # Assuming the original feature columns are consistent and can be derived from the initial data processing
    # If df was loaded and processed, we can use the current df to get columns except 'Disease'
    
    # Create a dummy DataFrame to get the feature order
    # This is critical for consistent input to the model
    dummy_df = pd.DataFrame(columns=['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable'])
    # The actual columns should match exactly what the model was trained on.
    # The order of columns in X_train was:
    # ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable']
    feature_columns_order = dummy_df.columns.tolist()

    print("Model and Label Encoders loaded successfully.")
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    best_rf_model = None
    label_encoders = None


@app.route('/predict', methods=['POST'])
def predict():
    if not best_rf_model or not label_encoders:
        return jsonify({'error': 'Model or encoders not loaded. Check server logs.'}), 500

    data = request.get_json(force=True)
    sample_symptoms = data.get('symptoms')

    if not sample_symptoms:
        return jsonify({'error': 'No symptoms provided in the request.'}), 400

    try:
        # Convert sample symptoms to a DataFrame
        sample_df = pd.DataFrame([sample_symptoms])

        # Preprocess the input data using the loaded label_encoders
        encoded_sample = sample_df.copy()
        for column in feature_columns_order: # Iterate through expected feature columns
            if column in encoded_sample.columns:
                le = label_encoders.get(column)
                if le and df[column].dtype == 'object': # Only encode if it was originally an object type and encoder exists
                    # Check if the input value is known by the encoder
                    if encoded_sample[column].iloc[0] in le.classes_:
                        encoded_sample[column] = le.transform(encoded_sample[column])
                    else:
                        # Handle unseen labels: for now, map to a default or raise error
                        # For simplicity, if an unseen categorical value is provided, we'll replace it with a placeholder like -1 or mode
                        # A more robust solution might involve returning an error or using a more sophisticated imputer
                        print(f"Warning: Unseen label '{encoded_sample[column].iloc[0]}' for column '{column}'. Setting to -1.")
                        encoded_sample[column] = -1
            elif column == 'Age': # Age is numerical, ensure it's handled as is
                encoded_sample[column] = pd.to_numeric(encoded_sample[column])
            else:
                # If a required feature column is missing, this is an error in input or processing
                return jsonify({'error': f"Missing or invalid feature: '{column}'"}), 400
        
        # Ensure the order of columns matches the training data
        encoded_sample = encoded_sample[feature_columns_order]

        # Make prediction and get probabilities
        probabilities = best_rf_model.predict_proba(encoded_sample)
        disease_labels_encoded = best_rf_model.classes_
        disease_names = label_encoders['Disease'].inverse_transform(disease_labels_encoded)

        # Create a dictionary of disease probabilities
        probability_dict = dict(zip(disease_names, probabilities[0].tolist()))

        # Sort probabilities in descending order
        sorted_probabilities = sorted(probability_dict.items(), key=lambda item: item[1], reverse=True)

        # Return top N predictions
        N = 10 # Number of top predictions to return
        top_predictions = {disease: prob for disease, prob in sorted_probabilities[:N]}

        return jsonify(top_predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # To run in Colab, you might need ngrok or a similar tool to expose the port
    # For local testing, you can run app.run(debug=True, port=5000)
    # In Colab, often a simple wrapper is used for quick testing or ngrok for external access
    # For this example, we'll provide instructions for typical local execution.
    print("To run this Flask app:\n1. Save the code above to a file named `app.py`.")
    print("2. Open your terminal or command prompt.")
    print("3. Navigate to the directory where `app.py` is saved.")
    print("4. Run the command: `python app.py`")
    print("5. The app will usually run on `http://127.0.0.1:5000/`")
    print("\nOnce running, you can send POST requests to `http://127.0.0.1:5000/predict` with JSON data.")
    print("Example POST request (using curl or Postman/Insomnia):\n")
    print("```bash")
    print("curl -X POST -H \"Content-Type: application/json\" -d '{\"symptoms\": {\"Fever\": \"Yes\", \"Cough\": \"No\", \"Fatigue\": \"Yes\", \"Difficulty Breathing\": \"No\", \"Age\": 45, \"Gender\": \"Male\", \"Blood Pressure\": \"High\", \"Cholesterol Level\": \"Normal\", \"Outcome Variable\": \"Positive\"}}' http://127.0.0.1:5000/predict")
    print("```")
