import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows requests from your GitHub Pages site

# Load model and scaler
MODEL_PATH = 'best_mlp_model.pkl'
SCALER_PATH = 'scaler.pkl'

with open(MODEL_PATH, 'rb') as f:
    best_mlp_model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "sqpAI Backend is running 24/7!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        user_hours = float(data.get("hours", 7.0))
        user_gender = data.get("gender", "Male")
        user_age = int(data.get("age", 30))
        user_stress = int(data.get("stress", 5))

        # 🛑 SAFEGUARD 1: Physical Limits
        if user_hours > 24 or user_hours < 0 or user_age < 0 or user_age > 120:
            return jsonify({"quality": "1/10 (Error: Input values physically impossible)"})

        # 🛑 SAFEGUARD 2: Medical Age-Group Sleep Thresholds
        if user_age <= 1:
            min_healthy, max_healthy = 11.0, 18.0
        elif user_age <= 5:
            min_healthy, max_healthy = 10.0, 14.0
        elif user_age <= 13:
            min_healthy, max_healthy = 9.0, 12.0
        elif user_age <= 17:
            min_healthy, max_healthy = 8.0, 11.0
        else:
            min_healthy, max_healthy = 7.0, 9.5

        if user_hours < min_healthy or user_hours > max_healthy:
            base_score = 7
            deviation = (min_healthy - user_hours) if user_hours < min_healthy else (user_hours - max_healthy)
            calculated_score = base_score - int(deviation * 1.5)
            
            if user_stress >= 9:
                calculated_score -= 2
            elif user_stress >= 7:
                calculated_score -= 1
                
            final_score = max(1, min(10, calculated_score))
            return jsonify({"quality": f"{final_score}/10 (Dynamic Age-Threshold Assessment)"})

        # --- DYNAMIC FEATURE RECONSTRUCTION ---
        # Extract exact feature names the scaler was trained on
        if hasattr(scaler, 'feature_names_in_'):
            expected_cols = list(scaler.feature_names_in_)
        elif hasattr(best_mlp_model, 'feature_names_in_'):
            expected_cols = list(best_mlp_model.feature_names_in_)
        else:
            return jsonify({"quality": "Error: Scaler missing feature_names_in_ metadata."})

        # Create zero-initialized DataFrame with exact training columns
        input_data = pd.DataFrame(0, index=[0], columns=expected_cols)

        # Fill numerical features if present
        if 'Sleep Duration' in input_data.columns: input_data['Sleep Duration'] = user_hours
        if 'Age' in input_data.columns: input_data['Age'] = user_age
        if 'Stress Level' in input_data.columns: input_data['Stress Level'] = user_stress
        if 'Physical Activity Level' in input_data.columns: input_data['Physical Activity Level'] = 55
        if 'Heart Rate' in input_data.columns: input_data['Heart Rate'] = 72
        if 'Daily Steps' in input_data.columns: input_data['Daily Steps'] = 6500
        if 'Systolic Blood Pressure' in input_data.columns: input_data['Systolic Blood Pressure'] = 120
        if 'Diastolic Blood Pressure' in input_data.columns: input_data['Diastolic Blood Pressure'] = 80

        # One-hot encoded categorical matching
        gender_col = f"Gender_{user_gender}"
        if gender_col in input_data.columns:
            input_data[gender_col] = 1

        # Default standard category values for auxiliary encoded features
        for default_cat in ['BMI Category_Normal', 'BMI Category_Normal Weight', 'Occupation_Doctor']:
            if default_cat in input_data.columns:
                input_data[default_cat] = 1
                break

        # Transform features and run prediction
        input_scaled = scaler.transform(input_data)
        prediction = best_mlp_model.predict(input_scaled)
        ai_quality_score = int(prediction[0])

        return jsonify({"quality": f"{ai_quality_score}/10"})
    
    except Exception as e:
        return jsonify({"quality": f"Error running AI calculation: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
