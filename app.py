import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables cross-origin requests from your GitHub Pages frontend

# Load the trained model and scaler on startup
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

        # --- STANDARD AI PATHWAY ---
        # Reconstruct feature structure matching training data
        feature_columns = getattr(best_mlp_model, 'feature_names_in_', None)
        
        if feature_columns is not None:
            input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
        else:
            # Fallback to standard columns if feature names aren't embedded
            cols = ['Sleep Duration', 'Age', 'Stress Level', 'Physical Activity Level', 
                    'Heart Rate', 'Daily Steps', 'Systolic Blood Pressure', 
                    'Diastolic Blood Pressure', 'Gender_Female', 'Gender_Male']
            input_data = pd.DataFrame(0, index=[0], columns=cols)

        # Baseline defaults
        for col, val in [('Physical Activity Level', 55), ('Heart Rate', 72), 
                         ('Daily Steps', 6500), ('Systolic Blood Pressure', 120), 
                         ('Diastolic Blood Pressure', 80)]:
            if col in input_data.columns:
                input_data[col] = val

        input_data['Sleep Duration'] = user_hours
        input_data['Age'] = user_age
        input_data['Stress Level'] = user_stress
        
        gender_col = f"Gender_{user_gender}"
        if gender_col in input_data.columns:
            input_data[gender_col] = 1

        input_scaled = scaler.transform(input_data)
        prediction = best_mlp_model.predict(input_scaled)
        ai_quality_score = int(prediction[0])

        return jsonify({"quality": f"{ai_quality_score}/10"})
    
    except Exception as e:
        return jsonify({"quality": f"Error running AI calculation: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
