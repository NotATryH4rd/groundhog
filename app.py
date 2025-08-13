from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

#rahhhh
# Load models & scalers
model1 = tf.keras.models.load_model("model1.keras")
model2 = tf.keras.models.load_model("model2.keras")
scaler_X1 = joblib.load("scaler_X1.pkl")
scaler_X2 = joblib.load("scaler_X2.pkl")
target_scaler_1 = joblib.load("target_scaler_1.pkl")
target_scaler_2 = joblib.load("target_scaler_2.pkl")

#features 
input_features_1 = ['Temperature', 'Humidity']
input_features_2 = ['ph', 'EC']
target_features = ['N', 'P', 'K']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    print("--- Predict function called! ---") #debugging
    data = request.get_json()
    print("Received data:", data)

    #check required features
    if not all(key in data for key in input_features_1 + input_features_2):
        return jsonify({"error": f"Missing input features. Required: {input_features_1 + input_features_2}"}), 400

    input_array_1 = np.array([[data[feature] for feature in input_features_1]])
    input_array_2 = np.array([[data[feature] for feature in input_features_2]])
    
    input_scaled_1 = scaler_X1.transform(input_array_1)
    input_scaled_2 = scaler_X2.transform(input_array_2)

    #predict
    prediction_scaled_1 = model1.predict(input_scaled_1)
    prediction_scaled_2 = model2.predict(input_scaled_2)

    prediction_1 = target_scaler_1.inverse_transform(prediction_scaled_1)
    prediction_2 = target_scaler_2.inverse_transform(prediction_scaled_2)

    #average out the 2 predictions
    final_prediction = (prediction_1 + prediction_2) / 2
    final_prediction = final_prediction[0]

    result = {
        "N": round(float(final_prediction[0]), 2),
        "P": round(float(final_prediction[1]), 2),
        "K": round(float(final_prediction[2]), 2)
    }

    return jsonify(result)

if __name__ == '__main__':

    app.run(debug=True)


