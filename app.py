from flask import Flask, jsonify, request
import pickle
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

with open(r'C:\Users\snjch\Desktop\predictive_maintenance_quality_control\models\predictive_model.pkl', 'rb') as f:
    predictive_model = pickle.load(f)

cnn_model = tf.keras.models.load_model(r'C:\Users\snjch\Desktop\predictive_maintenance_quality_control\models\cnn_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    predictive_result = {"prediction": "No predictive features provided.", "confidence": 0.0}
    image_result = {"prediction": "No image data provided.", "confidence": 0.0}

    if 'features' in data:
        print(f"Received features: {data['features']}")
        predictive_features = np.array([data['features']])

        if predictive_features.shape[1] == 6:
           
            predictive_prediction = predictive_model.predict(predictive_features).tolist()
            confidence = float(predictive_prediction[0])  

            if confidence > 0.5:
                predictive_result = {"prediction": "Maintenance Required", "confidence": confidence}
            else:
                predictive_result = {"prediction": "No Maintenance Needed", "confidence": confidence}
        else:
            return jsonify({'error': 'Predictive model requires exactly 6 features.'}), 400

    if 'image' in data:
        try:
            image = np.array(data['image'], dtype=np.uint8)

            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.resize(image, (64, 64))
                image = np.expand_dims(image, axis=0)
                image = image / 255.0

                image_prediction = cnn_model.predict(image)

                confidence = float(image_prediction[0][0]) 

                if confidence > 0.5:
                    image_result = {"prediction": "Defective", "confidence": confidence}
                else:
                    image_result = {"prediction": "Non-Defective", "confidence": confidence}
            else:
                return jsonify({'error': 'Image data must be a 3D array with three channels (RGB).'}), 400
        except Exception as e:
            return jsonify({'error': f'Error processing image data: {str(e)}'}), 500

    return jsonify({
        'predictive_maintenance': predictive_result,
        'quality_control': image_result
    })


if __name__ == '__main__':
    app.run(debug=True)
