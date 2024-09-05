from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get the expected input shape
input_shape = input_details[0]['shape']

# Preprocess the image
def preprocess_image(image):
    # Resize the image to the model's input size
    image = image.resize((input_shape[2], input_shape[1]))  # Note: TensorFlow Lite models often expect (height, width)
    image = np.array(image) / 255.0  # Normalize the image
    # Ensure the image has the correct number of channels (e.g., RGB)
    if len(image.shape) == 2:  # Grayscale to RGB conversion if necessary
        image = np.stack([image] * 3, axis=-1)
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
    return image


# Predict function
def predict(image):
    preprocessed_image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction

# Route to render the upload page
@app.route('/')
def upload_page():
    return render_template('upload.html')

# Route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        prediction = predict(image)
        # Apply threshold to get "yes" or "no" result
        result = "yes" if prediction[0][0] >= 0.5 else "no"
        response = {
            'prediction': result
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
