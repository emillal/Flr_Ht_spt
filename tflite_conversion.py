import os
import tensorflow as tf


MODEL_DIR = 'output_plots_and_metrics'  
H5_MODEL_PATH = os.path.join(MODEL_DIR, 'trained_model.h5')  
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'trained_model.tflite')  

# Function to convert .h5 model to .tflite
def convert_h5_to_tflite(h5_model_path, tflite_model_path):
    model = tf.keras.models.load_model(h5_model_path)
    print("Model loaded successfully from:", h5_model_path)
    
    # Converting the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model successfully converted to TensorFlow Lite format and saved as {tflite_model_path}")
    
convert_h5_to_tflite(H5_MODEL_PATH, TFLITE_MODEL_PATH)
