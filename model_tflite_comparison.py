
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.lite.python.interpreter import Interpreter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import h5py


def load_data(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        data = np.array(hf['data'])
        power_density = np.array(hf['power_density'])
        switching_activity = np.array(hf['switching_activity'])
        thermal_conductivity = np.array(hf['thermal_conductivity'])
        binary_hotspot = np.array(hf['binary_hotspot'])  
        num_macros = np.array(hf['num_macros'])
    return data, power_density, switching_activity, thermal_conductivity, binary_hotspot, num_macros

# Preprocessing the data for both models
def preprocess_data():
    data, power_density, switching_activity, thermal_conductivity, binary_hotspot, num_macros = load_data('floorplan_data.h5')

    scaler = StandardScaler()
    power_density = scaler.fit_transform(power_density.reshape(-1, 1)).reshape(-1, 32, 32, 1)
    switching_activity = scaler.fit_transform(switching_activity.reshape(-1, 1)).reshape(-1, 32, 32, 1)
    thermal_conductivity = scaler.fit_transform(thermal_conductivity.reshape(-1, 1)).reshape(-1, 32, 32, 1)
    features = np.concatenate([data[..., np.newaxis], power_density, switching_activity, thermal_conductivity], axis=-1)
    labels = binary_hotspot
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def evaluate_keras_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    return test_accuracy

def evaluate_tflite_model(tflite_model_path, X_test, y_test):
    interpreter = Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    X_test_input = X_test.astype(np.float32)
    y_test_input = y_test.astype(np.float32)

    predictions = []

    for i in range(len(X_test_input)):
        interpreter.set_tensor(input_details[0]['index'], X_test_input[i:i+1])
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0][0]) 

    predictions = np.array(predictions)
    accuracy = np.mean((predictions > 0.5) == y_test_input)
    return accuracy

def compare_models(h5_model_path, tflite_model_path):

    X_train, X_test, y_train, y_test = preprocess_data()
    keras_model = load_model(h5_model_path)

    keras_accuracy = evaluate_keras_model(keras_model, X_test, y_test)
    print(f"Keras model accuracy: {keras_accuracy:.4f}")

    tflite_accuracy = evaluate_tflite_model(tflite_model_path, X_test, y_test)
    print(f"TFLite model accuracy: {tflite_accuracy:.4f}")


MODEL_DIR = os.getcwd()  
H5_MODEL_PATH = os.path.join(MODEL_DIR, 'trained_model.h5')
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'trained_model.tflite')

# Running the comparison
compare_models(H5_MODEL_PATH, TFLITE_MODEL_PATH)
