import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv

OUTPUT_DIR = 'output_plots_and_metrics'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        data = np.array(hf['data'])
        power_density = np.array(hf['power_density'])
        switching_activity = np.array(hf['switching_activity'])
        thermal_conductivity = np.array(hf['thermal_conductivity'])
        binary_hotspot = np.array(hf['binary_hotspot'])  # Binary label for hotspot
        num_macros = np.array(hf['num_macros'])
    return data, power_density, switching_activity, thermal_conductivity, binary_hotspot, num_macros

#Function to load and preprocess the data
def load_and_preprocess_data():
    data, power_density, switching_activity, thermal_conductivity, binary_hotspot, num_macros = load_data('floorplan_data.h5')

    # Normalize the features
    scaler = StandardScaler()
    power_density = scaler.fit_transform(power_density.reshape(-1, 1)).reshape(-1, 32, 32, 1)
    switching_activity = scaler.fit_transform(switching_activity.reshape(-1, 1)).reshape(-1, 32, 32, 1)
    thermal_conductivity = scaler.fit_transform(thermal_conductivity.reshape(-1, 1)).reshape(-1, 32, 32, 1)

    features = np.concatenate([data[..., np.newaxis], power_density, switching_activity, thermal_conductivity], axis=-1)

    labels = binary_hotspot

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.3),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.3),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.4),

        Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        Dropout(0.4),
        
        Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        Dropout(0.5),

        Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        Dropout(0.5),
        
        Conv2D(1, (1, 1), activation='sigmoid', padding='same')  # Output (32, 32, 1)
    ])
    
    optimizer = Adam(learning_rate=0.0001)  # Lower learning rate
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = build_model(X_train.shape[1:])  # Use the shape of training data

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    model.save(os.path.join(OUTPUT_DIR, "trained_model.h5"))
    model.save(os.path.join(OUTPUT_DIR, "trained_model.keras"))
    print("Model saved as trained_model.h5")

    return model, history

# Saving to a csv file
def save_accuracies_to_csv(history, filename="training_validation_accuracies.csv"):
    epochs = range(1, len(history.history['accuracy']) + 1)
    train_accuracies = history.history['accuracy']
    val_accuracies = history.history['val_accuracy']
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']

    with open(os.path.join(OUTPUT_DIR, filename), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss'])
        for epoch, train_acc, val_acc, train_loss, val_loss in zip(epochs, train_accuracies, val_accuracies, train_losses, val_losses):
            writer.writerow([epoch, train_acc, val_acc, train_loss, val_loss])

def plot_training_history(history):
    # Ploting Accuracy
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.plot(epochs,history.history['accuracy'], label='Train Accuracy')
    plt.plot(epochs,history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(OUTPUT_DIR, "training_validation_accuracy_plot.png"))
    plt.close()

    # Ploting Loss
    plt.plot(epochs,history.history['loss'], label='Train Loss')
    plt.plot(epochs,history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(OUTPUT_DIR, "training_validation_loss_plot.png"))
    plt.close()


def main():

    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    model, history = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    save_accuracies_to_csv(history)

    plot_training_history(history)

if __name__ == '__main__':
    main()

