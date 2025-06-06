import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data into a pandas DataFrame
data = pd.read_csv('training_validation_accuracies.csv')

# Plot for Accuracy
plt.figure(figsize=(10, 10))
plt.plot(data['Epoch'], data['Training Accuracy'], label='Training Accuracy')
plt.plot(data['Epoch'], data['Validation Accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
# Set the x-axis to show whole numbers
plt.xticks(range(int(data['Epoch'].min()), int(data['Epoch'].max()) + 1))
plt.tight_layout()
# Save the accuracy plot
plt.savefig('accuracy_plot.png')
plt.close()

# Plot for Loss
plt.figure(figsize=(10, 10))
plt.plot(data['Epoch'], data['Training Loss'], label='Training Loss')
plt.plot(data['Epoch'], data['Validation Loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
# Set the x-axis to show whole numbers
plt.xticks(range(int(data['Epoch'].min()), int(data['Epoch'].max()) + 1))
plt.tight_layout()
# Save the loss plot
plt.savefig('loss_plot.png')
plt.close()

