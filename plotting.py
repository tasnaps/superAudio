import matplotlib.pyplot as plt

# Epochs
epochs = range(1, 11)

# Training losses
training_losses = [0.3282, 0.2151, 0.1898, 0.1698, 0.1540, 0.1398, 0.1277, 0.1162, 0.1069, 0.0978]

# Validation losses
validation_losses = [0.3457, 0.2298, 0.2069, 0.1893, 0.1655, 0.1551, 0.1477, 0.1382, 0.1290, 0.1232]

# Plot Training Loss
plt.plot(epochs, training_losses, 'r', label='Training Loss')

# Plot Validation Loss
plt.plot(epochs, validation_losses, 'b', label='Validation Loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
