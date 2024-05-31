from app.utils.unetmilesial import UNet  # replace 'your_module_directory' with the directory containing your U-Net code
import torch
n_channels = 3 # number of input channels
n_classes = 1 # number of output classes
bilinear = True  # use bilinear interpolation for upsampling

model = UNet(n_channels, n_classes, bilinear)

# Report the model's architecture
print(model)

# Test forward pass with dummy data
dummy_input = torch.randn(1, n_channels, 224, 224)  # replace 224 with your image size
dummy_output = model(dummy_input)

# Report the output shape
print(dummy_output.shape)
