import torch
from matplotlib import pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from app.utils.unetmilesial.unet_model import UNet

# Define the transformation for test images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = UNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load('unet_epoch_10.pth'))
model.eval()

# Load a test image
test_image_path = 'app/storage/validationDataset/361.png'
test_image = Image.open(test_image_path).convert('RGB')
test_image = transform(test_image).unsqueeze(0).to(device)  # Apply the transform

# Get prediction
with torch.no_grad():
    output = model(test_image)
    output = torch.sigmoid(output)  # Sigmoid to get probabilities
    output = output.squeeze().cpu().numpy()

# Display the image and the mask
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(Image.open(test_image_path))
plt.title('Input Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(output, cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.show()
