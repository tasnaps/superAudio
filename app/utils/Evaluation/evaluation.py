import torch
from app.utils.unetmilesial.unet_model import UNet

# Parameters for the UNet model
n_channels = 1
n_classes = 1
bilinear = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the UNet model
model = UNet(n_channels, n_classes, bilinear).to(device)

# Load the weights from the .pth file
trained_model_path = 'C:/Users/tapio/PycharmProjects/superAudio/app/storage/Epochs/unet_epoch_9.pth'
model.load_state_dict(torch.load(trained_model_path))
model.eval()  # set the model to evaluation mode

# DataLoader for test dataset
# You need to define test_dataset yourself.
# It should be a SpectrogramDatasets of your test data
# test_dataset = YourSpectrogramDataset(your_test_data...)
# test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

for i, (low_quality, high_quality) in enumerate(test_dataloader):
    low_quality = low_quality.to(device)
    high_quality = high_quality.to(device)

    # Forward pass
    outputs = model(low_quality)

    # Compute any evaluation metrics here...

# Print/return evaluation metric results...
