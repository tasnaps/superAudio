import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    #Perform manipulations on data using features of pyTorch
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.ReLU()):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.activation(self.batchnorm(self.conv(x)))

class Encoder(nn.Module):
    ##Contracting path of U-Net. Reduce spatian dimensionality, increase feature dimensionality
    #Basicly several ConvBlock units that progressively refine data
    # by repeatedly applying ConvBlock and MaxPool2d processes
    def __init__(self, channels):
        super(Encoder, self).__init__()
        self.enc_blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i+1])
            for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        block_outputs = []
        for block in self.enc_blocks:
            x = block(x)
            print(f'Encoder block output shape: {x.shape}')
            block_outputs.append(x)
            x = self.pool(x)
            print(f'After pooling shape: {x.shape}')
        return block_outputs

class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i + 1], 2, stride=2)
            for i in range(len(channels) - 1)])
        self.dec_blocks = nn.ModuleList([
            ConvBlock(channels[i] * 2, channels[i + 1])
            for i in range(len(channels) - 1)])

    def forward(self, x, enc_features):
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            print(f'Decoder upconv output shape: {x.shape}')
            enc_feat = enc_features[-i - 1]
            print(f'Encoder feature shape before interpolation: {enc_feat.shape}')
            if x.shape[2] != enc_feat.shape[2] or x.shape[3] != enc_feat.shape[3]:
                enc_feat = F.interpolate(enc_feat, size=(x.shape[2], x.shape[3]), mode='nearest')
            print(f'Encoder feature shape after interpolation: {enc_feat.shape}')
            x = torch.cat([x, enc_feat], dim=1)#Erroneus concatenation size.
            print(f'Concatenated shape: {x.shape}')
            x = self.dec_blocks[i](x)
            print(f'After dec_block shape: {x.shape}')
        return x


class UNet(nn.Module):
    #Main worker: apply encoder, run decoder on the output of encode, then final operation
    def __init__(self):
        super(UNet, self).__init__()
        channels = [1, 64, 128, 256, 512]
        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels[::-1])
        self.final_conv = nn.Conv2d(channels[1], 1, 1)

    def forward(self, x):
        enc_features = self.encoder(x)
        dec_output = self.decoder(enc_features[-1], enc_features)
        return self.final_conv(dec_output)


##Troubleshooting
model = UNet()
print(model)

# Example input for testing
input_tensor = torch.randn(1, 1, 256, 256)
output_tensor = model(input_tensor)
print(f'Output shape: {output_tensor.shape}')