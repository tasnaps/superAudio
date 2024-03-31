import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.ReLU()):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.activation(self.batchnorm(self.conv(x)))

class Encoder(nn.Module):
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
            block_outputs.append(x)
            x = self.pool(x)
        return block_outputs

class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i+1], 2, stride=2)
            for i in range(len(channels)-1)])
        self.dec_blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i+1])
            for i in range(len(channels)-1)])

    def forward(self, x, enc_features):
        for i in range(len(self.dec_blocks)):
            x = self.upconvs[i](x)
            enc_feat = enc_features[-i-2]
            x = torch.cat([x, enc_feat], dim=1)
            x = self.dec_blocks[i](x)
        return x

class UNet(nn.Module):
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