import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: compare the generated code to the original paper
class UNetBlock(nn.Module):
    """
    UNetBlock: Basic building block for the U-Net architecture used in the Generator
    It can be used for both downsampling and upsampling paths
    """

    def __init__(self, in_channels, out_channels, downsample=True, use_dropout=False):
        super(UNetBlock, self).__init__()
        # Use Conv2d for downsampling, ConvTranspose2d for upsampling
        self.conv = (
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
            if downsample
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        )
        self.norm = nn.BatchNorm2d(out_channels)  # Normalize the output
        # LeakyReLU for downsampling, ReLU for upsampling
        self.activation = nn.LeakyReLU(0.2) if downsample else nn.ReLU()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(
            0.5
        )  # Used in some upsampling blocks to prevent overfitting

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    """
    Generator: U-Net architecture for image-to-image translation
    U-Net allows for direct connections between layers at the same resolution in the encoder and decoder
    This helps preserve low-level information and improves gradient flow
    """

    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        # Encoder (downsampling) path
        self.down1 = UNetBlock(in_channels, 64, downsample=True)
        self.down2 = UNetBlock(64, 128)
        self.down3 = UNetBlock(128, 256)
        self.down4 = UNetBlock(256, 512)
        self.down5 = UNetBlock(512, 512)
        self.down6 = UNetBlock(512, 512)
        self.down7 = UNetBlock(512, 512)
        self.down8 = UNetBlock(512, 512, use_dropout=True)

        # Decoder (upsampling) path
        self.up1 = UNetBlock(512, 512, downsample=False, use_dropout=True)
        self.up2 = UNetBlock(1024, 512, downsample=False, use_dropout=True)
        self.up3 = UNetBlock(1024, 512, downsample=False)
        self.up4 = UNetBlock(1024, 512, downsample=False)
        self.up5 = UNetBlock(1024, 256, downsample=False)
        self.up6 = UNetBlock(512, 128, downsample=False)
        self.up7 = UNetBlock(256, 64, downsample=False)

        # Final layer to produce the output image
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh(),  # Tanh activation to ensure output is in range [-1, 1]
        )

    def forward(self, x):
        # Encoder path
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder path with skip connections
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))  # Skip connection
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return self.final(torch.cat([u7, d1], 1))

        # 6 channels: 3 for input image, 3 for target/generated image
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, padding=1),  # Output is a patch of predictions
        )

    def forward(self, x, y):
        # x: input image, y: target image or generated image
        return self.model(torch.cat([x, y], 1))


if __name__ == "__main__":
    # Test the Generator
    generator = Generator()
    gen_input = torch.randn(1, 3, 256, 256)
    gen_output = generator(gen_input)
    print(f"Generator output shape: {gen_output.shape}")

    # Test the Discriminator
    discriminator = Discriminator()
    disc_input_x = torch.randn(1, 3, 256, 256)  # Input image
    disc_input_y = torch.randn(1, 3, 256, 256)  # Target or generated image
    disc_output = discriminator(disc_input_x, disc_input_y)
    print(f"Discriminator output shape: {disc_output.shape}")
