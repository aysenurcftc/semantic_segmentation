import torch
import torch.nn as nn



class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(x)


class AttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

        self.Ws = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

        self.output = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s


class AttentionUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Encoder path
        self.enc1_conv = DoubleConv(in_channels, 64)
        self.enc1_pool = DownSample()

        self.enc2_conv = DoubleConv(64, 128)
        self.enc2_pool = DownSample()

        self.enc3_conv = DoubleConv(128, 256)
        self.enc3_pool = DownSample()

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder path
        self.dec1_up = UpConv(512, 256)
        self.dec1_ag = AttentionGate([256, 256], 256)
        self.dec1_conv = DoubleConv(512, 256)

        self.dec2_up = UpConv(256, 128)
        self.dec2_ag = AttentionGate([128, 128], 128)
        self.dec2_conv = DoubleConv(256, 128)

        self.dec3_up = UpConv(128, 64)
        self.dec3_ag = AttentionGate([64, 64], 64)
        self.dec3_conv = DoubleConv(128, 64)

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1_conv(x)
        p1 = self.enc1_pool(s1)

        s2 = self.enc2_conv(p1)
        p2 = self.enc2_pool(s2)

        s3 = self.enc3_conv(p2)
        p3 = self.enc3_pool(s3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d1 = self.dec1_up(b)
        s3 = self.dec1_ag(d1, s3)
        d1 = torch.cat([d1, s3], dim=1)
        d1 = self.dec1_conv(d1)

        d2 = self.dec2_up(d1)
        s2 = self.dec2_ag(d2, s2)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2_conv(d2)

        d3 = self.dec3_up(d2)
        s1 = self.dec3_ag(d3, s1)
        d3 = torch.cat([d3, s1], dim=1)
        d3 = self.dec3_conv(d3)

        # Output
        output = self.output(d3)
        return output



if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = AttentionUnet(in_channels=3, out_channels=1)
    output = model(x)
    print("Output shape:", output.shape)