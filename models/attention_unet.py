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
    """
        Attention Unet implementation
        Paper: https://arxiv.org/pdf/1804.03999
        """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        n1 = 64  # Initial number of filters
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # Encoder path
        self.down_conv1 = DoubleConv(in_channels, filters[0])
        self.down_sample1 = DownSample()

        self.down_conv2 = DoubleConv(filters[0], filters[1])
        self.down_sample2 = DownSample()

        self.down_conv3 = DoubleConv(filters[1], filters[2])
        self.down_sample3 = DownSample()

        # Bottleneck
        self.middle_conv = DoubleConv(filters[2], filters[3])

        # Decoder path
        self.up_sample1 = UpConv(filters[3], filters[2])
        self.dec1_ag = AttentionGate([filters[2], filters[2]], filters[2])
        self.dec1_conv = DoubleConv(filters[3], filters[2])

        self.up_sample2 = UpConv(filters[2], filters[1])
        self.dec2_ag = AttentionGate([filters[1], filters[1]], filters[1])
        self.dec2_conv = DoubleConv(filters[2], filters[2])

        self.up_sample3 = UpConv(filters[1], filters[0])
        self.dec3_ag = AttentionGate([filters[0], filters[0]], filters[0])
        self.dec3_conv = DoubleConv(filters[1], filters[0])

        # Output layer
        self.output = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.down_conv1(x)
        p1 = self.down_sample1(s1)

        s2 = self.down_conv2(p1)
        p2 = self.down_sample2(s2)

        s3 = self.down_conv3(p2)
        p3 = self.down_sample3(s3)

        # Bottleneck
        b = self.middle_conv(p3)

        # Decoder
        d1 = self.up_sample1(b)
        s3 = self.dec1_ag(d1, s3)
        d1 = torch.cat([d1, s3], dim=1)
        d1 = self.dec1_conv(d1)

        d2 = self.up_sample2(d1)
        s2 = self.dec2_ag(d2, s2)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2_conv(d2)

        d3 = self.up_sample3(d2)
        s1 = self.dec3_ag(d3, s1)
        d3 = torch.cat([d3, s1], dim=1)
        d3 = self.dec3_conv(d3)

        # Output
        output = self.output(d3)
        return output



