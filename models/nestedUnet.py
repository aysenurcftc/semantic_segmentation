import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MaxPool(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.pool(x)


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    def forward(self, x):
        return self.upsample(x)


class NestedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(NestedUNet, self).__init__()

        n1 = 64  # Initial number of filters
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]


        self.pool = MaxPool(kernel_size=2, stride=2)
        self.Up = Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Define the convolution blocks for the contracting path
        self.conv0_0 = DoubleConv(in_channels, filters[0])
        self.conv1_0 = DoubleConv(filters[0], filters[1])
        self.conv2_0 = DoubleConv(filters[1], filters[2])
        self.conv3_0 = DoubleConv(filters[2], filters[3])
        self.conv4_0 = DoubleConv(filters[3], filters[4])

        # Expanding path (nested convolutional blocks)
        self.conv0_1 = DoubleConv(filters[0] + filters[1], filters[0])
        self.conv1_1 = DoubleConv(filters[1] + filters[2], filters[1])
        self.conv2_1 = DoubleConv(filters[2] + filters[3], filters[2])
        self.conv3_1 = DoubleConv(filters[3] + filters[4], filters[3])

        self.conv0_2 = DoubleConv(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = DoubleConv(filters[1] * 2 + filters[2], filters[1])
        self.conv2_2 = DoubleConv(filters[2] * 2 + filters[3], filters[2])

        self.conv0_3 = DoubleConv(filters[0] * 3 + filters[1], filters[0])
        self.conv1_3 = DoubleConv(filters[1] * 3 + filters[2], filters[1])

        self.conv0_4 = DoubleConv(filters[0] * 4 + filters[1], filters[0])


        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)


    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], dim=1))

        # Final output layer
        output = self.final(x0_4)
        return output