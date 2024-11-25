import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms.functional


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        return self.pool(x)

class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    def forward(self, x: torch.Tensor):
        return self.up(x)


class CropAndConcat(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat([x, contracting_x], dim=1)
        return x

# UNet Architecture
class UNET(nn.Module):

    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/pdf/1505.04597
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        n1 = 64  # Initial number of filters
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # Downsampling layers
        self.down_conv1 = DoubleConv(in_channels, filters[0])
        self.down_sample1 = DownSample()

        self.down_conv2 = DoubleConv(filters[0], filters[1])
        self.down_sample2 = DownSample()

        self.down_conv3 = DoubleConv(filters[1], filters[2])
        self.down_sample3 = DownSample()

        self.down_conv4 = DoubleConv(filters[2], filters[3])
        self.down_sample4 = DownSample()

        # Middle layer
        self.middle_conv = DoubleConv(filters[3], filters[4])

        # Upsampling layers
        self.up_sample1 = UpSample(filters[4], filters[3])
        self.concat1 = CropAndConcat()
        self.up_conv1 = DoubleConv(filters[4], filters[3])

        self.up_sample2 = UpSample(filters[3], filters[2])
        self.concat2 = CropAndConcat()
        self.up_conv2 = DoubleConv(filters[3], filters[2])

        self.up_sample3 = UpSample(filters[2], filters[1])
        self.concat3 = CropAndConcat()
        self.up_conv3 = DoubleConv(filters[2], filters[1])

        self.up_sample4 = UpSample(filters[1], filters[0])
        self.concat4 = CropAndConcat()
        self.up_conv4 = DoubleConv(filters[1], filters[0])

        # Final layer
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):

        # Downsampling path
        pass_through = []

        x = self.down_conv1(x)
        pass_through.append(x)
        x = self.down_sample1(x)

        x = self.down_conv2(x)
        pass_through.append(x)
        x = self.down_sample2(x)

        x = self.down_conv3(x)
        pass_through.append(x)
        x = self.down_sample3(x)

        x = self.down_conv4(x)
        pass_through.append(x)
        x = self.down_sample4(x)

        # Middle layer
        x = self.middle_conv(x)

        # Upsampling path
        x = self.up_sample1(x)
        x = self.concat1(x, pass_through.pop())
        x = self.up_conv1(x)

        x = self.up_sample2(x)
        x = self.concat2(x, pass_through.pop())
        x = self.up_conv2(x)

        x = self.up_sample3(x)
        x = self.concat3(x, pass_through.pop())
        x = self.up_conv3(x)

        x = self.up_sample4(x)
        x = self.concat4(x, pass_through.pop())
        x = self.up_conv4(x)

        # Final layer
        x = self.final_conv(x)
        return x


