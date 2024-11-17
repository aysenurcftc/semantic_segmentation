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
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        n1 = 64  # Initial number of filters
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.down_conv = nn.ModuleList([DoubleConv(i, o) for i, o in
                                       [(in_channels, filters[0]), (filters[0], filters[1]), (filters[1], filters[2]), (filters[2], filters[3])]])


        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])


        self.middle_conv = DoubleConv(filters[3], filters[4])


        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                       [(filters[4], filters[3]), (filters[3], filters[2]), (filters[2], filters[1]), (filters[1], filters[0])]])


        self.up_conv = nn.ModuleList([DoubleConv(i, o) for i, o in
                                     [(filters[4], filters[3]), (filters[3], filters[2]), (filters[2], filters[1]), (filters[1], filters[0])]])


        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])


        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):

        pass_through = []


        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            pass_through.append(x)
            x = self.down_sample[i](x)

        x = self.middle_conv(x)


        for i in range(len(self.up_conv)):

            x = self.up_sample[i](x)
            x = self.concat[i](x, pass_through.pop())
            x = self.up_conv[i](x)

        x = self.final_conv(x)
        return x



