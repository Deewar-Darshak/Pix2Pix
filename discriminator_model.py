import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride,
                      bias=False, padding_mode="reflect" ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

# we concatenate x (i.e. the i/p image) and y (i.e. the o/p image) along the channels
# and then send it in to the discriminator (aka critic) to be judged about how real it is
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]): # 256 -> 26x26
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4,
                      stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]: # we are skipping the first element of features list
                                     # because we already use it in the initial block
            layers.append(CNNBlock(in_channels, feature,
                                   stride=1 if feature == features[-1] else 2),
                          ) #paper did it this way
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, 4, 1, 1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)

def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)

if __name__ == '__main__':
    test()


