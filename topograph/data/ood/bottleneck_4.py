class GatedBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeeze_conv = nn.Conv2d(32, 8, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.depth_conv = nn.Conv2d(8, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.unsqueeze_conv = nn.Conv2d(8, 64, 1)
        self.shortcut_conv = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        b = F.relu(self.bn1(self.squeeze_conv(x)))
        b = F.relu(self.bn2(self.depth_conv(b)))
        b = torch.sigmoid(self.unsqueeze_conv(b))
        return b * self.shortcut_conv(x)
