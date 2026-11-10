class PadCropNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3)
        self.mix = nn.Conv2d(6, 2, 1)

    def forward(self, x):
        y = F.pad(self.conv(x), (1, 1, 1, 1))
        return self.mix(torch.cat([y, x], dim=1))
