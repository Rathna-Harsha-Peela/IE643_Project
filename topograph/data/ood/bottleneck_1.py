class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(16, 8, 1)
        self.bn_r = nn.BatchNorm2d(8)
        self.spatial = nn.Conv2d(8, 8, 3, padding=1)
        self.bn_s = nn.BatchNorm2d(8)
        self.expand = nn.Conv2d(8, 32, 1)
        self.bn_e = nn.BatchNorm2d(32)
        self.proj = nn.Conv2d(16, 32, 1)
        self.bn_p = nn.BatchNorm2d(32)

    def forward(self, x):
        out = F.relu(self.bn_r(self.reduce(x)))
        out = F.relu(self.bn_s(self.spatial(out)))
        out = self.bn_e(self.expand(out))
        shortcut = self.bn_p(self.proj(x))
        return F.relu(out + shortcut)
