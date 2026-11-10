class DownBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(24, 12, 1)
        self.c2 = nn.Conv2d(12, 12, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(12, 48, 1)
        self.sc = nn.Conv2d(24, 48, 1, stride=2)

    def forward(self, feat_map):
        y = F.relu(self.c1(feat_map))
        y = F.relu(self.c2(y))
        y = self.c3(y)
        return F.relu(y + self.sc(feat_map))
