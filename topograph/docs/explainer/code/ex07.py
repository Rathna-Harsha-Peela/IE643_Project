class ResUnit(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_a = nn.Conv2d(c, c, 3, padding=1)
        self.conv_b = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, inp):
        h = F.relu(self.conv_a(inp))
        return F.relu(self.conv_b(h) + inp)


class TwoUnitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unit_a = ResUnit(8)
        self.unit_b = ResUnit(8)
        self.final = nn.Conv2d(8, 2, 1)

    def forward(self, feat):
        return self.final(self.unit_b(self.unit_a(feat)))
