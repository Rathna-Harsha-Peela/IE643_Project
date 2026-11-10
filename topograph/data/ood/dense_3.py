class DenseBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(4)
        self.conv0 = nn.Conv2d(4, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv1 = nn.Conv2d(8, 4, 3, padding=1)
        self.head = nn.Conv2d(12, 2, 1)

    def forward(self, x):
        g0 = self.conv0(torch.relu(self.bn0(x)))
        cat0 = torch.cat([x, g0], dim=1)
        g1 = self.conv1(torch.relu(self.bn1(cat0)))
        cat1 = torch.cat([cat0, g1], dim=1)
        return self.head(cat1)
