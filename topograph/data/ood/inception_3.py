class StemInception(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 12, 3, padding=1)
        self.b_short = nn.Conv2d(12, 6, 1)
        self.b_long1 = nn.Conv2d(12, 6, 1)
        self.b_long2 = nn.Conv2d(6, 6, 3, padding=1)
        self.b_long3 = nn.Conv2d(6, 6, 3, padding=1)
        self.bn = nn.BatchNorm2d(12)
        self.classifier = nn.Conv2d(12, 3, 1)

    def forward(self, x):
        s = torch.relu(self.stem(x))
        short = torch.relu(self.b_short(s))
        long = torch.relu(self.b_long1(s))
        long = torch.relu(self.b_long2(long))
        long = torch.relu(self.b_long3(long))
        y = self.bn(torch.cat([short, long], dim=1))
        return self.classifier(torch.relu(y))
