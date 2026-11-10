class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + self.short(x))


class ResNetish(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = Block(3, 12)
        self.b2 = Block(12, 24)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(24, 6)

    def forward(self, x):
        x = self.b2(self.b1(x))
        x = torch.flatten(self.pool(x), 1)
        return self.fc(x)
