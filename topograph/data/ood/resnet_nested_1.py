class BasicBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.layer1 = BasicBlock(16)
        self.layer2 = BasicBlock(16)
        self.head = nn.Conv2d(16, 10, 1)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)
