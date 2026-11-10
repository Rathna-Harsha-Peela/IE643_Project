class PreActBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        return out + x


class PreActNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_one = PreActBlock(16)
        self.block_two = PreActBlock(16)
        self.bn_final = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.block_two(self.block_one(x))
        return torch.relu(self.bn_final(x))
