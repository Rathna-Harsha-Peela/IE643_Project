class TwoLevelUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Conv2d(4, 8, 3, padding=1)
        self.e2 = nn.Conv2d(8, 16, 3, padding=1)
        self.e3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.up2 = nn.Upsample(scale_factor=2)
        self.d2 = nn.Conv2d(48, 16, 3, padding=1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.d1 = nn.Conv2d(24, 8, 3, padding=1)
        self.final = nn.Conv2d(8, 3, 1)

    def forward(self, x):
        s1 = torch.relu(self.e1(x))
        s2 = torch.relu(self.e2(self.pool1(s1)))
        bott = torch.relu(self.e3(self.pool2(s2)))
        y = torch.relu(self.d2(torch.cat([self.up2(bott), s2], dim=1)))
        y = torch.relu(self.d1(torch.cat([self.up1(y), s1], dim=1)))
        return self.final(y)
