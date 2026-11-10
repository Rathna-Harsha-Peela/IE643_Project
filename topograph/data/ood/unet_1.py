class UNetLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Conv2d(8, 16, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = nn.Conv2d(24, 8, 3, padding=1)
        self.out = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(self.pool(e1)))
        d1 = torch.cat([self.up(e2), e1], dim=1)
        d1 = torch.relu(self.dec1(d1))
        return self.out(d1)
