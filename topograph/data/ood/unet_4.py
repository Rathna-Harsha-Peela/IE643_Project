class AddSkipAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Conv2d(2, 10, 3, padding=1)
        self.shrink = nn.AvgPool2d(2)
        self.mid = nn.Conv2d(10, 10, 3, padding=1)
        self.decode = nn.Conv2d(10, 2, 3, padding=1)

    def forward(self, x):
        enc = torch.relu(self.encode(x))
        m = torch.relu(self.mid(self.shrink(enc)))
        m = F.interpolate(m, scale_factor=2.0)
        return self.decode(m + enc)
