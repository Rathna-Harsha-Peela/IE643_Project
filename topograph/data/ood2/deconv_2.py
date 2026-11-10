class DeconvSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(4, 8, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(8, 8, 3, padding=1)
        self.up = nn.ConvTranspose2d(8, 4, 2, stride=2)

    def forward(self, x):
        d = F.relu(self.down(x))
        m = F.relu(self.mid(d)) + d
        return self.up(m) + x
