class PermuteNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 6, 1)
        self.norm = nn.LayerNorm(6)

    def forward(self, x):
        y = self.conv(x)
        y = y.permute(0, 2, 3, 1)
        y = self.norm(y)
        return y.permute(0, 3, 1, 2)
