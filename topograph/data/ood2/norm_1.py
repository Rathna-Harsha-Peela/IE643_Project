class GNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 3, padding=1)
        self.gn = nn.GroupNorm(2, 8)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x))) + x
