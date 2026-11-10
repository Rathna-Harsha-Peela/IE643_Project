class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, padding=1)

    def forward(self, x):
        y = self.conv(x)
        return (y - y.mean()) / (y.std() + 1e-5)
