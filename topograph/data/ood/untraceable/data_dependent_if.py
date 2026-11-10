class GatedByMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.alt = nn.Conv2d(3, 8, 1)

    def forward(self, x):
        if x.mean() > 0:
            return torch.relu(self.conv(x))
        return self.alt(x)
