class SEBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(24, 12, 1)
        self.conv = nn.Conv2d(12, 12, 3, padding=1)
        self.expand = nn.Conv2d(12, 24, 1)
        self.se_down = nn.Linear(24, 6)
        self.se_up = nn.Linear(6, 24)

    def forward(self, x):
        y = self.expand(torch.relu(self.conv(torch.relu(self.reduce(x)))))
        z = y.mean((2, 3))
        z = torch.sigmoid(self.se_up(torch.relu(self.se_down(z))))
        y = y * z.unsqueeze(-1).unsqueeze(-1)
        return torch.relu(y + x)
