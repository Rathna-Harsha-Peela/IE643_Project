class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(8, 4, 3, padding=1)
        self.l2 = nn.Conv2d(12, 4, 3, padding=1)
        self.l3 = nn.Conv2d(16, 4, 3, padding=1)

    def forward(self, x):
        y1 = torch.relu(self.l1(x))
        c1 = torch.cat([x, y1], dim=1)
        y2 = torch.relu(self.l2(c1))
        c2 = torch.cat([x, y1, y2], dim=1)
        y3 = torch.relu(self.l3(c2))
        return torch.cat([x, y1, y2, y3], dim=1)
